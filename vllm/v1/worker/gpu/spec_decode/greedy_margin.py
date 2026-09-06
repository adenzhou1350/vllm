# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tie-stable top-1 token and top-2 logit margin for greedy decoding."""

import torch

from vllm.triton_utils import tl, triton

_VOCAB_BLOCK_SIZE = 8192


@triton.jit
def _compute_local_top2_kernel(
    logits_ptr,
    logits_stride,
    local_winner_ptr,
    local_winner_stride,
    local_max_ptr,
    local_max_stride,
    local_second_ptr,
    local_second_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_SIZE)
    token_ids = block_idx * BLOCK_SIZE + offsets
    mask = token_ids < vocab_size

    logits = tl.load(
        logits_ptr + row_idx * logits_stride + token_ids,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    # Match the rejection sampler's global reduction behavior: NaNs are not
    # eligible winners. This also keeps the winner in range for padded blocks.
    logits = tl.where(logits != logits, float("-inf"), logits)
    local_max = tl.max(logits, axis=0)

    # torch.argmax returns the lowest token id on ties. Do not rely on the
    # backend reduction's tie behavior: select the lowest matching offset
    # explicitly, then exclude only that occurrence when computing second max.
    winner_offset = tl.min(
        tl.where(mask & (logits == local_max), offsets, BLOCK_SIZE), axis=0
    )
    local_winner = block_idx * BLOCK_SIZE + winner_offset
    local_second = tl.max(
        tl.where(mask & (offsets != winner_offset), logits, float("-inf")),
        axis=0,
    )

    output_offset = row_idx * local_winner_stride + block_idx
    tl.store(local_winner_ptr + output_offset, local_winner)
    tl.store(
        local_max_ptr + row_idx * local_max_stride + block_idx,
        local_max,
    )
    tl.store(
        local_second_ptr + row_idx * local_second_stride + block_idx,
        local_second,
    )


@triton.jit
def _reduce_global_top2_kernel(
    local_winner_ptr,
    local_winner_stride,
    local_max_ptr,
    local_max_stride,
    local_second_ptr,
    local_second_stride,
    winner_ptr,
    margin_ptr,
    vocab_size,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    block_mask = blocks < vocab_num_blocks

    local_max = tl.load(
        local_max_ptr + row_idx * local_max_stride + blocks,
        mask=block_mask,
        other=float("-inf"),
    )
    local_winner = tl.load(
        local_winner_ptr + row_idx * local_winner_stride + blocks,
        mask=block_mask,
        other=vocab_size,
    ).to(tl.int64)
    global_max = tl.max(local_max, axis=0)
    winner = tl.min(
        tl.where(
            block_mask & (local_max == global_max),
            local_winner,
            vocab_size,
        ),
        axis=0,
    )

    winner_block = winner // BLOCK_SIZE
    local_second = tl.load(
        local_second_ptr + row_idx * local_second_stride + blocks,
        mask=block_mask,
        other=float("-inf"),
    )
    # For the winning block, its second value is the best token after removing
    # the selected winner. Every other block contributes its local maximum.
    global_second = tl.max(
        tl.where(blocks == winner_block, local_second, local_max), axis=0
    )
    margin = global_max - global_second
    # Equal infinities produce NaN on subtraction and represent an ambiguous
    # decision. Report zero so a margin-based policy takes its safe path.
    margin = tl.where(margin == margin, margin, 0.0)

    tl.store(winner_ptr + row_idx, winner)
    tl.store(margin_ptr + row_idx, margin)


def compute_greedy_top2_margin(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return greedy winners and the top-1 minus top-2 logit margin.

    The winner uses the same lowest-index tie rule as ``torch.argmax``. The
    second value excludes only that selected token, so a tied maximum has a
    zero margin. NaNs are treated as negative infinity, matching the existing
    speculative rejection sampler's guarded global argmax reduction.

    Args:
        logits: A two-dimensional CUDA tensor contiguous in its last dimension.

    Returns:
        A pair of ``(winner_ids, margins)`` with shapes ``[num_rows]``. Winner
        ids are int64 and margins are float32.
    """
    assert logits.is_cuda
    assert logits.ndim == 2 and logits.stride(-1) == 1
    num_rows, vocab_size = logits.shape
    assert vocab_size >= 2

    vocab_num_blocks = triton.cdiv(vocab_size, _VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    local_shape = (num_rows, vocab_num_blocks)
    local_winner = torch.empty(local_shape, device=logits.device, dtype=torch.int64)
    local_max = torch.empty(local_shape, device=logits.device, dtype=torch.float32)
    local_second = torch.empty(local_shape, device=logits.device, dtype=torch.float32)

    _compute_local_top2_kernel[(num_rows, vocab_num_blocks)](
        logits,
        logits.stride(0),
        local_winner,
        local_winner.stride(0),
        local_max,
        local_max.stride(0),
        local_second,
        local_second.stride(0),
        vocab_size,
        BLOCK_SIZE=_VOCAB_BLOCK_SIZE,
    )

    winners = torch.empty(num_rows, device=logits.device, dtype=torch.int64)
    margins = torch.empty(num_rows, device=logits.device, dtype=torch.float32)
    _reduce_global_top2_kernel[(num_rows,)](
        local_winner,
        local_winner.stride(0),
        local_max,
        local_max.stride(0),
        local_second,
        local_second.stride(0),
        winners,
        margins,
        vocab_size,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        BLOCK_SIZE=_VOCAB_BLOCK_SIZE,
        num_warps=1,
    )
    return winners, margins
