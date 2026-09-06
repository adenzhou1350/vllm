# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.greedy_margin import (
    compute_greedy_top2_margin,
)

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for greedy margin tests", allow_module_level=True)


def _reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clean = torch.nan_to_num(logits.float(), nan=float("-inf"))
    winners = clean.argmax(dim=-1)
    without_winner = clean.clone()
    without_winner.scatter_(1, winners[:, None], float("-inf"))
    second = without_winner.max(dim=-1).values
    margins = clean.gather(1, winners[:, None]).squeeze(1) - second
    margins = torch.nan_to_num(margins, nan=0.0)
    return winners, margins


@pytest.mark.parametrize("vocab_size", [2, 8191, 8192, 8193, 248320])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_matches_reference(vocab_size: int, dtype: torch.dtype):
    torch.manual_seed(vocab_size)
    logits = torch.randn(4, vocab_size, device="cuda", dtype=dtype)

    expected_winners, expected_margins = _reference(logits)
    winners, margins = compute_greedy_top2_margin(logits)

    torch.testing.assert_close(winners, expected_winners)
    torch.testing.assert_close(margins, expected_margins)


@pytest.mark.parametrize(
    ("indices", "expected_winner"),
    [
        ((3, 7), 3),
        ((8191, 8192), 8191),
        ((4, 16390), 4),
    ],
)
def test_ties_choose_lowest_token_and_report_zero_margin(
    indices: tuple[int, int], expected_winner: int
):
    logits = torch.full((1, 16391), -4.0, device="cuda", dtype=torch.bfloat16)
    logits[0, indices[0]] = 7.0
    logits[0, indices[1]] = 7.0

    winners, margins = compute_greedy_top2_margin(logits)

    assert winners.item() == expected_winner
    assert margins.item() == 0.0


def test_non_contiguous_rows_are_supported():
    base = torch.randn(4, 32, device="cuda", dtype=torch.float32)
    logits = base[::2]
    assert not logits.is_contiguous()
    assert logits.stride(-1) == 1

    expected_winners, expected_margins = _reference(logits)
    winners, margins = compute_greedy_top2_margin(logits)

    torch.testing.assert_close(winners, expected_winners)
    torch.testing.assert_close(margins, expected_margins)


def test_nan_is_not_an_eligible_winner():
    logits = torch.tensor(
        [[float("nan"), 2.0, 1.5], [float("nan"), float("nan"), float("nan")]],
        device="cuda",
    )

    expected_winners, expected_margins = _reference(logits)
    winners, margins = compute_greedy_top2_margin(logits)

    torch.testing.assert_close(winners, expected_winners)
    torch.testing.assert_close(margins, expected_margins)
