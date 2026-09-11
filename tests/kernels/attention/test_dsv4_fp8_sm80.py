# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exactness of the SM80 manual e4m3fn encode/decode helpers.

Wrong rounding here drifts every KV-cache byte and surfaces as gradual
output corruption, so the encoder must match ``torch.Tensor.to(float8_e4m3fn)``
bit-for-bit on the clamped domain the kernels feed it, and the ALU decoder
must invert every byte exactly.
"""

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    fused_indexer_q_rope_quant,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8_sm80 import (
    _e4m3fn_to_f32_alu,
    _f32_to_e4m3fn_u8,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernels"
)


@triton.jit
def _encode_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, _f32_to_e4m3fn_u8(x), mask=mask)


@triton.jit
def _decode_kernel(u_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    u = tl.load(u_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, _e4m3fn_to_f32_alu(u), mask=mask)


def _encode(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.uint8)
    n = x.numel()
    _encode_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out


def _decode(u: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(u, dtype=torch.float32)
    n = u.numel()
    _decode_kernel[(triton.cdiv(n, 1024),)](u, out, n, BLOCK=1024)
    return out


def _assert_bytes_equal(got: torch.Tensor, ref: torch.Tensor, inputs: torch.Tensor):
    if not torch.equal(got, ref):
        bad = (got != ref).nonzero()[:10].flatten()
        raise AssertionError(
            f"{bad.numel()}+ mismatches, first at inputs "
            f"{inputs.flatten()[bad].tolist()}: got "
            f"{got.flatten()[bad].tolist()} ref {ref.flatten()[bad].tolist()}"
        )


def test_encoder_matches_torch_on_clamped_domain():
    """Kernels always clamp to +-448 before encoding; on that domain the
    manual encoder must equal torch's cast bitwise (RNE incl. ties)."""
    torch.manual_seed(0)
    xs = [
        (torch.rand(1_000_000, device="cuda") * 2 - 1) * 448.0,
        # log-uniform sweep into the subnormal range
        torch.exp2(torch.rand(1_000_000, device="cuda") * 30 - 21)
        * torch.where(torch.rand(1_000_000, device="cuda") > 0.5, 1.0, -1.0),
    ]
    x = torch.cat(xs).float()
    ref = x.to(torch.float8_e4m3fn).view(torch.uint8)
    _assert_bytes_equal(_encode(x), ref, x)


def test_encoder_exact_ties_and_boundaries():
    vals = []
    # every representable value: exact roundtrip required
    all_bytes = torch.arange(256, dtype=torch.uint8, device="cuda")
    finite = all_bytes[(all_bytes & 0x7F) != 0x7F]
    vals.append(finite.view(torch.float8_e4m3fn).float())
    # exact midpoints between adjacent representables (RNE ties-to-even),
    # generated from consecutive finite positives
    pos = torch.sort(vals[0][vals[0] >= 0]).values
    mids = (pos[:-1] + pos[1:]) / 2
    vals += [mids, -mids]
    # boundary/overflow: values at and beyond max finite saturate to +-448
    vals.append(
        torch.tensor(
            [448.0, 449.0, 455.9, 456.0, 464.0, 1e30, -1e30, 0.0, -0.0],
            device="cuda",
        )
    )
    # deep subnormals and half-min-subnormal ties
    vals.append(
        torch.tensor(
            [2**-9, 2**-10, 1.5 * 2**-10, 2**-11, 2**-20, -(2**-10)],
            device="cuda",
        )
    )
    x = torch.cat([v.float().flatten() for v in vals])
    ref = x.to(torch.float8_e4m3fn).view(torch.uint8)
    _assert_bytes_equal(_encode(x), ref, x)


def test_alu_decoder_all_bytes():
    """ALU decode must match torch's fp8->f32 for all 256 bytes, incl. the
    two NaN encodings (NaN in, NaN out -- keeps NaN scrubbing identical)."""
    u = torch.arange(256, dtype=torch.uint8, device="cuda")
    got = _decode(u)
    ref = u.view(torch.float8_e4m3fn).float()
    assert torch.equal(got.nan_to_num(1234.5), ref.nan_to_num(1234.5))
    assert got[(u & 0x7F) == 0x7F].isnan().all()


def test_roundtrip_all_finite_bytes():
    u = torch.arange(256, dtype=torch.uint8, device="cuda")
    finite = u[(u & 0x7F) != 0x7F]
    assert torch.equal(_encode(_decode(finite)), finite)


@torch.inference_mode()
def test_fused_indexer_q_sm80_dispatch_uses_raw_fp8_storage():
    """The high-level SM80 path must not specialize its output pointer as fp8.

    Triton lowers an fp8-typed store before the software encoder can run, which
    fails to compile on SM80 even though the kernel itself emits encoded bytes.
    """
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip("SM80-family dispatch regression")

    torch.manual_seed(11)
    tokens, heads, head_dim, rope_dim = 3, 4, 128, 64
    nope_dim = head_dim - rope_dim
    positions = torch.tensor([1, 7, 15], dtype=torch.int64, device="cuda")
    q = torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(tokens, heads, dtype=torch.bfloat16, device="cuda")
    cache = torch.randn(32, rope_dim, dtype=torch.float32, device="cuda")
    softmax_scale, head_scale = 0.37, 1.25

    actual_q, actual_weights = fused_indexer_q_rope_quant(
        positions,
        q,
        cache,
        weights,
        softmax_scale,
        head_scale,
    )

    qf = q.float()
    cos = cache[positions, : rope_dim // 2][:, None, :]
    sin = cache[positions, rope_dim // 2 :][:, None, :]
    even = qf[..., nope_dim::2]
    odd = qf[..., nope_dim + 1 :: 2]
    rot_even = (even * cos - odd * sin).bfloat16().float()
    rot_odd = (odd * cos + even * sin).bfloat16().float()
    logical = qf.clone()
    logical[..., nope_dim::2] = rot_even
    logical[..., nope_dim + 1 :: 2] = rot_odd
    scale = (
        torch.maximum(logical.abs().amax(dim=-1), torch.tensor(1e-4, device="cuda"))
        / 448.0
    )
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    expected_q = (logical / scale[..., None]).to(torch.float8_e4m3fn)
    expected_weights = weights.float() * scale * softmax_scale * head_scale

    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    torch.testing.assert_close(actual_weights, expected_weights, atol=2e-7, rtol=2e-6)
