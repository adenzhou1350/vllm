# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.models.deepseek_v4 import compressor as dsv4_compressor
from vllm.models.deepseek_v4.nvidia import model as dsv4_model


class _SM80Capability:
    major = 8


class _AmpereAttention:
    pass


class _AttentionConfig:
    backend = None

    def __init__(self, indexer_kv_dtype: str) -> None:
        self.indexer_kv_dtype = indexer_kv_dtype

    def resolve_indexer_kv_dtype(self, default: str) -> str:
        if self.indexer_kv_dtype == "auto":
            return default
        return self.indexer_kv_dtype


def _config(indexer_kv_dtype: str):
    return SimpleNamespace(
        attention_config=_AttentionConfig(indexer_kv_dtype),
    )


def test_ampere_import_does_not_load_rocm_platform(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delitem(
        sys.modules,
        "vllm.models.deepseek_v4.ampere.ampere_sparse",
        raising=False,
    )
    monkeypatch.delitem(
        sys.modules,
        "vllm.models.deepseek_v4.amd.rocm",
        raising=False,
    )
    monkeypatch.delitem(sys.modules, "vllm.platforms.rocm", raising=False)

    importlib.import_module("vllm.models.deepseek_v4.ampere.ampere_sparse")

    assert "vllm.platforms.rocm" not in sys.modules


@pytest.fixture(autouse=True)
def _force_sm80(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        dsv4_model.current_platform,
        "get_device_capability",
        lambda: _SM80Capability(),
    )
    ampere_module = ModuleType("vllm.models.deepseek_v4.ampere.ampere_sparse")
    ampere_module.DeepseekV4AmpereMLAAttention = _AmpereAttention
    monkeypatch.setitem(
        sys.modules,
        "vllm.models.deepseek_v4.ampere.ampere_sparse",
        ampere_module,
    )


@pytest.mark.parametrize("indexer_kv_dtype", ["auto", "fp8"])
def test_sm80_attention_selection_accepts_fp8_indexer(indexer_kv_dtype: str):
    attention_cls = dsv4_model._select_dsv4_attn_cls(_config(indexer_kv_dtype))

    assert attention_cls is _AmpereAttention


@pytest.mark.parametrize("indexer_kv_dtype", ["bf16", "mxfp4", "nvfp4"])
def test_sm80_attention_selection_rejects_non_fp8_indexer(indexer_kv_dtype: str):
    with pytest.raises(
        ValueError,
        match=r"indexer_kv_dtype=.* is not supported for DeepSeek V4 on SM8x",
    ):
        dsv4_model._select_dsv4_attn_cls(_config(indexer_kv_dtype))


@pytest.mark.parametrize(
    ("is_cuda", "is_supported", "expected"),
    [(True, False, False), (True, True, True), (False, True, False)],
)
def test_dsv4_cutedsl_warmup_requires_cuda_and_supported_arch(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda: bool,
    is_supported: bool,
    expected: bool,
):
    monkeypatch.setattr(
        dsv4_attention.current_platform,
        "is_cuda",
        lambda: is_cuda,
    )
    monkeypatch.setattr(
        dsv4_attention,
        "is_cutedsl_supported",
        lambda: is_supported,
    )

    assert dsv4_attention._use_cutedsl_warmup() is expected


@pytest.mark.parametrize(
    ("head_dim", "is_cuda", "is_supported", "expected"),
    [
        (512, True, False, False),
        (512, True, True, True),
        (512, False, True, False),
        (128, True, True, False),
    ],
)
def test_dsv4_cutedsl_compressor_warmup_matches_dispatch_support(
    monkeypatch: pytest.MonkeyPatch,
    head_dim: int,
    is_cuda: bool,
    is_supported: bool,
    expected: bool,
):
    monkeypatch.setattr(
        dsv4_compressor.current_platform,
        "is_cuda",
        lambda: is_cuda,
    )
    monkeypatch.setattr(
        dsv4_compressor,
        "is_cutedsl_supported",
        lambda: is_supported,
    )

    assert dsv4_compressor._use_cutedsl_compressor(head_dim) is expected
