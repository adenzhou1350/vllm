# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest

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
