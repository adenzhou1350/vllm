# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


def test_qwen3_5_packed_gdn_input_projection_scope():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    config = SimpleNamespace(
        quant_config=None,
        lora_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    support = QwenGatedDeltaNetAttention.supports_packed_input_projection

    with (
        patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "current_platform.is_device_capability_family",
            return_value=True,
        ),
    ):
        assert support(config, gqa_interleaved_layout=False)
        assert not support(config, gqa_interleaved_layout=True)

        config.parallel_config.tensor_parallel_size = 2
        assert not support(config, gqa_interleaved_layout=False)
        config.parallel_config.tensor_parallel_size = 1

        config.quant_config = Mock()
        assert not support(config, gqa_interleaved_layout=False)
        config.quant_config = None

        config.lora_config = Mock()
        assert not support(config, gqa_interleaved_layout=False)

    config.lora_config = None
    with patch(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
        "current_platform.is_cuda",
        return_value=False,
    ):
        assert not support(config, gqa_interleaved_layout=False)

    with (
        patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "current_platform.is_cuda",
            return_value=True,
        ),
        patch(
            "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
            "current_platform.is_device_capability_family",
            return_value=False,
        ),
    ):
        assert not support(config, gqa_interleaved_layout=False)


def test_qwen3_5_lm_head_receives_quant_config():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config
    mock_vllm_config.lora_config = None

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5Model") as MockModel,
        patch("vllm.model_executor.models.qwen3_5.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()

        Qwen3_5ForCausalLMBase(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_mtp_lm_head_receives_quant_config():
    from vllm.config import CompilationMode
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.compilation_config.mode = CompilationMode.NONE
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor"),
        patch("vllm.model_executor.models.qwen3_5_mtp.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5_mtp.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3_5MTP(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_packed_gdn_input_projection_mapper():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    source_names = [
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.layers.0.linear_attn.in_proj_z.weight",
        "model.layers.0.linear_attn.in_proj_b.weight",
        "model.layers.0.linear_attn.in_proj_a.weight",
    ]
    weights = [(name, torch.empty(1)) for name in source_names]

    mapped = list(Qwen3_5Model.remap_packed_gdn_input_projections(weights))

    assert [name for name, _ in mapped] == [
        "model.layers.0.linear_attn.in_proj_qkvzba.weight",
    ] * 4
    assert [weight.shard_id for _, weight in mapped] == [(0, 1, 2), 3, 4, 5]


def test_qwen3_5_combined_checkpoint_packed_projection_mapper():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    qkvz = torch.arange(8).reshape(8, 1)
    ba = torch.arange(4).reshape(4, 1)
    weights = [
        ("model.layers.0.linear_attn.in_proj_qkvz.weight", qkvz),
        ("model.layers.0.linear_attn.in_proj_ba.weight", ba),
    ]

    mapped = list(Qwen3_5Model.remap_packed_gdn_input_projections(weights))

    assert [name for name, _ in mapped] == [
        "model.layers.0.linear_attn.in_proj_qkvzba.weight",
    ] * 3
    assert [weight.shard_id for _, weight in mapped] == [(0, 1, 2, 3), 4, 5]
    torch.testing.assert_close(mapped[0][1], qkvz)
    torch.testing.assert_close(mapped[1][1], ba[:2])
    torch.testing.assert_close(mapped[2][1], ba[2:])


def test_qwen3_5_combined_checkpoint_rejects_malformed_ba_projection():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

    weights = [
        ("model.layers.0.linear_attn.in_proj_ba.weight", torch.empty(3, 1)),
    ]

    with pytest.raises(ValueError, match="equally sized b/a shards"):
        list(Qwen3_5Model.remap_packed_gdn_input_projections(weights))
