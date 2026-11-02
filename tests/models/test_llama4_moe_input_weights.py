# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input router weights must precede the nonlinear Llama4 expert MLP."""

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Llama4ForCausalLM, Llama4TextConfig

from vllm import SamplingParams
from vllm.model_executor.models.llama4 import Llama4MoE
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability_family(100)
    or not has_flashinfer_trtllm_fused_moe(),
    reason="Requires FlashInfer TRTLLM BF16 experts on Blackwell",
)


@pytest.fixture
def tiny_llama4_path(tmp_path):
    config = Llama4TextConfig(
        vocab_size=64,
        hidden_size=256,
        intermediate_size=256,
        intermediate_size_mlp=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        num_local_experts=8,
        num_experts_per_tok=1,
        interleave_moe_layer_step=2,
        moe_layers=[1],
        max_position_embeddings=128,
        no_rope_layers=[1, 0],
        attn_temperature_tuning=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = Llama4ForCausalLM(config)
        # Some tiny checkpoints have zero routed weights and cannot expose this bug.
        for name, parameter in model.named_parameters():
            if ".experts." in name:
                torch.nn.init.normal_(parameter, std=0.02)
        model.save_pretrained(tmp_path)
    return str(tmp_path)


class RouterWeightProbe:
    def capture_router_weights(self):
        self.router_weight_captures: dict[str, dict[str, torch.Tensor]] = {}
        self.router_weight_handles = []
        for name, layer in self.get_model().named_modules():  # type: ignore[attr-defined]
            if not isinstance(layer, Llama4MoE):
                continue
            kernel = layer.experts.routed_experts.quant_method.moe_kernel
            assert type(kernel.fused_experts).__name__ in {
                "TrtLlmBf16ExpertsMonolithic",
                "TrtLlmBf16ExpertsModular",
            }
            captured = self.router_weight_captures[name] = {}

            def before(module, args, kwargs, captured=captured):
                if not captured:
                    captured["input"] = kwargs["hidden_states"].detach().clone()
                    captured["logits"] = kwargs["router_logits"].detach().clone()

            def after(module, args, kwargs, output, captured=captured):
                if "output" not in captured:
                    captured["output"] = output.detach().clone()

            self.router_weight_handles.extend(
                [
                    layer.experts.register_forward_pre_hook(before, with_kwargs=True),
                    layer.experts.register_forward_hook(after, with_kwargs=True),
                ]
            )

    def check_router_weights(self, checkpoint):
        for handle in self.router_weight_handles:
            handle.remove()
        assert len(self.router_weight_captures) == 1
        with safe_open(checkpoint, framework="pt", device="cpu") as weights:
            for name, captured in self.router_weight_captures.items():
                x = captured["input"].float()
                scores, ids = captured["logits"].float().max(dim=-1)
                gates = scores.sigmoid()

                def weight(suffix, name=name, device=x.device):
                    return (
                        weights.get_tensor(name + suffix)
                        .to(device=device, dtype=torch.bfloat16)
                        .float()
                    )

                w1 = weight(".experts.gate_up_proj").transpose(-1, -2)
                w2 = weight(".experts.down_proj").transpose(-1, -2)
                routed = torch.empty_like(x)
                for expert in ids.unique().tolist():
                    selected = ids == expert
                    a = x[selected] * gates[selected, None]
                    gate, up = F.linear(a, w1[expert]).chunk(2, dim=-1)
                    routed[selected] = F.linear(F.silu(gate) * up, w2[expert])
                shared = F.linear(
                    F.silu(F.linear(x, weight(".shared_expert.gate_proj.weight")))
                    * F.linear(x, weight(".shared_expert.up_proj.weight")),
                    weight(".shared_expert.down_proj.weight"),
                )
                expected = routed + shared
                assert routed.norm() > 1e-6
                error = (captured["output"].float() - expected).norm() / expected.norm()
                assert torch.isfinite(error) and error <= 0.05, error.item()


def test_llama4_bf16_router_weights(vllm_runner, tiny_llama4_path):
    with vllm_runner(
        tiny_llama4_path,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        kv_cache_memory_bytes=64 * 1024**2,
        gpu_memory_utilization=0.05,
        enforce_eager=True,
        skip_tokenizer_init=True,
        attention_backend="TRITON_ATTN",
        moe_backend="flashinfer_trtllm",
        worker_extension_cls=f"{__name__}.RouterWeightProbe",
    ) as runner:
        runner.collective_rpc("capture_router_weights")
        output = runner.llm.generate(
            [
                {"prompt_token_ids": list(range(1, 9))},
                {"prompt_token_ids": list(range(11, 19))},
            ],
            SamplingParams(temperature=0, max_tokens=1, detokenize=False),
            use_tqdm=False,
        )
        assert len(output) == 2 and all(
            len(x.outputs[0].token_ids) == 1 for x in output
        )
        runner.collective_rpc(
            "check_router_weights", args=(tiny_llama4_path + "/model.safetensors",)
        )
