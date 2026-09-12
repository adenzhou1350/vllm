# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
)


def test_triton_sparse_mla_advertises_only_bf16() -> None:
    assert TritonMLASparseBackend.supported_dtypes == [torch.bfloat16]
    assert TritonMLASparseBackend.supports_dtype(torch.bfloat16)
    assert not TritonMLASparseBackend.supports_dtype(torch.float16)
