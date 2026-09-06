# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for structured-output mask-to-logits mapping."""

import numpy as np

from vllm.v1.worker.gpu.structured_outputs import _build_grammar_mapping


def test_build_grammar_mapping_from_logit_offsets() -> None:
    mapping = _build_grammar_mapping(
        req_ids=["a", "b", "c"],
        grammar_req_ids=["c", "a"],
        cu_num_logits_np=np.asarray([0, 1, 4, 6], dtype=np.int32),
        num_draft_tokens_per_req=None,
        num_bonus_tokens=1,
        mask_stride=16,
    )

    assert mapping == [32, 33, 0]


def test_build_grammar_mapping_from_scheduled_draft_counts() -> None:
    mapping = _build_grammar_mapping(
        req_ids=["a", "b", "c"],
        grammar_req_ids=["b", "c"],
        # Adaptive verification may compact these actual logit offsets. The
        # mapping must continue to use the scheduled draft-token layout.
        cu_num_logits_np=np.asarray([0, 1, 2, 3], dtype=np.int64),
        num_draft_tokens_per_req=np.asarray([2, 0, 3], dtype=np.int32),
        num_bonus_tokens=1,
        mask_stride=16,
    )

    assert mapping == [16, 32, 33, 34, 35]
