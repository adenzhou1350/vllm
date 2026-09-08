# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue

from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.core import EngineCoreProc


def make_engine() -> EngineCoreProc:
    engine = EngineCoreProc.__new__(EngineCoreProc)
    engine.output_queue = queue.Queue()
    engine._output_coalesce_intervals = {"request-0": 4}
    engine._pending_coalesced_outputs = {}
    return engine


def make_outputs(*token_ids: int, finished: bool = False) -> EngineCoreOutputs:
    return EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                request_id="request-0",
                new_token_ids=list(token_ids),
                finish_reason=FinishReason.LENGTH if finished else None,
            )
        ]
    )


def test_coalesces_token_outputs_until_interval() -> None:
    engine = make_engine()

    for token_id in range(3):
        engine._enqueue_or_coalesce_output(0, make_outputs(token_id))
        assert engine.output_queue.empty()

    engine._enqueue_or_coalesce_output(0, make_outputs(3))

    client_index, outputs = engine.output_queue.get_nowait()
    assert client_index == 0
    assert outputs.outputs[0].new_token_ids == [0, 1, 2, 3]
    assert outputs.finished_requests is None
    assert engine.output_queue.empty()


def test_flushes_finished_output_before_interval() -> None:
    engine = make_engine()
    engine._enqueue_or_coalesce_output(0, make_outputs(10))
    engine._enqueue_or_coalesce_output(0, make_outputs(11, finished=True))

    _, outputs = engine.output_queue.get_nowait()
    assert outputs.outputs[0].new_token_ids == [10, 11]
    assert outputs.outputs[0].finish_reason == FinishReason.LENGTH
    assert outputs.finished_requests == {"request-0"}
    assert "request-0" not in engine._output_coalesce_intervals


def test_falls_back_and_preserves_order_for_side_channel_output() -> None:
    engine = make_engine()
    engine._enqueue_or_coalesce_output(0, make_outputs(20))
    output_with_logprobs = make_outputs(21)
    output_with_logprobs.outputs[0].new_logprobs = ([[], []], [[], []])

    engine._enqueue_or_coalesce_output(0, output_with_logprobs)

    _, flushed = engine.output_queue.get_nowait()
    _, fallback = engine.output_queue.get_nowait()
    assert flushed.outputs[0].new_token_ids == [20]
    assert fallback.outputs[0].new_token_ids == [21]
    assert "request-0" not in engine._output_coalesce_intervals
