"""Direct prompt-list AR diagnostic; public requests use the resident scheduler."""

from __future__ import annotations

import time
from typing import Any

from hipengine.generation.deadline import raise_if_generation_deadline_expired


def generate_ar_serving_slots(generator: Any, shared_runner: Any, request: Any, *, plan: Any):
    # Formatting and telemetry are shared with the resident implementation. Import
    # at invocation so the diagnostic does not create an initialization cycle.
    from hipengine.generation.qwen35_gguf import (
        GenerationOutput,
        _encode_prompt_timed,
        _gguf_finish_details,
        _gguf_last_batch_generation,
        _gguf_telemetry,
        _new_gguf_timing_batch_id,
        _timing_set,
        _with_batch_timing_ownership,
    )

    encoded_prompts: dict[int, list[int]] = {}
    tokenize_ms_by_request: dict[int, float] = {}
    for row_index, prompt in enumerate(request.prompts):
        raise_if_generation_deadline_expired(request)
        prompt_ids, tokenize_ms_by_request[row_index] = _encode_prompt_timed(
            generator.tokenizer,
            prompt,
        )
        if not prompt_ids:
            raise ValueError("GGUF prompt tokenization produced no token IDs")
        encoded_prompts[row_index] = prompt_ids

    slots = []
    try:
        slots = generator._open_ar_serving_slots(
            shared_runner,
            encoded_prompts,
            tokenize_ms_by_request,
            request,
        )
        generator._run_ar_serving_slots(slots, request)
        outputs = []
        prompt_rows_by_request: dict[int, list[int]] = {}
        generated_ids_by_request: dict[int, list[int]] = {}
        token_logprobs_by_request = {}
        for slot in sorted(slots, key=lambda item: item.request_id):
            row_timing = dict(slot.timing)
            generated_ids = list(slot.generated_ids)
            prompt_rows_by_request[slot.request_id] = list(slot.prompt_ids)
            generated_ids_by_request[slot.request_id] = generated_ids
            token_logprobs_by_request[slot.request_id] = []
            decode_text_start = time.perf_counter()
            text = generator.tokenizer.decode(generated_ids)
            _timing_set(row_timing, "decode_text_ms", decode_text_start)
            outputs.append(
                GenerationOutput(
                    text=text,
                    generated_token_ids=generated_ids,
                    finish_details=_gguf_finish_details(generated_ids, generator.tokenizer, request),
                    telemetry=_gguf_telemetry(
                        slot.prompt_ids,
                        generated_ids,
                        request,
                        row_index=slot.request_id,
                        timing=row_timing,
                        execution_path="gguf_packed_ar_server_decode",
                        native_compact_prefill=slot.native_compact_prefill,
                        native_caware_decode=slot.native_decode_steps > 0,
                        serial_decode_fallback=slot.serial_decode_steps > 0,
                        native_sampler_rows=False,
                    ),
                )
            )
        batch_id = _new_gguf_timing_batch_id("ar")
        outputs = _with_batch_timing_ownership(outputs, batch_id=batch_id)
        generator.last_generation_outputs = tuple(outputs)
        native_compact_prefill = bool(slots) and all(slot.native_compact_prefill for slot in slots)
        native_decode_steps = max((slot.native_decode_steps for slot in slots), default=0)
        serial_decode_fallback = any(slot.serial_decode_steps > 0 for slot in slots)
        generator.last_batch_generation = _gguf_last_batch_generation(
            generator.tokenizer,
            request,
            plan,
            prompt_rows_by_request,
            generated_ids_by_request,
            token_logprobs_by_request,
            outputs=generator.last_generation_outputs,
            execution_path="gguf_packed_ar_server_decode",
            native_compact_prefill=native_compact_prefill,
            native_decode_steps=native_decode_steps,
            native_caware_decode=native_decode_steps > 0,
            serial_decode_fallback=serial_decode_fallback,
        )
        generator.last_batch_generation.update(
            {
                "batch_id": batch_id,
                "group_rows": len(outputs),
                "timing_scope": "batch",
                "timing_owner": True,
            }
        )
        generator._close_ar_serving_slots(slots, reuse=True)
        slots = []
        return outputs
    except Exception:
        if slots:
            generator._close_ar_serving_slots(slots, reuse=False)
        raise
