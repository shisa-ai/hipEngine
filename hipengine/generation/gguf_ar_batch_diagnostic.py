"""Direct prompt-list AR diagnostic; public requests use the resident scheduler."""

from __future__ import annotations

import time
from typing import Any

from hipengine.generation.deadline import raise_if_generation_deadline_expired


def open_ar_serving_slots(
    generator: Any, shared_runner: Any, encoded_prompts: dict[int, list[int]],
    tokenize_ms_by_request: dict[int, float], request: Any,
) -> list[Any]:
    from hipengine.generation.qwen35_gguf import (
        _GGUFARServingSlot, _GGUF_PUBLIC_USE_GEMV_DECODE,
        _resident_session_wmma_prefill_default, _timing_add, _timing_set,
    )

    slots = []
    try:
        for row_index in range(len(request.prompts)):
            raise_if_generation_deadline_expired(request)
            prompt_ids = encoded_prompts[row_index]
            timing = {"tokenize_ms": float(tokenize_ms_by_request.get(row_index, 0.0))}
            session_open_start = time.perf_counter()
            session, session_pool_key, _session_reused = generator._acquire_shared_session(
                shared_runner, pool_name="ar_batch",
                use_wmma_prefill=_resident_session_wmma_prefill_default(),
                use_gemv_decode=_GGUF_PUBLIC_USE_GEMV_DECODE,
            )
            _timing_set(timing, "session_open_ms", session_open_start)
            slot = _GGUFARServingSlot(
                request_id=row_index, prompt_ids=list(prompt_ids), session=session,
                prev_token=0, seq_position=0, generated_ids=[], timing=timing,
                session_pool_key=session_pool_key,
            )
            slots.append(slot)
        if generator._try_prefill_ar_serving_slots_batch(slots, request):
            return slots
        for slot in slots:
            prefill_start = time.perf_counter()
            prefill_result = slot.session.prefill(slot.prompt_ids, return_logits=False)
            _timing_add(slot.timing, "prefill_ms", prefill_start)
            generator._finish_ar_serving_slot_prefill(slot, int(prefill_result.token_id), request)
    except Exception:
        generator._close_ar_serving_slots(slots, reuse=False)
        raise
    return slots


def try_prefill_ar_serving_slots_batch(generator: Any, slots: list[Any], request: Any) -> bool:
    from hipengine.generation.qwen35_gguf import (
        _timing_add_ms, _timing_ms_since,
    )

    if len(slots) <= 1 or not generator._ar_packed_prefill_enabled():
        return False
    chunks = generator._ar_serving_slot_chunks(slots)
    if any(len(chunk) <= 1 for chunk in chunks):
        return False
    for chunk in chunks:
        prefill_batch = getattr(chunk[0].session, "prefill_batch_native", None)
        if not callable(prefill_batch):
            return False
        if any(not slot.prompt_ids for slot in chunk):
            return False
    batch_start = time.perf_counter()
    batch_results_by_request: dict[int, Any] = {}
    chunk_ms_by_request: dict[int, float] = {}
    completed_chunks = 0
    serial_prefill_request_ids: set[int] = set()
    try:
        for chunk in chunks:
            prefill_batch = getattr(chunk[0].session, "prefill_batch_native")
            prompt_ids = [tuple(int(token) for token in slot.prompt_ids) for slot in chunk]
            sessions = [slot.session for slot in chunk]
            chunk_start = time.perf_counter()
            results = prefill_batch(prompt_ids, sessions=sessions, return_logits=False)
            if results is None:
                raise NotImplementedError("GGUF AR packed prefill returned no results")
            batch_results = list(results)
            if len(batch_results) != len(chunk):
                raise RuntimeError(
                    f"GGUF AR packed prefill returned {len(batch_results)} result(s) "
                    f"for {len(chunk)} live slot(s)"
                )
            chunk_ms = _timing_ms_since(chunk_start)
            for slot, result in zip(chunk, batch_results, strict=True):
                batch_results_by_request[int(slot.request_id)] = result
                chunk_ms_by_request[int(slot.request_id)] = chunk_ms
            completed_chunks += 1
    except NotImplementedError:
        if completed_chunks == 0:
            # Nothing was prefilled yet, so the caller's per-slot route can
            # serve the whole request; declining here keeps the packed
            # bound from rejecting a prompt the context admits.
            return False
        # A later width group hit the packed route's context bound. The
        # groups before it are already prefilled, so finish the remaining
        # slots on the strict per-session route instead of failing a
        # request whose prompt is inside the configured context.
        generator._fallback_reasons["packed_prefill_unsupported_k0"] += 1
        for chunk in chunks[completed_chunks:]:
            for slot in chunk:
                result = slot.session.prefill(slot.prompt_ids, return_logits=False)
                batch_results_by_request[int(slot.request_id)] = result
                chunk_ms_by_request[int(slot.request_id)] = _timing_ms_since(batch_start)
                serial_prefill_request_ids.add(int(slot.request_id))
    prefill_ms = _timing_ms_since(batch_start)
    for slot in slots:
        result = batch_results_by_request[int(slot.request_id)]
        _timing_add_ms(slot.timing, "prefill_ms", prefill_ms)
        _timing_add_ms(slot.timing, "prefill_batch_ms", prefill_ms)
        if len(chunks) > 1:
            _timing_add_ms(
                slot.timing, "prefill_batch_chunk_ms",
                float(chunk_ms_by_request.get(int(slot.request_id), 0.0)),
            )
        slot.native_compact_prefill = int(slot.request_id) not in serial_prefill_request_ids
        generator._finish_ar_serving_slot_prefill(slot, int(getattr(result, "token_id")), request)
    return True


def finish_ar_serving_slot_prefill(generator: Any, slot: Any, token_id: int, request: Any) -> None:
    from hipengine.generation.qwen35_gguf import _gguf_finished

    prev_token = int(token_id)
    generated_ids = [prev_token]
    slot.prev_token = prev_token
    slot.seq_position = int(slot.session.position)
    slot.generated_ids = generated_ids
    slot.done = (
        len(generated_ids) >= int(request.max_tokens)
        or _gguf_finished(generated_ids, generator.tokenizer, request)
    )


def try_step_ar_serving_slots_batch(generator: Any, live_slots: list[Any], request: Any) -> bool:
    from hipengine.generation.qwen35_gguf import _gguf_ar_stream_decode_enabled

    if len(live_slots) <= 1:
        return False
    chunks = generator._ar_serving_slot_chunks(live_slots)
    if len(chunks) > 1 and _gguf_ar_stream_decode_enabled():
        streamed = generator._try_step_ar_serving_slot_chunks_streams(chunks, request)
        if streamed:
            return True
    for chunk in chunks:
        if len(chunk) <= 1:
            generator._flush_ar_packed_decode_owners(chunk)
            for slot in chunk:
                generator._step_ar_serving_slot_serial(slot, request)
            continue
        if not generator._step_ar_serving_slot_chunk_packed(chunk, request):
            generator._flush_ar_packed_decode_owners(chunk)
            for slot in chunk:
                generator._step_ar_serving_slot_serial(slot, request)
            continue
    return True


def step_ar_serving_slot_chunk_packed(
    generator: Any, chunk: list[Any], request: Any, *,
    stream: int = 0, record_tokens: bool = True,
) -> list[Any] | None:
    from hipengine.generation.qwen35_gguf import _timing_add_ms, _timing_ms_since

    if len(chunk) <= 1:
        return None
    first_session = chunk[0].session
    step_batch = getattr(first_session, "step_batch_native", None)
    if not callable(step_batch):
        return None
    token_ids = [int(slot.prev_token) for slot in chunk]
    sessions = [slot.session for slot in chunk]
    positions = [int(slot.seq_position) for slot in chunk]
    decode_start = time.perf_counter()
    try:
        if stream:
            batch_result = step_batch(
                token_ids, sessions=sessions, positions=positions,
                return_logits=False, scatter_state=False, stream=int(stream),
            )
        else:
            generator._flush_ar_packed_decode_owners_if_chunk_changed(chunk)
            batch_result = step_batch(
                token_ids, sessions=sessions, positions=positions,
                return_logits=False, scatter_state=False,
            )
    except NotImplementedError:
        return None
    if batch_result is None:
        return None
    step_results = list(batch_result)
    if len(step_results) != len(chunk):
        raise RuntimeError(
            f"GGUF AR native batch decode returned {len(step_results)} result(s) "
            f"for {len(chunk)} live slot(s)"
        )
    if not record_tokens:
        return step_results
    decode_ms = _timing_ms_since(decode_start)
    for slot, step_result in zip(chunk, step_results, strict=True):
        _timing_add_ms(slot.timing, "decode_batch_ms", decode_ms)
        token = int(getattr(step_result, "token_id"))
        generator._record_ar_serving_token(slot, token, request)
        slot.packed_decode_owner = first_session
        slot.native_decode_steps += 1
    return step_results


def try_step_ar_serving_slots_streams(generator: Any, live_slots: list[Any], request: Any) -> bool:
    from hipengine.generation.qwen35_gguf import (
        _gguf_ar_stream_decode_enabled, _timing_add_ms, _timing_ms_since,
    )

    if len(live_slots) <= 1 or not _gguf_ar_stream_decode_enabled():
        return False
    generator._flush_ar_packed_decode_owners(live_slots)
    for slot in live_slots:
        if not callable(getattr(slot.session, "step_async_top1", None)):
            return False
        if not callable(getattr(slot.session, "read_top1_sample", None)):
            return False
        runtime = getattr(slot.session, "runtime", None)
        if not callable(getattr(runtime, "stream_create", None)):
            return False
        if not callable(getattr(runtime, "stream_synchronize", None)):
            return False
        if not callable(getattr(runtime, "stream_destroy", None)):
            return False
    launch_start = time.perf_counter()
    launched = []
    for slot in live_slots:
        if slot.decode_stream == 0:
            slot.decode_stream = int(slot.session.runtime.stream_create(nonblocking=True))
        slot.session.step_async_top1(int(slot.prev_token), position=int(slot.seq_position), stream=int(slot.decode_stream))
        launched.append(slot)
    for slot in launched:
        slot.session.runtime.stream_synchronize(int(slot.decode_stream))
    launch_ms = _timing_ms_since(launch_start)
    for slot in launched:
        result = slot.session.read_top1_sample()
        _timing_add_ms(slot.timing, "decode_stream_batch_ms", launch_ms)
        generator._record_ar_serving_token(slot, int(result.token_id), request)
        slot.native_decode_steps += 1
    return True


def try_step_ar_serving_slot_chunks_streams(generator: Any, chunks: list[list[Any]], request: Any) -> bool:
    import concurrent.futures
    from hipengine.generation.qwen35_gguf import _timing_add_ms, _timing_ms_since

    if len(chunks) <= 1:
        return False
    for chunk in chunks:
        if len(chunk) <= 1:
            return False
        owner_slot = chunk[0]
        step_batch = getattr(owner_slot.session, "step_batch_native", None)
        if not callable(step_batch):
            return False
        runtime = getattr(owner_slot.session, "runtime", None)
        if not callable(getattr(runtime, "stream_create", None)):
            return False
        if not callable(getattr(runtime, "stream_synchronize", None)):
            return False
        if not callable(getattr(runtime, "stream_destroy", None)):
            return False
    for chunk in chunks:
        generator._flush_ar_packed_decode_owners_if_chunk_changed(chunk)
        owner_slot = chunk[0]
        if owner_slot.decode_stream == 0:
            owner_slot.decode_stream = int(owner_slot.session.runtime.stream_create(nonblocking=True))

    def step_chunk(chunk: list[Any]) -> list[Any] | None:
        return generator._step_ar_serving_slot_chunk_packed(
            chunk, request, stream=int(chunk[0].decode_stream), record_tokens=False,
        )

    stream_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [(chunk, pool.submit(step_chunk, chunk)) for chunk in chunks]
        chunk_results = []
        for chunk, future in futures:
            result = future.result()
            if result is None:
                raise RuntimeError("streamed packed AR decode chunk failed after launch")
            chunk_results.append((chunk, result))
    stream_ms = _timing_ms_since(stream_start)
    for chunk, step_results in chunk_results:
        owner_session = chunk[0].session
        for slot, step_result in zip(chunk, step_results, strict=True):
            _timing_add_ms(slot.timing, "decode_batch_ms", stream_ms)
            _timing_add_ms(slot.timing, "decode_stream_chunks_ms", stream_ms)
            token = int(getattr(step_result, "token_id"))
            generator._record_ar_serving_token(slot, token, request)
            slot.packed_decode_owner = owner_session
            slot.native_decode_steps += 1
    return True


def ar_serving_slot_chunks(live_slots: list[Any]) -> list[list[Any]]:
    from hipengine.generation.qwen35_gguf import _GGUF_AR_NATIVE_MAX_SLOTS

    chunks = []
    index = 0
    while index < len(live_slots):
        remaining = len(live_slots) - index
        take = min(_GGUF_AR_NATIVE_MAX_SLOTS, remaining)
        if remaining > _GGUF_AR_NATIVE_MAX_SLOTS and remaining - take == 1:
            take -= 1
        chunks.append(live_slots[index:index + take])
        index += take
    return chunks


def record_ar_serving_token(generator: Any, slot: Any, token_id: int, request: Any) -> None:
    from hipengine.generation.qwen35_gguf import _gguf_finished

    token = int(token_id)
    slot.generated_ids.append(token)
    slot.prev_token = token
    slot.seq_position += 1
    slot.done = (
        len(slot.generated_ids) >= int(request.max_tokens)
        or _gguf_finished(slot.generated_ids, generator.tokenizer, request)
    )


def step_ar_serving_slot_serial(generator: Any, slot: Any, request: Any) -> None:
    from hipengine.generation.qwen35_gguf import _timing_add

    if slot.done:
        return
    generator._flush_ar_packed_decode_owners([slot])
    raise_if_generation_deadline_expired(request)
    decode_start = time.perf_counter()
    step = slot.session.step(slot.prev_token, return_logits=False)
    _timing_add(slot.timing, "decode_ms", decode_start)
    generator._record_ar_serving_token(slot, int(step.token_id), request)
    slot.serial_decode_steps += 1


def run_ar_serving_slots(generator: Any, slots: list[Any], request: Any) -> None:
    from hipengine.generation.qwen35_gguf import (
        _timing_add_ms,
        _timing_ms_since,
    )

    while any(not slot.done for slot in slots):
        live_slots = [slot for slot in slots if not slot.done]
        cycle_start = time.perf_counter()
        handled = False
        if generator._ar_packed_decode_enabled():
            handled = generator._try_step_ar_serving_slots_batch(live_slots, request)
        if not handled:
            handled = generator._try_step_ar_serving_slots_streams(live_slots, request)
        if not handled:
            for slot in live_slots:
                generator._step_ar_serving_slot_serial(slot, request)
        cycle_ms = _timing_ms_since(cycle_start)
        for slot in live_slots:
            _timing_add_ms(slot.timing, "slots_decode_phase_ms", cycle_ms)


def close_ar_serving_slots(generator: Any, slots: list[Any], *, reuse: bool = True) -> None:
    for slot in reversed(slots):
        if slot.decode_stream:
            slot.session.runtime.stream_destroy(int(slot.decode_stream))
            slot.decode_stream = 0
        if reuse:
            generator._release_shared_session(slot.session_pool_key, slot.session)
        else:
            slot.session.close()


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
