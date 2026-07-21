"""Reusable torch-free native sampler workspace over FP32 device logits."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from hipengine.core.dtype import DType
from hipengine.core.hip import HipRuntime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.generation.sampling import (
    RowSamplingState,
    SampleResult,
    SamplingMode,
    active_processor_names,
    normalize_logit_bias_pairs,
    sampler_fast_path_blockers,
    supports_native_gpu_sampling,
)
from hipengine.kernels.hip_gfx1100.sampling import (
    apply_processors_f32_rows,
    sample_temperature_f32_rows_i32,
    sample_temperature_top_logprobs_f32_rows_i32,
    sample_top_p_temperature_f32_rows_i32,
    sample_topk_temperature_f32_rows_i32,
)


class NativeSamplerWorkspace:
    """Own reusable buffers for supported native selection from device logits.

    The model runtime remains responsible for the FP32 logits projection. This
    workspace applies the shared sampler processor order and reads back only the
    selected id/value/logprob plus explicitly requested bounded metadata.
    Unsupported request shapes fail before any row state is mutated.
    """

    def __init__(
        self,
        *,
        runtime: HipRuntime,
        vocab_size: int,
        sampler_library: Any,
    ) -> None:
        if int(vocab_size) <= 0:
            raise ValueError("vocab_size must be positive")
        self.runtime = runtime
        self.vocab_size = int(vocab_size)
        self.sampler_library = sampler_library
        self._buffers: list[DeviceBuffer] = []
        self._named_buffers: dict[str, DeviceBuffer] = {}
        self._cached_uploads: dict[tuple[Any, ...], DeviceBuffer] = {}
        self.closed = False

    def sample(
        self,
        logits_f32_ptr: int,
        params: Any,
        state: RowSamplingState,
        *,
        out_index_i64_ptr: int | None = None,
        out_value_f32_ptr: int | None = None,
        stream: int = 0,
    ) -> SampleResult:
        """Select one supported stochastic row with tiny readback only."""

        return self.sample_rows(
            logits_f32_ptr,
            (params,),
            (state,),
            out_indices_i64_ptr=out_index_i64_ptr,
            out_values_f32_ptr=out_value_f32_ptr,
            stream=stream,
        )[0]

    def sample_rows(
        self,
        logits_f32_ptr: int,
        params_rows: Sequence[Any],
        states: Sequence[RowSamplingState],
        *,
        out_indices_i64_ptr: int | None = None,
        out_values_f32_ptr: int | None = None,
        stream: int = 0,
    ) -> tuple[SampleResult, ...]:
        """Select contiguous rows in one native launch when their shape matches.

        Current sampler kernels use launch-wide ``top_k``, ``top_logprobs``, and
        RNG step index values. Heterogeneous rows fail closed so the caller can
        split them into compatible groups or use the explicit host fallback.
        """

        if self.closed:
            raise RuntimeError("native sampler workspace is closed")
        if int(logits_f32_ptr) == 0:
            raise ValueError("logits_f32_ptr must be non-zero")
        params_tuple = tuple(params_rows)
        state_tuple = tuple(states)
        if not params_tuple:
            raise ValueError("native sampler rows must be non-empty")
        if len(params_tuple) != len(state_tuple):
            raise ValueError("native sampler params and states must align")
        if any(not supports_native_gpu_sampling(params) for params in params_tuple):
            raise NotImplementedError(
                "native sampler request is outside supports_native_gpu_sampling"
            )

        top_ks = {int(getattr(params, "top_k", 0)) for params in params_tuple}
        top_logprobs = {
            int(getattr(params, "top_logprobs", 0)) for params in params_tuple
        }
        step_indices = {int(state.step_index) for state in state_tuple}
        if len(top_ks) != 1 or len(top_logprobs) != 1 or len(step_indices) != 1:
            raise NotImplementedError(
                "native sampler batch requires common top_k, top_logprobs, and step_index"
            )

        for state in state_tuple:
            state.prepare_for_selection()
            if state.has_forced_tokens:
                raise NotImplementedError(
                    "native sampler does not admit dynamic forced-token queues"
                )

        processed_ptr = self._processed_logits_ptr_rows(
            int(logits_f32_ptr),
            params_tuple,
            state_tuple,
            stream=int(stream),
        )
        return self._sample_stochastic_rows(
            processed_ptr,
            params_tuple,
            state_tuple,
            top_k=next(iter(top_ks)),
            requested_top=next(iter(top_logprobs)),
            step_index=next(iter(step_indices)),
            out_indices_i64_ptr=out_indices_i64_ptr,
            out_values_f32_ptr=out_values_f32_ptr,
            stream=int(stream),
        )

    def close(self) -> None:
        if self.closed:
            return
        for buffer in reversed(self._buffers):
            free(buffer, runtime=self.runtime)
        self._buffers.clear()
        self._named_buffers.clear()
        self._cached_uploads.clear()
        self.closed = True

    def _sample_stochastic_rows(
        self,
        logits_ptr: int,
        params_rows: tuple[Any, ...],
        states: tuple[RowSamplingState, ...],
        *,
        top_k: int,
        requested_top: int,
        step_index: int,
        out_indices_i64_ptr: int | None,
        out_values_f32_ptr: int | None,
        stream: int,
    ) -> tuple[SampleResult, ...]:
        rows = len(params_rows)
        temperatures = self._cached_upload(
            (
                "temperatures_f32",
                tuple(float(getattr(params, "temperature", 0.0)) for params in params_rows),
            ),
            np.asarray(
                [float(getattr(params, "temperature", 0.0)) for params in params_rows],
                dtype=np.float32,
            ),
        )
        seeds = self._cached_upload(
            ("row_seeds_u64", tuple(int(state.seed) for state in states)),
            np.asarray(
                [int(state.seed) & ((1 << 64) - 1) for state in states],
                dtype=np.uint64,
            ),
        )
        out_indices = self._buffer(
            "out_indices_i32",
            rows * DType.INT32.itemsize,
        )
        out_logprobs = self._buffer(
            "out_logprobs_f32",
            rows * DType.FP32.itemsize,
        )
        committed_indices = (
            int(out_indices_i64_ptr)
            if out_indices_i64_ptr is not None
            else self._buffer(
                "out_indices_i64",
                rows * DType.INT64.itemsize,
            ).ptr
        )
        committed_values = (
            int(out_values_f32_ptr)
            if out_values_f32_ptr is not None
            else self._buffer(
                "out_values_f32",
                rows * DType.FP32.itemsize,
            ).ptr
        )

        top_ps = np.asarray(
            [float(getattr(params, "top_p", 1.0)) for params in params_rows],
            dtype=np.float32,
        )
        min_ps = np.asarray(
            [float(getattr(params, "min_p", 0.0)) for params in params_rows],
            dtype=np.float32,
        )
        uses_filter = bool(np.any(top_ps < 1.0) or np.any(min_ps > 0.0))
        top_width = top_k if top_k > 0 else requested_top
        out_top_indices = None
        out_top_logprobs = None
        if requested_top > 0:
            out_top_indices = self._buffer(
                "top_indices_i32",
                rows * top_width * DType.INT32.itemsize,
            )
            out_top_logprobs = self._buffer(
                "top_logprobs_f32",
                rows * top_width * DType.FP32.itemsize,
            )

        retained = None
        if top_k > 0:
            top_p_buf = (
                self._cached_upload(("top_ps_f32", top_ps.tobytes()), top_ps)
                if uses_filter
                else None
            )
            min_p_buf = (
                self._cached_upload(("min_ps_f32", min_ps.tobytes()), min_ps)
                if uses_filter
                else None
            )
            sample_topk_temperature_f32_rows_i32(
                logits_ptr,
                temperatures.ptr,
                seeds.ptr,
                out_indices.ptr,
                out_logprobs.ptr,
                None if out_top_indices is None else out_top_indices.ptr,
                None if out_top_logprobs is None else out_top_logprobs.ptr,
                rows,
                self.vocab_size,
                top_k,
                top_ps_f32_ptr=None if top_p_buf is None else top_p_buf.ptr,
                min_ps_f32_ptr=None if min_p_buf is None else min_p_buf.ptr,
                out_indices_i64_ptr=committed_indices,
                out_values_f32_ptr=committed_values,
                step_index=step_index,
                threads=128,
                stream=stream,
                library=self.sampler_library,
                runtime=self.runtime,
            )
        elif uses_filter:
            top_p_buf = self._cached_upload(("top_ps_f32", top_ps.tobytes()), top_ps)
            min_p_buf = self._cached_upload(("min_ps_f32", min_ps.tobytes()), min_ps)
            retained = self._buffer(
                "retained_counts_i32",
                rows * DType.INT32.itemsize,
            )
            sample_top_p_temperature_f32_rows_i32(
                logits_ptr,
                temperatures.ptr,
                top_p_buf.ptr,
                min_p_buf.ptr,
                seeds.ptr,
                out_indices.ptr,
                out_logprobs.ptr,
                retained.ptr,
                rows,
                self.vocab_size,
                out_top_indices_i32_ptr=(
                    None if out_top_indices is None else out_top_indices.ptr
                ),
                out_top_logprobs_f32_ptr=(
                    None if out_top_logprobs is None else out_top_logprobs.ptr
                ),
                top_logprobs=requested_top,
                out_indices_i64_ptr=committed_indices,
                out_values_f32_ptr=committed_values,
                step_index=step_index,
                threads=128,
                stream=stream,
                library=self.sampler_library,
                runtime=self.runtime,
            )
        else:
            sample_temperature_f32_rows_i32(
                logits_ptr,
                temperatures.ptr,
                seeds.ptr,
                out_indices.ptr,
                out_logprobs.ptr,
                rows,
                self.vocab_size,
                out_indices_i64_ptr=committed_indices,
                out_values_f32_ptr=committed_values,
                step_index=step_index,
                threads=128,
                stream=stream,
                library=self.sampler_library,
                runtime=self.runtime,
            )
            if requested_top > 0:
                assert out_top_indices is not None
                assert out_top_logprobs is not None
                sample_temperature_top_logprobs_f32_rows_i32(
                    logits_ptr,
                    temperatures.ptr,
                    out_top_indices.ptr,
                    out_top_logprobs.ptr,
                    rows,
                    self.vocab_size,
                    requested_top,
                    threads=128,
                    stream=stream,
                    library=self.sampler_library,
                    runtime=self.runtime,
                )

        self._synchronize(stream)
        selected_ids = self._read_pointer(
            out_indices.ptr,
            np.int32,
            rows,
        )
        selected_logprobs = self._read_pointer(
            out_logprobs.ptr,
            np.float32,
            rows,
        )
        selected_logits = self._read_pointer(
            committed_values,
            np.float32,
            rows,
        )
        retained_counts = (
            None
            if retained is None
            else self._read_pointer(retained.ptr, np.int32, rows)
        )
        top_ids = (
            None
            if out_top_indices is None
            else self._read_pointer(
                out_top_indices.ptr,
                np.int32,
                rows * top_width,
            ).reshape(rows, top_width)
        )
        top_values = (
            None
            if out_top_logprobs is None
            else self._read_pointer(
                out_top_logprobs.ptr,
                np.float32,
                rows * top_width,
            ).reshape(rows, top_width)
        )

        results: list[SampleResult] = []
        for row, (params, state) in enumerate(zip(params_rows, states, strict=True)):
            token_id = int(selected_ids[row])
            self._validate_token(token_id)
            top_pairs: list[tuple[int, float]] = []
            if requested_top > 0:
                assert top_ids is not None and top_values is not None
                for candidate_id, candidate_logprob in zip(
                    top_ids[row],
                    top_values[row],
                    strict=True,
                ):
                    candidate = int(candidate_id)
                    logprob = float(candidate_logprob)
                    if (
                        candidate < 0
                        or candidate >= self.vocab_size
                        or not np.isfinite(logprob)
                    ):
                        continue
                    top_pairs.append((candidate, logprob))
                    if len(top_pairs) >= requested_top:
                        break
            candidate_count = (
                int(retained_counts[row])
                if retained_counts is not None
                else (
                    min(top_k, self.vocab_size)
                    if top_k > 0
                    else self.vocab_size
                )
            )
            state.observe(token_id)
            results.append(
                SampleResult(
                    token_id=token_id,
                    logit=float(selected_logits[row]),
                    logprob=float(selected_logprobs[row]),
                    mode=SamplingMode.GPU_SAMPLE,
                    candidate_count=max(1, candidate_count),
                    top_logprobs=tuple(top_pairs),
                    active_processors=active_processor_names(params),
                    fast_path_blockers=sampler_fast_path_blockers(params),
                )
            )
        return tuple(results)

    def _processed_logits_ptr_rows(
        self,
        logits_ptr: int,
        params_rows: tuple[Any, ...],
        states: tuple[RowSamplingState, ...],
        *,
        stream: int,
    ) -> int:
        if not any(_needs_processors(params) for params in params_rows):
            return int(logits_ptr)

        rows = len(params_rows)
        processed = self._buffer(
            "processed_logits",
            rows * self.vocab_size * DType.FP32.itemsize,
        )
        bias_rows = tuple(
            normalize_logit_bias_pairs(getattr(params, "logit_bias", None))
            for params in params_rows
        )
        history_rows = tuple(
            tuple(
                (int(token), int(count))
                for token, count in sorted(state.history_counts().items())
                if 0 <= int(token) < self.vocab_size
            )
            for state in states
        )
        suppress_rows = tuple(_suppress_token_ids(params) for params in params_rows)
        for bias_pairs in bias_rows:
            for token_id, _bias in bias_pairs:
                self._validate_token(token_id, field="logit_bias token id")
        for suppress_ids in suppress_rows:
            for token_id in suppress_ids:
                self._validate_token(token_id, field="suppress_token_ids token id")

        min_tokens = np.asarray(
            [int(getattr(params, "min_tokens", 0)) for params in params_rows],
            dtype=np.int32,
        )
        eos_token_ids = np.full((rows,), -1, dtype=np.int32)
        for row, (params, minimum) in enumerate(zip(params_rows, min_tokens, strict=True)):
            if int(minimum) <= 0:
                continue
            raw_eos = getattr(params, "eos_token_id", None)
            if raw_eos is None:
                raise ValueError("min_tokens requires eos_token_id")
            eos_token_ids[row] = int(raw_eos)
            self._validate_token(
                int(eos_token_ids[row]),
                field="eos_token_id",
            )

        bias_offsets, bias_ids, bias_values = _flatten_pairs(bias_rows, np.float32)
        history_offsets, history_ids, history_counts = _flatten_pairs(
            history_rows,
            np.int32,
        )
        suppress_offsets, suppress_ids = _flatten_ids(suppress_rows)
        bias_offsets_buf = self._cached_upload(
            ("bias_offsets_i32", bias_offsets.tobytes()),
            bias_offsets,
        )
        history_offsets_buf = self._upload(
            "history_offsets_i32",
            history_offsets,
        )
        suppress_offsets_buf = self._cached_upload(
            ("suppress_offsets_i32", suppress_offsets.tobytes()),
            suppress_offsets,
        )
        bias_ids_buf = (
            None
            if bias_ids.size == 0
            else self._cached_upload(("bias_ids_i32", bias_ids.tobytes()), bias_ids)
        )
        bias_values_buf = (
            None
            if bias_values.size == 0
            else self._cached_upload(
                ("bias_values_f32", bias_values.tobytes()),
                bias_values,
            )
        )
        history_ids_buf = (
            None
            if history_ids.size == 0
            else self._upload("history_ids_i32", history_ids)
        )
        history_counts_buf = (
            None
            if history_counts.size == 0
            else self._upload("history_counts_i32", history_counts)
        )
        suppress_ids_buf = (
            None
            if suppress_ids.size == 0
            else self._cached_upload(
                ("suppress_ids_i32", suppress_ids.tobytes()),
                suppress_ids,
            )
        )
        repetition = np.asarray(
            [float(getattr(params, "repetition_penalty", 1.0)) for params in params_rows],
            dtype=np.float32,
        )
        presence = np.asarray(
            [float(getattr(params, "presence_penalty", 0.0)) for params in params_rows],
            dtype=np.float32,
        )
        frequency = np.asarray(
            [float(getattr(params, "frequency_penalty", 0.0)) for params in params_rows],
            dtype=np.float32,
        )
        repetition_buf = self._cached_upload(
            ("repetition_f32", repetition.tobytes()),
            repetition,
        )
        presence_buf = self._cached_upload(
            ("presence_f32", presence.tobytes()),
            presence,
        )
        frequency_buf = self._cached_upload(
            ("frequency_f32", frequency.tobytes()),
            frequency,
        )
        has_min_tokens = bool(np.any(min_tokens > 0))
        min_tokens_buf = (
            self._cached_upload(("min_tokens_i32", min_tokens.tobytes()), min_tokens)
            if has_min_tokens
            else None
        )
        eos_buf = (
            self._cached_upload(("eos_token_ids_i32", eos_token_ids.tobytes()), eos_token_ids)
            if has_min_tokens
            else None
        )
        step_indices = np.asarray(
            [int(state.step_index) for state in states],
            dtype=np.uint64,
        )
        step_buf = (
            self._upload("step_indices_u64", step_indices)
            if has_min_tokens
            else None
        )

        apply_processors_f32_rows(
            int(logits_ptr),
            processed.ptr,
            bias_offsets_buf.ptr,
            None if bias_ids_buf is None else bias_ids_buf.ptr,
            None if bias_values_buf is None else bias_values_buf.ptr,
            history_offsets_buf.ptr,
            None if history_ids_buf is None else history_ids_buf.ptr,
            None if history_counts_buf is None else history_counts_buf.ptr,
            repetition_buf.ptr,
            presence_buf.ptr,
            frequency_buf.ptr,
            rows,
            self.vocab_size,
            suppress_offsets_i32_ptr=suppress_offsets_buf.ptr,
            suppress_token_ids_i32_ptr=(
                None if suppress_ids_buf is None else suppress_ids_buf.ptr
            ),
            min_tokens_i32_ptr=(
                None if min_tokens_buf is None else min_tokens_buf.ptr
            ),
            eos_token_ids_i32_ptr=None if eos_buf is None else eos_buf.ptr,
            step_indices_u64_ptr=None if step_buf is None else step_buf.ptr,
            threads=128,
            stream=stream,
            library=self.sampler_library,
            runtime=self.runtime,
        )
        return int(processed.ptr)

    def _alloc(self, nbytes: int) -> DeviceBuffer:
        buffer = malloc(max(4, int(nbytes)), runtime=self.runtime)
        self._buffers.append(buffer)
        return buffer

    def _buffer(self, name: str, nbytes: int) -> DeviceBuffer:
        required = max(4, int(nbytes))
        current = self._named_buffers.get(name)
        if current is None or int(current.nbytes) < required:
            current = self._alloc(required)
            self._named_buffers[name] = current
        return current

    def _upload(self, name: str, array: np.ndarray) -> DeviceBuffer:
        host = np.ascontiguousarray(array)
        buffer = self._buffer(name, int(host.nbytes))
        copy_host_to_device(
            buffer,
            host_array_ptr(host),
            int(host.nbytes),
            runtime=self.runtime,
        )
        return buffer

    def _cached_upload(
        self,
        prefix: tuple[Any, ...],
        array: np.ndarray,
    ) -> DeviceBuffer:
        host = np.ascontiguousarray(array)
        key = (
            *prefix,
            str(host.dtype),
            tuple(int(dim) for dim in host.shape),
            host.tobytes(),
        )
        buffer = self._cached_uploads.get(key)
        if buffer is None:
            buffer = self._alloc(int(host.nbytes))
            copy_host_to_device(
                buffer,
                host_array_ptr(host),
                int(host.nbytes),
                runtime=self.runtime,
            )
            self._cached_uploads[key] = buffer
        return buffer

    def _read_pointer(
        self,
        ptr: int,
        dtype: Any,
        count: int,
    ) -> np.ndarray:
        host = np.empty((int(count),), dtype=dtype)
        copy_device_to_host(
            host_array_ptr(host),
            DeviceBuffer(int(ptr), int(host.nbytes)),
            int(host.nbytes),
            runtime=self.runtime,
        )
        return host

    def _synchronize(self, stream: int) -> None:
        if int(stream):
            self.runtime.stream_synchronize(int(stream))
        else:
            self.runtime.device_synchronize()

    def _validate_token(self, token_id: int, *, field: str = "token id") -> None:
        token = int(token_id)
        if token < 0 or token >= self.vocab_size:
            raise ValueError(
                f"{field} {token} is outside vocab size {self.vocab_size}"
            )


def _flatten_pairs(
    rows: Sequence[Sequence[tuple[int, Any]]],
    value_dtype: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = [0]
    token_ids: list[int] = []
    values: list[Any] = []
    for pairs in rows:
        token_ids.extend(int(token_id) for token_id, _value in pairs)
        values.extend(value for _token_id, value in pairs)
        offsets.append(len(token_ids))
    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(token_ids, dtype=np.int32),
        np.asarray(values, dtype=value_dtype),
    )


def _flatten_ids(
    rows: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    token_ids: list[int] = []
    for row in rows:
        token_ids.extend(int(token_id) for token_id in row)
        offsets.append(len(token_ids))
    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(token_ids, dtype=np.int32),
    )


def _suppress_token_ids(params: Any) -> tuple[int, ...]:
    raw_ids = getattr(params, "suppress_token_ids", None)
    if raw_ids is None:
        raw_ids = getattr(params, "suppress_tokens", ())
    return tuple(int(token) for token in (raw_ids or ()))


def _needs_processors(params: Any) -> bool:
    return bool(
        normalize_logit_bias_pairs(getattr(params, "logit_bias", None))
        or _suppress_token_ids(params)
        or int(getattr(params, "min_tokens", 0)) > 0
        or float(getattr(params, "repetition_penalty", 1.0)) != 1.0
        or float(getattr(params, "presence_penalty", 0.0)) != 0.0
        or float(getattr(params, "frequency_penalty", 0.0)) != 0.0
    )
