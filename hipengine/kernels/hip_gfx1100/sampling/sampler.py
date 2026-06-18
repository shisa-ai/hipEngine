"""Raw-pointer wrappers for native FP32 logits sampler kernels."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("sampler.hip")
_OUTPUT_NAME = "sampler.so"
_SYMBOL_PROCESSORS = "hipengine_sampler_apply_processors_f32_rows"
_SYMBOL_TEMPERATURE = "hipengine_sampler_temperature_f32_rows_i32"
_SYMBOL_TEMPERATURE_TOP_LOGPROBS = "hipengine_sampler_temperature_top_logprobs_f32_rows_i32"
_SYMBOL_TOPP_TEMPERATURE = "hipengine_sampler_top_p_temperature_f32_rows_i32"
_SYMBOL_TOPK_TEMPERATURE = "hipengine_sampler_topk_temperature_f32_rows_i32"
_ALLOWED_THREADS = {64, 128}
_MAX_TOPK = 64


def plan_sampler_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="sampler",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_sampler(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
    dry_run: bool = False,
    load: bool = True,
    require_cached: bool = False,
) -> ctypes.CDLL | BuildArtifact:
    return build_hip(
        sources=[_SOURCE],
        family="sampler",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def apply_processors_f32_rows(
    logits_f32_ptr: int,
    processed_f32_ptr: int,
    bias_offsets_i32_ptr: int,
    bias_token_ids_i32_ptr: int | None,
    bias_values_f32_ptr: int | None,
    history_offsets_i32_ptr: int,
    history_token_ids_i32_ptr: int | None,
    history_counts_i32_ptr: int | None,
    repetition_penalties_f32_ptr: int,
    presence_penalties_f32_ptr: int,
    frequency_penalties_f32_ptr: int,
    rows: int,
    vocab_size: int,
    *,
    suppress_offsets_i32_ptr: int | None = None,
    suppress_token_ids_i32_ptr: int | None = None,
    min_tokens_i32_ptr: int | None = None,
    eos_token_ids_i32_ptr: int | None = None,
    step_indices_u64_ptr: int | None = None,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Apply sampler logits processors row-wise on FP32 logits.

    ``processed`` receives a finite-clamped copy of ``logits`` followed by
    logit-bias additions, history penalties, and static suppression processors
    in the documented host order: repetition penalty, presence penalty,
    frequency penalty, suppress token ids, then min-token EOS suppression. Bias,
    history, and suppression inputs use CSR-style ``offsets[rows + 1]`` arrays
    with compact token id/value or token id/count payloads.
    """

    _check_rows_vocab(rows, vocab_size)
    _check_threads(threads)
    library = library or build_sampler(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_PROCESSORS)
    fn.argtypes = [
        ctypes.c_void_p,  # logits f32
        ctypes.c_void_p,  # processed f32
        ctypes.c_void_p,  # bias offsets i32 [rows+1]
        ctypes.c_void_p,  # bias token ids i32 (nullable when offsets are empty)
        ctypes.c_void_p,  # bias values f32 (nullable when offsets are empty)
        ctypes.c_void_p,  # history offsets i32 [rows+1]
        ctypes.c_void_p,  # history token ids i32 (nullable when offsets are empty)
        ctypes.c_void_p,  # history counts i32 (nullable when offsets are empty)
        ctypes.c_void_p,  # repetition penalties f32 [rows]
        ctypes.c_void_p,  # presence penalties f32 [rows]
        ctypes.c_void_p,  # frequency penalties f32 [rows]
        ctypes.c_void_p,  # suppress offsets i32 [rows+1] (nullable)
        ctypes.c_void_p,  # suppress token ids i32 (nullable)
        ctypes.c_void_p,  # min tokens i32 [rows] (nullable)
        ctypes.c_void_p,  # eos token ids i32 [rows] (nullable)
        ctypes.c_void_p,  # step indices u64 [rows] (nullable)
        ctypes.c_int64,   # rows
        ctypes.c_int64,   # vocab size
        ctypes.c_int64,   # threads
        ctypes.c_void_p,  # stream
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(processed_f32_ptr),
        ctypes.c_void_p(bias_offsets_i32_ptr),
        ctypes.c_void_p(bias_token_ids_i32_ptr) if bias_token_ids_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(bias_values_f32_ptr) if bias_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(history_offsets_i32_ptr),
        ctypes.c_void_p(history_token_ids_i32_ptr) if history_token_ids_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(history_counts_i32_ptr) if history_counts_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(repetition_penalties_f32_ptr),
        ctypes.c_void_p(presence_penalties_f32_ptr),
        ctypes.c_void_p(frequency_penalties_f32_ptr),
        ctypes.c_void_p(suppress_offsets_i32_ptr) if suppress_offsets_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(suppress_token_ids_i32_ptr) if suppress_token_ids_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(min_tokens_i32_ptr) if min_tokens_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(eos_token_ids_i32_ptr) if eos_token_ids_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(step_indices_u64_ptr) if step_indices_u64_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def sample_temperature_f32_rows_i32(
    logits_f32_ptr: int,
    temperatures_f32_ptr: int,
    row_seeds_u64_ptr: int,
    out_indices_i32_ptr: int,
    out_logprobs_f32_ptr: int | None,
    rows: int,
    vocab_size: int,
    *,
    out_indices_i64_ptr: int | None = None,
    out_values_f32_ptr: int | None = None,
    step_index: int = 0,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Sample one token per row from the full-vocab temperature distribution.

    This is the native sampler shape for public ``top_k=0`` when no top-p/min-p
    filter is active. The cumulative draw scans finite token ids in ascending id
    order; exact host RNG/order equality is not part of this standalone kernel
    contract.
    """

    _check_rows_vocab(rows, vocab_size)
    _check_threads(threads)
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    library = library or build_sampler(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TEMPERATURE)
    fn.argtypes = [
        ctypes.c_void_p,  # logits f32
        ctypes.c_void_p,  # temperatures f32
        ctypes.c_void_p,  # row seeds u64
        ctypes.c_void_p,  # out selected indices i32
        ctypes.c_void_p,  # out selected logprobs f32 (nullable)
        ctypes.c_void_p,  # out selected indices i64 commit (nullable)
        ctypes.c_void_p,  # out selected logits f32 commit (nullable)
        ctypes.c_int64,   # rows
        ctypes.c_int64,   # vocab size
        ctypes.c_uint64,  # step index
        ctypes.c_int64,   # threads
        ctypes.c_void_p,  # stream
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(temperatures_f32_ptr),
        ctypes.c_void_p(row_seeds_u64_ptr),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_void_p(out_logprobs_f32_ptr) if out_logprobs_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_indices_i64_ptr) if out_indices_i64_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(vocab_size),
        ctypes.c_uint64(step_index),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def sample_top_p_temperature_f32_rows_i32(
    logits_f32_ptr: int,
    temperatures_f32_ptr: int,
    top_ps_f32_ptr: int,
    min_ps_f32_ptr: int,
    row_seeds_u64_ptr: int,
    out_indices_i32_ptr: int,
    out_logprobs_f32_ptr: int | None,
    out_candidate_counts_i32_ptr: int | None,
    rows: int,
    vocab_size: int,
    *,
    out_top_indices_i32_ptr: int | None = None,
    out_top_logprobs_f32_ptr: int | None = None,
    top_logprobs: int = 0,
    out_indices_i64_ptr: int | None = None,
    out_values_f32_ptr: int | None = None,
    step_index: int = 0,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Sample from exact full-vocab top-p/min-p filtered temperature rows.

    This correctness-first standalone S7 kernel sorts finite logits by
    descending value with lower-token-id ties, applies exact nucleus/min-p
    retain-one semantics, and writes selected ids/logprobs plus optional retained
    candidate counts. Generation routes supported PARO native-sampler requests
    through this correctness-first kernel; a faster top-p selector remains future
    performance work.
    """

    _check_rows_vocab(rows, vocab_size)
    _check_threads(threads)
    _check_top_logprobs(top_logprobs)
    if (out_top_indices_i32_ptr is None) != (out_top_logprobs_f32_ptr is None):
        raise ValueError("top-logprob output buffers must be provided together")
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    library = library or build_sampler(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TOPP_TEMPERATURE)
    fn.argtypes = [
        ctypes.c_void_p,  # logits f32
        ctypes.c_void_p,  # temperatures f32
        ctypes.c_void_p,  # top_ps f32
        ctypes.c_void_p,  # min_ps f32
        ctypes.c_void_p,  # row seeds u64
        ctypes.c_void_p,  # out selected indices i32
        ctypes.c_void_p,  # out selected logprobs f32 (nullable)
        ctypes.c_void_p,  # out retained counts i32 (nullable)
        ctypes.c_void_p,  # out top indices i32 (nullable)
        ctypes.c_void_p,  # out top logprobs f32 (nullable)
        ctypes.c_void_p,  # out selected indices i64 commit (nullable)
        ctypes.c_void_p,  # out selected logits f32 commit (nullable)
        ctypes.c_int64,   # rows
        ctypes.c_int64,   # vocab size
        ctypes.c_int64,   # top logprobs
        ctypes.c_uint64,  # step index
        ctypes.c_int64,   # threads
        ctypes.c_void_p,  # stream
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(temperatures_f32_ptr),
        ctypes.c_void_p(top_ps_f32_ptr),
        ctypes.c_void_p(min_ps_f32_ptr),
        ctypes.c_void_p(row_seeds_u64_ptr),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_void_p(out_logprobs_f32_ptr) if out_logprobs_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_candidate_counts_i32_ptr) if out_candidate_counts_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_top_indices_i32_ptr) if out_top_indices_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_top_logprobs_f32_ptr) if out_top_logprobs_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_indices_i64_ptr) if out_indices_i64_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(top_logprobs),
        ctypes.c_uint64(step_index),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def sample_temperature_top_logprobs_f32_rows_i32(
    logits_f32_ptr: int,
    temperatures_f32_ptr: int,
    out_top_indices_i32_ptr: int,
    out_top_logprobs_f32_ptr: int,
    rows: int,
    vocab_size: int,
    top_logprobs: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Emit full-vocabulary top-logprob candidates for temperature rows.

    This metadata kernel preserves the selection semantics of
    :func:`sample_temperature_f32_rows_i32`; it only computes the sorted
    OpenAI-style top candidates over the full finite vocabulary distribution.
    """

    _check_rows_vocab(rows, vocab_size)
    _check_top_logprobs(top_logprobs, allow_zero=False)
    _check_threads(threads)
    library = library or build_sampler(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TEMPERATURE_TOP_LOGPROBS)
    fn.argtypes = [
        ctypes.c_void_p,  # logits f32
        ctypes.c_void_p,  # temperatures f32
        ctypes.c_void_p,  # out top indices i32
        ctypes.c_void_p,  # out top logprobs f32
        ctypes.c_int64,   # rows
        ctypes.c_int64,   # vocab size
        ctypes.c_int64,   # top logprobs
        ctypes.c_int64,   # threads
        ctypes.c_void_p,  # stream
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(temperatures_f32_ptr),
        ctypes.c_void_p(out_top_indices_i32_ptr),
        ctypes.c_void_p(out_top_logprobs_f32_ptr),
        ctypes.c_int64(rows),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(top_logprobs),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def sample_topk_temperature_f32_rows_i32(
    logits_f32_ptr: int,
    temperatures_f32_ptr: int,
    row_seeds_u64_ptr: int,
    out_indices_i32_ptr: int,
    out_logprobs_f32_ptr: int | None,
    out_top_indices_i32_ptr: int | None,
    out_top_logprobs_f32_ptr: int | None,
    rows: int,
    vocab_size: int,
    top_k: int,
    *,
    top_ps_f32_ptr: int | None = None,
    min_ps_f32_ptr: int | None = None,
    out_indices_i64_ptr: int | None = None,
    out_values_f32_ptr: int | None = None,
    step_index: int = 0,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Sample one token per row from a bounded top-k temperature distribution.

    ``logits`` is a row-major FP32 ``[rows, vocab_size]`` device buffer.
    ``temperatures`` and ``row_seeds`` are device arrays with one entry per row.
    ``top_k`` is common for the launch and currently limited to ``1 <= k <= 64``;
    Optional ``top_ps`` / ``min_ps`` apply host-order probability filtering over
    the bounded top-k candidate set before the draw. Full-vocab top-p/min-p
    sampling remains a separate native sampler track for public ``top_k=0``.

    Outputs:
    - ``out_indices_i32[rows]`` receives selected token ids.
    - ``out_logprobs_f32[rows]`` receives selected logprobs when non-null.
    - ``out_top_indices_i32[rows, top_k]`` receives sorted candidate ids when non-null.
    - ``out_top_logprobs_f32[rows, top_k]`` receives candidate logprobs when non-null.
    """

    _check_rows_vocab(rows, vocab_size)
    _check_topk(top_k)
    _check_threads(threads)
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    library = library or build_sampler(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, _SYMBOL_TOPK_TEMPERATURE)
    fn.argtypes = [
        ctypes.c_void_p,  # logits f32
        ctypes.c_void_p,  # temperatures f32
        ctypes.c_void_p,  # row seeds u64
        ctypes.c_void_p,  # top ps f32 [rows] (nullable)
        ctypes.c_void_p,  # min ps f32 [rows] (nullable)
        ctypes.c_void_p,  # out selected indices i32
        ctypes.c_void_p,  # out selected logprobs f32 (nullable)
        ctypes.c_void_p,  # out top indices i32 (nullable)
        ctypes.c_void_p,  # out top logprobs f32 (nullable)
        ctypes.c_void_p,  # out selected indices i64 commit (nullable)
        ctypes.c_void_p,  # out selected logits f32 commit (nullable)
        ctypes.c_int64,   # rows
        ctypes.c_int64,   # vocab size
        ctypes.c_int64,   # top k
        ctypes.c_uint64,  # step index
        ctypes.c_int64,   # threads
        ctypes.c_void_p,  # stream
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(logits_f32_ptr),
        ctypes.c_void_p(temperatures_f32_ptr),
        ctypes.c_void_p(row_seeds_u64_ptr),
        ctypes.c_void_p(top_ps_f32_ptr) if top_ps_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(min_ps_f32_ptr) if min_ps_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_indices_i32_ptr),
        ctypes.c_void_p(out_logprobs_f32_ptr) if out_logprobs_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_top_indices_i32_ptr) if out_top_indices_i32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_top_logprobs_f32_ptr) if out_top_logprobs_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_indices_i64_ptr) if out_indices_i64_ptr is not None else ctypes.c_void_p(),
        ctypes.c_void_p(out_values_f32_ptr) if out_values_f32_ptr is not None else ctypes.c_void_p(),
        ctypes.c_int64(rows),
        ctypes.c_int64(vocab_size),
        ctypes.c_int64(top_k),
        ctypes.c_uint64(step_index),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


def register_sampler_kernels(*, replace: bool = True) -> None:
    register(
        KernelKey("hip_gfx1100", "sampler", "f32", "processors_rows"),
        apply_processors_f32_rows,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "sampler", "f32", "temperature_rows_i32"),
        sample_temperature_f32_rows_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "sampler", "f32", "temperature_top_logprobs_rows_i32"),
        sample_temperature_top_logprobs_f32_rows_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "sampler", "f32", "top_p_temperature_rows_i32"),
        sample_top_p_temperature_f32_rows_i32,
        replace=replace,
    )
    register(
        KernelKey("hip_gfx1100", "sampler", "f32", "topk_temperature_rows_i32"),
        sample_topk_temperature_f32_rows_i32,
        replace=replace,
    )


def _check_rows_vocab(rows: int, vocab_size: int) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if vocab_size > 2**31 - 1:
        raise ValueError("vocab_size must fit int32 row-wise sampler outputs")


def _check_topk(top_k: int) -> None:
    if top_k <= 0 or top_k > _MAX_TOPK:
        raise ValueError(f"top_k must be in [1, {_MAX_TOPK}]")


def _check_top_logprobs(top_logprobs: int, *, allow_zero: bool = True) -> None:
    minimum = 0 if allow_zero else 1
    if top_logprobs < minimum or top_logprobs > _MAX_TOPK:
        raise ValueError(f"top_logprobs must be in [{minimum}, {_MAX_TOPK}]")


def _check_threads(threads: int) -> None:
    if threads not in _ALLOWED_THREADS:
        raise ValueError("threads must be one of 64 or 128")


register_sampler_kernels()
