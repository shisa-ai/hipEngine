"""Raw GGUF Q8_0/Q5_K/Q6_K GEMV and exact tiled-prefill wrappers."""

from __future__ import annotations

import ctypes
from pathlib import Path

from hipengine.core.build import BuildArtifact, ProfileName, build_hip, plan_hip_build
from hipengine.core.hip import HIP_SUCCESS, HipRuntime, get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

_SOURCE = Path(__file__).with_name("gguf_k_gemv.hip")

# Cached configured extern-C handles. Setting ``fn.argtypes`` and rebuilding
# ``ctypes.c_void_p``/``c_int64`` per call dominates the host launch cost
# (~88 us full vs ~1.4 us lean; see WORKLOG 2026-06-28). Configure each
# (library, symbol) once and call with raw ints thereafter.
_VOID = ctypes.c_void_p
_I64 = ctypes.c_int64
_CACHED_FNS: dict[tuple[int, str], ctypes._CFuncPtr] = {}


def _cached_fn(library: ctypes.CDLL, symbol: str, argtypes: list) -> ctypes._CFuncPtr:
    key = (id(library), symbol)
    fn = _CACHED_FNS.get(key)
    if fn is None:
        fn = getattr(library, symbol)
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        _CACHED_FNS[key] = fn
    return fn

_OUTPUT_NAME = "gguf_k_gemv.so"
_ALLOWED_THREADS = {64, 128, 256}
_QTYPE_BLOCK_SIZE = {"gguf_q8_0": 32, "gguf_q5_k": 256, "gguf_q6_k": 256}
_MIXED_ATTENTION_VARIANT = "mixed_pack8_gemv_decode_bf16_f32_out"
_MIXED_ATTENTION_Q6_FIXED_META_VARIANT = (
    "mixed_q6_fixed_meta_pack8_gemv_decode_bf16_f32_out"
)
_MIXED_ATTENTION_LOCAL32_FIXED_META_VARIANT = (
    "mixed_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out"
)
_MIXED_ATTENTION_PAIR_REUSE_LOCAL32_FIXED_META_VARIANT = (
    "mixed_pair_reuse_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out"
)
_MIXED_ATTENTION_LOCAL32_Q5_SWAR_PAIR_FIXED_META_VARIANT = (
    "mixed_local32_q5_swar_pair_fixed_meta_gemv_decode_bf16_f32_out"
)
_MIXED_ATTENTION_Q5_QG_QUANT = (
    "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k"
)
_MIXED_ATTENTION_Q6_QG_Q8_KV_QUANT = (
    "gguf_q6_k+gguf_q8_0+gguf_q8_0+gguf_q6_k"
)


def plan_gguf_k_gemv_build(
    *,
    cache_root: str | Path | None = None,
    compiler_version: str | None = None,
    profile: ProfileName = "decode",
) -> BuildArtifact:
    return plan_hip_build(
        sources=[_SOURCE],
        family="gguf_k_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
    )


def build_gguf_k_gemv(
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
        family="gguf_k_gemv",
        profile=profile,
        cache_root=cache_root,
        compiler_version=compiler_version,
        output_name=_OUTPUT_NAME,
        dry_run=dry_run,
        load=load,
        require_cached=require_cached,
    )


def _make_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch(quant, symbol, *args, **kwargs)

    return wrapper


def _validate_h7c_raw_q6_role(
    output_dtype: str,
    rows: int,
    in_features: int,
    out_features: int,
    threads: int,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if output_dtype == "bf16":
        if in_features not in {9_216, 12_288}:
            raise ValueError("in_features must be exactly 9216 or 12288")
        if out_features != 3_072:
            raise ValueError("out_features must be exactly 3072")
    elif output_dtype == "f32":
        if in_features != 3_072:
            raise ValueError("in_features must be exactly 3072")
        if out_features != 9_216:
            raise ValueError("out_features must be exactly 9216")
    else:
        raise ValueError("H7C output dtype must be 'bf16' or 'f32'")
    if threads != 128:
        raise ValueError("threads must be exactly 128")


def _launch_h7c_raw_q6(
    output_dtype: str,
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_h7c_raw_q6_role(
        output_dtype,
        rows,
        in_features,
        out_features,
        threads,
    )
    _launch(
        "gguf_q6_k",
        symbol,
        x_ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _validate_h7i_raw_q6_full_group_role(
    output_dtype: str,
    rows: int,
    in_features: int,
    out_features: int,
    threads: int,
) -> None:
    if rows != 512:
        raise ValueError("rows must be exactly 512")
    _validate_h7c_raw_q6_role(
        output_dtype,
        rows,
        in_features,
        out_features,
        threads,
    )


def _launch_h7i_raw_q6_full_group(
    output_dtype: str,
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _validate_h7i_raw_q6_full_group_role(
        output_dtype,
        rows,
        in_features,
        out_features,
        threads,
    )
    _launch(
        "gguf_q6_k",
        symbol,
        x_ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def _make_selected_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch_selected(quant, symbol, *args, **kwargs)

    return wrapper


def _make_selected_silu_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch_selected_silu(quant, symbol, *args, **kwargs)

    return wrapper


def _make_dual_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch_dual(quant, symbol, *args, **kwargs)

    return wrapper


def _make_dual_pack8_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch_dual(quant, symbol, *args, require_pack8=True, **kwargs)

    return wrapper


def _make_unequal_dual_pack8_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch_unequal_dual(quant, symbol, *args, require_pack8=True, **kwargs)

    return wrapper


def _make_wave32x2_wrapper(symbol: str, *, require_non_null: bool = False):
    def wrapper(*args, **kwargs) -> None:
        kwargs.setdefault("threads", 32)
        _launch_wave32x2(
            symbol, *args, require_non_null=require_non_null, **kwargs
        )

    return wrapper


def _make_unequal_wave32x2_wrapper(
    symbol: str,
    *,
    require_non_null: bool = False,
):
    def wrapper(*args, **kwargs) -> None:
        kwargs.setdefault("threads", 32)
        _launch_unequal_wave32x2(
            symbol, *args, require_non_null=require_non_null, **kwargs
        )

    return wrapper


def _make_pack8_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch(quant, symbol, *args, require_pack8=True, **kwargs)

    return wrapper


def _make_mixed_attention_wrapper(
    symbol: str,
    primary_roles: tuple[int, int],
    *,
    require_non_null: bool = False,
    require_primary_total_at_least_secondary: bool = False,
):
    def wrapper(*args, **kwargs) -> None:
        _launch_mixed_attention(
            symbol,
            *args,
            primary_roles=primary_roles,
            require_non_null=require_non_null,
            require_primary_total_at_least_secondary=require_primary_total_at_least_secondary,
            **kwargs,
        )

    return wrapper


def _make_selected_pack8_wrapper(quant: str, symbol: str):
    def wrapper(*args, **kwargs) -> None:
        _launch_selected(quant, symbol, *args, require_pack8=True, **kwargs)

    return wrapper


def _symbol(quant: str, variant: str) -> str:
    return f"hipengine_{quant}_{variant}"


def gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32(
    x_ptr: int,
    qweight_ptr: int,
    normalized_ptr: int,
    gate_ptr: int,
    mixed_ptr: int,
    rows: int,
    in_features: int,
    branches: int,
    hidden: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    """Launch exact raw-Q8 GR up, sigmoid gate, and branch mean."""

    _validate("gguf_q8_0", rows, in_features, branches * hidden, threads)
    if branches != 4:
        raise ValueError("branches must equal 4")
    if hidden <= 0 or hidden % 2:
        raise ValueError("hidden must be a positive multiple of 2")
    library = library or build_gguf_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(
        library,
        "hipengine_gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32",
        [_VOID, _VOID, _VOID, _VOID, _VOID,
         _I64, _I64, _I64, _I64, _I64, _VOID],
    )
    err = fn(
        x_ptr,
        qweight_ptr,
        normalized_ptr,
        gate_ptr,
        mixed_ptr,
        rows,
        in_features,
        branches,
        hidden,
        threads,
        stream,
    )
    _check_launch(runtime, err)


gguf_q8_0_gemv_f32_f32_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_f32_f32_out"))
gguf_q8_0_gemv_f32_fp16_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_f32_fp16_out"))
gguf_q8_0_gemv_fp16_f32_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_fp16_f32_out"))
gguf_q8_0_gemv_fp16_fp16_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_fp16_fp16_out"))
gguf_q8_0_gemv_bf16_f32_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_bf16_f32_out"))
gguf_q8_0_gemv_bf16_fp16_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_bf16_fp16_out"))
gguf_q8_0_gemv_bf16_bf16_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_bf16_bf16_out"))
gguf_q8_0_dual_gemv_f32_f32_out = _make_dual_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "dual_gemv_f32_f32_out"))
gguf_q8_0_dual_gemv_bf16_bf16_out = _make_dual_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "dual_gemv_bf16_bf16_out"))
gguf_q8_0_pack8_gemv_f32_f32_out = _make_pack8_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "pack8_gemv_f32_f32_out"))
gguf_q8_0_pack8_gemv_bf16_f32_out = _make_pack8_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "pack8_gemv_bf16_f32_out"))
gguf_q8_0_pack8_gemv_bf16_bf16_out = _make_pack8_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "pack8_gemv_bf16_bf16_out"))
gguf_q8_0_exact_prefill_tile8x2_bf16_bf16_out = _make_pack8_wrapper(
    "gguf_q8_0", _symbol("gguf_q8_0", "exact_prefill_tile8x2_bf16_bf16_out")
)
gguf_q8_0_exact_prefill_tile8x4_bf16_bf16_out = _make_pack8_wrapper(
    "gguf_q8_0", _symbol("gguf_q8_0", "exact_prefill_tile8x4_bf16_bf16_out")
)
gguf_q8_0_exact_prefill_tile16x4_bf16_bf16_out = _make_pack8_wrapper(
    "gguf_q8_0", _symbol("gguf_q8_0", "exact_prefill_tile16x4_bf16_bf16_out")
)
gguf_q8_0_selected_gemv_bf16_bf16_out = _make_selected_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "selected_gemv_bf16_bf16_out"))
gguf_q8_0_selected_pack8_gemv_bf16_bf16_out = _make_selected_pack8_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "selected_pack8_gemv_bf16_bf16_out"))
gguf_q8_0_prefill_f32_f32_out = gguf_q8_0_gemv_f32_f32_out
gguf_q8_0_prefill_f32_fp16_out = gguf_q8_0_gemv_f32_fp16_out
gguf_q8_0_prefill_fp16_f32_out = gguf_q8_0_gemv_fp16_f32_out
gguf_q8_0_prefill_fp16_fp16_out = gguf_q8_0_gemv_fp16_fp16_out
gguf_q8_0_prefill_bf16_f32_out = gguf_q8_0_gemv_bf16_f32_out
gguf_q8_0_prefill_bf16_fp16_out = gguf_q8_0_gemv_bf16_fp16_out
gguf_q8_0_prefill_bf16_bf16_out = gguf_q8_0_gemv_bf16_bf16_out

gguf_q5_k_gemv_f32_f32_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_f32_f32_out"))
gguf_q5_k_gemv_f32_fp16_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_f32_fp16_out"))
gguf_q5_k_gemv_fp16_f32_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_fp16_f32_out"))
gguf_q5_k_gemv_fp16_fp16_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_fp16_fp16_out"))
gguf_q5_k_gemv_bf16_f32_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_bf16_f32_out"))
gguf_q5_k_gemv_bf16_fp16_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_bf16_fp16_out"))
gguf_q5_k_gemv_bf16_bf16_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_bf16_bf16_out"))
gguf_q5_k_pack8_gemv_bf16_f32_out = _make_pack8_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "pack8_gemv_bf16_f32_out"))
gguf_q5_k_pack8_gemv_bf16_bf16_out = _make_pack8_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "pack8_gemv_bf16_bf16_out"))
gguf_q5_k_pack8_gemv_decode_bf16_f32_out = _make_pack8_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "pack8_gemv_decode_bf16_f32_out")
)
gguf_q5_k_pack8_gemv_decode_bf16_bf16_out = _make_pack8_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "pack8_gemv_decode_bf16_bf16_out")
)
gguf_q5_k_pair_pack8_gemv_decode_bf16_bf16_out = _make_dual_pack8_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "pair_pack8_gemv_decode_bf16_bf16_out")
)
gguf_q5_k_pair_pack8_gemv_decode_bf16_f32_out = _make_unequal_dual_pack8_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "pair_pack8_gemv_decode_bf16_f32_out")
)
gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out = _make_wave32x2_wrapper(
    _symbol("gguf_q5_k", "wave32x2_gemv_decode_bf16_bf16_out")
)
gguf_q5_k_wave32x2_gemv_decode_bf16_f32_out = _make_wave32x2_wrapper(
    _symbol("gguf_q5_k", "wave32x2_gemv_decode_bf16_f32_out")
)
gguf_q5_k_pair_wave32x2_gemv_decode_bf16_f32_out = _make_unequal_wave32x2_wrapper(
    _symbol("gguf_q5_k", "pair_wave32x2_gemv_decode_bf16_f32_out")
)
gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out = _make_wave32x2_wrapper(
    _symbol("gguf_q5_k", "wave32x2_fixed_meta_gemv_decode_bf16_bf16_out")
)
gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_f32_out = _make_wave32x2_wrapper(
    _symbol("gguf_q5_k", "wave32x2_fixed_meta_gemv_decode_bf16_f32_out")
)
gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out = (
    _make_unequal_wave32x2_wrapper(
        _symbol("gguf_q5_k", "pair_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out")
    )
)
gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_f32_out = (
    _make_unequal_wave32x2_wrapper(
        _symbol("gguf_q5_k", "pair_wave32x2_fixed_meta_gemv_decode_bf16_f32_out")
    )
)
gguf_q5_k_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out = (
    _make_wave32x2_wrapper(
        _symbol(
            "gguf_q5_k",
            "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out",
        ),
        require_non_null=True,
    )
)
gguf_q5_k_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_f32_out = (
    _make_wave32x2_wrapper(
        _symbol(
            "gguf_q5_k",
            "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_f32_out",
        ),
        require_non_null=True,
    )
)
gguf_q5_k_pair_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out = (
    _make_unequal_wave32x2_wrapper(
        _symbol(
            "gguf_q5_k",
            "pair_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out",
        ),
        require_non_null=True,
    )
)
gguf_q5_k_pair_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_f32_out = (
    _make_unequal_wave32x2_wrapper(
        _symbol(
            "gguf_q5_k",
            "pair_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_f32_out",
        ),
        require_non_null=True,
    )
)
gguf_q5_k_selected_gemv_bf16_bf16_out = _make_selected_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "selected_gemv_bf16_bf16_out"))
gguf_q5_k_selected_silu_gemv_bf16_bf16_out = _make_selected_silu_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "selected_silu_gemv_bf16_bf16_out")
)
gguf_q5_k_selected_pack8_gemv_bf16_bf16_out = _make_selected_pack8_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "selected_pack8_gemv_bf16_bf16_out"))
gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out = _make_selected_pack8_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "selected_pack8_gemv_q8_1_dp4a_bf16_bf16_out")
)
gguf_q5_k_prefill_f32_f32_out = gguf_q5_k_gemv_f32_f32_out
gguf_q5_k_prefill_f32_fp16_out = gguf_q5_k_gemv_f32_fp16_out
gguf_q5_k_prefill_fp16_f32_out = gguf_q5_k_gemv_fp16_f32_out
gguf_q5_k_prefill_fp16_fp16_out = gguf_q5_k_gemv_fp16_fp16_out
gguf_q5_k_prefill_bf16_f32_out = gguf_q5_k_gemv_bf16_f32_out
gguf_q5_k_prefill_bf16_fp16_out = gguf_q5_k_gemv_bf16_fp16_out
gguf_q5_k_prefill_bf16_bf16_out = gguf_q5_k_gemv_bf16_bf16_out

gguf_q6_k_gemv_f32_f32_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_f32_f32_out"))
gguf_q6_k_gemv_f32_fp16_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_f32_fp16_out"))
gguf_q6_k_gemv_fp16_f32_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_fp16_f32_out"))
gguf_q6_k_gemv_fp16_fp16_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_fp16_fp16_out"))
gguf_q6_k_gemv_bf16_f32_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_bf16_f32_out"))
gguf_q6_k_gemv_bf16_fp16_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_bf16_fp16_out"))
gguf_q6_k_gemv_bf16_bf16_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_bf16_bf16_out"))
gguf_q6_k_pack8_gemv_bf16_f32_out = _make_pack8_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "pack8_gemv_bf16_f32_out"))
gguf_q6_k_pack8_gemv_bf16_bf16_out = _make_pack8_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "pack8_gemv_bf16_bf16_out"))
gguf_q6_k_pair_pack8_gemv_decode_bf16_f32_out = _make_unequal_dual_pack8_wrapper(
    "gguf_q6_k", _symbol("gguf_q6_k", "pair_pack8_gemv_decode_bf16_f32_out")
)
gguf_q6_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out = _make_wave32x2_wrapper(
    _symbol("gguf_q6_k", "wave32x2_fixed_meta_gemv_decode_bf16_bf16_out")
)
_MIXED_ATTENTION_SYMBOL = (
    "hipengine_gguf_q5_q6_mixed_attention_pack8_gemv_decode_bf16_f32_out"
)
gguf_q5_q6_attention_q5_qg_mixed_gemv_decode_bf16_f32_out = (
    _make_mixed_attention_wrapper(_MIXED_ATTENTION_SYMBOL, (0, 3))
)
gguf_q5_q6_attention_q5_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out = (
    _make_mixed_attention_wrapper(
        "hipengine_gguf_q5_q6_mixed_q6_fixed_meta_attention_pack8_gemv_decode_bf16_f32_out",
        (0, 3),
    )
)
gguf_q5_q6_attention_q5_qg_mixed_local32_fixed_meta_gemv_decode_bf16_f32_out = (
    _make_mixed_attention_wrapper(
        "hipengine_gguf_q5_q6_mixed_local32_fixed_meta_attention_pack8_gemv_decode_bf16_f32_out",
        (0, 3),
    )
)
gguf_q5_q6_attention_q5_qg_mixed_local32_q5_swar_pair_fixed_meta_gemv_decode_bf16_f32_out = (
    _make_mixed_attention_wrapper(
        "hipengine_gguf_q5_q6_mixed_local32_q5_swar_pair_fixed_meta_attention_gemv_decode_bf16_f32_out",
        (0, 3),
        require_non_null=True,
    )
)
gguf_q5_q6_attention_q5_qg_mixed_pair_reuse_local32_fixed_meta_gemv_decode_bf16_f32_out = (
    _make_mixed_attention_wrapper(
        "hipengine_gguf_q5_q6_mixed_pair_reuse_local32_fixed_meta_attention_pack8_gemv_decode_bf16_f32_out",
        (0, 3),
        require_non_null=True,
        require_primary_total_at_least_secondary=True,
    )
)
gguf_q6_q8_attention_q6_qg_mixed_gemv_decode_bf16_f32_out = (
    _make_mixed_attention_wrapper(
        "hipengine_gguf_q6_q8_mixed_attention_pack8_gemv_decode_bf16_f32_out",
        (0, 3),
    )
)
gguf_q6_q8_attention_q6_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out = (
    _make_mixed_attention_wrapper(
        "hipengine_gguf_q6_q8_mixed_q6_fixed_meta_attention_pack8_gemv_decode_bf16_f32_out",
        (0, 3),
    )
)
gguf_q6_k_selected_gemv_bf16_bf16_out = _make_selected_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "selected_gemv_bf16_bf16_out"))
gguf_q6_k_selected_silu_gemv_bf16_bf16_out = _make_selected_silu_wrapper(
    "gguf_q6_k", _symbol("gguf_q6_k", "selected_silu_gemv_bf16_bf16_out")
)
gguf_q6_k_selected_pack8_gemv_bf16_bf16_out = _make_selected_pack8_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "selected_pack8_gemv_bf16_bf16_out"))
gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out = _make_selected_pack8_wrapper(
    "gguf_q6_k", _symbol("gguf_q6_k", "selected_pack8_gemv_q8_1_dp4a_bf16_bf16_out")
)
gguf_q6_k_prefill_f32_f32_out = gguf_q6_k_gemv_f32_f32_out
gguf_q6_k_prefill_f32_fp16_out = gguf_q6_k_gemv_f32_fp16_out
gguf_q6_k_prefill_fp16_f32_out = gguf_q6_k_gemv_fp16_f32_out
gguf_q6_k_prefill_fp16_fp16_out = gguf_q6_k_gemv_fp16_fp16_out
gguf_q6_k_prefill_bf16_f32_out = gguf_q6_k_gemv_bf16_f32_out
gguf_q6_k_prefill_bf16_fp16_out = gguf_q6_k_gemv_bf16_fp16_out
gguf_q6_k_prefill_bf16_bf16_out = gguf_q6_k_gemv_bf16_bf16_out

# Small-B weight-amortized row-tile variants (rows in [2, 8]); same launch ABI.
gguf_q8_0_gemv_rowtile_bf16_bf16_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_rowtile_bf16_bf16_out"))
gguf_q8_0_gemv_rowtile_bf16_f32_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_rowtile_bf16_f32_out"))
gguf_q8_0_gemv_rowtile_f32_f32_out = _make_wrapper("gguf_q8_0", _symbol("gguf_q8_0", "gemv_rowtile_f32_f32_out"))
gguf_q5_k_gemv_rowtile_bf16_bf16_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_rowtile_bf16_bf16_out"))
gguf_q5_k_gemv_rowtile_bf16_f32_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_rowtile_bf16_f32_out"))
gguf_q5_k_gemv_rowtile_f32_f32_out = _make_wrapper("gguf_q5_k", _symbol("gguf_q5_k", "gemv_rowtile_f32_f32_out"))
gguf_q6_k_gemv_rowtile_bf16_bf16_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_rowtile_bf16_bf16_out"))
gguf_q6_k_gemv_rowtile_bf16_f32_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_rowtile_bf16_f32_out"))
gguf_q6_k_gemv_rowtile_f32_f32_out = _make_wrapper("gguf_q6_k", _symbol("gguf_q6_k", "gemv_rowtile_f32_f32_out"))

# WPF-1 fixed-grid-Y weight-amortized variants for arbitrary prefill rows.
gguf_q8_0_gemv_rowbatch4_f32_f32_out = _make_wrapper(
    "gguf_q8_0", _symbol("gguf_q8_0", "gemv_rowbatch4_f32_f32_out")
)
gguf_q8_0_gemv_rowbatch8_f32_f32_out = _make_wrapper(
    "gguf_q8_0", _symbol("gguf_q8_0", "gemv_rowbatch8_f32_f32_out")
)
gguf_q8_0_gemv_rowbatch16_f32_f32_out = _make_wrapper(
    "gguf_q8_0", _symbol("gguf_q8_0", "gemv_rowbatch16_f32_f32_out")
)
gguf_q8_0_gemv_rowbatch32_f32_f32_out = _make_wrapper(
    "gguf_q8_0", _symbol("gguf_q8_0", "gemv_rowbatch32_f32_f32_out")
)
gguf_q8_0_gemv_coltile4_rowbatch8_f32_f32_out = _make_wrapper(
    "gguf_q8_0",
    _symbol("gguf_q8_0", "gemv_coltile4_rowbatch8_f32_f32_out"),
)
gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out = _make_wrapper(
    "gguf_q8_0",
    _symbol("gguf_q8_0", "gemv_coltile8_rowbatch4_f32_f32_out"),
)
gguf_q8_0_gemv_coltile8_rowbatch8_f32_f32_out = _make_wrapper(
    "gguf_q8_0",
    _symbol("gguf_q8_0", "gemv_coltile8_rowbatch8_f32_f32_out"),
)
gguf_q8_0_gemv_coltile16_rowbatch2_f32_f32_out = _make_wrapper(
    "gguf_q8_0",
    _symbol("gguf_q8_0", "gemv_coltile16_rowbatch2_f32_f32_out"),
)
gguf_q8_0_gemv_coltile16_rowbatch4_f32_f32_out = _make_wrapper(
    "gguf_q8_0",
    _symbol("gguf_q8_0", "gemv_coltile16_rowbatch4_f32_f32_out"),
)
gguf_q8_0_gemv_coltile32_rowbatch1_f32_f32_out = _make_wrapper(
    "gguf_q8_0",
    _symbol("gguf_q8_0", "gemv_coltile32_rowbatch1_f32_f32_out"),
)
gguf_q5_k_gemv_rowbatch4_bf16_bf16_out = _make_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "gemv_rowbatch4_bf16_bf16_out")
)
gguf_q5_k_gemv_rowbatch4_bf16_f32_out = _make_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "gemv_rowbatch4_bf16_f32_out")
)
gguf_q5_k_gemv_rowbatch8_bf16_bf16_out = _make_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "gemv_rowbatch8_bf16_bf16_out")
)
gguf_q5_k_gemv_rowbatch8_bf16_f32_out = _make_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "gemv_rowbatch8_bf16_f32_out")
)
gguf_q5_k_gemv_rowbatch16_bf16_bf16_out = _make_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "gemv_rowbatch16_bf16_bf16_out")
)
gguf_q5_k_gemv_rowbatch16_bf16_f32_out = _make_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "gemv_rowbatch16_bf16_f32_out")
)
gguf_q5_k_gemv_rowbatch32_bf16_bf16_out = _make_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "gemv_rowbatch32_bf16_bf16_out")
)
gguf_q5_k_gemv_rowbatch32_bf16_f32_out = _make_wrapper(
    "gguf_q5_k", _symbol("gguf_q5_k", "gemv_rowbatch32_bf16_f32_out")
)
gguf_q6_k_gemv_rowbatch4_bf16_bf16_out = _make_wrapper(
    "gguf_q6_k", _symbol("gguf_q6_k", "gemv_rowbatch4_bf16_bf16_out")
)
gguf_q6_k_gemv_rowbatch4_bf16_f32_out = _make_wrapper(
    "gguf_q6_k", _symbol("gguf_q6_k", "gemv_rowbatch4_bf16_f32_out")
)
gguf_q6_k_gemv_rowbatch8_bf16_bf16_out = _make_wrapper(
    "gguf_q6_k", _symbol("gguf_q6_k", "gemv_rowbatch8_bf16_bf16_out")
)
gguf_q6_k_gemv_rowbatch8_bf16_f32_out = _make_wrapper(
    "gguf_q6_k", _symbol("gguf_q6_k", "gemv_rowbatch8_bf16_f32_out")
)
gguf_q6_k_gemv_rowbatch16_bf16_bf16_out = _make_wrapper(
    "gguf_q6_k", _symbol("gguf_q6_k", "gemv_rowbatch16_bf16_bf16_out")
)
gguf_q6_k_gemv_rowbatch16_bf16_f32_out = _make_wrapper(
    "gguf_q6_k", _symbol("gguf_q6_k", "gemv_rowbatch16_bf16_f32_out")
)
gguf_q6_k_gemv_rowbatch32_bf16_bf16_out = _make_wrapper(
    "gguf_q6_k", _symbol("gguf_q6_k", "gemv_rowbatch32_bf16_bf16_out")
)
gguf_q6_k_gemv_rowbatch32_bf16_f32_out = _make_wrapper(
    "gguf_q6_k", _symbol("gguf_q6_k", "gemv_rowbatch32_bf16_f32_out")
)

# WPF-1T constant-32-accumulator Q5/Q6 output-column candidates.
gguf_q5_k_gemv_coltile2_rowbatch16_bf16_bf16_out = _make_wrapper(
    "gguf_q5_k",
    _symbol("gguf_q5_k", "gemv_coltile2_rowbatch16_bf16_bf16_out"),
)
gguf_q5_k_gemv_coltile2_rowbatch16_bf16_f32_out = _make_wrapper(
    "gguf_q5_k",
    _symbol("gguf_q5_k", "gemv_coltile2_rowbatch16_bf16_f32_out"),
)
gguf_q5_k_gemv_coltile4_rowbatch8_bf16_bf16_out = _make_wrapper(
    "gguf_q5_k",
    _symbol("gguf_q5_k", "gemv_coltile4_rowbatch8_bf16_bf16_out"),
)
gguf_q5_k_gemv_coltile4_rowbatch8_bf16_f32_out = _make_wrapper(
    "gguf_q5_k",
    _symbol("gguf_q5_k", "gemv_coltile4_rowbatch8_bf16_f32_out"),
)
gguf_q6_k_gemv_coltile2_rowbatch16_bf16_bf16_out = _make_wrapper(
    "gguf_q6_k",
    _symbol("gguf_q6_k", "gemv_coltile2_rowbatch16_bf16_bf16_out"),
)
gguf_q6_k_gemv_coltile2_rowbatch16_bf16_f32_out = _make_wrapper(
    "gguf_q6_k",
    _symbol("gguf_q6_k", "gemv_coltile2_rowbatch16_bf16_f32_out"),
)
gguf_q6_k_gemv_coltile4_rowbatch8_bf16_bf16_out = _make_wrapper(
    "gguf_q6_k",
    _symbol("gguf_q6_k", "gemv_coltile4_rowbatch8_bf16_bf16_out"),
)
gguf_q6_k_gemv_coltile4_rowbatch8_bf16_f32_out = _make_wrapper(
    "gguf_q6_k",
    _symbol("gguf_q6_k", "gemv_coltile4_rowbatch8_bf16_f32_out"),
)


def gguf_q6_k_gemv_dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out(
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_h7c_raw_q6(
        "bf16",
        _symbol(
            "gguf_q6_k",
            "gemv_dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out",
        ),
        x_ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_gemv_dpp_wave_reduction_coltile2_rowbatch16_bf16_f32_out(
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_h7c_raw_q6(
        "f32",
        _symbol(
            "gguf_q6_k",
            "gemv_dpp_wave_reduction_coltile2_rowbatch16_bf16_f32_out",
        ),
        x_ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_gemv_dpp_wave_reduction_full_group_compute_coltile4_rowbatch8_bf16_bf16_out(
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_h7i_raw_q6_full_group(
        "bf16",
        _symbol(
            "gguf_q6_k",
            "gemv_dpp_wave_reduction_full_group_compute_"
            "coltile4_rowbatch8_bf16_bf16_out",
        ),
        x_ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def gguf_q6_k_gemv_dpp_wave_reduction_full_group_compute_coltile2_rowbatch16_bf16_f32_out(
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    _launch_h7i_raw_q6_full_group(
        "f32",
        _symbol(
            "gguf_q6_k",
            "gemv_dpp_wave_reduction_full_group_compute_"
            "coltile2_rowbatch16_bf16_f32_out",
        ),
        x_ptr,
        qweight_ptr,
        out_ptr,
        rows,
        in_features,
        out_features,
        threads=threads,
        stream=stream,
        library=library,
        runtime=runtime,
    )


def register_gguf_k_gemv_kernels(*, replace: bool = True) -> None:
    for quant in ("gguf_q8_0", "gguf_q5_k", "gguf_q6_k"):
        for variant, fn in _WRAPPERS[quant].items():
            register(KernelKey("hip_gfx1100", "linear", quant, variant), fn, replace=replace)
    register(
        KernelKey(
            "hip_gfx1100",
            "linear+gr_gated_mean",
            "gguf_q8_0",
            "coltile2_branch4_rowbatch4_f32_exact",
        ),
        gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair",
            "gguf_q5_k",
            "pack8_gemv_decode_bf16_bf16_out",
        ),
        gguf_q5_k_pair_pack8_gemv_decode_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair",
            "gguf_q5_k",
            "pack8_gemv_decode_bf16_f32_out",
        ),
        gguf_q5_k_pair_pack8_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair",
            "gguf_q6_k",
            "pack8_gemv_decode_bf16_f32_out",
        ),
        gguf_q6_k_pair_pack8_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair",
            "gguf_q5_k",
            "wave32x2_gemv_decode_bf16_f32_out",
        ),
        gguf_q5_k_pair_wave32x2_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair",
            "gguf_q5_k",
            "wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
        ),
        gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair",
            "gguf_q5_k",
            "wave32x2_fixed_meta_gemv_decode_bf16_f32_out",
        ),
        gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "linear_pair",
            "gguf_q5_k",
            "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out",
        ),
        gguf_q5_k_pair_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "attention_projection_quad",
            _MIXED_ATTENTION_Q5_QG_QUANT,
            _MIXED_ATTENTION_VARIANT,
        ),
        gguf_q5_q6_attention_q5_qg_mixed_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "attention_projection_quad",
            _MIXED_ATTENTION_Q5_QG_QUANT,
            _MIXED_ATTENTION_Q6_FIXED_META_VARIANT,
        ),
        gguf_q5_q6_attention_q5_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "attention_projection_quad",
            _MIXED_ATTENTION_Q5_QG_QUANT,
            _MIXED_ATTENTION_LOCAL32_FIXED_META_VARIANT,
        ),
        gguf_q5_q6_attention_q5_qg_mixed_local32_fixed_meta_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "attention_projection_quad",
            _MIXED_ATTENTION_Q5_QG_QUANT,
            _MIXED_ATTENTION_PAIR_REUSE_LOCAL32_FIXED_META_VARIANT,
        ),
        gguf_q5_q6_attention_q5_qg_mixed_pair_reuse_local32_fixed_meta_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "attention_projection_quad",
            _MIXED_ATTENTION_Q5_QG_QUANT,
            _MIXED_ATTENTION_LOCAL32_Q5_SWAR_PAIR_FIXED_META_VARIANT,
        ),
        gguf_q5_q6_attention_q5_qg_mixed_local32_q5_swar_pair_fixed_meta_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "attention_projection_quad",
            _MIXED_ATTENTION_Q6_QG_Q8_KV_QUANT,
            _MIXED_ATTENTION_VARIANT,
        ),
        gguf_q6_q8_attention_q6_qg_mixed_gemv_decode_bf16_f32_out,
        replace=replace,
    )
    register(
        KernelKey(
            "hip_gfx1100",
            "attention_projection_quad",
            _MIXED_ATTENTION_Q6_QG_Q8_KV_QUANT,
            _MIXED_ATTENTION_Q6_FIXED_META_VARIANT,
        ),
        gguf_q6_q8_attention_q6_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out,
        replace=replace,
    )


def _launch(
    quant: str,
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    require_pack8: bool = False,
) -> None:
    _validate(quant, rows, in_features, out_features, threads, require_pack8=require_pack8)
    library = library or build_gguf_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(library, symbol, [_VOID, _VOID, _VOID, _I64, _I64, _I64, _I64, _VOID])
    err = fn(x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, threads, stream)
    _check_launch(runtime, err)


def _launch_dual(
    quant: str,
    symbol: str,
    x_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    require_pack8: bool = False,
) -> None:
    _validate(quant, rows, in_features, out_features, threads, require_pack8=require_pack8)
    library = library or build_gguf_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(library, symbol, [_VOID, _VOID, _VOID, _VOID, _VOID, _I64, _I64, _I64, _I64, _VOID])
    err = fn(
        x_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_a_ptr,
        out_b_ptr,
        rows,
        in_features,
        out_features,
        threads,
        stream,
    )
    _check_launch(runtime, err)


def _launch_unequal_dual(
    quant: str,
    symbol: str,
    x_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    out_features_b: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    require_pack8: bool = False,
) -> None:
    _validate(
        quant,
        rows,
        in_features,
        out_features,
        threads,
        require_pack8=require_pack8,
    )
    _validate(
        quant,
        rows,
        in_features,
        out_features_b,
        threads,
        require_pack8=require_pack8,
    )
    library = library or build_gguf_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(
        library,
        symbol,
        [_VOID, _VOID, _VOID, _VOID, _VOID, _I64, _I64, _I64, _I64, _I64, _VOID],
    )
    err = fn(
        x_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_a_ptr,
        out_b_ptr,
        rows,
        in_features,
        out_features,
        out_features_b,
        threads,
        stream,
    )
    _check_launch(runtime, err)


def _launch_wave32x2(
    symbol: str,
    x_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 32,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    require_non_null: bool = False,
) -> None:
    if require_non_null and not all((x_ptr, qweight_ptr, out_ptr)):
        raise ValueError("GGUF Q5_K SWAR pair pointers must be non-zero")
    _validate_wave32x2(rows, in_features, out_features, threads)
    library = library or build_gguf_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(library, symbol, [_VOID, _VOID, _VOID, _I64, _I64, _I64, _I64, _VOID])
    err = fn(x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, threads, stream)
    _check_launch(runtime, err)


def _launch_unequal_wave32x2(
    symbol: str,
    x_ptr: int,
    qweight_a_ptr: int,
    qweight_b_ptr: int,
    out_a_ptr: int,
    out_b_ptr: int,
    rows: int,
    in_features: int,
    out_features: int,
    out_features_b: int,
    *,
    threads: int = 32,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    require_non_null: bool = False,
) -> None:
    if require_non_null and not all(
        (x_ptr, qweight_a_ptr, qweight_b_ptr, out_a_ptr, out_b_ptr)
    ):
        raise ValueError("GGUF Q5_K SWAR pair pointers must be non-zero")
    _validate_wave32x2(rows, in_features, out_features, threads)
    _validate_wave32x2(rows, in_features, out_features_b, threads)
    library = library or build_gguf_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(
        library,
        symbol,
        [_VOID, _VOID, _VOID, _VOID, _VOID, _I64, _I64, _I64, _I64, _I64, _VOID],
    )
    err = fn(
        x_ptr,
        qweight_a_ptr,
        qweight_b_ptr,
        out_a_ptr,
        out_b_ptr,
        rows,
        in_features,
        out_features,
        out_features_b,
        threads,
        stream,
    )
    _check_launch(runtime, err)


def _launch_mixed_attention(
    symbol: str,
    x_ptr: int,
    qweight_q_ptr: int,
    qweight_k_ptr: int,
    qweight_v_ptr: int,
    qweight_gate_ptr: int,
    out_q_ptr: int,
    out_k_ptr: int,
    out_v_ptr: int,
    out_gate_ptr: int,
    rows: int,
    in_features: int,
    q_features: int,
    k_features: int,
    v_features: int,
    gate_features: int,
    *,
    primary_roles: tuple[int, int],
    require_non_null: bool = False,
    require_primary_total_at_least_secondary: bool = False,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    pointers = (
        x_ptr,
        qweight_q_ptr,
        qweight_k_ptr,
        qweight_v_ptr,
        qweight_gate_ptr,
        out_q_ptr,
        out_k_ptr,
        out_v_ptr,
        out_gate_ptr,
    )
    if require_non_null and any(not pointer for pointer in pointers):
        raise ValueError("mixed GGUF K attention pointers must be non-zero")
    if rows != 1:
        raise ValueError("rows must be exactly 1 for mixed GGUF K attention decode")
    if in_features <= 0 or in_features % _QTYPE_BLOCK_SIZE["gguf_q5_k"] != 0:
        raise ValueError("in_features must be positive and divisible by GGUF K block size 256")
    features = (q_features, k_features, v_features, gate_features)
    if any(value <= 0 or value % 8 != 0 for value in features):
        raise ValueError("all mixed GGUF K attention output features must be positive and divisible by 8")
    qweights = (qweight_q_ptr, qweight_k_ptr, qweight_v_ptr, qweight_gate_ptr)
    outputs = (out_q_ptr, out_k_ptr, out_v_ptr, out_gate_ptr)
    secondary_roles = tuple(index for index in range(4) if index not in primary_roles)
    if require_primary_total_at_least_secondary and sum(
        features[index] for index in primary_roles
    ) < sum(features[index] for index in secondary_roles):
        raise ValueError("Q5 total output features must be at least Q6 total output features")
    library = library or build_gguf_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(
        library,
        symbol,
        [_VOID] * 9 + [_I64] * 6 + [_VOID],
    )
    err = fn(
        x_ptr,
        *(qweights[index] for index in (*primary_roles, *secondary_roles)),
        *(outputs[index] for index in (*primary_roles, *secondary_roles)),
        rows,
        in_features,
        *(features[index] for index in (*primary_roles, *secondary_roles)),
        stream,
    )
    _check_launch(runtime, err)


def _launch_selected(
    quant: str,
    symbol: str,
    x_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
    require_pack8: bool = False,
) -> None:
    if x_rows <= 0:
        raise ValueError("x_rows must be positive")
    if rows <= 0 or rows % x_rows != 0:
        raise ValueError("rows must be positive and divisible by x_rows")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    _validate(quant, rows, in_features, out_features, threads, require_pack8=require_pack8)
    library = library or build_gguf_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = getattr(library, symbol)
    fn.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    fn.restype = ctypes.c_int
    err = fn(
        ctypes.c_void_p(x_ptr),
        ctypes.c_void_p(selected_ptr),
        ctypes.c_void_p(qweight_ptr),
        ctypes.c_void_p(out_ptr),
        ctypes.c_int64(x_rows),
        ctypes.c_int64(rows),
        ctypes.c_int64(num_experts),
        ctypes.c_int64(in_features),
        ctypes.c_int64(out_features),
        ctypes.c_int64(threads),
        ctypes.c_void_p(stream),
    )
    _check_launch(runtime, err)


def _launch_selected_silu(
    quant: str,
    symbol: str,
    gate_ptr: int,
    up_ptr: int,
    selected_ptr: int,
    qweight_ptr: int,
    out_ptr: int,
    x_rows: int,
    rows: int,
    num_experts: int,
    in_features: int,
    out_features: int,
    *,
    threads: int = 128,
    stream: int = 0,
    library: ctypes.CDLL | None = None,
    runtime: HipRuntime | None = None,
) -> None:
    if x_rows <= 0:
        raise ValueError("x_rows must be positive")
    if rows <= 0 or rows % x_rows != 0:
        raise ValueError("rows must be positive and divisible by x_rows")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    _validate(quant, rows, in_features, out_features, threads)
    library = library or build_gguf_k_gemv(load=True)
    runtime = runtime or get_hip_runtime()
    fn = _cached_fn(
        library,
        symbol,
        [_VOID, _VOID, _VOID, _VOID, _VOID, _I64, _I64, _I64, _I64, _I64, _I64, _VOID],
    )
    err = fn(
        gate_ptr,
        up_ptr,
        selected_ptr,
        qweight_ptr,
        out_ptr,
        x_rows,
        rows,
        num_experts,
        in_features,
        out_features,
        threads,
        stream,
    )
    _check_launch(runtime, err)


def _validate(
    quant: str,
    rows: int,
    in_features: int,
    out_features: int,
    threads: int,
    *,
    require_pack8: bool = False,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    block_size = _QTYPE_BLOCK_SIZE[quant]
    if in_features % block_size != 0:
        raise ValueError(f"in_features must be divisible by GGUF {quant} block size {block_size}")
    if require_pack8 and out_features % 8 != 0:
        raise ValueError("out_features must be divisible by 8 for GGUF K pack8")
    if threads not in _ALLOWED_THREADS:
        allowed = ", ".join(str(value) for value in sorted(_ALLOWED_THREADS))
        raise ValueError(f"threads must be one of {allowed}")


def _validate_wave32x2(
    rows: int,
    in_features: int,
    out_features: int,
    threads: int,
) -> None:
    if rows != 1:
        raise ValueError("rows must be exactly 1 for GGUF Q5_K wave32x2 decode")
    if in_features <= 0 or in_features % _QTYPE_BLOCK_SIZE["gguf_q5_k"] != 0:
        raise ValueError("in_features must be positive and divisible by GGUF Q5_K block size 256")
    if out_features <= 0:
        raise ValueError("out_features must be positive")
    if out_features % 2 != 0:
        raise ValueError("out_features must be divisible by 2 for GGUF Q5_K wave32x2")
    if threads != 32:
        raise ValueError("threads must be 32 for GGUF Q5_K wave32x2")


def _check_launch(runtime: HipRuntime, err: int) -> None:
    if int(err) != HIP_SUCCESS:
        runtime.check(int(err))


_WRAPPERS = {
    "gguf_q8_0": {
        "gemv_f32_f32_out": gguf_q8_0_gemv_f32_f32_out,
        "gemv_f32_fp16_out": gguf_q8_0_gemv_f32_fp16_out,
        "gemv_fp16_f32_out": gguf_q8_0_gemv_fp16_f32_out,
        "gemv_fp16_fp16_out": gguf_q8_0_gemv_fp16_fp16_out,
        "gemv_bf16_f32_out": gguf_q8_0_gemv_bf16_f32_out,
        "gemv_bf16_fp16_out": gguf_q8_0_gemv_bf16_fp16_out,
        "gemv_bf16_bf16_out": gguf_q8_0_gemv_bf16_bf16_out,
        "dual_gemv_f32_f32_out": gguf_q8_0_dual_gemv_f32_f32_out,
        "dual_gemv_bf16_bf16_out": gguf_q8_0_dual_gemv_bf16_bf16_out,
        "pack8_gemv_f32_f32_out": gguf_q8_0_pack8_gemv_f32_f32_out,
        "pack8_gemv_bf16_f32_out": gguf_q8_0_pack8_gemv_bf16_f32_out,
        "pack8_gemv_bf16_bf16_out": gguf_q8_0_pack8_gemv_bf16_bf16_out,
        "exact_prefill_tile8x2_bf16_bf16_out": gguf_q8_0_exact_prefill_tile8x2_bf16_bf16_out,
        "exact_prefill_tile8x4_bf16_bf16_out": gguf_q8_0_exact_prefill_tile8x4_bf16_bf16_out,
        "exact_prefill_tile16x4_bf16_bf16_out": gguf_q8_0_exact_prefill_tile16x4_bf16_bf16_out,
        "selected_gemv_bf16_bf16_out": gguf_q8_0_selected_gemv_bf16_bf16_out,
        "selected_pack8_gemv_bf16_bf16_out": gguf_q8_0_selected_pack8_gemv_bf16_bf16_out,
        "prefill_f32_f32_out": gguf_q8_0_prefill_f32_f32_out,
        "prefill_f32_fp16_out": gguf_q8_0_prefill_f32_fp16_out,
        "prefill_fp16_f32_out": gguf_q8_0_prefill_fp16_f32_out,
        "prefill_fp16_fp16_out": gguf_q8_0_prefill_fp16_fp16_out,
        "prefill_bf16_f32_out": gguf_q8_0_prefill_bf16_f32_out,
        "prefill_bf16_fp16_out": gguf_q8_0_prefill_bf16_fp16_out,
        "prefill_bf16_bf16_out": gguf_q8_0_prefill_bf16_bf16_out,
        "rowtile_bf16_bf16_out": gguf_q8_0_gemv_rowtile_bf16_bf16_out,
        "rowtile_bf16_f32_out": gguf_q8_0_gemv_rowtile_bf16_f32_out,
        "rowtile_f32_f32_out": gguf_q8_0_gemv_rowtile_f32_f32_out,
        "rowbatch4_f32_f32_out": gguf_q8_0_gemv_rowbatch4_f32_f32_out,
        "rowbatch8_f32_f32_out": gguf_q8_0_gemv_rowbatch8_f32_f32_out,
        "rowbatch16_f32_f32_out": gguf_q8_0_gemv_rowbatch16_f32_f32_out,
        "rowbatch32_f32_f32_out": gguf_q8_0_gemv_rowbatch32_f32_f32_out,
        "coltile4_rowbatch8_f32_f32_out": gguf_q8_0_gemv_coltile4_rowbatch8_f32_f32_out,
        "coltile8_rowbatch4_f32_f32_out": gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out,
        "coltile8_rowbatch8_f32_f32_out": gguf_q8_0_gemv_coltile8_rowbatch8_f32_f32_out,
        "coltile16_rowbatch2_f32_f32_out": gguf_q8_0_gemv_coltile16_rowbatch2_f32_f32_out,
        "coltile16_rowbatch4_f32_f32_out": gguf_q8_0_gemv_coltile16_rowbatch4_f32_f32_out,
        "coltile32_rowbatch1_f32_f32_out": gguf_q8_0_gemv_coltile32_rowbatch1_f32_f32_out,
    },
    "gguf_q5_k": {
        "gemv_f32_f32_out": gguf_q5_k_gemv_f32_f32_out,
        "gemv_f32_fp16_out": gguf_q5_k_gemv_f32_fp16_out,
        "gemv_fp16_f32_out": gguf_q5_k_gemv_fp16_f32_out,
        "gemv_fp16_fp16_out": gguf_q5_k_gemv_fp16_fp16_out,
        "gemv_bf16_f32_out": gguf_q5_k_gemv_bf16_f32_out,
        "gemv_bf16_fp16_out": gguf_q5_k_gemv_bf16_fp16_out,
        "gemv_bf16_bf16_out": gguf_q5_k_gemv_bf16_bf16_out,
        "pack8_gemv_bf16_f32_out": gguf_q5_k_pack8_gemv_bf16_f32_out,
        "pack8_gemv_bf16_bf16_out": gguf_q5_k_pack8_gemv_bf16_bf16_out,
        "pack8_gemv_decode_bf16_f32_out": gguf_q5_k_pack8_gemv_decode_bf16_f32_out,
        "pack8_gemv_decode_bf16_bf16_out": gguf_q5_k_pack8_gemv_decode_bf16_bf16_out,
        "wave32x2_gemv_decode_bf16_bf16_out": gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out,
        "wave32x2_fixed_meta_gemv_decode_bf16_bf16_out": gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
        "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out": gguf_q5_k_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out,
        "selected_gemv_bf16_bf16_out": gguf_q5_k_selected_gemv_bf16_bf16_out,
        "selected_silu_gemv_bf16_bf16_out": gguf_q5_k_selected_silu_gemv_bf16_bf16_out,
        "selected_pack8_gemv_bf16_bf16_out": gguf_q5_k_selected_pack8_gemv_bf16_bf16_out,
        "selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out": gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
        "prefill_f32_f32_out": gguf_q5_k_prefill_f32_f32_out,
        "prefill_f32_fp16_out": gguf_q5_k_prefill_f32_fp16_out,
        "prefill_fp16_f32_out": gguf_q5_k_prefill_fp16_f32_out,
        "prefill_fp16_fp16_out": gguf_q5_k_prefill_fp16_fp16_out,
        "prefill_bf16_f32_out": gguf_q5_k_prefill_bf16_f32_out,
        "prefill_bf16_fp16_out": gguf_q5_k_prefill_bf16_fp16_out,
        "prefill_bf16_bf16_out": gguf_q5_k_prefill_bf16_bf16_out,
        "rowtile_bf16_bf16_out": gguf_q5_k_gemv_rowtile_bf16_bf16_out,
        "rowtile_bf16_f32_out": gguf_q5_k_gemv_rowtile_bf16_f32_out,
        "rowtile_f32_f32_out": gguf_q5_k_gemv_rowtile_f32_f32_out,
        "rowbatch4_bf16_bf16_out": gguf_q5_k_gemv_rowbatch4_bf16_bf16_out,
        "rowbatch4_bf16_f32_out": gguf_q5_k_gemv_rowbatch4_bf16_f32_out,
        "rowbatch8_bf16_bf16_out": gguf_q5_k_gemv_rowbatch8_bf16_bf16_out,
        "rowbatch8_bf16_f32_out": gguf_q5_k_gemv_rowbatch8_bf16_f32_out,
        "rowbatch16_bf16_bf16_out": gguf_q5_k_gemv_rowbatch16_bf16_bf16_out,
        "rowbatch16_bf16_f32_out": gguf_q5_k_gemv_rowbatch16_bf16_f32_out,
        "rowbatch32_bf16_bf16_out": gguf_q5_k_gemv_rowbatch32_bf16_bf16_out,
        "rowbatch32_bf16_f32_out": gguf_q5_k_gemv_rowbatch32_bf16_f32_out,
        "coltile2_rowbatch16_bf16_bf16_out": gguf_q5_k_gemv_coltile2_rowbatch16_bf16_bf16_out,
        "coltile2_rowbatch16_bf16_f32_out": gguf_q5_k_gemv_coltile2_rowbatch16_bf16_f32_out,
        "coltile4_rowbatch8_bf16_bf16_out": gguf_q5_k_gemv_coltile4_rowbatch8_bf16_bf16_out,
        "coltile4_rowbatch8_bf16_f32_out": gguf_q5_k_gemv_coltile4_rowbatch8_bf16_f32_out,
    },
    "gguf_q6_k": {
        "gemv_f32_f32_out": gguf_q6_k_gemv_f32_f32_out,
        "gemv_f32_fp16_out": gguf_q6_k_gemv_f32_fp16_out,
        "gemv_fp16_f32_out": gguf_q6_k_gemv_fp16_f32_out,
        "gemv_fp16_fp16_out": gguf_q6_k_gemv_fp16_fp16_out,
        "gemv_bf16_f32_out": gguf_q6_k_gemv_bf16_f32_out,
        "gemv_bf16_fp16_out": gguf_q6_k_gemv_bf16_fp16_out,
        "gemv_bf16_bf16_out": gguf_q6_k_gemv_bf16_bf16_out,
        "pack8_gemv_bf16_f32_out": gguf_q6_k_pack8_gemv_bf16_f32_out,
        "pack8_gemv_bf16_bf16_out": gguf_q6_k_pack8_gemv_bf16_bf16_out,
        "standalone_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out": gguf_q6_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out,
        "selected_gemv_bf16_bf16_out": gguf_q6_k_selected_gemv_bf16_bf16_out,
        "selected_silu_gemv_bf16_bf16_out": gguf_q6_k_selected_silu_gemv_bf16_bf16_out,
        "selected_pack8_gemv_bf16_bf16_out": gguf_q6_k_selected_pack8_gemv_bf16_bf16_out,
        "selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out": gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out,
        "prefill_f32_f32_out": gguf_q6_k_prefill_f32_f32_out,
        "prefill_f32_fp16_out": gguf_q6_k_prefill_f32_fp16_out,
        "prefill_fp16_f32_out": gguf_q6_k_prefill_fp16_f32_out,
        "prefill_fp16_fp16_out": gguf_q6_k_prefill_fp16_fp16_out,
        "prefill_bf16_f32_out": gguf_q6_k_prefill_bf16_f32_out,
        "prefill_bf16_fp16_out": gguf_q6_k_prefill_bf16_fp16_out,
        "prefill_bf16_bf16_out": gguf_q6_k_prefill_bf16_bf16_out,
        "rowtile_bf16_bf16_out": gguf_q6_k_gemv_rowtile_bf16_bf16_out,
        "rowtile_bf16_f32_out": gguf_q6_k_gemv_rowtile_bf16_f32_out,
        "rowtile_f32_f32_out": gguf_q6_k_gemv_rowtile_f32_f32_out,
        "rowbatch4_bf16_bf16_out": gguf_q6_k_gemv_rowbatch4_bf16_bf16_out,
        "rowbatch4_bf16_f32_out": gguf_q6_k_gemv_rowbatch4_bf16_f32_out,
        "rowbatch8_bf16_bf16_out": gguf_q6_k_gemv_rowbatch8_bf16_bf16_out,
        "rowbatch8_bf16_f32_out": gguf_q6_k_gemv_rowbatch8_bf16_f32_out,
        "rowbatch16_bf16_bf16_out": gguf_q6_k_gemv_rowbatch16_bf16_bf16_out,
        "rowbatch16_bf16_f32_out": gguf_q6_k_gemv_rowbatch16_bf16_f32_out,
        "rowbatch32_bf16_bf16_out": gguf_q6_k_gemv_rowbatch32_bf16_bf16_out,
        "rowbatch32_bf16_f32_out": gguf_q6_k_gemv_rowbatch32_bf16_f32_out,
        "coltile2_rowbatch16_bf16_bf16_out": gguf_q6_k_gemv_coltile2_rowbatch16_bf16_bf16_out,
        "coltile2_rowbatch16_bf16_f32_out": gguf_q6_k_gemv_coltile2_rowbatch16_bf16_f32_out,
        "coltile4_rowbatch8_bf16_bf16_out": gguf_q6_k_gemv_coltile4_rowbatch8_bf16_bf16_out,
        "coltile4_rowbatch8_bf16_f32_out": gguf_q6_k_gemv_coltile4_rowbatch8_bf16_f32_out,
        "dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out": gguf_q6_k_gemv_dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out,
        "dpp_wave_reduction_coltile2_rowbatch16_bf16_f32_out": gguf_q6_k_gemv_dpp_wave_reduction_coltile2_rowbatch16_bf16_f32_out,
        "dpp_wave_reduction_full_group_compute_coltile4_rowbatch8_bf16_bf16_out": gguf_q6_k_gemv_dpp_wave_reduction_full_group_compute_coltile4_rowbatch8_bf16_bf16_out,
        "dpp_wave_reduction_full_group_compute_coltile2_rowbatch16_bf16_f32_out": gguf_q6_k_gemv_dpp_wave_reduction_full_group_compute_coltile2_rowbatch16_bf16_f32_out,
    },
}

register_gguf_k_gemv_kernels()


__all__ = [
    "build_gguf_k_gemv",
    "gguf_q8_0_gemv_rowbatch4_f32_f32_out",
    "gguf_q8_0_gemv_rowbatch8_f32_f32_out",
    "gguf_q8_0_gemv_rowbatch16_f32_f32_out",
    "gguf_q8_0_gemv_rowbatch32_f32_f32_out",
    "gguf_q8_0_gemv_coltile4_rowbatch8_f32_f32_out",
    "gguf_q8_0_gemv_coltile8_rowbatch4_f32_f32_out",
    "gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_f32",
    "gguf_q8_0_gemv_coltile8_rowbatch8_f32_f32_out",
    "gguf_q8_0_gemv_coltile16_rowbatch2_f32_f32_out",
    "gguf_q8_0_gemv_coltile16_rowbatch4_f32_f32_out",
    "gguf_q8_0_gemv_coltile32_rowbatch1_f32_f32_out",
    "gguf_q5_k_gemv_f32_f32_out",
    "gguf_q5_q6_attention_q5_qg_mixed_gemv_decode_bf16_f32_out",
    "gguf_q5_q6_attention_q5_qg_mixed_local32_fixed_meta_gemv_decode_bf16_f32_out",
    "gguf_q5_q6_attention_q5_qg_mixed_local32_q5_swar_pair_fixed_meta_gemv_decode_bf16_f32_out",
    "gguf_q5_q6_attention_q5_qg_mixed_pair_reuse_local32_fixed_meta_gemv_decode_bf16_f32_out",
    "gguf_q5_q6_attention_q5_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out",
    "gguf_q6_q8_attention_q6_qg_mixed_gemv_decode_bf16_f32_out",
    "gguf_q6_q8_attention_q6_qg_mixed_q6_fixed_meta_gemv_decode_bf16_f32_out",
    "gguf_q5_k_gemv_f32_fp16_out",
    "gguf_q5_k_gemv_fp16_f32_out",
    "gguf_q5_k_gemv_fp16_fp16_out",
    "gguf_q5_k_gemv_bf16_f32_out",
    "gguf_q5_k_gemv_bf16_fp16_out",
    "gguf_q5_k_gemv_bf16_bf16_out",
    "gguf_q5_k_gemv_rowbatch4_bf16_bf16_out",
    "gguf_q5_k_gemv_rowbatch4_bf16_f32_out",
    "gguf_q5_k_gemv_rowbatch8_bf16_bf16_out",
    "gguf_q5_k_gemv_rowbatch8_bf16_f32_out",
    "gguf_q5_k_gemv_rowbatch16_bf16_bf16_out",
    "gguf_q5_k_gemv_rowbatch16_bf16_f32_out",
    "gguf_q5_k_gemv_rowbatch32_bf16_bf16_out",
    "gguf_q5_k_gemv_rowbatch32_bf16_f32_out",
    "gguf_q5_k_gemv_coltile2_rowbatch16_bf16_bf16_out",
    "gguf_q5_k_gemv_coltile2_rowbatch16_bf16_f32_out",
    "gguf_q5_k_gemv_coltile4_rowbatch8_bf16_bf16_out",
    "gguf_q5_k_gemv_coltile4_rowbatch8_bf16_f32_out",
    "gguf_q5_k_pack8_gemv_decode_bf16_bf16_out",
    "gguf_q5_k_pack8_gemv_decode_bf16_f32_out",
    "gguf_q5_k_pair_pack8_gemv_decode_bf16_bf16_out",
    "gguf_q5_k_pair_pack8_gemv_decode_bf16_f32_out",
    "gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
    "gguf_q5_k_pair_wave32x2_fixed_meta_gemv_decode_bf16_f32_out",
    "gguf_q5_k_pair_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out",
    "gguf_q5_k_pair_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_f32_out",
    "gguf_q5_k_pair_wave32x2_gemv_decode_bf16_f32_out",
    "gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
    "gguf_q5_k_wave32x2_fixed_meta_gemv_decode_bf16_f32_out",
    "gguf_q5_k_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out",
    "gguf_q5_k_wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_f32_out",
    "gguf_q5_k_wave32x2_gemv_decode_bf16_bf16_out",
    "gguf_q5_k_wave32x2_gemv_decode_bf16_f32_out",
    "gguf_q5_k_selected_gemv_bf16_bf16_out",
    "gguf_q5_k_selected_silu_gemv_bf16_bf16_out",
    "gguf_q5_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out",
    "gguf_q5_k_prefill_f32_f32_out",
    "gguf_q5_k_prefill_f32_fp16_out",
    "gguf_q5_k_prefill_fp16_f32_out",
    "gguf_q5_k_prefill_fp16_fp16_out",
    "gguf_q5_k_prefill_bf16_f32_out",
    "gguf_q5_k_prefill_bf16_fp16_out",
    "gguf_q5_k_prefill_bf16_bf16_out",
    "gguf_q6_k_gemv_f32_f32_out",
    "gguf_q6_k_gemv_f32_fp16_out",
    "gguf_q6_k_gemv_fp16_f32_out",
    "gguf_q6_k_gemv_fp16_fp16_out",
    "gguf_q6_k_gemv_bf16_f32_out",
    "gguf_q6_k_gemv_bf16_fp16_out",
    "gguf_q6_k_gemv_bf16_bf16_out",
    "gguf_q6_k_gemv_rowbatch4_bf16_bf16_out",
    "gguf_q6_k_gemv_rowbatch4_bf16_f32_out",
    "gguf_q6_k_gemv_rowbatch8_bf16_bf16_out",
    "gguf_q6_k_gemv_rowbatch8_bf16_f32_out",
    "gguf_q6_k_gemv_rowbatch16_bf16_bf16_out",
    "gguf_q6_k_gemv_rowbatch16_bf16_f32_out",
    "gguf_q6_k_gemv_rowbatch32_bf16_bf16_out",
    "gguf_q6_k_gemv_rowbatch32_bf16_f32_out",
    "gguf_q6_k_gemv_coltile2_rowbatch16_bf16_bf16_out",
    "gguf_q6_k_gemv_coltile2_rowbatch16_bf16_f32_out",
    "gguf_q6_k_gemv_coltile4_rowbatch8_bf16_bf16_out",
    "gguf_q6_k_gemv_coltile4_rowbatch8_bf16_f32_out",
    "gguf_q6_k_gemv_dpp_wave_reduction_coltile2_rowbatch16_bf16_f32_out",
    "gguf_q6_k_gemv_dpp_wave_reduction_coltile4_rowbatch8_bf16_bf16_out",
    "gguf_q6_k_gemv_dpp_wave_reduction_full_group_compute_coltile2_rowbatch16_bf16_f32_out",
    "gguf_q6_k_gemv_dpp_wave_reduction_full_group_compute_coltile4_rowbatch8_bf16_bf16_out",
    "gguf_q6_k_pair_pack8_gemv_decode_bf16_f32_out",
    "gguf_q6_k_wave32x2_fixed_meta_gemv_decode_bf16_bf16_out",
    "gguf_q6_k_selected_gemv_bf16_bf16_out",
    "gguf_q6_k_selected_silu_gemv_bf16_bf16_out",
    "gguf_q6_k_selected_pack8_q8_1_dp4a_gemv_bf16_bf16_out",
    "gguf_q6_k_prefill_f32_f32_out",
    "gguf_q6_k_prefill_f32_fp16_out",
    "gguf_q6_k_prefill_fp16_f32_out",
    "gguf_q6_k_prefill_fp16_fp16_out",
    "gguf_q6_k_prefill_bf16_f32_out",
    "gguf_q6_k_prefill_bf16_fp16_out",
    "gguf_q6_k_prefill_bf16_bf16_out",
    "gguf_q8_0_gemv_f32_f32_out",
    "gguf_q8_0_gemv_f32_fp16_out",
    "gguf_q8_0_gemv_fp16_f32_out",
    "gguf_q8_0_gemv_fp16_fp16_out",
    "gguf_q8_0_gemv_bf16_f32_out",
    "gguf_q8_0_gemv_bf16_fp16_out",
    "gguf_q8_0_gemv_bf16_bf16_out",
    "gguf_q8_0_dual_gemv_f32_f32_out",
    "gguf_q8_0_dual_gemv_bf16_bf16_out",
    "gguf_q8_0_exact_prefill_tile16x4_bf16_bf16_out",
    "gguf_q8_0_exact_prefill_tile8x2_bf16_bf16_out",
    "gguf_q8_0_exact_prefill_tile8x4_bf16_bf16_out",
    "gguf_q8_0_prefill_f32_f32_out",
    "gguf_q8_0_prefill_f32_fp16_out",
    "gguf_q8_0_prefill_fp16_f32_out",
    "gguf_q8_0_prefill_fp16_fp16_out",
    "gguf_q8_0_prefill_bf16_f32_out",
    "gguf_q8_0_prefill_bf16_fp16_out",
    "gguf_q8_0_prefill_bf16_bf16_out",
    "plan_gguf_k_gemv_build",
    "register_gguf_k_gemv_kernels",
]
