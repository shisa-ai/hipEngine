"""RED contracts for WPF-H6H bounded source-F16 Q6 raw fallbacks."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest

import hipengine.runtime.gguf_linear as gguf_linear_module
import hipengine.runtime.laguna_gguf_runner as runner_module
from hipengine.kernels import hip_gfx1100, hip_gfx1151
from hipengine.kernels.hip_gfx1100.quant import (
    gguf_q6_k_f16_rocblas_prefill as q6_f16,
)
from hipengine.kernels.registry import KernelKey, is_registered, register, resolve
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF
from hipengine.runtime.gguf_linear import GGUFLinearDispatch
from hipengine.runtime.laguna_gguf_runner import (
    LagunaPrefillScratchPlan,
    LagunaQ5F32OrderedScratch,
)

_H6H_POLICY = frozenset(
    {
        ("bf16", 9_216, 3_072),
        ("bf16", 12_288, 3_072),
        ("f32", 3_072, 9_216),
    }
)
_H6E_ORDERED_POLICY = {
    ("bf16", 3_072, 1_024): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch5"
    ),
    ("bf16", 1_024, 3_072): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch4"
    ),
    ("f32", 3_072, 72): "coltile8_rowbatch4",
    ("f32", 3_072, 1_024): (
        "weight_major_row_major_activation_tile_k_row_"
        "coltile16_rowbatch5"
    ),
}
_BROAD_H4_ONLY = (
    ("bf16", 1_024, 3_072),
    ("f32", 3_072, 72),
    ("f32", 3_072, 1_024),
)
_WEIGHT_F16_NBYTES = 75_497_472
_INPUT_F16_NBYTES = 12_582_912
_OUTPUT_F16_NBYTES = 9_437_184
_REQUIRED_NBYTES = 97_517_568
_ORDERED_SCRATCH_NBYTES = 161_120_256
_REMAINING_NBYTES = 63_602_688
_ROOT = Path(__file__).parents[1]


def _base(
    output_dtype: str,
    *,
    backend: str = "hip_gfx1100",
    quant: str = "gguf_q6_k",
) -> GGUFLinearDispatch:
    return GGUFLinearDispatch(
        KernelKey(
            backend,
            "linear",
            quant,
            f"prefill_bf16_{output_dtype}_out",
        ),
        "raw",
    )


def _selected(output_dtype: str) -> GGUFLinearDispatch:
    return GGUFLinearDispatch(
        KernelKey(
            "hip_gfx1100",
            "linear",
            "gguf_q6_k",
            f"f16_rocblas_source_bf16_{output_dtype}_out",
        ),
        "raw_q6_f16_rocblas",
    )


def test_h6h_bounded_runtime_policy_alias_dispatch_and_launch_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeze H6H before restoring the rejected broad-H4 runtime surface."""

    # Source production remains exact: the separately named H6H capability is
    # default-off, and gfx1151 must not inherit it through alias registration.
    assert hip_gfx1100.GGUF_Q6_F16_ROCBLAS_PREFILL_POLICY == frozenset()
    assert hip_gfx1100.GGUF_Q6_F16_ROCBLAS_PREFILL_H6H_POLICY == _H6H_POLICY
    assert hip_gfx1151.GGUF_Q6_F16_ROCBLAS_PREFILL_POLICY == frozenset()
    assert not hasattr(hip_gfx1151, "GGUF_Q6_F16_ROCBLAS_PREFILL_H6H_POLICY")
    assert hip_gfx1100.GGUF_Q6_F32_ORDERED_PREFILL_POLICY == (
        _H6E_ORDERED_POLICY
    )
    assert _H6H_POLICY.isdisjoint(_H6E_ORDERED_POLICY)
    assert _H6H_POLICY.isdisjoint(_BROAD_H4_ONLY)

    # H6H reuses the retained leaf byte-for-byte. It adds no kernel body and
    # keeps the mandatory unfused producer/cast chain registered on gfx1100.
    source_hashes = {
        "hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_f16_rocblas_prefill.py": (
            "4345f8a01a9bb0b56934d0738d858100010d192147b42db5a46bc765930ad313"
        ),
        "hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_f16_rocblas_prefill.hip": (
            "2d1cbd2e99a082e1a347308b338536b4a097a7512f53272374b02bf16b71c90f"
        ),
    }
    for relative, expected in source_hashes.items():
        assert hashlib.sha256((_ROOT / relative).read_bytes()).hexdigest() == expected
    q6_f16.register_gguf_q6_k_f16_rocblas_prefill_kernels(replace=True)
    for output_dtype in ("bf16", "f32"):
        key = _selected(output_dtype).key
        assert is_registered(key)
        assert not is_registered(
            KernelKey("hip_gfx1151", key.layer, key.quant, key.variant)
        )
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "dequant",
            "gguf_q6_k",
            "raw_f16_source_local64",
        )
    )
    assert is_registered(
        KernelKey(
            "hip_gfx1100",
            "dequant_cast",
            "gguf_q6_k",
            "raw_f16_bf16_input_source_local64",
        )
    )

    # The old H4 dedicated scratch owner/public selector must not return. H6H
    # aliases three non-overlapping slices of H5Y/H6E's existing allocation;
    # admission and device allocation totals therefore remain unchanged.
    assert not hasattr(runner_module, "LagunaQ6F16RocblasScratch")
    assert "use_q6_f16_rocblas" not in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters
    assert [field.name for field in fields(LagunaPrefillScratchPlan)] == [
        "matrix_rows",
        "attention_rows",
        "rows_nbytes",
        "moe_nbytes",
        "q5_f32_ordered_nbytes",
    ]
    assert LagunaQ5F32OrderedScratch.planned_nbytes(
        max_rows=512,
        use_activation_tile_k_row=True,
    ) == _ORDERED_SCRATCH_NBYTES
    scratch_base = 0x1000_0000
    scratch = LagunaQ5F32OrderedScratch(
        max_rows=512,
        buffer=SimpleNamespace(
            ptr=scratch_base,
            nbytes=_ORDERED_SCRATCH_NBYTES,
        ),
        activation_bf16_nbytes=10_125_312,
    )
    planes = scratch.q6_f16_rocblas_plane_slices()
    assert planes == (
        (scratch_base, _WEIGHT_F16_NBYTES),
        (scratch_base + _WEIGHT_F16_NBYTES, _INPUT_F16_NBYTES),
        (
            scratch_base + _WEIGHT_F16_NBYTES + _INPUT_F16_NBYTES,
            _OUTPUT_F16_NBYTES,
        ),
    )
    assert sum(nbytes for _, nbytes in planes) == _REQUIRED_NBYTES
    assert scratch.nbytes - _REQUIRED_NBYTES == _REMAINING_NBYTES

    Q6F16RocblasPrefillSession = getattr(
        gguf_linear_module,
        "Q6F16RocblasPrefillSession",
    )
    q6_f16_rocblas_prefill_session = getattr(
        gguf_linear_module,
        "q6_f16_rocblas_prefill_session",
    )
    dispatch_candidate = getattr(
        gguf_linear_module,
        "_q6_f16_rocblas_prefill_dispatch",
    )
    resolve_policy = getattr(
        runner_module,
        "_resolve_laguna_q6_f16_rocblas_policy",
    )

    assert resolve_policy("hip_gfx1100") == frozenset()
    monkeypatch.setattr(
        hip_gfx1100,
        "GGUF_Q6_F16_ROCBLAS_PREFILL_POLICY",
        _H6H_POLICY,
    )
    assert resolve_policy("hip_gfx1100") == _H6H_POLICY
    monkeypatch.setattr(
        hip_gfx1100,
        "GGUF_Q6_F16_ROCBLAS_PREFILL_POLICY",
        {("bf16", 9_216)},
    )
    with pytest.raises(ValueError, match="triples"):
        resolve_policy("hip_gfx1100")
    monkeypatch.setattr(
        hip_gfx1100,
        "GGUF_Q6_F16_ROCBLAS_PREFILL_POLICY",
        frozenset(),
    )

    session = Q6F16RocblasPrefillSession(
        min_rows=512,
        max_rows=512,
        shape_policy=_H6H_POLICY,
        weight_f16_ptr=planes[0][0],
        weight_f16_nbytes=planes[0][1],
        x_f16_ptr=planes[1][0],
        x_f16_nbytes=planes[1][1],
        out_f16_ptr=planes[2][0],
        out_f16_nbytes=planes[2][1],
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
    )

    # No context, every non-M512 request, all three broad-H4-only roles, wrong
    # quant/backend, capacity failure, and a registration miss retain exact raw
    # dispatch. Only the three fixed M512 roles may select source-F16.
    for output_dtype, in_features, out_features in _H6H_POLICY:
        base = _base(output_dtype)
        assert (
            dispatch_candidate(
                base,
                rows=512,
                in_features=in_features,
                out_features=out_features,
            )
            is base
        )
    with q6_f16_rocblas_prefill_session(session):
        for output_dtype, in_features, out_features in _H6H_POLICY:
            assert dispatch_candidate(
                _base(output_dtype),
                rows=512,
                in_features=in_features,
                out_features=out_features,
            ) == _selected(output_dtype)
        for rows, output_dtype, in_features, out_features in (
            (511, "bf16", 9_216, 3_072),
            (513, "bf16", 9_216, 3_072),
            *((512, *role) for role in _BROAD_H4_ONLY),
        ):
            base = _base(output_dtype)
            assert (
                dispatch_candidate(
                    base,
                    rows=rows,
                    in_features=in_features,
                    out_features=out_features,
                )
                is base
            )
        wrong_quant = _base("bf16", quant="gguf_q5_k")
        assert (
            dispatch_candidate(
                wrong_quant,
                rows=512,
                in_features=9_216,
                out_features=3_072,
            )
            is wrong_quant
        )
        wrong_backend = _base("bf16", backend="hip_gfx1151")
        assert (
            dispatch_candidate(
                wrong_backend,
                rows=512,
                in_features=9_216,
                out_features=3_072,
            )
            is wrong_backend
        )

    undersized = Q6F16RocblasPrefillSession(
        min_rows=512,
        max_rows=512,
        shape_policy=_H6H_POLICY,
        weight_f16_ptr=planes[0][0],
        weight_f16_nbytes=_WEIGHT_F16_NBYTES - 1,
        x_f16_ptr=planes[1][0],
        x_f16_nbytes=planes[1][1],
        out_f16_ptr=planes[2][0],
        out_f16_nbytes=planes[2][1],
        dequant_library="dequant-library",
        cast_library="cast-library",
        rocblas="rocblas-handle",
    )
    with q6_f16_rocblas_prefill_session(undersized):
        base = _base("bf16")
        assert (
            dispatch_candidate(
                base,
                rows=512,
                in_features=12_288,
                out_features=3_072,
            )
            is base
        )

    missing_key = _selected("bf16").key
    original_is_registered = gguf_linear_module.is_registered
    monkeypatch.setattr(
        gguf_linear_module,
        "is_registered",
        lambda key: key != missing_key and original_is_registered(key),
    )
    with q6_f16_rocblas_prefill_session(session):
        base = _base("bf16")
        assert (
            dispatch_candidate(
                base,
                rows=512,
                in_features=9_216,
                out_features=3_072,
            )
            is base
        )
    monkeypatch.setattr(
        gguf_linear_module,
        "is_registered",
        original_is_registered,
    )

    # The composed launch must pass only the raw resident weight and the three
    # aliased planes through the retained source-F16 ABI on the caller stream.
    key = _selected("f32").key
    original = resolve(
        backend=key.backend,
        layer=key.layer,
        quant=key.quant,
        variant=key.variant,
    )
    calls: list[tuple[tuple, dict]] = []

    def candidate(*args, **kwargs):
        calls.append((args, kwargs))

    raw = SimpleNamespace(tensor=SimpleNamespace(ptr=0x2000))
    weight = SimpleNamespace(
        backend="hip_gfx1100",
        spec=SimpleNamespace(layout=LAYOUT_RAW_GGUF, quant_key="gguf_q6_k"),
        allocation=lambda name: raw,
    )
    register(key, candidate, replace=True)
    gguf_linear_module.clear_gguf_linear_dispatch_cache()
    try:
        with q6_f16_rocblas_prefill_session(session):
            gguf_linear_module.launch_gguf_linear(
                weight,
                x_ptr=0x3000,
                out_ptr=0x4000,
                rows=512,
                in_features=3_072,
                out_features=9_216,
                output_dtype="f32",
                backend="hip_gfx1100",
                stream=7,
                runtime="runtime-sentinel",
            )
    finally:
        register(key, original, replace=True)
        gguf_linear_module.clear_gguf_linear_dispatch_cache()

    assert calls == [
        (
            (
                0x3000,
                0x2000,
                0x4000,
                planes[1][0],
                planes[0][0],
                planes[2][0],
                512,
                3_072,
                9_216,
            ),
            {
                "stream": 7,
                "dequant_library": "dequant-library",
                "cast_library": "cast-library",
                "rocblas": "rocblas-handle",
                "runtime": "runtime-sentinel",
            },
        )
    ]
