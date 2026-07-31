"""WPF-H6L exact IQ2 pair16 grouped rowbatch16 contract."""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.backends import load_backend_kernel_package
from hipengine.kernels.registry import KernelKey, is_registered, resolve
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime.laguna_gguf_runner import (
    LagunaPrefillChunkPolicy,
    LagunaPrefillScratchPlan,
)
from hipengine.runtime.laguna_moe import resolve_laguna_moe_plan
from tests._laguna_synthetic import make_laguna_info
from tests.test_gguf_iq2_xs_gemv import HIP_AVAILABLE, _make_iq2_xs_weight
from tests.test_gguf_iq_gemv import (
    _bf16_u16_to_f32,
    _f32_to_bf16_u16,
    _make_x,
    _selected_reference,
)
from tests.test_gguf_iq_selected_prefill import _compact_meta, _run_dual_grouped

_IN_FEATURES = 3072
_OUT_FEATURES = 1024
_NUM_EXPERTS = 256
_ROW_BATCH_CONTROL = 8
_ROW_BATCH_CANDIDATE = 16
_WRAPPER_NAME = (
    "gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_k3072_n1024_"
    "e256_pair16_rowbatch16_bf16_bf16_out"
)
_SYMBOL = f"hipengine_{_WRAPPER_NAME}"
_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_k3072_n1024_e256_pair16_"
    "rowbatch16_bf16_bf16_out"
)
_CONTROL_VARIANT = (
    "selected_dual_silu_grouped_prefill_compact_pair16_rowbatch8_bf16_bf16_out"
)
_TEMPLATE_NAME = (
    "gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_kernel"
)
_BODY_HASHES = {
    "load_iq2_pair16": "a87eeae5027bd71139844f0b9a35afd2a227a1042b8bebe00fd80b78fe9fccc7",
    "dot_iq2_pair16": "c08941c9a8a32560de4cf29192bd73a05da7dbcfbe5b5c8026887db0abee4db0",
    "reduce_local64_pairs_batched": (
        "446b1dbdc6acbfead726e38b08a752cdf2eda51922d889b925aa2eab6ed89b90"
    ),
    "pair16_grouped_kernel_template": (
        "8f869bc7483934808681a7299d2cf8035550113e2734948131f6422da084282c"
    ),
}
_BODY_TOKENS = {
    "load_iq2_pair16": "__device__ inline IQ2Pair16 load_iq2_pair16(",
    "dot_iq2_pair16": "__device__ inline float dot_iq2_pair16(",
    "reduce_local64_pairs_batched": (
        "__device__ inline void reduce_local64_pairs_batched("
    ),
    "pair16_grouped_kernel_template": f"__global__ void {_TEMPLATE_NAME}(",
}
_SAMPLE_COLS = np.asarray([0, 1, 511, 512, 1022, 1023], dtype=np.int64)
_PRODUCTION_WORKSPACE_BYTES = 161_120_256
_PRODUCTION_TOTAL_SCRATCH_BYTES = 600_141_856


def _module():
    return importlib.import_module(
        "hipengine.kernels.hip_gfx1100.quant.gguf_iq_selected_prefill"
    )


def _candidate():
    return getattr(_module(), _WRAPPER_NAME)


def _candidate_key(backend: str = "hip_gfx1100") -> KernelKey:
    return KernelKey(backend, "moe_linear", "gguf_iq2_xs", _VARIANT)


def _function_source(source: str, declaration: str) -> str:
    start = source.index(declaration)
    body_start = source.index("{", start)
    depth = 0
    for offset in range(body_start, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise AssertionError(f"unterminated function: {declaration}")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(scope="module")
def grouped_library():
    if not HIP_AVAILABLE:
        pytest.skip("HIP runtime is not available")
    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    compiler_version = Path(version_file).read_text() if version_file else None
    return _module().build_gguf_iq_selected_prefill(
        load=True,
        compiler_version=compiler_version,
        require_cached=os.environ.get("HIPENGINE_REQUIRE_CACHED_BUILD") == "1",
    )


@pytest.fixture(scope="module")
def iq2_weights() -> tuple[np.ndarray, np.ndarray]:
    gate = _make_iq2_xs_weight(
        1,
        _OUT_FEATURES,
        _IN_FEATURES,
        seed=0x6A10,
    )
    up = _make_iq2_xs_weight(
        1,
        _OUT_FEATURES,
        _IN_FEATURES,
        seed=0x6A11,
    )
    return gate, up


def test_h6l_registry_source_and_production_ownership() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151

    assert hip_gfx1100.LAGUNA_SELECTED_GATE_UP_MODE == "grouped_pair16"
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    plan = resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    assert plan.grouped_pair16_gate_up_keys["gguf_iq2_xs"].variant == _VARIANT
    assert plan.grouped_pair16_gate_up_routes["gguf_iq2_xs"].abi == (
        "grouped_raw_iq_dual_silu"
    )
    policy = LagunaPrefillChunkPolicy.resolve(
        context_length=4096,
        matrix_rows=512,
        attention_rows=128,
    )
    scratch = LagunaPrefillScratchPlan.build(
        config,
        plan,
        policy=policy,
        use_q5_f32_ordered=True,
        use_q5_activation_tile_k_row=True,
    )
    assert scratch.q5_f32_ordered_nbytes == _PRODUCTION_WORKSPACE_BYTES
    assert scratch.total_nbytes == _PRODUCTION_TOTAL_SCRATCH_BYTES

    load_backend_kernel_package("hip_gfx1151")
    assert not is_registered(_candidate_key("hip_gfx1151"))
    assert hip_gfx1151.LAGUNA_SELECTED_GATE_UP_MODE == (
        "mmq128x32_d8_f32_wavecols_direct_doublebuf_rawprefetch_ge512"
    )

    source = Path(_module().__file__).with_suffix(".hip").read_text()
    actual_hashes = {
        name: _sha256(_function_source(source, declaration))
        for name, declaration in _BODY_TOKENS.items()
    }
    assert actual_hashes == _BODY_HASHES

    candidate = _candidate()
    assert candidate.__name__ == _WRAPPER_NAME
    assert resolve(
        backend="hip_gfx1100",
        layer="moe_linear",
        quant="gguf_iq2_xs",
        variant=_VARIANT,
    ) is candidate
    assert source.count(_SYMBOL) == 1
    assert source.count(f"({_TEMPLATE_NAME}<16>)") == 1
    wrapper_source = _function_source(source, f'extern "C" int {_SYMBOL}(')
    assert "in_features != 3072" in wrapper_source
    assert "out_features != 1024" in wrapper_source
    assert "num_experts != 256" in wrapper_source
    assert f"({_TEMPLATE_NAME}<16>)" in wrapper_source
    assert "64," in wrapper_source


def test_h6l_strict_shape_preflight_rejects_before_hip_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    candidate = _candidate()
    launch_attempts = 0

    def fail_if_launched(*_: object, **__: object) -> None:
        nonlocal launch_attempts
        launch_attempts += 1
        raise AssertionError("invalid H6L shape reached the HIP launch helper")

    monkeypatch.setattr(module, "_launch_grouped_dual", fail_if_launched)
    common = dict(
        compact_rows=17,
        in_features=_IN_FEATURES,
        out_features=_OUT_FEATURES,
        num_experts=_NUM_EXPERTS,
    )
    for changed, message in (
        ({"compact_rows": 0}, "compact_rows must be positive"),
        ({"in_features": 1024}, "exactly 3072"),
        ({"out_features": 512}, "exactly 1024"),
        ({"num_experts": 255}, "exactly 256"),
    ):
        with pytest.raises(ValueError, match=message):
            candidate(1, 2, 3, 4, 5, **(common | changed))
    assert launch_attempts == 0


def _cpu_fused_samples(
    x_bf16: np.ndarray,
    selected: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    sample_rows: np.ndarray,
) -> np.ndarray:
    gate_bits = _selected_reference(
        x_bf16[sample_rows],
        selected[sample_rows],
        gate[:, _SAMPLE_COLS, :],
        GGMLQuantizationType.IQ2_XS,
    )
    up_bits = _selected_reference(
        x_bf16[sample_rows],
        selected[sample_rows],
        up[:, _SAMPLE_COLS, :],
        GGMLQuantizationType.IQ2_XS,
    )
    gate_f32 = _bf16_u16_to_f32(gate_bits)
    up_f32 = _bf16_u16_to_f32(up_bits)
    return _f32_to_bf16_u16(
        gate_f32
        * (np.float32(1.0) / (np.float32(1.0) + np.exp(-gate_f32)))
        * up_f32
    )


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
@pytest.mark.parametrize(
    "rows_for_expert_zero",
    [
        pytest.param(1, id="rows1"),
        pytest.param(7, id="rows7"),
        pytest.param(8, id="rows8"),
        pytest.param(9, id="rows9"),
        pytest.param(15, id="rows15"),
        pytest.param(16, id="rows16"),
        pytest.param(17, id="rows17"),
        pytest.param(512, id="rows512"),
    ],
)
def test_h6l_complete_outputs_match_rowbatch8_and_cpu_at_boundaries(
    grouped_library,
    iq2_weights: tuple[np.ndarray, np.ndarray],
    rows_for_expert_zero: int,
) -> None:
    gate, up = iq2_weights
    meta = _compact_meta([rows_for_expert_zero] + [0] * (_NUM_EXPERTS - 1))
    assert meta.num_experts == _NUM_EXPERTS
    assert meta.counts[1] == 0
    x_bf16 = _f32_to_bf16_u16(_make_x(meta.compact_rows, _IN_FEATURES))
    control = getattr(
        _module(),
        "gguf_iq2_xs_selected_dual_silu_grouped_prefill_compact_pair16_"
        "rowbatch8_bf16_bf16_out",
    )
    expected = _run_dual_grouped(
        control,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
        fused_silu=True,
    )
    sample_rows = np.unique(
        np.asarray(
            [
                0,
                7,
                8,
                15,
                16,
                rows_for_expert_zero // 2,
                rows_for_expert_zero - 1,
            ]
        ).clip(0, rows_for_expert_zero - 1)
    )
    cpu = _cpu_fused_samples(
        x_bf16,
        meta.selected,
        gate,
        up,
        sample_rows,
    )
    np.testing.assert_array_equal(expected[np.ix_(sample_rows, _SAMPLE_COLS)], cpu)
    assert np.isfinite(_bf16_u16_to_f32(cpu)).all()

    candidate = _candidate()
    actual = _run_dual_grouped(
        candidate,
        grouped_library,
        x_bf16=x_bf16,
        meta=meta,
        gate=gate,
        up=up,
        wmma=False,
        fused_silu=True,
    )
    np.testing.assert_array_equal(actual, expected)
