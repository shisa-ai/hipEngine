"""WPF-H8B exact scoped activation-pack reuse RED contract."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_TARGET_ARTIFACT = _ROOT / (
    "benchmarks/results/2026-08-03-gfx1100-laguna-q2-xl-post-h8a-"
    "activation-pack-reuse-target.json"
)
_TARGET_ARTIFACT_SHA256 = (
    "e0cf5d4bd92539435020b8c1c6269d33b5719295851197ca50700be07153856a"
)
_HIP_SOURCE = _ROOT / (
    "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.hip"
)
_HIP_SOURCE_SHA256 = (
    "1a06011ea6e7bda8e0b48fd357cbcbadaff76793a1b5c49bd217cc83d32b7110"
)
_PRE_IMPLEMENTATION_SHA256 = {
    "hipengine/kernels/hip_gfx1100/__init__.py": (
        "f0198b23989072062e785b9eb3887360e39a4ec1423754d5b4ca4d3ed05915e0"
    ),
    "hipengine/kernels/hip_gfx1100/quant/gguf_q5_k_f32_rocblas_prefill.py": (
        "0e018f37c0fe60ae3121019a4f67184a58ca2a05d49f41de05c4b52ff8878b2d"
    ),
    "hipengine/runtime/gguf_linear.py": (
        "f9ebb089b31937dcaea27f8bb43bfc2936b294d541c2841465c498d6f6dbd363"
    ),
    "hipengine/runtime/laguna_gguf_runner.py": (
        "c7c865094a375d0445e1f7cc72be15e9cd76b0f07991104a13fb5decda51f803"
    ),
    "hipengine/runtime/laguna_moe.py": (
        "295890f46bf198cbc7e8c99de80dd1d17b6c62a18f8fe244e3e172d5c9c59eb6"
    ),
}
_SUPPORTED_CAPABILITY = "LAGUNA_ACTIVATION_PACK_REUSE_SUPPORTED"
_SOURCE_CAPABILITY = "LAGUNA_ACTIVATION_PACK_REUSE"
_SESSION_PARAMETER = "use_activation_pack_reuse"
_EXPECTED_CLASSES = {
    "full_attention_qkv": (12, 3, 24),
    "swa_kv": (35, 2, 35),
    "shared_q5_gate_up": (46, 2, 46),
    "dense_q5_gate_up": (1, 2, 1),
    "layer47_shared_q6_gate_up": (1, 2, 1),
}
_EXPECTED_TOPOLOGY = {
    "packs_before": 330,
    "packs_after": 223,
    "redundant_packs": 107,
    "dispatches_before": 2_262,
    "dispatches_after": 2_155,
    "queues": 1,
    "streams": 1,
    "compiler_processes_allowed": 0,
}
_EXPECTED_STATE = {
    "next_token_id": 2930,
    "position": 511,
    "logits_sha256": (
        "e347d89b1be6548975a689383e99b8115d715f61542193b8376c3957ae140f70"
    ),
    "final_hidden_sha256": (
        "fc75ce0ac332e5210e72be904c7e65ba59e15e5124340363232ac502bb23896b"
    ),
    "post_layer_hidden_sha256": (
        "e41008b4b2ae4f6b734d85cdc3bc18beea70b1f2bebed38b6c570b4b6f201cce"
    ),
    "kv_sha256": (
        "f27787ed2a278a1d3b1666016aea5c4abc87a974859a11c398c7db7f5f502ef5"
    ),
}
_NO_SALVAGE = (
    "attention-only",
    "shared-only",
    "layer",
    "role",
    "prompt",
    "token",
    "length",
    "key-relaxation",
    "favorable-rerun",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scope_api():
    module = importlib.import_module("hipengine.kernels.activation_pack")
    return (
        getattr(module, "activation_pack_reuse_scope"),
        getattr(module, "launch_scoped_activation_pack"),
    )


def test_h8b_frozen_target_sources_state_topology_and_admission_contract() -> None:
    artifact_bytes = _TARGET_ARTIFACT.read_bytes()
    assert hashlib.sha256(artifact_bytes).hexdigest() == _TARGET_ARTIFACT_SHA256
    artifact = json.loads(artifact_bytes)

    assert artifact["status"] == "accepted_target_only_no_candidate_implementation"
    assert artifact["target"]["id"] == "WPF-H8B"
    assert artifact["target"]["implemented"] is False
    assert artifact["target"]["candidate_executed"] is False
    assert artifact["target"]["new_device_body"] is False
    assert artifact["target"]["new_jit_object"] is False
    assert artifact["target"]["new_allocation_bytes"] == 0
    assert artifact["target"]["new_workspace_bytes"] == 0
    assert artifact["target"]["source_default_changed"] is False
    assert artifact["target"]["cache_key"] == [
        "input_ptr",
        "activation_ptr",
        "rows",
        "in_features",
        "row_batch",
        "stream",
    ]
    assert "only after" in artifact["target"]["publication"]
    assert "scope exit always invalidates" in artifact["target"]["lifetime"]

    audit = artifact["execution_unchanged_audit"]
    classes = {
        item["name"]: (
            item["groups"],
            item["calls_per_group"],
            item["redundant_packs"],
        )
        for item in audit["classes"]
    }
    assert classes == _EXPECTED_CLASSES
    assert sum(item[0] for item in classes.values()) == 95
    assert sum(item[2] for item in classes.values()) == 107
    assert audit["current_pack_calls"] == _EXPECTED_TOPOLOGY["packs_before"]
    assert audit["minimal_pack_calls"] == _EXPECTED_TOPOLOGY["packs_after"]
    assert audit["redundant_pack_calls"] == _EXPECTED_TOPOLOGY["redundant_packs"]
    assert audit["state"] == {**_EXPECTED_STATE, "finite": True}

    model = artifact["profile_cost_model"]
    assert model["dispatches_before"] == _EXPECTED_TOPOLOGY["dispatches_before"]
    assert model["dispatches_after_model"] == _EXPECTED_TOPOLOGY["dispatches_after"]
    assert model["dispatch_delta_model"] == -107
    assert model["removable_total_ms"] == pytest.approx(2.342313031914895)
    assert model["not_a_performance_claim"] is True
    assert artifact["admission"]["red_first"] is True
    assert "all 95" in artifact["admission"]["complete_correctness"]
    assert "223 packs" in artifact["admission"]["trace"]
    assert "2,155" in artifact["admission"]["trace"]
    assert "no class/layer/role" in artifact["admission"]["no_salvage"]
    assert _NO_SALVAGE == (
        "attention-only",
        "shared-only",
        "layer",
        "role",
        "prompt",
        "token",
        "length",
        "key-relaxation",
        "favorable-rerun",
    )

    assert _sha256(_HIP_SOURCE) == _HIP_SOURCE_SHA256
    target_sources = artifact["source_sha256"]
    for relative, expected in _PRE_IMPLEMENTATION_SHA256.items():
        if relative in target_sources:
            assert target_sources[relative] == expected


def test_h8b_package_and_runtime_owner_follow_source_default() -> None:
    from hipengine.kernels import hip_gfx1100, hip_gfx1151
    from hipengine.runtime import laguna_gguf_runner as runner

    assert getattr(hip_gfx1100, _SUPPORTED_CAPABILITY) is True
    assert getattr(hip_gfx1100, _SOURCE_CAPABILITY) is True
    assert _SUPPORTED_CAPABILITY in hip_gfx1100.__all__
    assert _SOURCE_CAPABILITY in hip_gfx1100.__all__
    assert not hasattr(hip_gfx1151, _SUPPORTED_CAPABILITY)
    assert not hasattr(hip_gfx1151, _SOURCE_CAPABILITY)

    resolver = getattr(runner, "resolve_laguna_activation_pack_reuse")
    assert resolver("hip_gfx1100", None) is True
    assert resolver("hip_gfx1100", False) is False
    assert resolver("hip_gfx1100", True) is True
    assert resolver("hip_gfx1151", None) is False
    assert resolver("hip_gfx1151", False) is False
    with pytest.raises(ValueError, match="not supported"):
        resolver("hip_gfx1151", True)

    parameters = inspect.signature(runner.LagunaGGUFResidentSession.__init__).parameters
    assert _SESSION_PARAMETER in parameters
    init_source = inspect.getsource(runner.LagunaGGUFResidentSession.__init__)
    assert "resolve_laguna_activation_pack_reuse(" in init_source
    assert "self.use_activation_pack_reuse" in init_source


def test_h8b_scope_publishes_only_after_success_and_isolates_every_key_axis() -> None:
    activation_pack_reuse_scope, launch_scoped_activation_pack = _scope_api()
    calls: list[tuple[int, int, int, int, int]] = []

    def producer(
        input_ptr: int,
        activation_ptr: int,
        rows: int,
        in_features: int,
        *,
        stream: int,
    ) -> None:
        calls.append(
            (
                int(input_ptr),
                int(activation_ptr),
                int(rows),
                int(in_features),
                int(stream),
            )
        )

    def launch(key: tuple[int, int, int, int, int, int]) -> bool:
        input_ptr, activation_ptr, rows, hidden, row_batch, stream = key
        return launch_scoped_activation_pack(
            producer,
            input_ptr,
            activation_ptr,
            rows,
            hidden,
            row_batch=row_batch,
            stream=stream,
        )

    base = (0x1000, 0x2000, 512, 3_072, 5, 7)

    # No owner and a disabled owner always execute the retained producer.
    assert launch(base) is True
    assert launch(base) is True
    with activation_pack_reuse_scope(enabled=False):
        assert launch(base) is True
        assert launch(base) is True

    # An exact key reuses only after the successful first publication.
    with activation_pack_reuse_scope(enabled=True):
        assert launch(base) is True
        assert launch(base) is False
        for axis in range(6):
            changed = list(base)
            changed[axis] += 1
            changed_key = tuple(changed)
            assert launch(changed_key) is True
            assert launch(changed_key) is False

    # Serial scopes never inherit a publication.
    for _ in range(2):
        with activation_pack_reuse_scope(enabled=True):
            assert launch(base) is True
            assert launch(base) is False

    # A nested scope may alias and mutate the same activation plane. Entering
    # and leaving it therefore invalidates both child and parent publications.
    with activation_pack_reuse_scope(enabled=True):
        assert launch(base) is True
        assert launch(base) is False
        with activation_pack_reuse_scope(enabled=True):
            assert launch(base) is True
            assert launch(base) is False
        assert launch(base) is True
        assert launch(base) is False

    failures = 0

    def failing_producer(*args, **kwargs) -> None:
        nonlocal failures
        failures += 1
        if failures == 1:
            raise RuntimeError("injected pack failure")
        producer(*args, **kwargs)

    with activation_pack_reuse_scope(enabled=True):
        with pytest.raises(RuntimeError, match="injected pack failure"):
            launch_scoped_activation_pack(
                failing_producer,
                base[0],
                base[1],
                base[2],
                base[3],
                row_batch=base[4],
                stream=base[5],
            )
        assert (
            launch_scoped_activation_pack(
                failing_producer,
                base[0],
                base[1],
                base[2],
                base[3],
                row_batch=base[4],
                stream=base[5],
            )
            is True
        )
        assert (
            launch_scoped_activation_pack(
                failing_producer,
                base[0],
                base[1],
                base[2],
                base[3],
                row_batch=base[4],
                stream=base[5],
            )
            is False
        )
    assert failures == 2


def test_h8b_q5_q6_composites_share_exact_pack_without_changing_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation_pack_reuse_scope, _ = _scope_api()
    from hipengine.kernels.hip_gfx1100.quant import (
        gguf_q5_k_f32_rocblas_prefill as q5_f32,
    )

    events: list[tuple[str, tuple[int, ...]]] = []

    def pack(x_ptr, activation_ptr, rows, hidden, **kwargs) -> None:
        events.append(
            (
                "pack",
                (
                    int(x_ptr),
                    int(activation_ptr),
                    int(rows),
                    int(hidden),
                    int(kwargs["stream"]),
                ),
            )
        )

    def dequant(qweight_ptr, weight_ptr, hidden, outputs, **_kwargs) -> None:
        events.append(
            (
                "dequant",
                (int(qweight_ptr), int(weight_ptr), int(hidden), int(outputs)),
            )
        )

    def primitive(activation_ptr, weight_ptr, out_ptr, rows, hidden, outputs, **_kwargs):
        events.append(
            (
                "compute",
                (
                    int(activation_ptr),
                    int(weight_ptr),
                    int(out_ptr),
                    int(rows),
                    int(hidden),
                    int(outputs),
                ),
            )
        )

    resident_q5 = q5_f32._make_q5_resident_padded_compute_composite(
        pack,
        primitive,
        col_tile=16,
        row_batch=5,
        output_dtype="f32",
        weight_layout="tile_k_col",
    )
    monkeypatch.setattr(q5_f32, "gguf_q6_k_dequantize_f32_exact", dequant)
    q6 = q5_f32._make_q6_activation_tile_k_row_composite(
        pack,
        primitive,
        col_tile=16,
        row_batch=5,
        output_dtype="f32",
    )
    q5_pair = q5_f32._make_q5_padded_compute_composite(
        pack,
        dequant,
        primitive,
        col_tile=8,
        row_batch=4,
        output_dtype="bf16",
        weight_layout="tile_k_col",
        roles=((8, 4, "bf16", "tile_k_col", 3_072, 1_024),),
        role_label="H8B fixture",
    )

    common = {
        "rows": 512,
        "in_features": 3_072,
        "stream": 9,
        "library": object(),
        "runtime": object(),
    }
    with activation_pack_reuse_scope(enabled=True):
        resident_q5(
            0x1000,
            0x3000,
            0x4000,
            0x2000,
            out_features=6_144,
            **common,
        )
        q6(
            0x1000,
            0x5000,
            0x6000,
            0x7000,
            0x2000,
            out_features=1_024,
            **common,
        )
        q6(
            0x1000,
            0x8000,
            0x9000,
            0xA000,
            0x2000,
            out_features=1_024,
            **common,
        )
    assert [name for name, _ in events].count("pack") == 1
    assert [name for name, _ in events].count("dequant") == 2
    assert [name for name, _ in events].count("compute") == 3

    events.clear()
    with activation_pack_reuse_scope(enabled=True):
        for qweight, weight, out in ((0xB000, 0xC000, 0xD000), (0xE000, 0xF000, 0x11000)):
            q5_pair(
                0x12000,
                qweight,
                out,
                weight,
                0x13000,
                out_features=1_024,
                **common,
            )
    assert [name for name, _ in events].count("pack") == 1
    assert [name for name, _ in events].count("dequant") == 2
    assert [name for name, _ in events].count("compute") == 2
    assert _sha256(_HIP_SOURCE) == _HIP_SOURCE_SHA256


def test_h8b_runtime_owns_all_attention_dense_and_shared_groups_together() -> None:
    from hipengine.runtime import laguna_gguf_runner as runner
    from hipengine.runtime import laguna_moe

    attention_source = inspect.getsource(
        runner.LagunaGGUFResidentSession._launch_attention_projections_rows
    )
    dense_source = inspect.getsource(
        runner.LagunaGGUFResidentSession._run_dense_ffn_rows
    )
    shared_source = inspect.getsource(laguna_moe._launch_laguna_shared_rows)
    sparse_source = inspect.getsource(
        runner.LagunaGGUFResidentSession._run_sparse_ffn_rows
    )

    for source in (attention_source, dense_source, shared_source):
        assert "activation_pack_reuse_scope(" in source
    assert "enabled=self.use_activation_pack_reuse" in attention_source
    assert "enabled=self.use_activation_pack_reuse" in dense_source
    assert "enabled=use_activation_pack_reuse" in shared_source
    assert "use_activation_pack_reuse=self.use_activation_pack_reuse" in sparse_source
    assert "use_activation_pack_reuse: bool = False" in inspect.getsource(
        laguna_moe.run_laguna_moe_rows
    )

    # One policy bit owns the complete architecture-defined class set. There
    # must be no layer/role/prompt/token/length allowlist in the runtime.
    combined = "\n".join((attention_source, dense_source, shared_source, sparse_source))
    for forbidden in (
        "layer_id in",
        "layer_id ==",
        "attn_q-only",
        "shared-only",
        "next_token_id",
        "position == 511",
    ):
        assert forbidden not in combined
    assert sum(groups for groups, _calls, _removed in _EXPECTED_CLASSES.values()) == 95
    assert sum(removed for _groups, _calls, removed in _EXPECTED_CLASSES.values()) == 107
    assert _EXPECTED_TOPOLOGY["packs_before"] - 107 == _EXPECTED_TOPOLOGY["packs_after"]
    assert (
        _EXPECTED_TOPOLOGY["dispatches_before"] - 107
        == _EXPECTED_TOPOLOGY["dispatches_after"]
    )
