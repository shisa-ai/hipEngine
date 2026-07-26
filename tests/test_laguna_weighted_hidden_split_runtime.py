from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner_module
from hipengine.runtime import laguna_moe as moe_module
from scripts import laguna_target_ar_bench as benchmark
from tests._laguna_synthetic import make_laguna_info

_VARIANT = "laguna_top10_routed_hidden_out"
_KEY = KernelKey(
    "hip_gfx1100",
    "weighted_sum+moe_tail",
    "bf16",
    _VARIANT,
)


def _buffer(ptr: int) -> SimpleNamespace:
    return SimpleNamespace(ptr=ptr)


def test_weighted_hidden_split_owner_is_explicit_default_off_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = getattr(moe_module, "resolve_laguna_weighted_hidden_split", None)
    assert callable(resolver), "weighted-hidden runtime resolver must be present"
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_WEIGHTED_HIDDEN_SPLIT",
        None,
    ) is False
    assert backend_package_capability(
        "hip_gfx1151",
        "LAGUNA_WEIGHTED_HIDDEN_SPLIT",
        None,
    ) is None
    assert not resolver("hip_gfx1100")
    assert resolver("hip_gfx1100", True)
    assert not resolver("hip_gfx1100", False)
    assert not resolver("hip_gfx1151", True)

    config = laguna_gguf_config_from_metadata(make_laguna_info())
    default = moe_module.resolve_laguna_moe_plan(config, backend="hip_gfx1100")
    candidate = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_weighted_hidden_split=True,
    )
    direct_weighted = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        iq3_c1_down_schedule="serial_weighted",
        use_weighted_hidden_split=True,
    )
    unsupported = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1151",
        use_weighted_hidden_split=True,
    )
    assert default.weighted_hidden_split_key == _KEY
    assert default.weighted_hidden_split is None
    assert candidate.weighted_hidden_split_key == _KEY
    assert callable(candidate.weighted_hidden_split)
    assert _KEY not in default.kernel_keys
    assert _KEY in candidate.kernel_keys
    assert direct_weighted.weighted_hidden_split is None
    assert unsupported.weighted_hidden_split is None

    original_is_registered = moe_module.is_registered
    monkeypatch.setattr(
        moe_module,
        "is_registered",
        lambda key: False if key == _KEY else original_is_registered(key),
    )
    missing = moe_module.resolve_laguna_moe_plan(
        config,
        backend="hip_gfx1100",
        use_weighted_hidden_split=True,
    )
    assert missing.weighted_hidden_split is None
    assert _KEY not in missing.kernel_keys


def test_weighted_hidden_split_defers_only_c1_routed_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class StopRouted(Exception):
        pass

    class StopShared(Exception):
        pass

    def routed_sum(*_args, **_kwargs) -> None:
        calls.append("routed_sum")
        raise StopRouted

    plan = SimpleNamespace(
        backend="hip_gfx1100",
        hidden_size=3_072,
        expert_count=256,
        top_k=10,
        shared_ffn_size=1_024,
        routed_scaling_factor=2.5,
        weighted_hidden_split=object(),
        router_logits=lambda *_args, **_kwargs: None,
        router_select=lambda *_args, **_kwargs: None,
        routed_sum=routed_sum,
    )
    scratch = SimpleNamespace(
        plan=plan,
        max_rows=1,
        router_logits=_buffer(10),
        routing_scores=_buffer(11),
        selection_scores=_buffer(12),
        selected_experts=_buffer(13),
        routing_weights=_buffer(14),
        scaled_routing_weights=_buffer(15),
        expert_down=_buffer(16),
        routed_output=_buffer(17),
        shared_gate=_buffer(18),
        shared_up=_buffer(19),
    )
    weights = {
        name: SimpleNamespace(
            allocation=lambda _kind, ptr=100 + index: SimpleNamespace(tensor=_buffer(ptr))
        )
        for index, name in enumerate(
            (
                "ffn_gate_inp",
                "exp_probs_b",
                "ffn_gate_shexp",
                "ffn_up_shexp",
                "ffn_down_shexp",
            )
        )
    }
    layer = SimpleNamespace(weight=lambda name: weights[name])
    monkeypatch.setattr(moe_module, "validate_laguna_moe_layer", lambda *_args: None)
    monkeypatch.setattr(
        moe_module,
        "_launch_selected_gate_up",
        lambda *_args, **_kwargs: calls.append("selected_gate_up"),
    )
    monkeypatch.setattr(
        moe_module,
        "_launch_weighted_selected_down",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        moe_module,
        "_launch_selected_down",
        lambda *_args, **_kwargs: calls.append("selected_down"),
    )

    def stop_shared(*_args, **_kwargs):
        calls.append("shared_pair")
        raise StopShared

    monkeypatch.setattr(moe_module, "launch_gguf_linear_pair", stop_shared)

    with pytest.raises(StopRouted):
        moe_module.run_laguna_moe_c1_components(1, layer, scratch)
    assert calls == ["selected_gate_up", "selected_down", "routed_sum"]

    calls.clear()
    with pytest.raises(StopShared):
        moe_module.run_laguna_moe_c1_components(
            1,
            layer,
            scratch,
            defer_routed_sum=True,
        )
    assert calls == ["selected_gate_up", "selected_down", "shared_pair"]
    assert "defer_routed_sum" not in inspect.signature(
        moe_module.run_laguna_moe_rows
    ).parameters


def test_sparse_runner_uses_candidate_then_registered_rms_and_preserves_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    routed = _buffer(31)
    shared = _buffer(32)

    def components(*_args, **kwargs):
        calls.append(("components", kwargs["defer_routed_sum"]))
        return routed, shared

    def candidate(*args, **kwargs) -> None:
        calls.append(("candidate", args, kwargs))

    def rmsnorm(*args, **kwargs) -> None:
        calls.append(("rmsnorm", args, kwargs))

    def control_tail(*args, **kwargs) -> None:
        calls.append(("control_tail", args, kwargs))

    monkeypatch.setattr(moe_module, "run_laguna_moe_c1_components", components)
    monkeypatch.setattr(runner_module, "run_laguna_moe_c1_components", components)
    monkeypatch.setattr(
        runner_module,
        "launch_laguna_moe_tail_next_rmsnorm",
        control_tail,
    )

    session = object.__new__(runner_module.LagunaGGUFResidentSession)
    session.weights = SimpleNamespace(
        config=SimpleNamespace(block_count=48, hidden_size=3_072, rms_norm_eps=1e-6),
        layer=lambda _layer_id: SimpleNamespace(
            weight=lambda name: SimpleNamespace(
                allocation=lambda _kind: SimpleNamespace(tensor=_buffer(99))
            )
        ),
        root=lambda _name: SimpleNamespace(
            allocation=lambda _kind: SimpleNamespace(tensor=_buffer(100))
        ),
    )
    session.scratch = SimpleNamespace(
        norm=_buffer(10),
        post_attention=_buffer(11),
        hidden=_buffer(12),
        final_norm=_buffer(13),
    )
    session.moe_scratch = SimpleNamespace(
        plan=SimpleNamespace(top_k=10),
        expert_down=_buffer(20),
        scaled_routing_weights=_buffer(21),
        output=_buffer(22),
    )
    session.kernel_plan = SimpleNamespace(
        rmsnorm=rmsnorm,
        moe_tail_next_rmsnorm=object(),
        add=object(),
    )
    session.moe_plan = SimpleNamespace(weighted_hidden_split=candidate)
    session.libraries = SimpleNamespace(
        moe={},
        routed_sum="combine-library",
        gguf_ops="gguf-library",
    )
    session.runtime = "runtime"
    session._q5_shared_pair_variant = None

    session._run_sparse_ffn(1, SimpleNamespace(), stream=7)
    assert calls[0] == ("components", True)
    assert calls[1][0] == "candidate"
    assert calls[1][1] == (20, 21, 32, 11, 31, 12, 1, 10, 3_072)
    assert calls[1][2] == {
        "stream": 7,
        "library": "combine-library",
        "runtime": "runtime",
    }
    assert calls[2][0] == "rmsnorm"
    assert calls[2][1] == (12, 99, 10, 1, 3_072, 1e-6)
    assert calls[2][2] == {
        "stream": 7,
        "library": "gguf-library",
        "runtime": "runtime",
    }
    assert all(call[0] != "control_tail" for call in calls)

    calls.clear()
    session.moe_plan = SimpleNamespace(weighted_hidden_split=None)
    session._run_sparse_ffn(1, SimpleNamespace(), stream=7)
    assert calls[0] == ("components", False)
    assert calls[1][0] == "control_tail"
    assert all(call[0] not in {"candidate", "rmsnorm"} for call in calls)


def test_weighted_hidden_split_session_and_cli_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "use_weighted_hidden_split" in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters

    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert not benchmark._parse_args().enable_weighted_hidden_split

    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-weighted-hidden-split"],
    )
    args = benchmark._parse_args()
    assert args.enable_weighted_hidden_split

    captured: dict[str, object] = {}

    def session_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(benchmark, "LagunaGGUFResidentSession", session_factory)
    owner = SimpleNamespace(weights=object(), runtime=object())
    benchmark._session(owner, args)
    assert captured["use_weighted_hidden_split"] is True
