"""Explicit layer-execution boundary for the attention and FFN components.

These are behavioural CPU regressions for the standalone extraction of one
Qwen hybrid GGUF layer's attention and feed-forward halves
(``hipengine/runtime/qwen35_gguf_layer_attn.py`` and
``qwen35_gguf_layer_ffn.py``). They pin four things:

* the dependency boundary - neither module imports the runner or a session, and
  no entry point accepts a runner, session, proxy or mixin;
* the launch contract - the leaves and the free bodies reach the same
  launchers, in the same order, with the same scalar arguments, driven only by
  the explicit weights/config/geometry/views/state plus the attention selectors
  and policy predicates (attention) and the kernel/ops/flags tables (FFN);
* branch-dependent laziness - the minimal attention norm and alpha/beta leaves
  need only the hidden geometry (no full-attention views, no GDN state, no
  scratch, no weights), and the DMS full-attention override returns before any
  ordinary scratch, weight or selector is touched;
* freshness - nothing is memoised inside either module, so a selector or a
  module-level kernel patched between two calls is observed by the second call,
  and the split-K buffers are read after the growth callback rather than from a
  snapshot taken before it.

Every callable a path is not expected to use raises ``AssertionError``, so an
unexpected dependency fails the test instead of silently reaching a real
kernel. No GPU, no ROCm runtime, no environment mutation.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from hipengine.core.dtype import DType
from hipengine.runtime import qwen35_gguf_layer_attn as attn
from hipengine.runtime import qwen35_gguf_layer_ffn as ffn


@pytest.mark.parametrize("selected", [False, True])
def test_runner_dense_policy_memo_is_lazy_and_published(monkeypatch, selected) -> None:
    from hipengine.runtime import qwen35_gguf_runner as runner_module

    runner = NS()
    calls = []

    def resolve(owner, **kwargs):
        assert owner is runner
        calls.append(kwargs)
        return selected

    monkeypatch.setattr(runner_module, "_gguf_dense_down_residual_decode_fused", resolve)
    ops = runner_module._layer_ffn_ops(runner)
    assert not calls
    assert not hasattr(runner, "_dense_down_residual_decode_c1")
    assert ops.dense_down_residual_fused(rows=1, in_features=64, out_features=32) is selected
    assert runner._dense_down_residual_decode_c1 is selected
    assert calls == [dict(rows=1, in_features=64, out_features=32)]

_MODULES = (attn, ffn)
_RUNTIME = NS(name="fake-hip-runtime")
_BACKEND = "hip_gfx1100"


# --------------------------------------------------------------------------
# recording and failing callables
# --------------------------------------------------------------------------
class _Log:
    """Ordered call log; entries are ``(label, args, kwargs)``."""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple, dict]] = []

    def fn(self, label: str, *, returns=None):
        def call(*args, **kwargs):
            self.events.append((label, args, kwargs))
            return returns

        return call

    def value(self, label: str, result):
        """A lookup whose *return value* is the point, not a launch."""

        def call(*args, **kwargs):
            self.events.append((label, args, kwargs))
            return result

        return call

    def labels(self) -> list[str]:
        return [label for label, _, _ in self.events]

    def slots(self) -> list[tuple[str, str | None]]:
        """``(label, first-argument weight slot)`` for weight-first launches."""
        return [
            (label, getattr(args[0], "slot", None) if args else None)
            for label, args, _ in self.events
        ]

    def one(self, label: str) -> tuple[tuple, dict]:
        matches = [(args, kwargs) for name, args, kwargs in self.events if name == label]
        assert len(matches) == 1, (label, self.labels())
        return matches[0]


def _fail(name: str):
    def call(*args, **kwargs):
        raise AssertionError(f"{name} must not be called on this path")

    return call


def _buf(ptr: int, **extra):
    return NS(ptr=int(ptr), **extra)


def _weight(slot: str, ptr: int, *, quant: str = "gguf_q8_0_t16_v1", layout: str = "gguf_q8_0_t16"):
    allocation = NS(tensor=NS(ptr=int(ptr)))
    return NS(
        slot=slot,
        ptr=int(ptr),
        backend=_BACKEND,
        spec=NS(quant_key=quant, layout=layout),
        allocation=lambda name="raw": allocation,
    )


class _Layer:
    """Only the weight slots the exercised path is allowed to read exist."""

    def __init__(self, **weights) -> None:
        self._weights = dict(weights)

    def weight(self, slot: str):
        try:
            return self._weights[slot]
        except KeyError:
            raise AssertionError(f"unexpected weight slot {slot!r}") from None


class _Stage:
    def __init__(self) -> None:
        self.marks: list[str] = []

    def mark(self, name: str, *aliases: str) -> None:
        self.marks.append(name)
        self.marks.extend(aliases)


# --------------------------------------------------------------------------
# module boundary
# --------------------------------------------------------------------------
def _module_functions(module) -> list[ast.FunctionDef]:
    tree = ast.parse(Path(module.__file__).read_text())
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)]


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.rsplit(".", 1)[-1])
def test_layer_modules_do_not_import_the_runner_or_a_session(module) -> None:
    tree = ast.parse(Path(module.__file__).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    forbidden = {
        "hipengine.runtime.qwen35_gguf_runner",
        "hipengine.runtime.qwen35_gguf_prefix_state",
        "hipengine.runtime.prefill",
        "hipengine.runtime.gguf_native_spec_cycle",
    }
    assert forbidden.isdisjoint(imported), sorted(forbidden.intersection(imported))
    # The two layer components are siblings: neither may import the other.
    siblings = {
        "hipengine.runtime.qwen35_gguf_layer_attn",
        "hipengine.runtime.qwen35_gguf_layer_ffn",
    } - {module.__name__}
    assert siblings.isdisjoint(imported), sorted(siblings.intersection(imported))


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.rsplit(".", 1)[-1])
def test_layer_modules_take_no_runner_session_or_proxy_argument(module) -> None:
    forbidden = {"self", "runner", "session", "host", "ctx", "context", "proxy", "mixin"}
    for node in _module_functions(module):
        names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
        assert not (names & forbidden), (node.name, sorted(names & forbidden))
        assert node.args.kwarg is None, node.name
        assert node.args.vararg is None, node.name


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.rsplit(".", 1)[-1])
def test_layer_modules_do_not_resolve_runner_globals_or_owner_attributes(module) -> None:
    tree = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "globals", "module rebinds through globals()"
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"__dict__", "_dms_decode_owner"}, node.attr
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "_dms_decode_owner" not in node.value, node.value


@pytest.mark.parametrize("module", _MODULES, ids=lambda module: module.__name__.rsplit(".", 1)[-1])
def test_layer_modules_hold_no_mixin_or_mutable_shared_state(module) -> None:
    tree = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        assert not node.name.endswith(("Mixin", "Proxy")), node.name
        bases = {base.id for base in node.bases if isinstance(base, ast.Name)}
        assert bases <= {"object", "Protocol"}, (node.name, bases)
    # The explicit input records are frozen: they cannot act as a mutable session.
    for name in ("AttentionGeometry", "AttentionSelectors", "AttentionPolicy"):
        cls = getattr(attn, name)
        assert cls.__dataclass_params__.frozen, name
    for name in ("FfnGeometry", "FfnKernels", "FfnOps", "FfnFlags"):
        cls = getattr(ffn, name)
        assert cls.__dataclass_params__.frozen, name


def test_attention_norm_leaf_has_no_views_scratch_or_layer_input() -> None:
    parameters = set(inspect.signature(attn.run_attention_norm_rows).parameters)
    assert parameters.isdisjoint({"views", "scratch", "layer", "layer_id", "cfg", "backend"})


# --------------------------------------------------------------------------
# attention: shared leaves
# --------------------------------------------------------------------------
def _attn_geometry(**overrides) -> attn.AttentionGeometry:
    fields = dict(
        hidden_size=256,
        q_width=512,
        kv_width=256,
        linear_qkv_width=768,
        ssm_value_dim=512,
        fp16_recurrent_state=None,
    )
    fields.update(overrides)
    return attn.AttentionGeometry(**fields)


def _policy(**overrides) -> attn.AttentionPolicy:
    fields = {
        name: _fail(f"policy.{name}")
        for name in attn.AttentionPolicy.__dataclass_fields__
    }
    fields.update(overrides)
    return attn.AttentionPolicy(**fields)


def _selectors(**overrides) -> attn.AttentionSelectors:
    fields = {
        name: _fail(f"selectors.{name}")
        for name in attn.AttentionSelectors.__dataclass_fields__
    }
    # A policy value, not a lazily resolved callable.
    fields["q8_t16_rowtile_all"] = None
    fields.update(overrides)
    return attn.AttentionSelectors(**fields)


def _attn_config(**overrides):
    fields = dict(
        hidden_size=256,
        rms_norm_eps=1e-6,
        head_count=4,
        head_count_kv=2,
        key_length=64,
        rope_dimension_count=32,
        ssm_time_step_rank=8,
        ssm_conv_kernel=4,
        ssm_inner_size=768,
        ssm_group_count=2,
        ssm_state_size=64,
        is_moe=False,
    )
    fields.update(overrides)
    return NS(**fields)


def _norm_call(**overrides) -> dict:
    kwargs = dict(
        hidden_ptr=0x10,
        hidden_f32_ptr=None,
        weight_ptr=0x20,
        out_ptr=0x30,
        rows=3,
        eps=1e-6,
        stream=7,
        runtime=_RUNTIME,
        geometry=_attn_geometry(),
        selectors=_selectors(),
        policy=_policy(),
    )
    kwargs.update(overrides)
    return kwargs


def test_attention_norm_leaf_selects_then_forwards_exact_scalars() -> None:
    log = _Log()
    selections: list[dict] = []

    def selector(**kwargs):
        selections.append(kwargs)
        return log.fn("kernel")

    out = attn.run_attention_norm_rows(
        **_norm_call(selectors=_selectors(norm_residual_kernel=selector))
    )

    assert out is None
    assert selections == [dict(layer="rmsnorm", rows=3, hidden_size=256)]
    assert log.events == [
        (
            "kernel",
            (0x10, 0x20, 0x30),
            dict(rows=3, hidden_size=256, eps=1e-6, stream=7, runtime=_RUNTIME),
        )
    ]


def test_attention_norm_leaf_f32_input_uses_the_plain_leaf_without_the_flag(
    monkeypatch,
) -> None:
    log = _Log()
    monkeypatch.setattr(attn, "gguf_rmsnorm_f32_f32_weight", log.fn("f32_in"))
    monkeypatch.setattr(attn, "gguf_rmsnorm_f32_f32_weight_out_f32", _fail("f32_in_out"))
    monkeypatch.setattr(attn, "f32_to_bf16", _fail("cast"))

    out = attn.run_attention_norm_rows(
        **_norm_call(
            hidden_f32_ptr=0x11,
            out_f32_ptr=0x33,
            policy=_policy(verify_f32_attention_norm=lambda: False),
        )
    )

    assert out is None
    assert log.events == [
        (
            "f32_in",
            (0x11, 0x20, 0x30),
            dict(rows=3, hidden_size=256, eps=1e-6, stream=7, runtime=_RUNTIME),
        )
    ]


def test_attention_norm_leaf_f32_verify_writes_f32_then_casts(monkeypatch) -> None:
    log = _Log()
    monkeypatch.setattr(attn, "gguf_rmsnorm_f32_f32_weight", _fail("f32_in"))
    monkeypatch.setattr(attn, "gguf_rmsnorm_f32_f32_weight_out_f32", log.fn("f32_in_out"))
    monkeypatch.setattr(attn, "f32_to_bf16", log.fn("cast"))

    out = attn.run_attention_norm_rows(
        **_norm_call(
            hidden_f32_ptr=0x11,
            out_f32_ptr=0x33,
            policy=_policy(verify_f32_attention_norm=lambda: True),
            selectors=_selectors(cast_library=lambda: "cast-lib"),
        )
    )

    assert out == 0x33
    assert log.labels() == ["f32_in_out", "cast"]
    assert log.one("f32_in_out") == (
        (0x11, 0x20, 0x33),
        dict(rows=3, hidden_size=256, eps=1e-6, stream=7, runtime=_RUNTIME),
    )
    assert log.one("cast") == (
        (0x33, 0x30, 3 * 256),
        dict(stream=7, library="cast-lib", runtime=_RUNTIME),
    )


def test_attention_norm_leaf_verify_flag_without_f32_out_stays_bf16(monkeypatch) -> None:
    log = _Log()
    monkeypatch.setattr(attn, "gguf_rmsnorm_f32_f32_weight", log.fn("f32_in"))
    monkeypatch.setattr(attn, "gguf_rmsnorm_f32_f32_weight_out_f32", _fail("f32_in_out"))

    out = attn.run_attention_norm_rows(
        **_norm_call(
            hidden_f32_ptr=0x11,
            out_f32_ptr=None,
            policy=_policy(verify_f32_attention_norm=lambda: True),
        )
    )

    assert out is None
    assert log.labels() == ["f32_in"]


def test_attention_norm_leaf_needs_only_hidden_geometry() -> None:
    """The minimal leaf never reads a full-attention field, scratch or weight.

    ``AttentionGeometry`` leaves every unread field unsupplied, so the leaf is
    exercised with ``hidden_size`` alone rather than with placeholder values.
    """

    log = _Log()
    out = attn.run_attention_norm_rows(
        **_norm_call(
            geometry=attn.AttentionGeometry(hidden_size=256),
            selectors=_selectors(norm_residual_kernel=lambda **_: log.fn("kernel")),
            policy=_policy(),
        )
    )

    assert out is None
    assert log.labels() == ["kernel"]


def _alpha_beta_layer(**overrides) -> _Layer:
    fields = dict(
        ssm_alpha=_weight("ssm_alpha", 0x21),
        ssm_beta=_weight("ssm_beta", 0x22),
    )
    fields.update(overrides)
    return _Layer(**fields)


def _alpha_beta_call(**overrides) -> dict:
    kwargs = dict(
        layer=_alpha_beta_layer(),
        norm_ptr=0x40,
        norm_f32_ptr=0x41,
        views=_alpha_beta_views(),
        cfg=_attn_config(),
        geometry=_attn_geometry(),
        rows=1,
        stream=7,
        runtime=_RUNTIME,
        selectors=_selectors(),
        policy=_policy(),
    )
    kwargs.update(overrides)
    return kwargs


def _alpha_beta_views(**overrides) -> attn.LinearAttentionViews:
    """The projection leaf's own buffers; the GDN state is deliberately absent."""

    fields = dict(
        norm=_buf(0x101),
        norm_f32=None,
        linear_qkv=_buf(0x102),
        linear_z=_buf(0x103),
        linear_alpha=_buf(0x31),
        linear_beta=_buf(0x32),
        linear_alpha_f32=_buf(0x51),
        linear_beta_f32=_buf(0x52),
        conv_out=_buf(0x106),
        recurrent_out=_buf(0x107),
        recurrent_bf16=_buf(0x108),
        conv_state=None,
        recurrent_state=None,
        q8_1_workspace_owner=None,
    )
    fields.update(overrides)
    return attn.LinearAttentionViews(**fields)


def _f32_out_support(monkeypatch, log: _Log) -> None:
    monkeypatch.setattr(attn, "launch_gguf_linear", log.fn("linear"))
    monkeypatch.setattr(attn, "f32_to_bf16", log.fn("cast"))
    monkeypatch.setattr(attn, "launch_gguf_linear_pair", _fail("pair"))


def test_alpha_beta_leaf_f32_singletons_f32_out_route(monkeypatch) -> None:
    log = _Log()
    _f32_out_support(monkeypatch, log)
    layer = _alpha_beta_layer()

    route = attn.run_linear_attention_alpha_beta_rows(
        **_alpha_beta_call(
            layer=layer,
            selectors=_selectors(
                cast_library=lambda: "cast-lib",
                linear_supports_f32_activation_f32_output=lambda weight, *, rows: True,
            ),
            policy=_policy(verify_f32_linear_projections=lambda: True),
        )
    )

    assert route == "f32_singletons_f32_out"
    assert log.slots() == [
        ("linear", "ssm_alpha"),
        ("linear", "ssm_beta"),
        ("cast", None),
        ("cast", None),
    ]
    assert log.events[0][1] == (layer.weight("ssm_alpha"), 0x41, 0x51)
    assert log.events[0][2] == dict(
        rows=1,
        in_features=256,
        out_features=8,
        activation_dtype="f32",
        output_dtype="f32",
        stream=7,
        runtime=_RUNTIME,
        q8_t16_rowtile_all=None,
    )
    assert log.events[1][1] == (layer.weight("ssm_beta"), 0x41, 0x52)
    assert log.events[2][1] == (0x51, 0x31, 1 * 8)
    assert log.events[3][1] == (0x52, 0x32, 1 * 8)


def test_alpha_beta_leaf_dense_q8_dp4a_f32_out_route(monkeypatch) -> None:
    log = _Log()
    _f32_out_support(monkeypatch, log)
    workspace = NS(name="q8_1-workspace-owner")
    seen: list[tuple] = []

    def dp4a_out(*args, **kwargs):
        seen.append((args, kwargs))
        return True

    route = attn.run_linear_attention_alpha_beta_rows(
        **_alpha_beta_call(
            views=_alpha_beta_views(q8_1_workspace_owner=workspace),
            selectors=_selectors(
                cast_library=lambda: "cast-lib",
                linear_supports_f32_activation_f32_output=lambda weight, *, rows: False,
                try_launch_dense_q8_pair_dp4a_f32_out=dp4a_out,
            ),
            policy=_policy(verify_f32_linear_projections=lambda: True),
        )
    )

    assert route == "dense_q8_dp4a_f32_out"
    assert len(seen) == 1
    args, kwargs = seen[0]
    # The q8_1 owner reaches the launcher unchanged in the workspace slot.
    assert args[5] is workspace
    assert kwargs == dict(
        rows=1,
        in_features=256,
        out_features_a=8,
        out_features_b=8,
        stream=7,
        runtime=_RUNTIME,
    )
    assert log.labels() == ["cast", "cast"]


def test_alpha_beta_leaf_dense_q8_dp4a_f32_route(monkeypatch) -> None:
    log = _Log()
    monkeypatch.setattr(attn, "launch_gguf_linear", _fail("linear"))
    workspace = NS(name="q8_1-workspace-owner")

    route = attn.run_linear_attention_alpha_beta_rows(
        **_alpha_beta_call(
            views=_alpha_beta_views(
                linear_alpha_f32=None,
                linear_beta_f32=None,
                q8_1_workspace_owner=workspace,
            ),
            selectors=_selectors(
                try_launch_dense_q8_pair_dp4a_f32=lambda *a, **k: a[5] is workspace
            ),
            policy=_policy(
                verify_f32_linear_projections=lambda: False,
                verify_f32_alpha_beta=lambda: True,
            ),
        )
    )

    assert route == "dense_q8_dp4a_f32"
    assert log.labels() == []


def test_alpha_beta_leaf_f32_singletons_route(monkeypatch) -> None:
    log = _Log()
    monkeypatch.setattr(attn, "launch_gguf_linear", log.fn("linear"))
    monkeypatch.setattr(attn, "launch_gguf_linear_pair", _fail("pair"))

    route = attn.run_linear_attention_alpha_beta_rows(
        **_alpha_beta_call(
            views=_alpha_beta_views(linear_alpha_f32=None, linear_beta_f32=None),
            selectors=_selectors(
                try_launch_dense_q8_pair_dp4a_f32=lambda *a, **k: False,
                linear_supports_f32_activation=lambda weight: True,
            ),
            policy=_policy(
                verify_f32_linear_projections=lambda: False,
                verify_f32_alpha_beta=lambda: True,
            ),
        )
    )

    assert route == "f32_singletons"
    assert log.slots() == [("linear", "ssm_alpha"), ("linear", "ssm_beta")]
    assert log.events[0][2]["activation_dtype"] == "f32"
    assert "output_dtype" not in log.events[0][2]


def test_alpha_beta_leaf_moe_singletons_route(monkeypatch) -> None:
    log = _Log()
    monkeypatch.setattr(attn, "launch_gguf_linear", log.fn("linear"))
    monkeypatch.setattr(attn, "launch_gguf_linear_pair", _fail("pair"))

    route = attn.run_linear_attention_alpha_beta_rows(
        **_alpha_beta_call(
            norm_f32_ptr=None,
            cfg=_attn_config(is_moe=True),
            views=_alpha_beta_views(linear_alpha_f32=None, linear_beta_f32=None),
        )
    )

    assert route == "singletons"
    assert [args[1] for _, args, _ in log.events] == [0x40, 0x40]
    assert log.events[0][2]["out_features"] == 8


def test_alpha_beta_leaf_pair_route_and_fallback_route(monkeypatch) -> None:
    for pair_ok, route_name, expected in (
        (True, "pair", ["pair"]),
        (False, "fallback", ["pair", "linear", "linear"]),
    ):
        log = _Log()
        monkeypatch.setattr(attn, "launch_gguf_linear_pair", log.fn("pair", returns=pair_ok))
        monkeypatch.setattr(attn, "launch_gguf_linear", log.fn("linear"))

        route = attn.run_linear_attention_alpha_beta_rows(
            **_alpha_beta_call(
                norm_f32_ptr=None,
                # The projection leaf reads hidden_size alone: no attention width,
                # no SSM value dim and no recurrent-state dtype is supplied.
                geometry=attn.AttentionGeometry(hidden_size=256),
                views=_alpha_beta_views(linear_alpha_f32=None, linear_beta_f32=None),
            )
        )

        assert route == route_name
        assert log.labels() == expected
        if not pair_ok:
            assert [args[1] for _, args, _ in log.events[1:]] == [0x40, 0x40]


# --------------------------------------------------------------------------
# attention: c=1 linear-attention body
# --------------------------------------------------------------------------
def _linear_views(**overrides) -> attn.LinearAttentionViews:
    fields = dict(
        norm=_buf(0x101),
        norm_f32=None,
        linear_qkv=_buf(0x102),
        linear_z=_buf(0x103),
        linear_alpha=_buf(0x31),
        linear_beta=_buf(0x32),
        linear_alpha_f32=_buf(0x51),
        linear_beta_f32=_buf(0x52),
        conv_out=_buf(0x106),
        recurrent_out=_buf(0x107),
        recurrent_bf16=_buf(0x108),
        conv_state=_buf(0x201),
        recurrent_state=_buf(0x202),
        q8_1_workspace_owner=None,
    )
    fields.update(overrides)
    return attn.LinearAttentionViews(**fields)


def _linear_attn_layer(**overrides) -> _Layer:
    fields = dict(
        attn_norm=_weight("attn_norm", 0x50, quant="gguf_f32"),
        attn_qkv=_weight("attn_qkv", 0x51),
        attn_gate=_weight("attn_gate", 0x52),
        ssm_alpha=_weight("ssm_alpha", 0x53),
        ssm_beta=_weight("ssm_beta", 0x54),
        ssm_conv1d=_weight("ssm_conv1d", 0x55, quant="gguf_f32"),
        ssm_dt_bias=_weight("ssm_dt_bias", 0x56, quant="gguf_f32"),
        ssm_a=_weight("ssm_a", 0x57, quant="gguf_f32"),
        ssm_norm=_weight("ssm_norm", 0x58, quant="gguf_f32"),
        ssm_out=_weight("ssm_out", 0x59),
    )
    fields.update(overrides)
    return _Layer(**fields)


def _linear_attn_call(**overrides) -> dict:
    kwargs = dict(
        layer_id=3,
        layer=_linear_attn_layer(),
        hidden_ptr=0x1000,
        attn_out_ptr=0x2000,
        views=_linear_views(),
        cfg=_attn_config(),
        geometry=_attn_geometry(),
        backend=_BACKEND,
        runtime=_RUNTIME,
        selectors=_selectors(),
        policy=_policy(),
        stream=7,
    )
    kwargs.update(overrides)
    return kwargs


def _patch_linear_launches(monkeypatch, log: _Log, *, pair_ok: bool = True) -> None:
    monkeypatch.setattr(attn, "launch_gguf_linear_pair", log.fn("pair", returns=pair_ok))
    monkeypatch.setattr(attn, "launch_gguf_linear", log.fn("linear"))
    monkeypatch.setattr(attn, "qwen35_linear_attn_conv_decode_bf16", log.fn("conv"))


_LOOKUPS = frozenset({"fusion_lookup", "cast_lookup"})


def _launch_slots(log: _Log) -> list[tuple[str, str | None]]:
    """The recorded launches, without the per-weight policy lookups around them."""

    return [entry for entry in log.slots() if entry[0] not in _LOOKUPS]


def _gdn_selectors(log: _Log, *, fusion=None, cast=None, conv_fused: bool = True, **overrides):
    fields = dict(
        norm_residual_kernel=lambda **_: log.fn("norm_kernel"),
        cast_library=lambda: "cast-lib",
        try_launch_dense_f32_alpha_beta_conv_decode=log.fn("alpha_beta_conv", returns=conv_fused),
        gdn_decode_output_fusion_for_weight=log.fn("fusion_lookup", returns=fusion),
        gdn_decode_output_cast_for_weight=log.fn("cast_lookup", returns=cast),
        gdn_decode_gate_kernel=lambda use_fp16_state: log.fn("gdn"),
    )
    fields.update(overrides)
    return _selectors(**fields)


def test_linear_attention_body_fused_path_launch_order(monkeypatch) -> None:
    log = _Log()
    _patch_linear_launches(monkeypatch, log)
    stage = _Stage()
    layer = _linear_attn_layer()
    views = _linear_views()

    attn.run_linear_attention_attn_only(
        **_linear_attn_call(
            layer=layer,
            views=views,
            selectors=_gdn_selectors(log),
            gpu_stage_recorder=stage,
        )
    )

    assert _launch_slots(log) == [
        ("norm_kernel", None),
        ("pair", "attn_qkv"),
        ("alpha_beta_conv", "ssm_alpha"),
        ("gdn", None),
        ("linear", "ssm_out"),
    ]
    # Both output-policy lookups run, in the order the body needs them.
    assert [label for label, _ in log.slots() if label in _LOOKUPS] == [
        "fusion_lookup",
        "cast_lookup",
    ]
    assert stage.marks == [
        "decode_linear_attn_norm",
        "decode_linear_attn_qkv_gate",
        "decode_linear_attn_alpha_beta",
        "decode_linear_attn_conv",
        "decode_linear_attn_gdn",
        "decode_linear_attn_ssm_out",
    ]
    pair_args, pair_kwargs = log.one("pair")
    assert [w.slot for w in pair_args[:2]] == ["attn_qkv", "attn_gate"]
    assert pair_args[2:] == (views.norm.ptr, views.linear_qkv.ptr, views.linear_z.ptr)
    assert pair_kwargs == dict(
        rows=1,
        in_features=256,
        out_features=768,
        out_features_b=768,
        stream=7,
        runtime=_RUNTIME,
        q8_t16_rowtile_all=None,
    )
    conv_args, conv_kwargs = log.one("alpha_beta_conv")
    assert conv_args[2:] == (
        views.norm.ptr,
        views.linear_alpha.ptr,
        views.linear_beta.ptr,
        views.linear_qkv.ptr,
        views.conv_state.ptr,
        0x55,
        views.conv_out.ptr,
    )
    assert conv_kwargs == dict(
        rows=1,
        in_features=256,
        out_features=8,
        channels=768,
        kernel_size=4,
        backend=_BACKEND,
        stream=7,
        runtime=_RUNTIME,
    )
    gdn_args, gdn_kwargs = log.one("gdn")
    assert gdn_args == (
        views.conv_out.ptr,
        views.linear_z.ptr,
        views.linear_alpha.ptr,
        views.linear_beta.ptr,
        0x56,
        0x57,
        0x58,
        views.recurrent_state.ptr,
        views.recurrent_out.ptr,
        1e-6,
        2,
        8,
        64,
        512,
    )
    assert gdn_kwargs == dict(stream=7, runtime=_RUNTIME)
    out_args, out_kwargs = log.one("linear")
    assert out_args == (layer.weight("ssm_out"), views.recurrent_out.ptr, 0x2000)
    assert out_kwargs == dict(
        rows=1,
        in_features=768,
        out_features=256,
        activation_dtype="f32",
        stream=7,
        runtime=_RUNTIME,
        q8_t16_rowtile_all=None,
    )


def test_linear_attention_body_unfused_alpha_beta_and_conv(monkeypatch) -> None:
    log = _Log()
    _patch_linear_launches(monkeypatch, log)
    views = _linear_views()

    attn.run_linear_attention_attn_only(
        **_linear_attn_call(
            views=views,
            selectors=_gdn_selectors(log, conv_fused=False),
        )
    )

    assert _launch_slots(log) == [
        ("norm_kernel", None),
        ("pair", "attn_qkv"),
        ("alpha_beta_conv", "ssm_alpha"),
        ("pair", "ssm_alpha"),
        ("conv", None),
        ("gdn", None),
        ("linear", "ssm_out"),
    ]
    assert log.one("conv")[0] == (
        views.linear_qkv.ptr,
        views.conv_state.ptr,
        0x55,
        views.conv_out.ptr,
        768,
        4,
    )
    assert log.one("conv")[1] == dict(stream=7, runtime=_RUNTIME)


def test_linear_attention_body_prefused_norm_skips_the_leaf(monkeypatch) -> None:
    log = _Log()
    _patch_linear_launches(monkeypatch, log)

    attn.run_linear_attention_attn_only(
        **_linear_attn_call(
            input_norm_ptr=0x101,
            selectors=_gdn_selectors(log, norm_residual_kernel=_fail("norm_kernel")),
        )
    )

    assert log.labels()[0] == "pair"


def test_linear_attention_body_rejects_a_foreign_prefused_norm(monkeypatch) -> None:
    _patch_linear_launches(monkeypatch, _Log())

    with pytest.raises(ValueError, match="must use scratch.norm"):
        attn.run_linear_attention_attn_only(
            **_linear_attn_call(input_norm_ptr=0x999, selectors=_gdn_selectors(_Log()))
        )


def test_linear_attention_body_fused_gdn_consumes_bf16_recurrent(monkeypatch) -> None:
    log = _Log()
    _patch_linear_launches(monkeypatch, log)
    views = _linear_views()

    attn.run_linear_attention_attn_only(
        **_linear_attn_call(
            views=views,
            selectors=_gdn_selectors(log, fusion=log.fn("gdn_fused")),
        )
    )

    assert "gdn" not in log.labels()
    fused_args, _ = log.one("gdn_fused")
    assert fused_args[9] == views.recurrent_bf16.ptr
    out_args, out_kwargs = log.one("linear")
    assert out_args[1] == views.recurrent_bf16.ptr
    assert out_kwargs["activation_dtype"] == "bf16"


def test_linear_attention_body_output_cast_feeds_bf16_ssm_out(monkeypatch) -> None:
    log = _Log()
    _patch_linear_launches(monkeypatch, log)
    views = _linear_views()

    attn.run_linear_attention_attn_only(
        **_linear_attn_call(
            views=views,
            selectors=_gdn_selectors(log, cast=log.fn("out_cast")),
        )
    )

    cast_args, cast_kwargs = log.one("out_cast")
    assert cast_args == (views.recurrent_out.ptr, views.recurrent_bf16.ptr, 768)
    assert cast_kwargs == dict(stream=7, library="cast-lib", runtime=_RUNTIME)
    out_args, out_kwargs = log.one("linear")
    assert out_args[1] == views.recurrent_bf16.ptr
    assert out_kwargs["activation_dtype"] == "bf16"


def test_linear_attention_body_requires_layer_state() -> None:
    with pytest.raises(ValueError, match="layer 3 has no linear-attention state"):
        attn.run_linear_attention_attn_only(
            **_linear_attn_call(
                views=_linear_views(conv_state=None),
                selectors=_selectors(),
            )
        )


# --------------------------------------------------------------------------
# attention: c=1 full-attention body
# --------------------------------------------------------------------------
class _FullScratch:
    """The two mutable-in-body scratch responsibilities: position and split-K."""

    def __init__(self, *, position: int = 0, split_count: int = 8) -> None:
        self.position_host = [int(position)]
        self.full_attn_split_count = int(split_count)
        self.full_attn_split_partial = _buf(0x401)
        self.full_attn_split_m = _buf(0x402)
        self.full_attn_split_l = _buf(0x403)
        self.positions: list[tuple[int, object]] = []

    def set_full_attention_position(self, position: int, runtime) -> None:
        self.positions.append((int(position), runtime))
        self.position_host[0] = int(position)


def _full_views(**overrides) -> attn.FullAttentionViews:
    fields = dict(
        norm=_buf(0x111),
        full_q=_buf(0x112),
        full_k=_buf(0x113),
        full_v=_buf(0x114),
        full_query_raw=_buf(0x115),
        full_key_raw=_buf(0x116),
        full_query=_buf(0x117),
        full_key=_buf(0x118),
        full_gate=_buf(0x119),
        full_gated=_buf(0x11A),
        full_attn_context=_buf(0x11B),
        cos_table=_buf(0x11C),
        sin_table=_buf(0x11D),
        position_tensor=_buf(0x11E),
        max_positions=4096,
        block_size=16,
        kv_storage_dtype=DType.BF16,
        int8_kv_value_bf16=False,
        append_spans=NS(name="append-spans"),
        decode_spans=NS(name="decode-spans"),
        key_cache=_buf(0x301),
        value_cache=_buf(0x302),
        layer_append_spans=NS(name="layer-append-spans", scale_metadata=None),
        layer_decode_spans=NS(name="layer-decode-spans", scale_metadata=None),
        bf16_mirror_cache=None,
    )
    fields.update(overrides)
    return attn.FullAttentionViews(**fields)


def _full_attn_layer(**overrides) -> _Layer:
    fields = dict(
        attn_norm=_weight("attn_norm", 0x60, quant="gguf_f32"),
        attn_q=_weight("attn_q", 0x61),
        attn_k=_weight("attn_k", 0x62),
        attn_v=_weight("attn_v", 0x63),
        attn_q_norm=_weight("attn_q_norm", 0x64, quant="gguf_f32"),
        attn_k_norm=_weight("attn_k_norm", 0x65, quant="gguf_f32"),
        attn_output=_weight("attn_output", 0x66),
    )
    fields.update(overrides)
    return _Layer(**fields)


def _full_attn_call(**overrides) -> dict:
    kwargs = dict(
        layer_id=5,
        layer=_full_attn_layer(),
        hidden_ptr=0x3000,
        attn_out_ptr=0x4000,
        views=_full_views(),
        scratch=_FullScratch(),
        cfg=_attn_config(),
        geometry=_attn_geometry(),
        backend=_BACKEND,
        runtime=_RUNTIME,
        selectors=_selectors(),
        policy=_policy(),
        ensure_full_attn_split_rows=_fail("ensure_full_attn_split_rows"),
        external_dms_full_attention=None,
        position=4095,
        stream=7,
    )
    kwargs.update(overrides)
    return kwargs


def _patch_full_launches(monkeypatch, log: _Log) -> None:
    monkeypatch.setattr(attn, "launch_gguf_linear_triple", log.fn("qkv_triple", returns=True))
    monkeypatch.setattr(attn, "launch_gguf_linear_pair", log.fn("kv_pair", returns=True))
    monkeypatch.setattr(attn, "launch_gguf_linear", log.fn("linear"))
    monkeypatch.setattr(attn, "qwen35_split_qgate_bf16", log.fn("split_qgate"))
    monkeypatch.setattr(attn, "bf16_to_f32", log.fn("bf16_to_f32"))
    monkeypatch.setattr(
        attn, "gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight", log.fn("rope")
    )
    monkeypatch.setattr(attn, "qwen35_write_paged_kv_mixed_value_bf16_spans", log.fn("kv_write"))
    monkeypatch.setattr(attn, "qwen35_full_attn_gate_mul_bf16", log.fn("gate_mul"))
    monkeypatch.setattr(
        attn, "qwen35_paged_full_attn_decode_context_bf16_spans", log.fn("context_decode")
    )
    monkeypatch.setattr(attn, "backend_package_capability", _fail("backend_package_capability"))


def _full_selectors(log: _Log, *, qk=None, short_batch=None, attn_library="attn-lib", **overrides):
    fields = dict(
        cast_library=lambda: "cast-lib",
        paged_attn_decode_library=lambda: attn_library,
        paged_kv_write_library=lambda: "kv-lib",
        full_attn_qk_postprocess_fn=lambda: qk,
        full_attn_decode_short_batch_fn=(
            short_batch if short_batch is not None else _fail("full_attn_decode_short_batch_fn")
        ),
        norm_residual_kernel=lambda **_: log.fn("norm_kernel"),
    )
    fields.update(overrides)
    return _selectors(**fields)


def _split_policy(log: _Log, *, short_split=False, full_split=True, **overrides):
    fields = dict(
        use_short_full_attention_split_decode=lambda *a, **k: short_split,
        use_full_attention_split_decode=lambda context: full_split,
        full_attention_split_gate_fn=log.fn(
            "split_gate_lookup", returns=log.fn("split_kernel")
        ),
    )
    fields.update(overrides)
    return _policy(**fields)


class _TrapScratch:
    """A scratch whose every attribute access is a contract violation."""

    def __getattr__(self, name: str):
        raise AssertionError(f"scratch.{name} must not be read on this path")


def test_full_attention_dms_override_touches_nothing_else(monkeypatch) -> None:
    """The DMS seam returns before weights, scratch, views or any selector."""

    _patch_full_launches(monkeypatch, _Log())
    calls: list[dict] = []
    scratch = _TrapScratch()

    def dms(**kwargs):
        calls.append(kwargs)

    attn.run_full_attention_attn_only(
        **_full_attn_call(
            layer=_fail("layer"),
            views=None,
            scratch=scratch,
            selectors=_selectors(),
            policy=_policy(),
            ensure_full_attn_split_rows=_fail("ensure_full_attn_split_rows"),
            external_dms_full_attention=dms,
            position=17,
        )
    )

    assert calls == [
        dict(
            layer_id=5,
            hidden_ptr=0x3000,
            attn_out_ptr=0x4000,
            scratch=scratch,
            position=17,
            hidden_f32_ptr=None,
            input_norm_ptr=None,
            stream=7,
            stage_prefix="decode_full_attn",
            gpu_stage_recorder=None,
        )
    ]


def test_full_attention_split_path_launch_order(monkeypatch) -> None:
    log = _Log()
    _patch_full_launches(monkeypatch, log)
    stage = _Stage()
    growth: list[int] = []
    scratch = _FullScratch()
    views = _full_views()

    attn.run_full_attention_attn_only(
        **_full_attn_call(
            views=views,
            scratch=scratch,
            selectors=_full_selectors(log),
            policy=_split_policy(log),
            ensure_full_attn_split_rows=growth.append,
            gpu_stage_recorder=stage,
        )
    )

    assert log.labels() == [
        "norm_kernel",
        "qkv_triple",
        "split_qgate",
        "bf16_to_f32",
        "rope",
        "kv_write",
        "split_gate_lookup",
        "split_kernel",
        "linear",
    ]
    assert growth == [1]
    assert scratch.positions == [(4095, _RUNTIME)]
    assert stage.marks == [
        "decode_full_attn_norm",
        "decode_full_attn_qkv_head_norm_rope",
        "decode_full_attn_kv_write",
        "decode_full_attn_core_gate",
        "decode_full_attn_output",
    ]
    triple_args, triple_kwargs = log.one("qkv_triple")
    assert [w.slot for w in triple_args[:3]] == ["attn_q", "attn_k", "attn_v"]
    assert triple_args[3:] == (
        views.norm.ptr,
        views.full_q.ptr,
        views.full_k.ptr,
        views.full_v.ptr,
    )
    assert triple_kwargs == dict(
        rows=1,
        in_features=256,
        out_features=1024,
        out_features_b=256,
        out_features_c=256,
        stream=7,
        runtime=_RUNTIME,
        rowtile_all=None,
    )
    kernel_args, kernel_kwargs = log.one("split_kernel")
    assert kernel_args == (
        views.full_query.ptr,
        views.key_cache.ptr,
        views.value_cache.ptr,
        views.full_gate.ptr,
        views.full_gated.ptr,
        scratch.full_attn_split_partial.ptr,
        scratch.full_attn_split_m.ptr,
        scratch.full_attn_split_l.ptr,
        views.layer_decode_spans,
        16,
        8,
        16,
        4,
        2,
        64,
        64,
        1,
        64**-0.5,
    )
    assert kernel_kwargs == dict(stream=7, library="attn-lib", runtime=_RUNTIME)
    out_args, out_kwargs = log.one("linear")
    assert out_args[1:] == (0x11A, 0x4000)
    assert out_kwargs == dict(
        rows=1,
        in_features=512,
        out_features=256,
        stream=7,
        runtime=_RUNTIME,
        q8_t16_rowtile_all=None,
    )


def test_full_attention_reads_split_buffers_after_growth(monkeypatch) -> None:
    """A growth callback that reallocates the partials must be observed."""

    log = _Log()
    _patch_full_launches(monkeypatch, log)
    scratch = _FullScratch()

    def grow(query_rows: int) -> None:
        assert query_rows == 1
        scratch.full_attn_split_partial = _buf(0x499)
        scratch.full_attn_split_m = _buf(0x49A)
        scratch.full_attn_split_l = _buf(0x49B)

    attn.run_full_attention_attn_only(
        **_full_attn_call(
            scratch=scratch,
            selectors=_full_selectors(log),
            policy=_split_policy(log),
            ensure_full_attn_split_rows=grow,
        )
    )

    kernel_args, _ = log.one("split_kernel")
    assert kernel_args[5:8] == (0x499, 0x49A, 0x49B)


def test_full_attention_reuses_an_already_published_position(monkeypatch) -> None:
    log = _Log()
    _patch_full_launches(monkeypatch, log)
    scratch = _FullScratch(position=4095)

    attn.run_full_attention_attn_only(
        **_full_attn_call(
            scratch=scratch,
            selectors=_full_selectors(log),
            policy=_split_policy(log),
            ensure_full_attn_split_rows=lambda rows: None,
        )
    )

    assert scratch.positions == []


def test_full_attention_composite_qk_postprocess_replaces_the_split_leaves(
    monkeypatch,
) -> None:
    log = _Log()
    _patch_full_launches(monkeypatch, log)
    monkeypatch.setattr(attn, "qwen35_split_qgate_bf16", _fail("split_qgate"))
    monkeypatch.setattr(
        attn, "gguf_qwen35_head_rmsnorm_partial_rotary_position_f32_weight", _fail("rope")
    )

    attn.run_full_attention_attn_only(
        **_full_attn_call(
            selectors=_full_selectors(log, qk=log.fn("qk_composite")),
            policy=_split_policy(log),
            ensure_full_attn_split_rows=lambda rows: None,
        )
    )

    assert log.labels()[1:3] == ["qkv_triple", "qk_composite"]
    assert "split_qgate" not in log.labels() and "rope" not in log.labels()
    args, kwargs = log.one("qk_composite")
    # The composite reads the raw q/k the triple wrote and emits q/k/gate.
    assert args[0:2] == (0x112, 0x113)
    assert args[7:10] == (0x117, 0x118, 0x119)
    assert args[10:] == (1e-6, 4, 2, 64, 32, 4096)
    assert kwargs == dict(stream=7, runtime=_RUNTIME)


def test_full_attention_short_batch_leaf_sees_the_layer_decode_spans(monkeypatch) -> None:
    log = _Log()
    _patch_full_launches(monkeypatch, log)
    monkeypatch.setattr(attn, "backend_package_capability", lambda *a, **k: 4096)
    views = _full_views()
    seen: list[object] = []

    def short_batch(spans):
        seen.append(spans)
        return log.fn("short_batch")

    attn.run_full_attention_attn_only(
        **_full_attn_call(
            views=views,
            selectors=_full_selectors(log, short_batch=short_batch),
            policy=_split_policy(log, full_split=False),
        )
    )

    assert seen == [views.layer_decode_spans]
    args, kwargs = log.one("short_batch")
    assert args[0:4] == (0x117, 0x301, 0x302, 0x11B)
    assert args[4] is views.layer_decode_spans
    assert args[5:] == (1, 4096, 16, 4, 2, 64, 64**-0.5)
    assert kwargs == dict(stream=7, library="attn-lib", runtime=_RUNTIME)
    assert log.one("gate_mul")[0] == (0x11B, 0x119, 0x11A, 512)


def test_full_attention_context_decode_when_the_short_batch_is_unavailable(
    monkeypatch,
) -> None:
    log = _Log()
    _patch_full_launches(monkeypatch, log)
    # The backend declares no short-batch cap, so the context kernel must run.
    monkeypatch.setattr(attn, "backend_package_capability", lambda *a, **k: 0)
    views = _full_views()

    attn.run_full_attention_attn_only(
        **_full_attn_call(
            views=views,
            selectors=_full_selectors(log),
            policy=_split_policy(log, full_split=False),
        )
    )

    args, kwargs = log.one("context_decode")
    assert args[0:4] == (0x117, 0x301, 0x302, 0x11B)
    assert args[4] is views.layer_decode_spans
    assert args[5:] == (4096, 16, 4, 2, 64, 64**-0.5)
    assert kwargs == dict(stream=7, library="attn-lib", runtime=_RUNTIME)


def test_full_attention_kv_write_only_stops_before_decode(monkeypatch) -> None:
    log = _Log()
    _patch_full_launches(monkeypatch, log)

    attn.run_full_attention_attn_only(
        **_full_attn_call(
            kv_write_only=True,
            input_norm_ptr=0x111,
            selectors=_full_selectors(log),
            policy=_split_policy(log),
            ensure_full_attn_split_rows=_fail("ensure_full_attn_split_rows"),
        )
    )

    # Only the query projection is skipped: k/v must still be written.
    assert log.labels() == ["kv_pair", "split_qgate", "bf16_to_f32", "rope", "kv_write"]
    assert "linear" not in log.labels()
    assert "gate_mul" not in log.labels() and "split_kernel" not in log.labels()


def test_full_attention_validates_the_context_cap(monkeypatch) -> None:
    log = _Log()
    _patch_full_launches(monkeypatch, log)
    with pytest.raises(ValueError, match="must cover the current decode position"):
        attn.run_full_attention_attn_only(
            **_full_attn_call(
                attention_max_context_len=100,
                selectors=_full_selectors(log),
                policy=_policy(),
            )
        )
    with pytest.raises(ValueError, match="exceeds GGUF resident cache capacity"):
        attn.run_full_attention_attn_only(
            **_full_attn_call(
                attention_max_context_len=8192,
                selectors=_full_selectors(log),
                policy=_policy(),
            )
        )


def test_full_attention_int8_kv_uses_the_granularity_selectors(monkeypatch) -> None:
    """An int8 key with a bf16 value routes each span through its own selector."""

    log = _Log()
    _patch_full_launches(monkeypatch, log)
    metadata = NS(
        k_scale=_buf(0x501),
        v_scale=_buf(0x502),
        granularity="per_token_head",
    )
    spans = NS(name="int8-append-spans", scale_metadata=metadata)
    decode_spans = NS(name="int8-decode-spans", scale_metadata=metadata)
    monkeypatch.setattr(
        attn,
        "qwen35_write_paged_kv_int8_key_bf16_value_spans",
        log.fn("int8_append"),
    )
    monkeypatch.setattr(
        attn,
        "qwen35_paged_attn_decode_int8_key_bf16_value_gqa_splitk_gate_bf16_spans",
        log.fn("int8_decode"),
    )
    views = _full_views(
        int8_kv_value_bf16=True,
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        append_spans=spans,
        decode_spans=decode_spans,
        layer_append_spans=NS(name="layer-int8-append", scale_metadata=metadata),
        layer_decode_spans=NS(name="layer-int8-decode", scale_metadata=metadata),
    )

    attn.run_full_attention_attn_only(
        **_full_attn_call(
            views=views,
            selectors=_full_selectors(
                log,
                int8_kv_append_write_fn=lambda meta: log.fn("int8_append"),
                int8_kv_decode_gate_fn=lambda meta: log.fn("int8_decode"),
            ),
            policy=_split_policy(log, full_split=False),
            ensure_full_attn_split_rows=lambda rows: None,
        )
    )

    assert "kv_write" not in log.labels()
    append_args, append_kwargs = log.one("int8_append")
    assert append_args[0:6] == (0x118, 0x116, 0x301, 0x302, 0x501, views.layer_append_spans)
    assert append_args[6:] == (16, 2, 64)
    assert append_kwargs["library"] == "kv-lib"
    decode_args, _ = log.one("int8_decode")
    assert decode_args[0:7] == (0x117, 0x301, 0x302, 0x501, 0x119, 0x11A, 0x401)
    assert decode_args[9:12] == (views.layer_decode_spans, 16, 8)


def test_full_attention_int8_kv_plain_value_resolves_both_selectors(monkeypatch) -> None:
    log = _Log()
    _patch_full_launches(monkeypatch, log)
    metadata = NS(
        k_scale=_buf(0x501),
        v_scale=_buf(0x502),
        granularity="per_tensor",
    )
    views = _full_views(
        int8_kv_value_bf16=False,
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        append_spans=NS(name="int8-append-spans", scale_metadata=metadata),
        decode_spans=NS(name="int8-decode-spans", scale_metadata=metadata),
        layer_append_spans=NS(name="layer-int8-append", scale_metadata=metadata),
        layer_decode_spans=NS(name="layer-int8-decode", scale_metadata=metadata),
    )

    attn.run_full_attention_attn_only(
        **_full_attn_call(
            views=views,
            selectors=_full_selectors(
                log,
                int8_kv_append_write_fn=lambda meta: log.fn("int8_append"),
                int8_kv_decode_gate_fn=lambda meta: log.fn("int8_decode"),
            ),
            policy=_split_policy(log, full_split=False),
            ensure_full_attn_split_rows=lambda rows: None,
        )
    )

    assert "kv_write" not in log.labels()
    append_args, append_kwargs = log.one("int8_append")
    assert append_args == (
        0x118,
        0x116,
        0x301,
        0x302,
        0x501,
        0x502,
        views.layer_append_spans,
        16,
        2,
        64,
    )
    assert append_kwargs == dict(stream=7, library="kv-lib", runtime=_RUNTIME)
    decode_args, decode_kwargs = log.one("int8_decode")
    assert decode_args[0:8] == (0x117, 0x301, 0x302, 0x501, 0x502, 0x119, 0x11A, 0x401)
    assert decode_args[8:12] == (0x402, 0x403, views.layer_decode_spans, 16)
    assert decode_kwargs == dict(stream=7, library="attn-lib", runtime=_RUNTIME)


# --------------------------------------------------------------------------
# freshness: selectors and module kernels are consulted per call
# --------------------------------------------------------------------------
class _LazyNormAdapter:
    """The runner adapter shape: a thunk that resolves a kernel at call time."""

    def __init__(self) -> None:
        self.kernel = _fail("kernel not published")

    def __call__(self, **kwargs):
        return self.kernel


def test_attention_norm_leaf_reconsults_its_selector_every_call() -> None:
    adapter = _LazyNormAdapter()
    log = _Log()
    calls = dict(_norm_call(selectors=_selectors(norm_residual_kernel=adapter)))

    adapter.kernel = log.fn("first")
    attn.run_attention_norm_rows(**calls)
    adapter.kernel = log.fn("second")
    attn.run_attention_norm_rows(**calls)

    assert log.labels() == ["first", "second"]


def test_attention_norm_leaf_observes_a_kernel_patched_between_calls(monkeypatch) -> None:
    log = _Log()
    calls = dict(
        _norm_call(
            hidden_f32_ptr=0x11,
            policy=_policy(verify_f32_attention_norm=lambda: False),
        )
    )

    monkeypatch.setattr(attn, "gguf_rmsnorm_f32_f32_weight", log.fn("first"))
    attn.run_attention_norm_rows(**calls)
    monkeypatch.setattr(attn, "gguf_rmsnorm_f32_f32_weight", log.fn("second"))
    attn.run_attention_norm_rows(**calls)

    assert log.labels() == ["first", "second"]


def test_full_attention_reconsults_the_split_policy_every_call(monkeypatch) -> None:
    log = _Log()
    _patch_full_launches(monkeypatch, log)
    decisions: list[int] = []

    def split_policy(context: int) -> bool:
        decisions.append(context)
        return True

    policy = _policy(
        use_short_full_attention_split_decode=lambda *a, **k: False,
        use_full_attention_split_decode=split_policy,
        full_attention_split_gate_fn=log.fn(
            "split_gate_lookup", returns=log.fn("split_kernel")
        ),
    )
    calls = dict(
        _full_attn_call(
            selectors=_full_selectors(log),
            policy=policy,
            ensure_full_attn_split_rows=lambda rows: None,
        )
    )

    attn.run_full_attention_attn_only(**calls)
    attn.run_full_attention_attn_only(**calls)

    assert decisions == [4096, 4096]
    assert log.labels().count("split_kernel") == 2


# --------------------------------------------------------------------------
# feed-forward: free bodies with explicit inputs and selected operations
# --------------------------------------------------------------------------
def _ffn_kernels(**overrides) -> ffn.FfnKernels:
    fields = {
        name: _fail(f"kernels.{name}") for name in ffn.FfnKernels.__dataclass_fields__
    }
    fields.update(overrides)
    return ffn.FfnKernels(**fields)


def _ffn_ops(**overrides) -> ffn.FfnOps:
    fields = {name: _fail(f"ops.{name}") for name in ffn.FfnOps.__dataclass_fields__}
    fields.update(overrides)
    return ffn.FfnOps(**fields)


def _ffn_flags(**overrides) -> ffn.FfnFlags:
    fields = {name: _fail(f"flags.{name}") for name in ffn.FfnFlags.__dataclass_fields__}
    fields.update(overrides)
    return ffn.FfnFlags(**fields)


def _ffn_config(*, is_moe: bool = False, **overrides):
    fields = dict(
        hidden_size=256,
        rms_norm_eps=1e-6,
        feed_forward_length=1024,
        is_moe=is_moe,
        expert_count=8,
        expert_used_count=2,
        expert_feed_forward_length=64,
        expert_shared_feed_forward_length=128,
    )
    fields.update(overrides)
    return NS(**fields)


def _dense_layer(**overrides) -> _Layer:
    fields = dict(
        post_attention_norm=_weight("post_attention_norm", 0x104, quant="gguf_f32"),
        ffn_gate=_weight("ffn_gate", 0x101),
        ffn_up=_weight("ffn_up", 0x102),
        ffn_down=_weight("ffn_down", 0x103),
    )
    fields.update(overrides)
    return _Layer(**fields)


def _ffn_scratch(**overrides):
    fields = dict(
        post_norm=_buf(0x201),
        residual=_buf(0x202),
        ffn_gate_up=_buf(0x203, nbytes=4096),
        ffn_intermediate=_buf(0x204),
        ffn_intermediate_f32=_buf(0x205),
        ffn_down=_buf(0x206),
    )
    fields.update(overrides)
    return NS(**fields)


def _dense_ffn_call(**overrides) -> dict:
    kwargs = dict(
        layer=_dense_layer(),
        config=_ffn_config(),
        geometry=ffn.FfnGeometry(hidden_size=256, backend=_BACKEND, ffn_size=1024),
        kernels=_ffn_kernels(),
        ops=_ffn_ops(),
        flags=_ffn_flags(),
        layer_id=2,
        hidden_ptr=0x10,
        attn_out_ptr=0x20,
        out_ptr=0x30,
        scratch=_ffn_scratch(),
        rows=1,
        dense_down_residual_decode_c1=False,
        runtime=_RUNTIME,
        stream=19,
    )
    kwargs.update(overrides)
    return kwargs


def _dense_ops(log: _Log, **overrides) -> ffn.FfnOps:
    def mark_sync_stage(runtime, timings, sync, name, t0):
        log.events.append((f"stage:{name}", (), {}))
        return t0

    fields = dict(
        norm_residual_kernel=lambda **_: log.fn("add_norm"),
        dense_pair_silu_variant=log.value("variant_lookup", None),
        optional_q8_1_workspace_ptr=log.value("q8_1_lookup", None),
        dense_down_residual_fused=log.value("down_policy_lookup", False),
        mark_sync_stage=mark_sync_stage,
    )
    fields.update(overrides)
    return _ffn_ops(**fields)


def _dense_kernels(log: _Log, *, pair_ok: bool = False, residual_ok: bool = False, **overrides):
    fields = dict(
        launch_gguf_linear_pair=log.fn("pair", returns=pair_ok),
        launch_gguf_linear=log.fn("linear"),
        launch_gguf_linear_residual=log.fn("residual", returns=residual_ok),
        silu_mul_separate_out_bf16=log.fn("silu"),
        gguf_bf16_add=log.fn("add"),
    )
    fields.update(overrides)
    return _ffn_kernels(**fields)


def test_dense_ffn_rows_launch_order_and_scalars() -> None:
    log = _Log()

    ffn.run_post_attention_ffn_rows(
        **_dense_ffn_call(ops=_dense_ops(log), kernels=_dense_kernels(log))
    )

    assert log.labels() == [
        "add_norm",
        "stage:target_block_ffn_post_norm_residual",
        "variant_lookup",
        "q8_1_lookup",
        "pair",
        "linear",
        "linear",
        "stage:target_block_ffn_gate_up",
        "silu",
        "stage:target_block_ffn_silu",
        "residual",
        "linear",
        "add",
        "stage:target_block_ffn_down_residual",
    ]
    assert log.one("add_norm")[0] == (0x10, 0x20, 0x104, 0x201, 0x202)
    assert log.one("add_norm")[1] == dict(
        rows=1, hidden_size=256, eps=1e-6, stream=19, runtime=_RUNTIME
    )
    assert log.one("pair")[0][3:] == (0x203, 0x203 + 1024 * 1 * 2)
    assert log.one("pair")[1] == dict(
        rows=1, in_features=256, out_features=1024, stream=19, runtime=_RUNTIME
    )
    assert [args[2] for label, args, _ in log.events if label == "linear"] == [
        0x203,
        0x203 + 1024 * 1 * 2,
        0x206,
    ]
    assert log.one("silu")[0] == (0x203, 0x203 + 1024 * 1 * 2, 0x204)
    assert log.one("silu")[1] == dict(rows=1, features=1024, stream=19, runtime=_RUNTIME)
    residual_args, residual_kwargs = log.one("residual")
    assert residual_args[0].slot == "ffn_down"
    assert residual_args[1:] == (0x204, 0x202, 0x30, 1, 1024, 256)
    assert residual_kwargs == dict(
        stream=19, runtime=_RUNTIME, registered_decode=False
    )
    assert log.one("add")[0] == (0x202, 0x206, 0x30, 1 * 256)


def test_dense_ffn_rows_fused_gate_up_pair_replaces_the_singles() -> None:
    log = _Log()

    ffn.run_post_attention_ffn_rows(
        **_dense_ffn_call(ops=_dense_ops(log), kernels=_dense_kernels(log, pair_ok=True))
    )

    assert log.labels().count("pair") == 1
    assert [label for label in log.labels() if label == "linear"] == ["linear"]
    assert log.one("silu")[0][0] == 0x203


def test_dense_ffn_rows_fused_down_residual_skips_the_add() -> None:
    log = _Log()

    ffn.run_post_attention_ffn_rows(
        **_dense_ffn_call(
            ops=_dense_ops(log), kernels=_dense_kernels(log, residual_ok=True)
        )
    )

    assert "add" not in log.labels()
    assert log.labels().count("linear") == 2
    residual_args, residual_kwargs = log.one("residual")
    assert residual_args[0].slot == "ffn_down"
    assert residual_args[1:] == (0x204, 0x202, 0x30, 1, 1024, 256)
    assert residual_kwargs == dict(
        stream=19, runtime=_RUNTIME, registered_decode=False
    )


def test_dense_ffn_rows_bulk_rows_uses_the_pair_silu_composite() -> None:
    log = _Log()
    kernels = _dense_kernels(
        log,
        launch_gguf_linear_pair_silu=log.fn("pair_silu", returns=True),
    )

    ffn.run_post_attention_ffn_rows(
        **_dense_ffn_call(rows=4, ops=_dense_ops(log), kernels=kernels)
    )

    assert log.labels()[:4] == [
        "add_norm",
        "stage:target_block_ffn_post_norm_residual",
        "variant_lookup",
        "q8_1_lookup",
    ]
    assert log.labels()[4] == "pair_silu"
    assert "pair" not in log.labels()
    assert "silu" not in log.labels()
    args, kwargs = log.one("pair_silu")
    assert args[2:] == (0x201, 0x204)
    assert kwargs == dict(
        rows=4,
        in_features=256,
        out_features=1024,
        stream=19,
        runtime=_RUNTIME,
        use_gemv_decode=None,
        registered_decode_variant=None,
        layer_id=2,
        q8_1_workspace_ptr=None,
        pair_workspace_ptr=0x203,
        pair_workspace_nbytes=4096,
    )
    assert log.one("q8_1_lookup")[1] == dict(enabled=False, planes=2)


def test_dense_ffn_rows_decode_variant_drives_the_q8_1_workspace() -> None:
    log = _Log()
    variant = "dense_dual_q8_1x2_dp4a_bf16_bf16_out"
    ops = _dense_ops(
        log,
        dense_pair_silu_variant=lambda **_: variant,
        optional_q8_1_workspace_ptr=log.fn("q8_1_lookup", returns=0x777),
    )
    kernels = _dense_kernels(log, launch_gguf_linear_pair_silu=log.fn("pair_silu", returns=True))

    ffn.run_post_attention_ffn_rows(
        **_dense_ffn_call(ops=ops, kernels=kernels)
    )

    assert log.one("q8_1_lookup")[1] == dict(enabled=True, planes=2)
    assert log.one("pair_silu")[1]["registered_decode_variant"] == variant
    assert log.one("pair_silu")[1]["use_gemv_decode"] is True
    assert log.one("pair_silu")[1]["q8_1_workspace_ptr"] == 0x777


def test_dense_ffn_rows_f32_residual_uses_the_f32_norm_and_add() -> None:
    log = _Log()
    flags = _ffn_flags(
        verify_f32_attn_out_enabled=lambda: True,
        verify_f32_post_norm_enabled=lambda: False,
    )
    kernels = _dense_kernels(
        log,
        gguf_add_rmsnorm_f32_f32_f32_weight=log.fn("f32_add_norm"),
        gguf_f32_bf16_add_out_f32=log.fn("f32_add"),
        f32_to_bf16=log.fn("cast"),
        launch_gguf_linear_residual=_fail("residual"),
    )
    ops = _dense_ops(log, cast_library=lambda: "cast-lib")

    ffn.run_post_attention_ffn_rows(
        **_dense_ffn_call(
            ops=ops,
            kernels=kernels,
            flags=flags,
            hidden_f32_ptr=0x40,
            out_f32_ptr=0x41,
            attn_out_f32_ptr=0x42,
        )
    )

    assert log.one("f32_add_norm")[0] == (0x40, 0x42, 0x104, 0x201, 0x41)
    assert log.one("f32_add")[0] == (0x41, 0x206, 0x41, 1 * 256)
    assert log.one("cast")[0] == (0x41, 0x30, 1 * 256)
    assert log.one("cast")[1] == dict(stream=19, library="cast-lib", runtime=_RUNTIME)
    assert "add" not in log.labels()


def test_dense_ffn_rows_next_norm_uses_the_rounded_leaf() -> None:
    log = _Log()
    rounded = log.fn("rounded_next_rms")
    ops = _dense_ops(log, rounded_next_rmsnorm=lambda *, rows: rounded)

    ffn.run_post_attention_ffn_rows(
        **_dense_ffn_call(
            ops=ops,
            kernels=_dense_kernels(log, gguf_bf16_add=_fail("add")),
            next_norm_weight_ptr=0x50,
            next_norm_out_ptr=0x51,
        )
    )

    assert log.one("rounded_next_rms")[0] == (0x202, 0x206, 0x50, 0x51, 0x30, 1, 256, 1e-6)


def test_dense_ffn_rows_rejects_a_half_next_norm_pair() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        ffn.run_post_attention_ffn_rows(
            **_dense_ffn_call(
                ops=_dense_ops(_Log()),
                kernels=_dense_kernels(_Log()),
                next_norm_weight_ptr=0x50,
            )
        )


def test_dense_ffn_rows_resolves_the_down_policy_when_the_memo_is_absent() -> None:
    log = _Log()

    ffn.run_post_attention_ffn_rows(
        **_dense_ffn_call(
            ops=_dense_ops(log),
            kernels=_dense_kernels(log),
            dense_down_residual_decode_c1=None,
        )
    )

    assert log.one("down_policy_lookup")[1] == dict(
        rows=1, in_features=1024, out_features=256
    )


def test_dense_ffn_rows_uses_the_kernel_bundle_it_is_given() -> None:
    """No bundle is cached in the module: each call uses its own callables."""

    first, second = _Log(), _Log()
    calls = dict(_dense_ffn_call())

    ffn.run_post_attention_ffn_rows(
        **{**calls, "ops": _dense_ops(first), "kernels": _dense_kernels(first)}
    )
    # The second bundle fuses the gate/up pair, which the first one declines.
    ffn.run_post_attention_ffn_rows(
        **{
            **calls,
            "ops": _dense_ops(second),
            "kernels": _dense_kernels(second, pair_ok=True),
        }
    )

    assert first.labels().count("pair") == 1
    # Unfused: gate, up and down each take their own linear launch.
    assert first.labels().count("linear") == 3
    assert second.labels().count("pair") == 1
    # Fused gate/up leaves only the down projection.
    assert second.labels().count("linear") == 1
    assert first.labels().count("silu") == second.labels().count("silu") == 1


def _moe_layer(**overrides) -> _Layer:
    fields = dict(
        post_attention_norm=_weight("post_attention_norm", 0x104, quant="gguf_f32"),
        ffn_gate_inp=_weight("ffn_gate_inp", 0x110, quant="gguf_f32"),
        ffn_gate_inp_shexp=_weight("ffn_gate_inp_shexp", 0x111, quant="gguf_f32"),
        ffn_gate_exps=_weight("ffn_gate_exps", 0x120),
        ffn_up_exps=_weight("ffn_up_exps", 0x121),
        ffn_down_exps=_weight("ffn_down_exps", 0x122),
        ffn_gate_shexp=_weight("ffn_gate_shexp", 0x130),
        ffn_up_shexp=_weight("ffn_up_shexp", 0x131),
        ffn_down_shexp=_weight("ffn_down_shexp", 0x132),
    )
    fields.update(overrides)
    return _Layer(**fields)


def _moe_scratch(**overrides):
    fields = dict(
        post_norm=_buf(0x201),
        residual=_buf(0x202),
        ffn_gate_up=_buf(0x203, nbytes=4096),
        ffn_intermediate=_buf(0x204),
        ffn_intermediate_f32=_buf(0x205),
        ffn_down=_buf(0x206),
        moe_router_logits=_buf(0x207),
        moe_selected_experts=_buf(0x208),
        moe_routing_weights=_buf(0x209),
        moe_router_counter=_buf(0x20A),
        moe_selected_host=NS(shape=(8,)),
        moe_down_out=_buf(0x20B),
        moe_down_out_f32=_buf(0x20C),
        moe_shared_gate=_buf(0x20D),
        moe_shared_up=_buf(0x20E),
        moe_shared_intermediate=_buf(0x20F),
        moe_shared_out=_buf(0x210),
        moe_shared_out_f32=_buf(0x211),
    )
    fields.update(overrides)
    return NS(**fields)


def _moe_flags(**overrides) -> ffn.FfnFlags:
    fields = dict(
        router_f32w_coop_enabled=lambda: False,
        compact_moe_c1_enabled=lambda: False,
        fused_moe_ffn_enabled=lambda: False,
        q4k_selected_dual_dp4a_enabled=lambda: False,
        t16_selected_dp4a_enabled=lambda: False,
        raw_selected_dp4a_enabled=lambda: False,
    )
    fields.update(overrides)
    return _ffn_flags(**fields)


def test_moe_c1_free_path_runs_on_explicit_inputs(monkeypatch) -> None:
    log = _Log()
    ops = _ffn_ops(
        router_logits_bf16_hidden=log.fn("router"),
        use_f32_selected_down=log.fn("f32_selected_down", returns=False),
        use_f32_selected_intermediate=log.fn("f32_selected_intermediate", returns=False),
        optional_q8_1_workspace_ptr=log.value("q8_1_lookup", None),
        selected_pair_requires_q8_1_input=log.fn("pair_q8_1", returns=False),
        selected_gemv_requires_q8_1_input=log.fn("gemv_q8_1", returns=False),
        selected_raw_moe_dual=log.fn("selected_dual"),
        selected_raw_moe_linear=log.fn("selected_linear"),
        shared_gate_up_from_f32_post_norm=log.fn("shared_from_f32", returns=False),
        use_f32_shared_down=log.fn("f32_shared_down", returns=False),
    )
    kernels = _ffn_kernels(
        qwen35_router_select=log.fn("router_select"),
        selected_ffn_modes=log.fn("selected_modes", returns=("dual", "single")),
        silu_mul_separate_out_bf16=log.fn("silu"),
        launch_gguf_linear_pair_concat=log.fn("shared_pair_concat", returns=True),
        silu_mul_dual_out_bf16=log.fn("silu_dual"),
        launch_gguf_linear=log.fn("shared_down"),
        weighted_sum_shared_gate_combine_residual_out_bf16_f32w=log.fn("combine"),
    )

    ffn.run_post_attention_moe_c1(
        _moe_layer(),
        _ffn_config(is_moe=True),
        # A MoE layer needs no dense FFN width.
        ffn.FfnGeometry(hidden_size=256, backend=_BACKEND),
        kernels,
        ops,
        _moe_flags(),
        0x30,
        _moe_scratch(),
        runtime=_RUNTIME,
        stream=19,
    )

    assert log.labels() == [
        "router",
        "router",
        "router_select",
        "f32_selected_down",
        "f32_selected_intermediate",
        "selected_modes",
        "pair_q8_1",
        "q8_1_lookup",
        "selected_dual",
        "silu",
        "gemv_q8_1",
        "q8_1_lookup",
        "selected_linear",
        "shared_from_f32",
        "shared_pair_concat",
        "silu_dual",
        "f32_shared_down",
        "shared_down",
        "combine",
    ]
    expert_router_args, _ = log.events[0][1], log.events[0][2]
    assert expert_router_args[0] == 0x201
    assert expert_router_args[2] == 0x207
    assert expert_router_args[3:] == (1, 256, 8)
    shared_router_args = log.events[1][1]
    assert shared_router_args[2] == 0x207 + 8 * DType.FP32.itemsize
    assert shared_router_args[3:] == (1, 256, 1)
    assert log.one("router_select")[0] == (0x207, 0x208, 0x209, 1, 8, 8, 2)
    assert log.one("router_select")[1] == dict(threads=256, stream=19, runtime=_RUNTIME)
    assert log.one("selected_modes")[1] == dict(
        allow_legacy_silu=True,
        f32_gate=False,
        f32_intermediate=False,
        f32_down=False,
    )
    dual_args, dual_kwargs = log.one("selected_dual")
    assert dual_args[2:] == (0x201, 0x208, 0x203, 0x203 + 2 * 64 * DType.BF16.itemsize)
    assert dual_kwargs["rows"] == 2
    assert dual_kwargs["in_features"] == 256
    assert dual_kwargs["out_features"] == 64
    assert log.one("selected_linear")[1]["x_rows"] == 2
    assert log.one("selected_linear")[1]["out_features"] == 256
    assert log.one("combine")[0] == (0x20B, 0x209, 0x210, 0x207 + 8 * 4, 0x202, 0x30, 2, 256)


def test_moe_c1_refuses_a_dense_config_and_a_small_scratch() -> None:
    with pytest.raises(ValueError, match="requires qwen35moe GGUF config"):
        ffn.run_post_attention_moe_c1(
            _moe_layer(),
            _ffn_config(is_moe=False),
            ffn.FfnGeometry(hidden_size=256, backend=_BACKEND),
            _ffn_kernels(),
            _ffn_ops(),
            _moe_flags(),
            0x30,
            _moe_scratch(),
            runtime=_RUNTIME,
        )
    with pytest.raises(ValueError, match="top-k capacity is too small"):
        ffn.run_post_attention_moe_c1(
            _moe_layer(),
            _ffn_config(is_moe=True),
            ffn.FfnGeometry(hidden_size=256, backend=_BACKEND),
            _ffn_kernels(),
            _ffn_ops(),
            _moe_flags(),
            0x30,
            _moe_scratch(moe_selected_host=NS(shape=(1,))),
            runtime=_RUNTIME,
        )


def test_moe_c1_observes_ops_republished_between_calls() -> None:
    """A resolver republished on the adapter is consulted by the next call."""

    log = _Log()
    resolver = {"fn": _fail("router not published")}
    ops = _ffn_ops(
        router_logits_bf16_hidden=lambda *a, **k: resolver["fn"](*a, **k),
        use_f32_selected_down=lambda *a, **k: False,
        use_f32_selected_intermediate=lambda *a, **k: False,
        optional_q8_1_workspace_ptr=lambda *a, **k: None,
        selected_pair_requires_q8_1_input=lambda *a, **k: False,
        selected_gemv_requires_q8_1_input=lambda *a, **k: False,
        selected_raw_moe_dual=lambda *a, **k: None,
        selected_raw_moe_linear=lambda *a, **k: None,
        shared_gate_up_from_f32_post_norm=lambda *a, **k: False,
        use_f32_shared_down=lambda *a, **k: False,
    )
    kernels = _ffn_kernels(
        qwen35_router_select=lambda *a, **k: None,
        selected_ffn_modes=lambda *a, **k: ("dual", "single"),
        silu_mul_separate_out_bf16=lambda *a, **k: None,
        launch_gguf_linear_pair_concat=lambda *a, **k: True,
        silu_mul_dual_out_bf16=lambda *a, **k: None,
        launch_gguf_linear=lambda *a, **k: None,
        weighted_sum_shared_gate_combine_residual_out_bf16_f32w=lambda *a, **k: None,
    )
    call = dict(
        layer=_moe_layer(),
        config=_ffn_config(is_moe=True),
        geometry=ffn.FfnGeometry(hidden_size=256, backend=_BACKEND),
        kernels=kernels,
        ops=ops,
        flags=_moe_flags(),
        out_ptr=0x30,
        scratch=_moe_scratch(),
        runtime=_RUNTIME,
    )

    resolver["fn"] = log.fn("first")
    ffn.run_post_attention_moe_c1(**call)
    resolver["fn"] = log.fn("second")
    ffn.run_post_attention_moe_c1(**call)

    assert log.labels() == ["first", "first", "second", "second"]
