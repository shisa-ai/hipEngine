"""Guards for the retained disabled-family sweep driver.

``scripts/qwen4exp_disabled_family_sweep.py`` is the reproduction path named by
two cited packets (``benchmarks/results/2026-09-16-disabled-family-recoverable-time/``
and ``benchmarks/results/2026-09-16-q8-wmma-layers-recoverable-time/``). The
decision that keeps it is
``worklog/entries/20261031T200314.348659Z-subagent-refactor-r0037-retain-family-sweep-c5c9fe.md``.
The GPU run needs the model,
so these tests replay the interface on host: the arm table, the child command it
builds for each axis, and the per-family JSON it aggregates.

Two arms had silently stopped measuring what they name:

* the no-op control carried ``HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE=0``, which was
  inert while nothing bound that key. The 2026-09-17 wide-route promotion bound
  it to ``1`` over layers 13-47, so the "control" became an arm that disables the
  live default route - and the sweep's own procedure subtracts it from every
  family's saving.
* the three ``Q8_WMMA_LAYERS_*`` arms bound a WMMA scope while the wide route
  stayed bound over layers 13-47. ``_q8_dense_wide_claims_parent`` accepts the
  WMMA-rewritten parent, so the wide route claims those dispatches and the arm
  measures it instead. The 2026-09-29 WMMA gate reported a green pass for a route
  that never ran for exactly this reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import qwen4exp_disabled_family_sweep as sweep

WIDE_ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE"
WIDE_LAYERS_ENV = "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS"
WMMA_ENV = "HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS"

# The six axes the ledger names, in addition to the families both harnesses share.
REPRODUCTION_AXES = (
    "Q8_DENSE_WIDE_16_47",
    "Q8_WMMA_LAYERS_16_47",
    "Q8_WMMA_LAYERS_32_47",
    "Q8_WMMA_LAYERS_0_47",
    "Q4_DP4A64",
    "NOOP_OVERRIDE_CONTROL",
)

# The route environment a named profile binder leaves behind for this lane. Only
# keys the binder writes can be repeated as an inert override.
BOUND_ROUTE_ENV = {
    "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1",
    "HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS": "35,36,37,38,39,40,41,42,43,44,45,46,47",
    "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT": "1",
    "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE": "1",
    "HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE_LAYERS": ",".join(map(str, range(13, 48))),
}

WALLS = {"fallback": 10.0, "NOOP_OVERRIDE_CONTROL": 10.05}

#: A route key no binder writes. Deliberately not a ``HIPENGINE_*`` literal:
#: ``scripts/check_envs_docs.py`` scans test files for environment names, and a
#: test-only token must not read as an undocumented production key.
UNDOCUMENTED_ROUTE_KEY = "TEST_ONLY_EXTRA_ROUTE_KEY"


def _family(name: str) -> dict:
    return next(item for item in sweep.FAMILIES if item["family"] == name)


def _override_pairs(command: list[str]) -> dict[str, str]:
    pairs = {}
    for index, token in enumerate(command):
        if token == "--override":
            key, _, value = command[index + 1].partition("=")
            pairs[key] = value
    return pairs


def _owner_pairs(command: list[str]) -> dict[str, str]:
    owner = {}
    for flag, field in (
        ("--q4-dp4a-decode", "q4_dp4a_decode"),
        ("--q4-dp4a-layers", "q4_dp4a_layers"),
        ("--q8-wmma-layers", "q8_wmma_layers"),
        ('--q8-dense-wide', 'q8_dense_wide'),
        ('--q8-dense-wide-layers', 'q8_dense_wide_layers'),
        ("--production-moe-prefill", "production_moe_prefill"),
        ("--q8-mmq-prefill", "q8_mmq_prefill"),
    ):
        if flag in command:
            owner[field] = command[command.index(flag) + 1]
    return owner


def _fake_child(monkeypatch: pytest.MonkeyPatch, calls: list[list[str]]) -> None:
    """Replay ``qwen4exp_profile_gap.py``'s report without a model."""

    def fake_run(command, **kwargs):
        calls.append(list(command))
        output = Path(command[command.index("--output") + 1])
        arm = output.stem
        overrides = _override_pairs(command)
        owner = _owner_pairs(command)
        effective = {**BOUND_ROUTE_ENV, **overrides}
        wall = WALLS.get(arm, 9.0)
        payload = {
            "wall_summary": {"median": wall, "tok_s": 4096.0 / wall},
            "wall_seconds": [wall],
            "bound_route_env": dict(BOUND_ROUTE_ENV),
            "route_env": effective,
            "configuration_class": (
                "diagnostic_post_binder_override"
                if overrides or owner
                else "named_profile"
            ),
            "fell_back_to_strict": False,
            "named_profile_intact": not overrides and not owner,
            "q8_wmma_layers": (
                None
                if "q8_wmma_layers" not in owner
                else {
                    "mode": "layers",
                    "layers": sorted({
                        int(part)
                        for part in owner["q8_wmma_layers"].split(",")
                        if part.strip()
                    }),
                }
            ),
            "q8_dense_wide_policy": {
                'enabled': owner.get('q8_dense_wide', 'on') == 'on',
                'layers': sorted({int(x) for x in owner.get('q8_dense_wide_layers', ','.join(map(str, range(13, 48)))).split(',') if x}) or None,
            },
            "risk_diagnostics": {
                "calls": 1,
                "by_role": {
                    "q4_iu8_exact:expert_gate_up": {
                        "outputs": 1000,
                        "risk": 7,
                        "max_repair_rate": 0.007,
                        "over_capacity_calls": 0,
                    }
                },
            },
            "logits_sha256": "fallback" if arm == "fallback" else arm,
            "token_id": 248068,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)


def _argv(tmp_path: Path, output: Path, *extra: str) -> list[str]:
    return [
        "--model-root", str(tmp_path / "model"),
        "--fixture", str(tmp_path / "fixture.json"),
        "--case-id", "code-p4096",
        "--repetitions", "1",
        "--scratch", str(tmp_path / "scratch"),
        "--output", str(output),
        *extra,
    ]


# --- the retained axes ------------------------------------------------------


def test_the_cited_reproduction_axes_are_present() -> None:
    names = [item["family"] for item in sweep.FAMILIES]
    missing = [name for name in REPRODUCTION_AXES if name not in names]
    assert not missing, missing


def test_the_noop_control_override_carries_the_binders_own_value() -> None:
    """A control whose value differs from the binder's is a route change.

    ``HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE=0`` was inert only until the wide route
    was bound to ``1`` over layers 13-47; from then on the arm disabled the live
    default route while still being subtracted from every family's saving.
    """

    fallback = {"ok": True, "route_env": dict(BOUND_ROUTE_ENV)}
    assert sweep._control_override(fallback) == {
        sweep.CONTROL_KEY: BOUND_ROUTE_ENV[sweep.CONTROL_KEY]
    }
    assert sweep.CONTROL_KEY in BOUND_ROUTE_ENV, (
        "the control key must be one the binder writes, or no inert value exists"
    )


def test_the_noop_control_is_skipped_when_no_inert_value_exists() -> None:
    assert sweep._control_override({"ok": True, "route_env": {}}) is None
    assert sweep._control_override({"ok": False, "returncode": 1}) is None
    assert sweep._control_override({"ok": True, "route_env": None}) is None


def test_a_moved_route_value_is_reported() -> None:
    neutral = {
        "bound_route_env": dict(BOUND_ROUTE_ENV),
        "route_env": dict(BOUND_ROUTE_ENV),
    }
    assert sweep._env_override_changed_route(neutral) is False

    moved = {
        "bound_route_env": dict(BOUND_ROUTE_ENV),
        "route_env": {**BOUND_ROUTE_ENV, WIDE_ENV: "0"},
    }
    assert sweep._env_override_changed_route(moved) is True

    added = {
        "bound_route_env": dict(BOUND_ROUTE_ENV),
        "route_env": {**BOUND_ROUTE_ENV, UNDOCUMENTED_ROUTE_KEY: "1"},
    }
    assert sweep._env_override_changed_route(added) is True
    assert sweep._env_override_changed_route({}) is None


def test_the_wmma_scope_arms_clear_the_wide_route() -> None:
    """The wide route claims the WMMA-rewritten parent wherever it is scoped.

    ``_q8_dense_wide_claims_parent`` accepts ``wmma_<parent>`` under the
    ``wmma_raw`` ABI, and the production plan binds the wide route over layers
    13-47, so a WMMA scope arm that leaves it bound measures the wide route.
    """

    for name in (
        "Q8_WMMA_LAYERS_16_47",
        "Q8_WMMA_LAYERS_32_47",
        "Q8_WMMA_LAYERS_0_47",
    ):
        env = _family(name)["env"]
        assert env == {}, name
        assert _family(name)['owner']['q8_dense_wide'] == 'off'
        assert _family(name)['owner']['q8_dense_wide_layers'] == ''
        assert WMMA_ENV not in env, (
            f"{name}: the scope is owner state, not an environment value"
        )


def test_the_owner_scope_engagement_is_checked_against_the_request() -> None:
    """A scope arm must report the scope it asked for, not just any scope.

    The child resolves the owner scope after the bind and reports it as
    ``q8_wmma_layers``. An arm that records a different scope measured a
    different route under the arm's name.
    """

    owner = {"q8_wmma_layers": "16,17,18"}
    bound = {"q8_wmma_layers": {"mode": "layers", "layers": [16, 17, 18]}}
    assert sweep._owner_scope_bound(bound, owner) is True

    other = {"q8_wmma_layers": {"mode": "layers", "layers": [0, 1, 2]}}
    assert sweep._owner_scope_bound(other, owner) is False

    empty = {"q8_wmma_layers": {"mode": "empty", "layers": None}}
    assert sweep._owner_scope_bound(empty, owner) is False

    # Not an owner-transported arm, or the child did not report the scope.
    assert sweep._owner_scope_bound(bound, {}) is None
    assert sweep._owner_scope_bound(bound, {"q4_dp4a_decode": "on"}) is None
    assert sweep._owner_scope_bound({}, owner) is None


# --- the interface, replayed on host ---------------------------------------


def test_each_axis_builds_the_child_command_it_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    _fake_child(monkeypatch, calls)
    output = tmp_path / "artifact.json"
    assert sweep.main(_argv(tmp_path, output)) == 0

    by_family = {
        Path(command[command.index("--output") + 1]).stem: command
        for command in calls
    }
    assert set(by_family) == {"fallback"} | {
        item["family"] for item in sweep.FAMILIES
    }

    # Owner-transported axes carry no environment override for the owner state.
    dp4a = _override_pairs(by_family["Q4_DP4A64"])
    assert dp4a == {}
    assert _owner_pairs(by_family["Q4_DP4A64"]) == {
        "q4_dp4a_decode": "on",
        "q4_dp4a_layers": _family("Q4_DP4A64")["owner"]["q4_dp4a_layers"],
    }

    for name in ("Q8_WMMA_LAYERS_16_47", "Q8_WMMA_LAYERS_32_47", "Q8_WMMA_LAYERS_0_47"):
        overrides = _override_pairs(by_family[name])
        assert overrides == {}, name
        assert _owner_pairs(by_family[name])['q8_dense_wide'] == 'off'
        assert _owner_pairs(by_family[name])['q8_dense_wide_layers'] == ''
        assert WMMA_ENV not in overrides, name
        assert _owner_pairs(by_family[name])["q8_wmma_layers"] == (
            _family(name)["owner"]["q8_wmma_layers"]
        )

    wide = _owner_pairs(by_family["Q8_DENSE_WIDE_16_47"])
    assert wide['q8_dense_wide'] == "on"
    assert wide['q8_dense_wide_layers'] == ",".join(map(str, range(16, 48)))

    # The control repeats the binder's value, so it is inert by construction.
    assert _override_pairs(by_family["NOOP_OVERRIDE_CONTROL"]) == {
        sweep.CONTROL_KEY: BOUND_ROUTE_ENV[sweep.CONTROL_KEY]
    }


def test_the_artifact_keeps_every_axis_and_records_control_neutrality(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    _fake_child(monkeypatch, calls)
    output = tmp_path / "artifact.json"
    assert sweep.main(_argv(tmp_path, output)) == 0
    artifact = json.loads(output.read_text())

    assert artifact["kind"] == "qwen4exp_disabled_family_recoverable_time"
    assert artifact["performance_claim"] is False
    assert artifact["numerics_evaluated"] is False
    names = [row["family"] for row in artifact["families"]]
    for name in REPRODUCTION_AXES:
        assert name in names, name
    assert artifact["control_route_neutral"] is True
    assert artifact["fallback_median_s"] == pytest.approx(10.0)

    control = next(
        row for row in artifact["families"]
        if row["family"] == "NOOP_OVERRIDE_CONTROL"
    )
    assert control["measured"] is True
    assert control["env"] == {sweep.CONTROL_KEY: BOUND_ROUTE_ENV[sweep.CONTROL_KEY]}
    assert control["env_override_changed_route"] is False

    # The ranking consumer still reads the fields the cited packet used.
    ranked = {row["family"]: row for row in artifact["ranked_by_seconds_recoverable"]}
    assert "Q8_DENSE_WIDE_16_47" in ranked
    assert ranked["Q8_DENSE_WIDE_16_47"]["seconds_recoverable"] == pytest.approx(1.0)

    # Owner-transported axes record the scope the child actually resolved.
    for name in ("Q8_WMMA_LAYERS_16_47", "Q8_WMMA_LAYERS_32_47", "Q8_WMMA_LAYERS_0_47"):
        row = next(item for item in artifact["families"] if item["family"] == name)
        assert row["owner_scope_bound"] is True, name
    dp4a = next(
        row for row in artifact["families"] if row["family"] == "Q4_DP4A64"
    )
    assert dp4a["owner_scope_bound"] is None


def test_the_control_is_flagged_when_its_override_moves_a_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A route-changing control must not be presented as a subtraction baseline."""

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        overrides = _override_pairs(command)
        if output.stem == sweep.CONTROL_FAMILY:
            overrides = {**overrides, WIDE_ENV: "0"}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "wall_summary": {"median": 10.0},
            "wall_seconds": [10.0],
            "bound_route_env": dict(BOUND_ROUTE_ENV),
            "route_env": {**BOUND_ROUTE_ENV, **overrides},
            "risk_diagnostics": {"by_role": {}},
            "logits_sha256": output.stem,
            "token_id": 1,
        }))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    output = tmp_path / "artifact.json"
    assert sweep.main(_argv(tmp_path, output, "--only", sweep.CONTROL_FAMILY)) == 0
    artifact = json.loads(output.read_text())
    assert artifact["control_route_neutral"] is False
    # A warning is not enough: a mismatched control must not be readable as a
    # measured saving by a JSON consumer.
    ranked = [row["family"] for row in artifact["ranked_by_seconds_recoverable"]]
    assert sweep.CONTROL_FAMILY not in ranked
    assert len(artifact["ranking_excludes"]) == 1
    excluded = artifact["ranking_excludes"][0]
    assert excluded["family"] == sweep.CONTROL_FAMILY
    assert excluded["reason"].startswith("the control's override moved")
    assert artifact["total_seconds_recoverable"] == 0.0


def test_a_requested_value_the_child_did_not_apply_is_not_attributed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A WMMA arm whose wide-route clear did not land measured the wide route.

    The coarse check asked whether *any* route value moved, so an unrelated key
    changing made the arm look engaged while
    ``HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE`` still claimed the parent. Every
    requested pair has to appear in the route environment as asked.
    """

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        arm = output.stem
        owner = _owner_pairs(command)
        requested = (
            sorted(
                int(part) for part in owner["q8_wmma_layers"].split(",")
                if part.strip()
            )
            if "q8_wmma_layers" in owner
            else None
        )
        # Wide disable is NOT applied; an unrelated key moves instead.
        effective = dict(BOUND_ROUTE_ENV)
        if arm != "fallback":
            effective["HIPENGINE_QWEN4_EXP_GR_IU8"] = "1"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "wall_summary": {"median": 10.0 if arm == "fallback" else 9.0},
            "wall_seconds": [10.0],
            "bound_route_env": dict(BOUND_ROUTE_ENV),
            "route_env": effective,
            "q8_wmma_layers": (
                None if requested is None
                else {"mode": "layers", "layers": requested}
            ),
            'q8_dense_wide_policy': {'enabled': True, 'layers': list(range(13, 48))},
            "risk_diagnostics": {"by_role": {}},
            "logits_sha256": arm,
            "token_id": 1,
        }))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    output = tmp_path / "artifact.json"
    assert sweep.main(_argv(
        tmp_path, output, "--only", "Q8_WMMA_LAYERS_16_47"
    )) == 0
    artifact = json.loads(output.read_text())

    row = artifact["families"][0]
    assert row["family"] == "Q8_WMMA_LAYERS_16_47"
    # The scope matched; that must not rescue an unapplied owner disable.
    assert row["owner_scope_bound"] is True
    assert row["attribution_verified"] is False
    assert 'dense-wide owner' in row["attribution_note"]
    assert artifact["ranked_by_seconds_recoverable"] == []
    assert [item["family"] for item in artifact["ranking_excludes"]] == [
        "Q8_WMMA_LAYERS_16_47"
    ]
    assert artifact["total_seconds_recoverable"] == 0.0


def test_a_requested_policy_absent_from_the_report_is_not_attributed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A requested owner policy the child never reports cannot be called applied."""

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        arm = output.stem
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "wall_summary": {"median": 10.0 if arm == "fallback" else 9.0},
            "wall_seconds": [10.0],
            "bound_route_env": dict(BOUND_ROUTE_ENV),
            # An unrelated key moves, but the requested wide clear is missing.
            "route_env": {
                "HIPENGINE_QWEN4_EXP_GR_IU8": "1",
                "HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT": "1",
            },
            "risk_diagnostics": {"by_role": {}},
            "logits_sha256": arm,
            "token_id": 1,
        }))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    output = tmp_path / "artifact.json"
    assert sweep.main(_argv(
        tmp_path, output, "--only", "Q8_WMMA_LAYERS_16_47"
    )) == 0
    artifact = json.loads(output.read_text())

    row = artifact["families"][0]
    assert row["attribution_verified"] is False
    assert 'did not report' in row["attribution_note"]
    assert artifact["ranked_by_seconds_recoverable"] == []


def test_the_q4_dp4a_transport_row_is_not_a_savings_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A decode route cannot measure a prefill gain.

    The arm stays in the table for transport coverage and its measurement stays
    in ``families`` as taken, but it is never ranked or totalled as a saving.
    """

    calls: list[list[str]] = []
    _fake_child(monkeypatch, calls)
    output = tmp_path / "artifact.json"
    assert sweep.main(_argv(tmp_path, output, "--only", "Q4_DP4A64")) == 0
    artifact = json.loads(output.read_text())

    row = artifact["families"][0]
    assert row["family"] == "Q4_DP4A64"
    assert row["measured"] is True
    assert row["savings_eligible"] is False
    assert row["candidate_median_s"] == 9.0
    assert artifact["ranked_by_seconds_recoverable"] == []
    assert artifact["ranking_excludes"] == [{
        "family": "Q4_DP4A64",
        "seconds_recoverable": 1.0,
        "reason": "a decode-only route cannot measure a prefill saving",
    }]
    assert artifact["total_seconds_recoverable"] == 0.0


def test_an_owner_scope_mismatch_is_excluded_from_the_ranking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A WMMA arm that resolved another scope measured another route."""

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        arm = output.stem
        overrides = _override_pairs(command)
        owner = _owner_pairs(command)
        requested = (
            sorted(
                int(part) for part in owner["q8_wmma_layers"].split(",")
                if part.strip()
            )
            if "q8_wmma_layers" in owner
            else None
        )
        resolved = {
            "mode": "layers",
            "layers": (
                [0, 1, 2] if arm == "Q8_WMMA_LAYERS_16_47" else requested
            ),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "wall_summary": {"median": 10.0 if arm == "fallback" else 9.0},
            "wall_seconds": [10.0],
            "bound_route_env": dict(BOUND_ROUTE_ENV),
            "route_env": {**BOUND_ROUTE_ENV, **overrides},
            "q8_wmma_layers": resolved if "q8_wmma_layers" in owner else None,
            'q8_dense_wide_policy': {'enabled': False, 'layers': None},
            "risk_diagnostics": {"by_role": {}},
            "logits_sha256": arm,
            "token_id": 1,
        }))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    output = tmp_path / "artifact.json"
    assert sweep.main(_argv(
        tmp_path, output, "--only", "Q8_WMMA_LAYERS_16_47",
        "--only", "Q8_WMMA_LAYERS_32_47",
    )) == 0
    artifact = json.loads(output.read_text())

    rows = {row["family"]: row for row in artifact["families"]}
    assert rows["Q8_WMMA_LAYERS_16_47"]["owner_scope_bound"] is False
    assert rows["Q8_WMMA_LAYERS_16_47"]["attribution_verified"] is False
    assert rows["Q8_WMMA_LAYERS_16_47"]["attribution_note"] == (
        "the owner scope resolved to different layers"
    )
    # The measurement is left as taken rather than rewritten or dropped.
    assert rows["Q8_WMMA_LAYERS_16_47"]["candidate_median_s"] == 9.0
    assert rows["Q8_WMMA_LAYERS_16_47"]["seconds_recoverable"] == 1.0

    ranked = [row["family"] for row in artifact["ranked_by_seconds_recoverable"]]
    assert ranked == ["Q8_WMMA_LAYERS_32_47"]
    assert [row["family"] for row in artifact["ranking_excludes"]] == [
        "Q8_WMMA_LAYERS_16_47"
    ]
    assert artifact["total_seconds_recoverable"] == 1.0


def test_historical_reuse_stays_unknown_and_is_not_rewritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A payload written before ``bound_route_env`` was reported is unknown.

    The cited 2026-09-16 per-arm JSONs have no bound-route record, so their
    attribution cannot be checked. They stay measured with their numbers and
    stay in the ranking, labelled unknown rather than verified or excluded. The
    one exception is the decode-only ``Q4_DP4A64`` row, which keeps its archived
    measurement but is never ranked as a prefill saving.
    """

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        arm = output.stem
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "wall_summary": {"median": 10.0 if arm == "fallback" else 9.0},
            "wall_seconds": [10.0],
            "route_env": {"HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1"},
            "risk_diagnostics": {"by_role": {}},
            "logits_sha256": arm,
            "token_id": 1,
        }))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    scratch = tmp_path / "scratch"
    first = tmp_path / "first.json"
    assert sweep.main(_argv(tmp_path, first)) == 0

    second = tmp_path / "second.json"
    assert sweep.main(_argv(tmp_path, second, "--reuse-scratch")) == 0
    artifact = json.loads(second.read_text())

    for row in artifact["families"]:
        assert row["attribution_verified"] is None, row["family"]
        assert row["attribution_note"], row["family"]
        assert row["seconds_recoverable"] == 1.0, row["family"]
    assert [item["family"] for item in artifact["ranking_excludes"]] == [
        "Q4_DP4A64"
    ]
    assert artifact["ranking_excludes"][0]["seconds_recoverable"] == 1.0
    assert len(artifact["ranked_by_seconds_recoverable"]) == len(
        artifact["families"]
    ) - 1
    assert artifact["control_route_neutral"] is None
    assert scratch.is_dir()


def test_the_per_arm_json_files_rebuild_the_ranking_without_remeasuring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    _fake_child(monkeypatch, calls)
    scratch = tmp_path / "scratch"
    first = tmp_path / "first.json"
    assert sweep.main(_argv(tmp_path, first)) == 0
    measured = len(calls)

    written = sorted(path.name for path in scratch.glob("*.json"))
    assert "fallback.json" in written
    for name in REPRODUCTION_AXES:
        assert f"{name}.json" in written, name

    second = tmp_path / "second.json"
    assert sweep.main(_argv(tmp_path, second, "--reuse-scratch")) == 0
    assert len(calls) == measured, "the reuse path must not launch a child"

    rebuilt = json.loads(second.read_text())
    original = json.loads(first.read_text())
    assert rebuilt["ranked_by_seconds_recoverable"] == (
        original["ranked_by_seconds_recoverable"]
    )
    assert rebuilt["fallback_median_s"] == original["fallback_median_s"]
    assert rebuilt["control_route_neutral"] is True


def test_every_arm_builds_a_command_the_child_parser_accepts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An arm the child rejects would fail every run; an arm it accepts
    silently is the more expensive mistake, so both are checked here."""

    from scripts import qwen4exp_profile_gap as child

    seen: list[list[str]] = []

    def fake_run(command, **kwargs):
        seen.append(list(command))
        return SimpleNamespace(returncode=1, stderr="stopped")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    for entry in sweep.FAMILIES:
        env = entry["env"]
        if entry["family"] == sweep.CONTROL_FAMILY:
            env = sweep._control_override(
                {"ok": True, "route_env": dict(BOUND_ROUTE_ENV)}
            )
            assert env is not None
        sweep._run(
            "python", Path("/tmp/model"), Path("/tmp/fixture.json"),
            "case", 1, env, tmp_path / f"{entry['family']}.json",
            owner=entry.get("owner"),
        )

    assert len(seen) == len(sweep.FAMILIES)
    parser = child.build_parser()
    for entry, command in zip(sweep.FAMILIES, seen):
        args = parser.parse_args(command[2:])
        child._parse_overrides(args.override)
        assert args.mode == "prefill"
        assert args.risk_diagnostics is True
        if entry["family"] == "Q4_DP4A64":
            assert args.q4_dp4a_decode == "on"
            assert args.q4_dp4a_layers == entry["owner"]["q4_dp4a_layers"]
        if entry["family"].startswith("Q8_WMMA_LAYERS_"):
            assert args.q8_wmma_layers == entry["owner"]["q8_wmma_layers"]


def test_an_unresolvable_control_is_skipped_rather_than_faked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({
            "wall_summary": {"median": 10.0},
            "wall_seconds": [10.0],
            # A binder that wrote no value for the control key.
            "bound_route_env": {"HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT": "1"},
            "route_env": {"HIPENGINE_QWEN4_EXP_Q4_IU8_EXACT": "1"},
            "risk_diagnostics": {"by_role": {}},
            "logits_sha256": "x",
            "token_id": 1,
        }))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)
    output = tmp_path / "artifact.json"
    assert sweep.main(_argv(tmp_path, output, "--only", sweep.CONTROL_FAMILY)) == 0
    artifact = json.loads(output.read_text())

    assert len(calls) == 1, "only the fallback may run"
    row = artifact["families"][0]
    assert row["family"] == sweep.CONTROL_FAMILY
    assert row["measured"] is False
    assert sweep.CONTROL_KEY in row["skipped"]
    assert artifact["control_route_neutral"] is None
