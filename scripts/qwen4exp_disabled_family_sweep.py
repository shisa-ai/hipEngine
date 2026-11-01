#!/usr/bin/env python3
"""Measure recoverable time for each disabled Qwen4Exp prefill family.

For every family in ``PRODUCTION_ARITHMETIC_RECOVERY_FLAGS`` that the current
default leaves off, run the same production prefill twice: once as the current
default (the fallback) and once with that family's fast path enabled. A family
whose switch is request-local runner state passes an explicit ``owner`` argument
instead of publishing an env key. Record the wall time and, where the family
routes through the exact iu8 risk+repair instrument, the repair rate the fast
path would have to repair.

The ranking is by seconds recoverable on the measured prefill, not by any
historical headline gain. A family that is fast but numerically inadmissible
still appears here with its speedup, because the speedup is what a future
corrected arithmetic would be worth; admissibility is a separate axis and is
recorded from the project's own rejection evidence rather than re-derived.

This driver is retained as the reproduction path for two cited packets:
``benchmarks/results/2026-09-16-disabled-family-recoverable-time/`` and
``benchmarks/results/2026-09-16-q8-wmma-layers-recoverable-time/``. The decision
that keeps it is
``worklog/entries/20261031T200314.348659Z-subagent-refactor-r0037-retain-family-sweep-c5c9fe.md``.
``scripts/qwen4exp_disabled_selector_ranking.py``
asks the same question of the arithmetic-recovery selectors, but it carries
neither the owner-transported axes here (the three f16 WMMA layer scopes and
``Q4_DP4A64``) nor the override control, it does not collect the risk/repair
instrument, and it writes a per-arm report instead of this per-family one. Do
not delete this driver while those packets are cited as its output.

The ``Q8_WMMA_LAYERS_*`` arms disable the explicit dense-wide owner. The
production plan binds the wide-row route over layers 13-47 and that route claims
the WMMA-rewritten parent wherever its own scope covers the layer, so an arm
that leaves it bound measures the wide route under a WMMA name. Clearing it is
what makes these the 2026-09-16 packet's route, not a composition with the route
that replaced it. Their saving is therefore measured against a fallback that has
the wide route bound, and is not a recoverable-time figure for enabling WMMA
alone.

``Q4_DP4A64`` is a decode route. In prefill mode its owner override cannot move
the measured pass, so that arm is a transport-and-spread row, not a family
saving: it is never ranked or totalled, and its archived 2026-09-16 row stays in
``families`` as legacy data rather than as a reproduced current saving.

Every measured row records ``attribution_verified``, which says whether the
arm's requested transport is confirmed **at the level the child reports**: for
an environment arm, that every requested pair appears in the child's effective
route environment as asked and at least one of those keys moved from the value
the named profile binder wrote; for the WMMA scope arms, also that the owner
scope resolved to the requested layers; for the control, that its override
repeated the bound values and nothing else moved. It is a binding-level
statement, not a launch census: a resolved scope does not prove the kernel ran,
and an ``unknown`` row (the MoE, MMQ and DP4A owner arms carry no confirmation
field this driver reads) is not a verified one. Rows whose attribution is
``False`` are excluded from ``ranked_by_seconds_recoverable`` and from
``total_seconds_recoverable`` and listed in ``ranking_excludes`` with their
reason, so a JSON consumer cannot read a mismatched arm as a valid saving. Their
measurements are left as taken rather than rewritten.

This is a diagnostic. Its runs are not rate rows and it does not claim a
performance result.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The same-code control's arm name.
CONTROL_FAMILY = "NOOP_OVERRIDE_CONTROL"

#: A process-owned route key the profile binder always writes and
#: ``qwen4exp_profile_gap.py`` reports, so the fallback's ``route_env`` carries
#: the exact value an inert override can repeat.
CONTROL_KEY = "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL"

#: Clearing the wide owner makes a WMMA scope arm run WMMA. The companion
#: enablement and layer arguments are explicit owner controls, not process keys.
WMMA_ARM_ENV: dict[str, str] = {}

# Each entry: family name, the env key that enables it, the value that enables
# it, and any companion key/value the family needs. The GDN peer prefill family
# is absent: its layer scope is request-local runner state now, and the
# columnwarp owner ahead of it in the same chain is open on this lane, so an
# environment-only arm measured the incumbent rather than the peer owner. The
# layer2 gate does not reach it either: its candidates bind the peer scope but
# also bind colwarps on, so the columnwarp owner still takes the chunk.
FAMILIES: tuple[dict, ...] = (
    {
        "family": "PRODUCTION_MOE_PREFILL",
        "env": {},
        "owner": {"production_moe_prefill": "on"},
    },
    {
        "family": "Q4_IU8_PREFILL",
        "env": {
            "HIPENGINE_QWEN4_EXP_Q4_IU8_PREFILL": "1",
            "HIPENGINE_QWEN4_EXP_Q4_IU8_LAYERS": ",".join(map(str, range(35, 48))),
        },
    },
    {
        "family": "Q8_MMQ_PREFILL",
        "env": {},
        "owner": {"q8_mmq_prefill": "on"},
    },
    # The f16 WMMA dense Q8_0 selector is a layer list rather than a flag, and
    # it is not one of the recovery flags, so it was absent from the original
    # ranking. Its numerical envelope is gated in
    # benchmarks/results/2026-09-16-q8-wmma-dense-prefill-layers-gate/ at both
    # scopes; these arms supply the matching performance half.
    {
        # Control: an override that repeats the value the named profile binder
        # already wrote, so it moves no route. It measures the spread between a
        # child started without --override and one started with it, which is
        # what has to be subtracted from every family's saving.
        #
        # The value cannot be a constant. This arm read
        # HIPENGINE_QWEN4_EXP_Q8_DENSE_WIDE=0 while nothing bound that key; the
        # 2026-09-17 wide-route promotion bound it to 1 over layers 13-47, which
        # turned the "control" into an arm that disables the live default route.
        # The 2026-10-31 owner migration also removed the
        # configure_mmq_prefill_resources() re-run this arm used to price. Its
        # environment is resolved from the fallback's own route env below, and
        # the row records whether the override stayed inert.
        "family": CONTROL_FAMILY,
        "env": {},
    },
    {
        # 16-47 is the scope the 2026-09-17 promotion certified. The production
        # plan now binds 13-47 (PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS), so this
        # arm is a narrowing of the live route rather than its current scope.
        "family": "Q8_DENSE_WIDE_16_47",
        "env": {},
        "owner": {
            "q8_dense_wide": "on",
            "q8_dense_wide_layers": ",".join(map(str, range(16, 48))),
        },
    },
    {
        "family": "Q8_WMMA_LAYERS_16_47",
        "env": dict(WMMA_ARM_ENV),
        "owner": {
            "q8_wmma_layers": ",".join(map(str, range(16, 48))),
            "q8_dense_wide": "off",
            "q8_dense_wide_layers": "",
        },
    },
    {
        "family": "Q8_WMMA_LAYERS_32_47",
        "env": dict(WMMA_ARM_ENV),
        "owner": {
            "q8_wmma_layers": ",".join(map(str, range(32, 48))),
            "q8_dense_wide": "off",
            "q8_dense_wide_layers": "",
        },
    },
    {
        "family": "Q8_WMMA_LAYERS_0_47",
        "env": dict(WMMA_ARM_ENV),
        "owner": {
            "q8_wmma_layers": ",".join(map(str, range(0, 48))),
            "q8_dense_wide": "off",
            "q8_dense_wide_layers": "",
        },
    },
    {
        "family": "GR_IU8",
        "env": {"HIPENGINE_QWEN4_EXP_GR_IU8": "1"},
    },
    {
        "family": "GR_IU8_DOWN",
        "env": {"HIPENGINE_QWEN4_EXP_GR_IU8_DOWN": "1"},
    },
    {
        "family": "GDN_COLWARPS_PREFILL",
        "env": {},
        "owner": {
            "gdn_colwarps_prefill": "on",
            "gdn_colwarps_layers": ",".join(map(str, range(27, 48))),
        },
    },
    {
        # Transport coverage only. ``Q4_DP4A64`` is a decode route, so its owner
        # override cannot move the measured prefill pass; a new measurement of
        # this arm is never ranked as a prefill saving.
        "family": "Q4_DP4A64",
        "env": {},
        "savings_eligible": False,
        "owner": {
            "q4_dp4a_decode": "on",
            "q4_dp4a_layers": ",".join(
                map(str, (0, 2, 5, 6, 8, 9, 10, 11) + tuple(range(13, 48)))
            ),
        },
    },
)


def _run(
    python: str,
    model_root: Path,
    fixture: Path,
    case_id: str,
    repetitions: int,
    overrides: dict[str, str],
    output: Path,
    reuse: bool = False,
    owner: dict[str, str] | None = None,
) -> dict:
    if reuse and output.is_file():
        payload = json.loads(output.read_text())
        return _summarize(payload, ["reused", str(output)])
    command = [
        python,
        str(ROOT / "scripts" / "qwen4exp_profile_gap.py"),
        "--model-root", str(model_root),
        "--mode", "prefill",
        "--fixture", str(fixture),
        "--case-id", case_id,
        "--repetitions", str(repetitions),
        "--risk-diagnostics",
        "--output", str(output),
    ]
    for key, value in overrides.items():
        command += ["--override", f"{key}={value}"]
    if owner:
        if "q4_dp4a_decode" in owner:
            command += ["--q4-dp4a-decode", owner["q4_dp4a_decode"]]
            command += ["--q4-dp4a-layers", owner["q4_dp4a_layers"]]
        if "production_moe_prefill" in owner:
            command += [
                "--production-moe-prefill", owner["production_moe_prefill"]]
        if "q8_mmq_prefill" in owner:
            command += ["--q8-mmq-prefill", owner["q8_mmq_prefill"]]
        if "q8_wmma_layers" in owner:
            command += ["--q8-wmma-layers", owner["q8_wmma_layers"]]
        if 'q8_dense_wide' in owner:
            command += ['--q8-dense-wide', owner['q8_dense_wide']]
        if 'q8_dense_wide_layers' in owner:
            command += ['--q8-dense-wide-layers', owner['q8_dense_wide_layers']]
        if 'gdn_colwarps_layers' in owner:
            command += ['--gdn-colwarps-layers', owner['gdn_colwarps_layers']]
        if "gdn_colwarps_prefill" in owner:
            command += [
                "--gdn-colwarps-prefill", owner["gdn_colwarps_prefill"]]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=3600
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "returncode": completed.returncode,
            "stderr_tail": completed.stderr[-1500:],
            "command": command,
        }
    return _summarize(json.loads(output.read_text()), command)


def _control_override(fallback: dict) -> dict[str, str] | None:
    """An override that repeats the value the named profile binder wrote.

    ``None`` means there is no inert value to send - the fallback failed, or the
    binder wrote nothing for ``CONTROL_KEY``. Sending a value anyway is the
    defect this replaces: an override that differs from the binder's is a route
    change, and this arm is subtracted from every family's saving.
    """

    if not fallback.get("ok"):
        return None
    route = fallback.get("route_env")
    if not isinstance(route, dict):
        return None
    value = route.get(CONTROL_KEY)
    if value is None:
        return None
    return {CONTROL_KEY: str(value)}


def _owner_scope_bound(result: dict, owner: dict | None) -> bool | None:
    """Whether the child resolved the owner scope this arm asked for.

    The environment axes are checked by comparing ``bound_route_env`` with
    ``route_env``. An owner-transported axis has no environment value to
    compare, so its engagement record is the scope the child resolved after the
    bind (``q8_wmma_layers``). An arm that reports a different scope measured a
    different route under the arm's name. ``None`` when the arm does not carry
    that owner axis or the child did not report it.
    """

    requested = (owner or {}).get("q8_wmma_layers")
    if requested is None:
        return None
    report = result.get("q8_wmma_layers")
    if not isinstance(report, dict):
        return None
    layers = [
        int(part) for part in str(requested).split(",") if part.strip()
    ]
    return report.get("layers") == sorted(set(layers))


def _env_override_changed_route(payload: dict) -> bool | None:
    """Whether ``--override`` moved a value the named profile binder wrote.

    ``bound_route_env`` is the route environment right after the binder and
    ``route_env`` is the same set after the override is applied, so a control arm
    needs them to be identical. ``None`` when the child did not report both,
    which is also the shape of a report written before this field existed.
    """

    bound = payload.get("bound_route_env")
    effective = payload.get("route_env")
    if not isinstance(bound, dict) or not isinstance(effective, dict):
        return None
    return any(
        bound.get(key) != effective.get(key)
        for key in set(bound) | set(effective)
    )


def _env_request_applied(
    result: dict, env: dict[str, str], *, control: bool,
) -> tuple[bool | None, str]:
    """Whether every requested pair is in the child's effective route env.

    ``None`` when the child reported no bound/effective pair, which is also the
    shape of a payload written before those fields existed: an unverifiable arm
    is unknown, not engaged. A coarse "some route value moved" is not enough - an
    unrelated key changing would otherwise make an arm whose own override never
    landed look engaged, which is how a WMMA arm could be attributed while the
    wide route still claimed the parent.
    """

    bound = result.get("bound_route_env")
    effective = result.get("route_env")
    if not isinstance(bound, dict) or not isinstance(effective, dict):
        return None, "the child did not report the bound route environment"
    for key, value in env.items():
        if key not in effective:
            return (
                False,
                f"the requested key {key} is absent from the route environment",
            )
        if effective[key] != value:
            return (
                False,
                f"{key} is {effective[key]!r} in the route environment, not the "
                f"requested {value!r}",
            )
    if control:
        moved = sorted(
            key for key in set(bound) | set(effective)
            if bound.get(key) != effective.get(key)
        )
        if moved:
            return False, f"the control's override moved {', '.join(moved)}"
        return True, "the control's override repeated every bound value"
    if not any(bound.get(key) != effective.get(key) for key in env):
        return (
            False,
            "no requested key moved from its bound value, so the arm changed "
            "nothing",
        )
    return True, "every requested pair is in the route environment as asked"


def _attribution(result: dict, entry: dict, env: dict[str, str]) -> tuple[bool | None, str]:
    """Whether the arm's requested transport is confirmed at binding level.

    Returns ``(verified, note)``. ``True``/``False`` mean a check ran and the
    requested transport did or did not take effect; ``None`` means no field this
    driver reads can tell, which is also what a per-arm payload written before
    ``bound_route_env`` was reported looks like. This is not a launch census: a
    resolved scope says the binding took the request, not that the kernel ran.
    """

    owner = entry.get("owner") or {}
    checks: list[bool | None] = []
    notes: list[str] = []
    if entry["family"] == CONTROL_FAMILY:
        verified, note = _env_request_applied(result, env, control=True)
        checks.append(verified)
        notes.append(note)
    elif env:
        gr_axes = {'HIPENGINE_QWEN4_EXP_GR_IU8': 'up',
                   'HIPENGINE_QWEN4_EXP_GR_IU8_DOWN': 'down'}
        process_env = {key: value for key, value in env.items() if key not in gr_axes}
        if process_env:
            verified, note = _env_request_applied(result, process_env, control=False)
            checks.append(verified)
            notes.append(note)
        policy = result.get('gr_iu8_policy')
        for key, axis in gr_axes.items():
            if key in env:
                expected = env[key] not in {'', '0', 'false', 'False'}
                selected = policy.get(axis) if isinstance(policy, dict) else None
                checks.append(None if selected is None else selected is expected)
                notes.append(f'the GR {axis} owner matches the requested choice' if checks[-1] is True
                             else f'the child did not confirm the requested GR {axis} choice')
    if "q8_wmma_layers" in owner:
        # An arm can carry both an environment override and an owner scope. Both
        # have to hold: the scope is what decides whether this arm ran WMMA.
        bound = _owner_scope_bound(result, owner)
        if bound is None:
            checks.append(None)
            notes.append("the child did not report the resolved owner scope")
        elif bound:
            checks.append(True)
            notes.append("the owner scope resolved to the requested layers")
        else:
            checks.append(False)
            notes.append("the owner scope resolved to different layers")
    if 'q8_dense_wide' in owner or 'q8_dense_wide_layers' in owner:
        policy = result.get('q8_dense_wide_policy')
        if not isinstance(policy, dict):
            checks.append(False if isinstance(result.get('bound_route_env'), dict) else None)
            notes.append('the child did not report the resolved dense-wide policy')
        else:
            matches = (
                ('q8_dense_wide' not in owner or policy.get('enabled') is (owner['q8_dense_wide'] == 'on'))
                and ('q8_dense_wide_layers' not in owner or policy.get('layers') ==
                     (sorted({int(x) for x in owner['q8_dense_wide_layers'].split(',') if x}) or None)))
            checks.append(matches)
            notes.append('the dense-wide owner resolved to the requested policy' if matches
                         else 'the dense-wide owner resolved to a different policy')
    if 'gdn_colwarps_layers' in owner:
        selected = result.get('gdn_colwarps_layers')
        checks.append(None if selected is None else selected == owner['gdn_colwarps_layers'])
        notes.append('the GDN layer owner matches the requested scope' if checks[-1] is True
                     else 'the child did not confirm the requested GDN layer scope')
    if 'gdn_colwarps_prefill' in owner:
        selected = result.get('gdn_colwarps_prefill')
        checks.append(None if selected is None else selected is (owner['gdn_colwarps_prefill'] == 'on'))
        notes.append('the GDN admission owner matches the requested gate' if checks[-1] is True
                     else 'the child did not confirm the requested GDN admission gate')
    if not checks:
        return None, (
            "this arm's owner state has no confirmation field this driver reads"
        )
    if False in checks:
        return False, notes[checks.index(False)]
    if None in checks:
        return None, notes[checks.index(None)]
    return True, "; ".join(notes)


def _summarize(payload: dict, command) -> dict:
    risk = payload.get("risk_diagnostics") or {}
    # The instrument reports per role and per layer; there is no precomputed
    # total, so one is derived from the outputs and risk counts it records.
    by_role = risk.get("by_role") or {}
    outputs = sum(int(v.get("outputs") or 0) for v in by_role.values())
    risky = sum(int(v.get("risk") or 0) for v in by_role.values())
    max_rate = max(
        (float(v["max_repair_rate"]) for v in by_role.values()
         if v.get("max_repair_rate") is not None),
        default=None,
    )
    over_capacity = sum(
        int(v.get("over_capacity_calls") or 0) for v in by_role.values()
    )
    return {
        "ok": True,
        "command": command,
        "wall_summary": payload.get("wall_summary"),
        "wall_seconds": payload.get("wall_seconds"),
        "route_env": payload.get("route_env"),
        "bound_route_env": payload.get("bound_route_env"),
        "configuration_class": payload.get("configuration_class"),
        "fell_back_to_strict": payload.get("fell_back_to_strict"),
        "named_profile_intact": payload.get("named_profile_intact"),
        "q8_wmma_layers": payload.get("q8_wmma_layers"),
        "gdn_colwarps_layers": payload.get('gdn_colwarps_layers'),
        "gdn_colwarps_prefill": payload.get('gdn_colwarps_prefill'),
        "q8_dense_wide_policy": payload.get('q8_dense_wide_policy'),
        "gr_iu8_policy": payload.get('gr_iu8_policy'),
        "risk_calls": risk.get("calls"),
        "risk_outputs": outputs,
        "risk_risky": risky,
        "aggregate_repair_rate": (risky / outputs) if outputs else None,
        "max_repair_rate": max_rate,
        "over_capacity_calls": over_capacity,
        "risk_by_role": {k: v.get("aggregate_repair_rate") for k, v in by_role.items()},
        "logits_sha256": payload.get("logits_sha256"),
        "token_id": payload.get("token_id"),
        "env_override_changed_route": _env_override_changed_route(payload),
    }


def _median_wall(result: dict) -> float | None:
    if not result.get("ok"):
        return None
    summary = result.get("wall_summary") or {}
    value = summary.get("median")
    return float(value) if value is not None else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--case-id", default="code-p4096")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--only", action="append", default=None,
                        help="restrict to these family names")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument(
        "--reuse-scratch",
        action="store_true",
        help=(
            "Aggregate the runs already present in --scratch instead of "
            "re-measuring. The per-family JSONs are the measurement; this "
            "only rebuilds the ranking from them."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    args.scratch.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else None
    families = [f for f in FAMILIES if wanted is None or f["family"] in wanted]

    print("=== fallback (current default) ===", flush=True)
    fallback = _run(
        args.python, args.model_root, args.fixture, args.case_id,
        args.repetitions, {}, args.scratch / "fallback.json",
        reuse=args.reuse_scratch,
    )
    fallback_wall = _median_wall(fallback)
    print(f"  median prefill {fallback_wall} s  ok={fallback.get('ok')}", flush=True)

    rows = []
    for entry in families:
        name = entry["family"]
        env = entry["env"]
        if name == CONTROL_FAMILY:
            env = _control_override(fallback)
            if env is None:
                print(
                    f"  SKIPPED: the fallback recorded no value for {CONTROL_KEY}, "
                    "so no inert override exists",
                    flush=True,
                )
                rows.append({
                    "family": name,
                    "env": {},
                    "owner": entry.get("owner"),
                    "measured": False,
                    "skipped": (
                        f"the fallback recorded no value for {CONTROL_KEY}; "
                        "any override would move a route"
                    ),
                })
                continue
        print(f"=== {name} ===", flush=True)
        result = _run(
            args.python, args.model_root, args.fixture, args.case_id,
            args.repetitions, env, args.scratch / f"{name}.json",
            reuse=args.reuse_scratch, owner=entry.get("owner"),
        )
        wall = _median_wall(result)
        if wall is None or fallback_wall is None:
            print(f"  FAILED: {json.dumps(result)[:300]}", flush=True)
            rows.append({
                "family": name,
                "env": env,
                "owner": entry.get("owner"),
                "measured": False,
                "failure": result.get("stderr_tail") or result.get("returncode"),
            })
            continue
        saved = fallback_wall - wall
        owner_bound = _owner_scope_bound(result, entry.get("owner"))
        verified, note = _attribution(result, entry, env)
        savings_eligible = entry.get("savings_eligible", True)
        rows.append({
            "family": name,
            "env": env,
            "owner": entry.get("owner"),
            "measured": True,
            "candidate_median_s": round(wall, 4),
            "fallback_median_s": round(fallback_wall, 4),
            "seconds_recoverable": round(saved, 4),
            "speedup": round(fallback_wall / wall, 4),
            "aggregate_repair_rate": result.get("aggregate_repair_rate"),
            "max_repair_rate": result.get("max_repair_rate"),
            "over_capacity_calls": result.get("over_capacity_calls"),
            "logits_sha256": result.get("logits_sha256"),
            "same_logits_as_fallback": (
                result.get("logits_sha256") == fallback.get("logits_sha256")
            ),
            "named_profile_intact": result.get("named_profile_intact"),
            "fell_back_to_strict": result.get("fell_back_to_strict"),
            "env_override_changed_route": result.get(
                "env_override_changed_route"),
            "owner_scope_bound": owner_bound,
            "gr_iu8_policy": result.get('gr_iu8_policy'),
            "attribution_verified": verified,
            "attribution_note": note,
            "savings_eligible": savings_eligible,
        })
        print(
            f"  median {wall:.3f} s  saved {saved:+.3f} s  "
            f"speedup {fallback_wall / wall:.3f}x  "
            f"repair {result.get('aggregate_repair_rate')}",
            flush=True,
        )
        if not savings_eligible:
            print(
                "  note: decode-only route; this row is not a prefill saving",
                flush=True,
            )
        elif name == CONTROL_FAMILY and verified is False:
            print(
                "  WARNING: the control's override moved a route value; do "
                "not subtract this arm from any family's saving",
                flush=True,
            )
        elif verified is False:
            print(
                f"  WARNING: {note}; this row is excluded from the ranking",
                flush=True,
            )

    def _exclusion_reason(row: dict) -> str:
        if not row["savings_eligible"]:
            return "a decode-only route cannot measure a prefill saving"
        return row["attribution_note"]

    measured = [r for r in rows if r["measured"]]
    excluded = [
        r for r in measured
        if not r["savings_eligible"] or r["attribution_verified"] is False
    ]
    ranked = sorted(
        (r for r in measured if r not in excluded),
        key=lambda r: -r["seconds_recoverable"],
    )
    control_row = next(
        (r for r in measured if r["family"] == CONTROL_FAMILY), None
    )
    control_route_neutral = (
        None if control_row is None else control_row["attribution_verified"]
    )
    artifact = {
        "schema": 1,
        "kind": "qwen4exp_disabled_family_recoverable_time",
        "performance_claim": False,
        "numerics_evaluated": False,
        "why": (
            "Each disabled family is worth whatever time its fast path saves on a "
            "real production prefill, but only if a corrected arithmetic can be "
            "made admissible. This measures the first half; admissibility is a "
            "separate axis recorded from the project's own rejection evidence."
        ),
        "model_root": str(args.model_root),
        "fixture": str(args.fixture),
        "case_id": args.case_id,
        "repetitions": args.repetitions,
        "fallback": fallback,
        "fallback_median_s": fallback_wall,
        "families": rows,
        "ranked_by_seconds_recoverable": [
            {"family": r["family"], "seconds_recoverable": r["seconds_recoverable"],
             "speedup": r["speedup"],
             "attribution_verified": r["attribution_verified"]}
            for r in ranked
        ],
        # Measured rows whose requested transport is confirmed not to have taken
        # effect. Their measurements stay in ``families`` as taken; they are not
        # savings any consumer may read, so they are kept out of the ranking and
        # the total and named here with the reason.
        "ranking_excludes": [
            {"family": r["family"],
             "seconds_recoverable": r["seconds_recoverable"],
             "reason": _exclusion_reason(r)}
            for r in excluded
        ],
        "total_seconds_recoverable": round(
            sum(r["seconds_recoverable"] for r in ranked), 4
        ),
        "all_candidates_bit_identical": all(
            r["same_logits_as_fallback"] for r in ranked
        ) if ranked else None,
        # Whether the control arm's override repeated the binder's value. Only
        # then is its spread the difference to subtract from the other arms.
        "control_route_neutral": control_route_neutral,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=1) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
