#!/usr/bin/env python3
"""Gate Qwen3.8 packed-AR direct INT8 decode across C-width transitions.

This is the IKV-C2 serving-lifecycle gate.  The steady-state model gate
(``qwen38_int8_batch_decode_gate.py``) proves that a fixed four-row packed
group matches independent c1 exactly.  It does not exercise the case that
actually stresses the row-batched INT8 consumer: a packed group whose active
lanes change while the physical width stays fixed.

That transition is the risky one because the direct INT8 split-K producer and
its strided reducer address rows by physical lane, and retirement/admission
rebuilds the slot->session mapping underneath them.  This gate therefore:

1. prefills six independent sessions through the exact scalar path and records
   each one's c1 token/logit/state trajectory;
2. replays a four-lane schedule that retires two lanes and admits two
   newcomers into the freed physical lanes, advancing every session's c1
   reference on exactly the steps where that session is active;
3. requires every active row's token, logits, and post-step state to equal its
   own c1 reference, and requires a retired lane's state to stay byte-frozen;
4. requires every packed step to report the direct batch route at physical
   width 4 with zero host row iterations.

It is a correctness/ownership harness, never a performance benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.core.dtype import DType  # noqa: E402
from hipengine.kernels.backends import hip_target_arch_for_backend  # noqa: E402
from hipengine.kvcache import FixedPagedKVPolicy  # noqa: E402
from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.models.kv_capabilities import KVCapabilityKey, model_artifact_identity  # noqa: E402
from hipengine.models.qwen35 import Qwen35GGUFModel  # noqa: E402
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession  # noqa: E402
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402
from scripts.gguf_mtp_bench import build_chat_prompt  # noqa: E402
from scripts.qwen38_int8_batch_decode_gate import (  # noqa: E402
    _capture_state,
    _live_kv_payload_nbytes,
    _load_prompt_rows,
    _logit_metrics,
    _sha256_bytes,
    _state_mismatches,
    _temporary_env,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
DEFAULT_LENGTHS = (255, 256, 257, 512)
_REQUIRED_CATEGORIES = ("code", "general_en", "general_ja", "mixed_ja_en")
_CAPTURE_PREFILL_GDN_ENV = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"
_GDN_PREFILL_MODE_ENV = "HIPENGINE_GGUF_GDN_PREFILL_MODE"
_NEWCOMER_LENGTHS = (511, 258)
_NEWCOMER_CATEGORIES = ("code", "general_ja")

# Physical lane -> session role at each decode step.  Lanes 1 and 2 retire after
# step 2 and are refilled by the newcomers from step 4; lanes 0 and 3 hold their
# session for the whole schedule so a steady lane and a recycled lane are
# observed in the same packed group.
_TRANSITION_SCHEDULE: tuple[tuple[tuple[int, str], ...], ...] = (
    ((0, "orig0"), (1, "orig1"), (2, "orig2"), (3, "orig3")),
    ((0, "orig0"), (1, "orig1"), (2, "orig2"), (3, "orig3")),
    ((0, "orig0"), (1, "orig1"), (2, "orig2"), (3, "orig3")),
    ((0, "orig0"), (3, "orig3")),
    ((0, "orig0"), (1, "new0"), (2, "new1"), (3, "orig3")),
    ((0, "orig0"), (1, "new0"), (2, "new1"), (3, "orig3")),
    ((0, "orig0"), (1, "new0"), (2, "new1"), (3, "orig3")),
    ((0, "orig0"), (1, "new0"), (2, "new1"), (3, "orig3")),
)

_RETIRED_ROLES = ("orig1", "orig2")
_RETIRE_AFTER_STEP = 2


def _quant_key(info: Any) -> str:
    name = str(getattr(info, "file_type_name", "") or "").strip().lower()
    if name.startswith("mostly_"):
        name = name[len("mostly_") :]
    if not name:
        raise ValueError("GGUF metadata does not expose file_type_name")
    return f"gguf_{name}"


def _build_prompts(
    tokenizer: Qwen35GGUFTokenizer,
    rows: Sequence[Mapping[str, Any]],
    lengths: Sequence[int],
) -> tuple[tuple[tuple[int, ...], ...], list[dict[str, Any]]]:
    if len(rows) != len(lengths):
        raise ValueError("prompt rows and lengths must align")
    prompts: list[tuple[int, ...]] = []
    manifest: list[dict[str, Any]] = []
    for row, length in zip(rows, lengths, strict=True):
        target = int(length)
        if target <= 0:
            raise ValueError("prompt lengths must be positive")
        expanded = "\n".join([str(row["content"])] * 128)
        tokens = tuple(int(token) for token in build_chat_prompt(tokenizer, expanded))
        if len(tokens) < target:
            raise ValueError(f"expanded prompt {row['id']!r} has only {len(tokens)} tokens")
        selected = tokens[:target]
        prompts.append(selected)
        manifest.append(
            {
                "id": row["id"],
                "category": row["category"],
                "source_text_sha256": _sha256_bytes(str(row["content"]).encode("utf-8")),
                "tokens": target,
                "token_ids_sha256": _sha256_bytes(np.asarray(selected, dtype=np.int64).tobytes()),
            }
        )
    return tuple(prompts), manifest


def _session_policy() -> FixedPagedKVPolicy:
    return FixedPagedKVPolicy(
        block_size=256,
        storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        scale_granularity="per_token_head",
    )


def _state_diff(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """Name the differing state components so a mismatch is actionable."""
    diff: list[str] = []
    if int(actual["position"]) != int(expected["position"]):
        diff.append(f"position:{actual['position']}!={expected['position']}")
    for component, parts in (("linear", ("conv", "recurrent")), ("kv", ("key_payload", "value_payload"))):
        actual_layers = {int(row["layer"]): row for row in actual[component]}
        expected_layers = {int(row["layer"]): row for row in expected[component]}
        for layer in sorted(set(actual_layers) | set(expected_layers)):
            got = actual_layers.get(layer)
            want = expected_layers.get(layer)
            if got is None or want is None:
                diff.append(f"{component}.layer{layer}:presence")
                continue
            for part in parts:
                if part in got or part in want:
                    if got.get(part) != want.get(part):
                        diff.append(f"{component}.layer{layer}.{part}")
    return diff


def _packed_replay(
    owner: Qwen35GGUFResidentSession,
    roles: Mapping[str, Qwen35GGUFResidentSession],
    prompts: Mapping[str, Sequence[int]],
    *,
    rows: int,
    steps: int,
    layer_ids: tuple[int, ...],
    reference: Mapping[str, Mapping[str, Any]],
    max_kl: float,
) -> dict[str, Any]:
    """Replay the transition schedule as packed steps and diff every row.

    Must run under the same exact-prefill environment as the c1 references:
    the packed group imports the per-slot state that the reference prefill
    produced, so preflighting the replay under a different prefill policy would
    compare two different contracts.
    """
    for session in roles.values():
        session.reset()
    current_tokens: dict[str, int] = {}
    for role, session in roles.items():
        first = session.prefill(prompts[role], return_logits=True)
        current_tokens[role] = int(first.token_id)

    retired_frozen: dict[str, dict[str, Any]] = {}
    step_records: list[dict[str, Any]] = []
    token_mismatches: list[dict[str, Any]] = []
    logit_mismatches: list[dict[str, Any]] = []
    hidden_mismatches: list[dict[str, Any]] = []
    state_mismatches: list[dict[str, Any]] = []
    retired_mutations: list[dict[str, Any]] = []
    # Each role's c1 reference index: how many results it has consumed.
    consumed: dict[str, int] = {role: 0 for role in roles}

    for step, lane_map in enumerate(_TRANSITION_SCHEDULE):
        lane_roles = [role for _lane, role in lane_map]
        lane_indices = tuple(int(lane) for lane, _role in lane_map)
        active_sessions = [roles[role] for role in lane_roles]
        try:
            results = owner.step_batch_native(
                [current_tokens[role] for role in lane_roles],
                sessions=active_sessions,
                return_logits=True,
                require_logits=True,
                scatter_state=True,
                capture_layer_output_hidden=layer_ids,
                physical_rows=rows,
                active_slot_indices=lane_indices,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"packed step {step} failed for lanes {lane_indices} "
                f"roles {lane_roles}: {type(exc).__name__}: {exc}"
            ) from exc
        manifest = copy.deepcopy(owner.last_packed_execution_manifest)
        step_record: dict[str, Any] = {
            "step": step,
            "active_lanes": list(lane_indices),
            "active_roles": list(lane_roles),
            "physical_rows": manifest.get("physical_rows"),
            "active_rows": manifest.get("active_rows"),
            "active_mask": manifest.get("active_mask"),
            "route": manifest.get("full_attention_decode_path"),
            "host_model_row_iterations": manifest.get("model_step", {}).get(
                "host_model_row_iterations"
            ),
            "rows": [],
        }
        for role, session, result in zip(lane_roles, active_sessions, results, strict=True):
            current_tokens[role] = int(result.token_id)
            consumed[role] += 1
            expected = reference[role]
            expected_token = int(expected["tokens"][consumed[role]])
            expected_logits = expected["logits"][consumed[role]]
            actual_logits = np.asarray(result.logits, dtype=np.float32)
            metrics = _logit_metrics(actual_logits, expected_logits)
            actual_hidden = {
                int(layer): _sha256_bytes(np.asarray(array, dtype=np.float32).tobytes())
                for layer, array in session.last_layer_output_hidden.items()
            }
            lane = lane_indices[lane_roles.index(role)]
            if int(result.token_id) != expected_token:
                token_mismatches.append(
                    {
                        "step": step,
                        "role": role,
                        "lane": lane,
                        "packed": int(result.token_id),
                        "c1": expected_token,
                    }
                )
            if not metrics["top1_match"] or float(metrics["kl"]) > float(max_kl):
                logit_mismatches.append({"step": step, "role": role, **metrics})
            if actual_hidden != expected["hidden"][step]:
                hidden_mismatches.append({"step": step, "role": role})
            actual_state = _capture_state(session, kv_live_nbytes=_live_kv_payload_nbytes(session))
            state_diff = _state_diff(actual_state, expected["states"][step])
            if state_diff:
                state_mismatches.append(
                    {"step": step, "role": role, "diff": state_diff[:12]}
                )
            step_record["rows"].append(
                {
                    "role": role,
                    "lane": lane,
                    "token": int(result.token_id),
                    "token_match": int(result.token_id) == expected_token,
                    "top1_match": bool(metrics["top1_match"]),
                    "kl": metrics["kl"],
                    "max_abs": metrics["max_abs"],
                    "state_match": actual_state == expected["states"][step],
                }
            )
        step_records.append(step_record)

        if step == _RETIRE_AFTER_STEP:
            # Freeze the excluded lanes here.  They are not members of any
            # later packed group, so a later packed step that still touches
            # them would be an ownership bug.
            for role in _RETIRED_ROLES:
                retired_frozen[role] = _capture_state(roles[role], kv_live_nbytes=_live_kv_payload_nbytes(roles[role]))

    # A retired lane that is excluded from every later packed group must keep
    # its post-retirement state byte-for-byte.  Newcomers reusing those physical
    # lanes must match their own c1 references (checked per step), which is what
    # proves the freed lane did not hand over the old state.
    for role in _RETIRED_ROLES:
        if _capture_state(roles[role], kv_live_nbytes=_live_kv_payload_nbytes(roles[role])) != retired_frozen[role]:
            retired_mutations.append(
                {
                    "role": role,
                    "detail": "packed steps mutated a lane excluded from the group",
                }
            )

    route_ok = all(
        record["route"] == "kv_live_spans_int8_batch"
        and record["physical_rows"] == rows
        and int(record["host_model_row_iterations"] or 0) == 0
        for record in step_records
    )
    final_state_mismatches = _state_mismatches(
        [_capture_state(roles[role], kv_live_nbytes=_live_kv_payload_nbytes(roles[role])) for role in ("orig0", "new0", "new1", "orig3")],
        [reference[role]["final_state"] for role in ("orig0", "new0", "new1", "orig3")],
    )
    return {
        "step_records": step_records,
        "token_mismatches": token_mismatches,
        "logit_mismatches": logit_mismatches,
        "hidden_mismatches": hidden_mismatches,
        "state_mismatches": state_mismatches,
        "final_state_mismatches": final_state_mismatches,
        "retired_lane_mutations": retired_mutations,
        "route_ok": route_ok,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"model does not exist: {model}")
    rows = int(args.physical_rows)
    if rows < 4:
        raise ValueError("physical-rows must be at least 4 for the transition schedule")
    lengths = tuple(int(value) for value in str(args.prompt_lengths).split(","))
    if len(lengths) != 4:
        raise ValueError("--prompt-lengths must provide four lengths for the four lanes")
    steps = len(_TRANSITION_SCHEDULE)

    info = scan_gguf(model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    prompt_rows = _load_prompt_rows(Path(args.prompts))
    by_category = {str(row["category"]): row for row in prompt_rows}
    missing = [category for category in _REQUIRED_CATEGORIES if category not in by_category]
    if missing:
        raise ValueError(f"prompt file is missing categories: {missing}")

    originals = tuple(by_category[category] for category in _REQUIRED_CATEGORIES)
    newcomers = tuple(
        dict(by_category[category], id=f"{by_category[category]['id']}_newcomer")
        for category in _NEWCOMER_CATEGORIES
    )
    original_prompts, original_manifest = _build_prompts(tokenizer, originals, lengths)
    newcomer_prompts, newcomer_manifest = _build_prompts(
        tokenizer,
        newcomers,
        _NEWCOMER_LENGTHS,
    )
    prompts = {f"orig{index}": prompt for index, prompt in enumerate(original_prompts)}
    prompts.update({f"new{index}": prompt for index, prompt in enumerate(newcomer_prompts)})

    identity = model_artifact_identity(model)
    if not identity.content_verified:
        raise ValueError(f"model identity unavailable: {identity.error}")
    key = KVCapabilityKey(
        artifact_sha256=identity.sha256,
        artifact_size_bytes=identity.size_bytes,
        backend=str(args.backend),
        target_arch=hip_target_arch_for_backend(str(args.backend)),
        weight_quant=_quant_key(info),
        kv_storage="int8_per_token_head",
        storage_layout="uniform",
        scale_dtype="fp32",
        scale_granularity="per_token_head",
    )
    resolution = Qwen35GGUFModel().resolve_kv_capability(key=key, artifact=identity)
    if not resolution.promotion_eligible:
        raise ValueError(f"artifact is not qualified for INT8 KV: {resolution.reason}")
    capability = copy.deepcopy(resolution.as_dict())
    evidence = capability.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("qualified capability has no evidence payload")
    admitted_rows = int(evidence.get("max_direct_rows", 0))
    diagnostic_override = admitted_rows < rows
    if diagnostic_override:
        if int(args.diagnostic_direct_rows) < rows:
            raise ValueError(
                f"artifact admits c{admitted_rows}; pass --diagnostic-direct-rows {rows} "
                "for a pre-promotion gate"
            )
        evidence["max_direct_rows"] = rows
        evidence["max_serial_resident_rows"] = max(
            rows,
            int(evidence.get("max_serial_resident_rows", 0)),
        )

    compiler_version = None
    if args.compiler_version_file is not None:
        path = Path(args.compiler_version_file).expanduser()
        if not path.is_file():
            raise ValueError(f"compiler version file does not exist: {path}")
        compiler_version = path.read_text(encoding="utf-8").strip()

    max_sequence_length = (
        max(
            max(len(prompt) for prompt in prompts.values()),
            max(_NEWCOMER_LENGTHS),
            max(lengths),
        )
        + steps
        + 4
    )

    with ExitStack() as stack:
        session_kwargs: dict[str, Any] = {
            "backend": str(args.backend),
            "max_sequence_length": max_sequence_length,
            "max_batch_size": rows,
            "kv_scale_dtype": DType.FP32,
            "kv_scale_granularity": "per_token_head",
            "compiler_version": compiler_version,
            "require_cached_build": bool(args.require_cached_build),
        }
        owner = stack.enter_context(
            Qwen35GGUFResidentSession(
                model,
                kv_policy=_session_policy(),
                kv_capability=copy.deepcopy(capability),
                **session_kwargs,
            )
        )
        sessions = [owner]
        for _ in range(rows - 1):
            sessions.append(
                stack.enter_context(
                    Qwen35GGUFResidentSession(
                        model,
                        runtime=owner.runtime,
                        shared_runner=owner.runner,
                        kv_policy=_session_policy(),
                        kv_capability=copy.deepcopy(capability),
                        **session_kwargs,
                    )
                )
            )
        if owner.runner is None or owner.runner.weights is None:
            raise RuntimeError("resident owner failed to load")
        if len(sessions) < rows:
            raise RuntimeError(f"expected at least {rows} sessions, got {len(sessions)}")

        roles: dict[str, Qwen35GGUFResidentSession] = {}
        # Extra sessions beyond the physical lanes carry the newcomers; they are
        # never part of the owner's packed group until the admission step.
        spare = [
            stack.enter_context(
                Qwen35GGUFResidentSession(
                    model,
                    runtime=owner.runtime,
                    shared_runner=owner.runner,
                    kv_policy=_session_policy(),
                    kv_capability=copy.deepcopy(capability),
                    **session_kwargs,
                )
            )
            for _ in range(2)
        ]
        for index, session in enumerate(sessions):
            roles[f"orig{index}"] = session
        for index, session in enumerate(spare):
            roles[f"new{index}"] = session

        layer_ids = tuple(range(len(owner.runner.weights.config.layer_types)))
        limits = [int(session.packed_decode_max_rows) for session in sessions]
        kernels = [callable(session._retained_decode_kernel) for session in sessions]
        if min(limits) < rows or not all(kernels):
            raise RuntimeError(
                f"direct batch route did not resolve: limits={limits}, kernels={kernels}"
            )

        # ---- independent c1 references, advanced on the same active steps ----
        reference: dict[str, dict[str, Any]] = {}
        with _temporary_env(
            {
                _CAPTURE_PREFILL_GDN_ENV: "1",
                _GDN_PREFILL_MODE_ENV: "exact",
            }
        ):
            active_by_step: dict[str, set[int]] = {role: set() for role in roles}
            for step, lane_map in enumerate(_TRANSITION_SCHEDULE):
                for _lane, role in lane_map:
                    active_by_step[role].add(step)
            for role, session in roles.items():
                first = session.prefill(prompts[role], return_logits=True)
                current = int(first.token_id)
                tokens = [current]
                logits = [np.asarray(first.logits, dtype=np.float32).copy()]
                states: dict[int, dict[str, Any]] = {}
                hidden: dict[int, dict[int, str]] = {}
                for step in range(steps):
                    if step not in active_by_step[role]:
                        continue
                    result = session.step(
                        current,
                        return_logits=True,
                        capture_layer_output_hidden=layer_ids,
                    )
                    current = int(result.token_id)
                    tokens.append(current)
                    logits.append(np.asarray(result.logits, dtype=np.float32).copy())
                    states[step] = _capture_state(session, kv_live_nbytes=_live_kv_payload_nbytes(session))
                    hidden[step] = {
                        int(layer): _sha256_bytes(np.asarray(array, dtype=np.float32).tobytes())
                        for layer, array in session.last_layer_output_hidden.items()
                    }
                reference[role] = {
                    "prompt_tokens": len(prompts[role]),
                    "prefill_token_id": int(first.token_id),
                    "tokens": tokens,
                    "logits": logits,
                    "states": states,
                    "hidden": hidden,
                    "final_state": _capture_state(session, kv_live_nbytes=_live_kv_payload_nbytes(session)),
                    "active_steps": sorted(active_by_step[role]),
                }

            # ---- packed replay of the same schedule ----
            replay = _packed_replay(
                owner,
                roles,
                prompts,
                rows=rows,
                steps=steps,
                layer_ids=layer_ids,
                reference=reference,
                max_kl=float(args.max_kl),
            )

        step_records = replay["step_records"]
        token_mismatches = replay["token_mismatches"]
        logit_mismatches = replay["logit_mismatches"]
        hidden_mismatches = replay["hidden_mismatches"]
        state_mismatches = replay["state_mismatches"]
        final_state_mismatches = replay["final_state_mismatches"]
        retired_mutations = replay["retired_lane_mutations"]
        route_ok = replay["route_ok"]
        passed = bool(
            route_ok
            and not token_mismatches
            and not logit_mismatches
            and not hidden_mismatches
            and not state_mismatches
            and not final_state_mismatches
            and not retired_mutations
        )
        command = shlex.join(
            [
                *(f"{key}={os.environ[key]}" for key in ("HIP_VISIBLE_DEVICES",) if key in os.environ),
                sys.executable,
                *sys.argv,
            ]
        )
        visible_devices = os.environ.get("HIP_VISIBLE_DEVICES")
        return {
            "schema": 1,
            "kind": "qwen38_int8_packed_transition_correctness",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "passed" if passed else "failed",
            "performance_claim": False,
            "model": identity.as_dict(),
            "backend": {
                "backend": str(args.backend),
                "target_arch": hip_target_arch_for_backend(str(args.backend)),
                "hip_visible_devices": visible_devices,
            },
            "capability": {
                "capability_id": resolution.capability_id,
                "admitted_max_direct_rows": admitted_rows,
                "executed_direct_rows": rows,
                "diagnostic_width_override": diagnostic_override,
                "decode_batch_variant": evidence.get("decode_batch_variant"),
            },
            "workload": {
                "physical_rows": rows,
                "decode_steps": steps,
                "schedule": [
                    {"step": index, "lanes": [lane for lane, _role in lane_map],
                     "roles": [role for _lane, role in lane_map]}
                    for index, lane_map in enumerate(_TRANSITION_SCHEDULE)
                ],
                "retire_after_step": _RETIRE_AFTER_STEP,
                "retired_roles": list(_RETIRED_ROLES),
                "original_prompts": original_manifest,
                "newcomer_prompts": newcomer_manifest,
            },
            "correctness": {
                "passed": passed,
                "token_mismatches": token_mismatches,
                "logit_mismatches": logit_mismatches,
                "hidden_mismatches": hidden_mismatches,
                "state_mismatches": state_mismatches,
                "final_state_mismatches": final_state_mismatches,
                "retired_lane_mutations": retired_mutations,
            },
            "execution": {
                "route_ok": route_ok,
                "step_records": step_records,
            },
            "command": command,
            "elapsed_seconds": time.perf_counter() - started,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument(
        "--prompt-lengths",
        default=",".join(str(value) for value in DEFAULT_LENGTHS),
        help="four comma-separated prompt lengths for the four physical lanes",
    )
    parser.add_argument("--physical-rows", type=int, default=4)
    parser.add_argument("--diagnostic-direct-rows", type=int, default=0)
    parser.add_argument("--max-kl", type=float, default=0.05)
    parser.add_argument(
        "--compiler-version-file",
        type=Path,
        default=Path("/tmp/hipengine-hipcc-version.txt"),
    )
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
