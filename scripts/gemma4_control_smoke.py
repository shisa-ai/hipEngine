#!/usr/bin/env python3
"""Produce the Gemma 4 execution-profile control packet and evaluate it.

Gemma 4 analogue of ``scripts/execution_profile_gguf_control_smoke.py`` --
the Qwen3.6 template ``docs/TESTING.md`` names for building the
``scripts/execution_profile_gate.py`` packet, which the Gemma4 campaign
recorded as missing for this model. It reuses the template's model-agnostic
capture/fixture assembly and swaps only the model couplings:

* **Session**: ``Gemma4ResidentSession`` over the resident ``Gemma4Runner``
  (the ``prefill``/``step`` contract whose arithmetic transparency the live
  test ``tests/test_live_gemma4_resident_session.py`` pins byte-for-byte).
* **Route env**: ``HIPENGINE_GEMMA4_MOE_PREFILL`` is ``grouped`` for strict
  and ``auto`` for production -- the pin ``gemma4_experts._prefill_mode()``
  reads per prefill call, and the arms the registered plans declare. The
  Qwen router/rowtile envs do not apply to this model.
* **Plans**: the registered ``(gemma4_gguf, hip_gfx1100, gguf_q4_k_m)``
  strict and production plans from ``hipengine.generation.gemma4_profiles``;
  both variant manifests are written from them before any trajectory runs.
* **Prompts**: suite JSONL in the shared format. ``messages[]`` rows render
  through the artifact's own chat template (``scripts.gemma4_real_probe``);
  plain ``prompt`` rows encode raw. The format is recorded per run in
  ``smoke-env.json``. The binding packet runs the frozen campaign corpus
  (``benchmarks/prompts/gemma4-campaign-corpus.jsonl``, pinned to
  ``gemma4_campaign_bench.CORPUS`` by test): the registered prompt identity,
  not a suite chosen for how a verdict came out.
* **Rows**: both arms prefill the prompt's cycled chain prefix and
  teacher-force its tail, so every row is a paired position on that chain --
  the row definition ``scripts/gemma4_teacher_forced_gate.py`` freezes
  ("the frozen prompt ids are forced into every arm, so rows are paired
  positions rather than each arm's own sampled trajectory"), not each arm's
  own sampled trajectory. The split, and the tail both arms receive, come
  from ``_corpus_forced_split``.
* **No GDN mode / bulk attention mode**: Gemma 4 has neither, and the session
  accepts those kwargs only to honour the shared call-site contract. No
  ``--gdn-mode`` flag exists here rather than one that does nothing.
* **Prompt width**: the strict/production split only exists at width. The
  MMQ/WMMA prefill plans gate on compact lanes (``lanes >=
  _WMMA_PREFILL_MIN_LANES_PER_EXPERT * expert_count``) and the lanes are
  ``tokens * expert_used_count``, so on the shipped artifact the route first
  engages at 256 tokens (2048 lanes / 8); the grouped folds need 512 lanes.
  Below those both arms take the same fallback and the packet certifies
  identity, not the promoted route. ``--prompt-tokens`` therefore defaults to
  the campaign gate's registered chain width (2048, matching the
  ``--prompt 2048`` teacher-forced recipe) or the derived route minimum when
  that is larger, padding each suite row by cycling its own user text (the
  campaign corpus-cycling practice); the target, the derived minimum and the
  source are recorded in ``smoke-env.json``. Pass ``0`` to keep prompts as-is.

Task-artifact mode (``--task-artifact`` in the template) is not carried over:
it reports the Qwen lane's retained-evidence note and artifact kind. This
script's main mode already writes the ``task-results.json`` greedy
strict-vs-production equality the gate consumes.

Run a small packet::

    .venv/bin/python scripts/gemma4_control_smoke.py \\
        --model <artifact>.gguf \\
        --prompts benchmarks/prompts/gemma4-campaign-corpus.jsonl \\
        --limit 1 --decode-steps 3 --output-dir /tmp/gemma4-c1

Omit ``--skip-gate`` to invoke ``execution_profile_gate.py`` on the packet it
just wrote (mirroring the template's gate wiring, including the isolation
fixture as the comparison control).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.benchmark.control_capture import schedule_c1_control_records
from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation import register_builtin_generators
from hipengine.generation.gemma4_profiles import (
    GEMMA4_GGUF_BACKEND,
    GEMMA4_GGUF_MODEL,
    GEMMA4_GGUF_QUANT,
    MOE_PREFILL_ENV,
)
from hipengine.kernels.backends import hip_target_arch_for_backend
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts import (
    _WMMA_PREFILL_MIN_LANES_PER_EXPERT,
)
from hipengine.runtime.gemma4_session import Gemma4ResidentSession
from hipengine.tokenization.gguf import Gemma4GGUFTokenizer
from scripts.execution_profile_gguf_control_smoke import (
    ISOLATION_SCENARIO_SUFFIX,
    SMOKE_TASK_NAME,
    _assemble_capture,
    _selected_ids,
    _sha256_file,  # noqa: F401  (re-exported for parity with the template)
    _token_seq_hash,
    _write_fixture,
)
from scripts.gemma4_campaign_bench import _resolve_generator
from scripts.gemma4_real_probe import render

DEFAULT_SCENARIO_ID = "gemma4_gguf_c1_smoke"
DEFAULT_RUN_ID = "gemma4-c1-smoke"
# The campaign gate's registered teacher-forced chain width: the frozen
# baselines are ``--prompt 2048 --prefill 1024``. The packet defaults here so
# its chain length is protocol-fixed, never chosen after seeing a verdict.
CAMPAIGN_GATE_CHAIN_TOKENS = 2048
# The registered prefill of that recipe: rows are the chain positions right
# after it, the band the campaign baselines already score.
CAMPAIGN_GATE_PREFILL_TOKENS = 1024

# The arms the registered plans declare: strict pins the grouped exact route,
# production the auto route the default path already resolves to.
_STRICT_ROUTE_ENV = {MOE_PREFILL_ENV: "grouped"}
_PRODUCTION_ROUTE_ENV = {MOE_PREFILL_ENV: "auto"}


@contextmanager
def _route_env(values: Mapping[str, str | None]) -> Iterator[None]:
    """Apply the Gemma 4 route pin for one trajectory and restore the caller.

    Both plans' binders write ``MOE_PREFILL_ENV`` at resolve time, but this
    script resolves both plans before running either arm, so each trajectory
    must pin its own arm's route explicitly -- exactly what makes the capture
    manifest true for the run that produced it.
    """

    keys = (MOE_PREFILL_ENV,)
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            value = values.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _prompt_rows(prompts_paths: list[Path], *, limit: int) -> list[dict]:
    """Load suite rows keeping their original prompt shape.

    Like the template's loader this dedups ids across suites and enforces
    exactly one of ``prompt``/``messages``, but it retains ``messages`` so the
    caller can render them through the model's own chat template instead of
    flattening to raw user text.
    """

    rows: list[dict] = []
    seen: set[str] = set()
    for prompts_path in prompts_paths:
        with prompts_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                prompt_id = str(raw["id"])
                if prompt_id in seen:
                    raise SystemExit(f"duplicate prompt id across suites: {prompt_id}")
                seen.add(prompt_id)
                has_prompt = "prompt" in raw
                has_messages = "messages" in raw
                has_ids = "ids" in raw
                if sum((has_prompt, has_messages, has_ids)) != 1:
                    raise SystemExit(
                        f"{prompt_id}: expected exactly one of prompt, "
                        "messages[], or ids"
                    )
                messages = raw.get("messages")
                if has_messages and (not isinstance(messages, list) or not messages):
                    raise SystemExit(f"{prompt_id}: expected prompt or messages[]")
                if has_ids:
                    ids = raw.get("ids")
                    if (
                        not isinstance(ids, list)
                        or not ids
                        or not all(isinstance(token, int) for token in ids)
                    ):
                        raise SystemExit(
                            f"{prompt_id}: ids must be a non-empty integer list"
                        )
                row = {
                    "id": prompt_id,
                    "category": str(raw.get("category", "general_en")),
                    "messages": messages if has_messages else None,
                    "prompt": str(raw["prompt"]) if has_prompt else None,
                    "ids": [int(token) for token in raw["ids"]] if has_ids else None,
                    "source_prompt": str(raw.get("source_prompt", "")),
                }
                if not row["ids"] and not row["messages"] and not (row["prompt"] or "").strip():
                    raise SystemExit(f"{prompt_id}: prompt text is empty")
                rows.append(row)
                if len(rows) >= limit:
                    return rows
    if not rows:
        raise SystemExit("no prompts loaded")
    return rows


def _row_source(row: Mapping) -> str:
    """The prompt text a row contributes, flattened for width cycling."""

    messages = row.get("messages")
    if messages:
        return "\n\n".join(
            str(message["content"])
            for message in messages
            if message.get("role") == "user" and message.get("content")
        )
    return str(row["prompt"])


def _render_text(tokenizer, row: Mapping, *, repeats: int) -> str:
    messages = row.get("messages")
    if messages:
        if repeats == 1:
            # Faithful render of the row's own turn structure.
            return render(tokenizer.chat_template, list(messages))
        source = _row_source(row)
        return render(
            tokenizer.chat_template,
            [{"role": "user", "content": source * repeats}],
        )
    return _row_source(row) * repeats


def _prompt_token_ids(tokenizer, row: Mapping, *, min_tokens: int = 0) -> list[int]:
    """Chat-template rows render through the artifact's own template; the rest
    encode raw. When ``min_tokens`` exceeds the natural length, the row's own
    source text is cycled (re-rendered through the template for messages) until
    the prefill crosses the MoE route width. Rows that carry pre-tokenized
    ``ids`` return them unchanged: the registered chain is exact ids, and
    text cycling cannot reproduce its sentence-boundary tokenization."""

    built_ids = row.get("ids")
    if built_ids is not None:
        ids = [int(token) for token in built_ids]
        if not ids:
            raise SystemExit("pre-tokenized prompt ids are empty")
        if min_tokens > len(ids):
            raise SystemExit(
                f"pre-tokenized prompt chain has {len(ids)} ids, below the "
                f"{min_tokens}-token width target"
            )
        return ids

    text = _render_text(tokenizer, row, repeats=1)
    ids = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
    if min_tokens <= len(ids):
        return ids
    source = _row_source(row)
    source_ids = tokenizer.encode(source, add_special_tokens=False)
    if not source_ids:
        raise SystemExit("prompt source produced no tokens; cannot pad to --prompt-tokens")
    repeats = max(2, -(-int(min_tokens) // len(source_ids)))
    for _ in range(64):
        text = _render_text(tokenizer, row, repeats=repeats)
        ids = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
        if len(ids) >= int(min_tokens):
            return ids
        repeats = max(repeats + 1, -(-int(min_tokens) // max(1, len(ids))) + 1)
    raise SystemExit(
        f"could not pad prompt to {min_tokens} tokens after 64 re-encodes"
    )


def _moe_prefill_width(metadata: Mapping) -> int:
    """Token width at which the MMQ/WMMA prefill plans first engage, or 0.

    The kernel gates on compact lanes (``lanes >= _WMMA_PREFILL_MIN_LANES_PER_EXPERT
    * expert_count``) and ``lanes = tokens * expert_used_count`` (gemma4_experts
    line 226), so the token width is ``ceil(16 * expert_count /
    expert_used_count)`` -- 256 on the shipped 128/8 artifact. Returns 0 when
    the artifact declares no MoE routing to size against, which leaves the
    caller on the campaign chain width.
    """

    try:
        experts = int(metadata.get("gemma4.expert_count") or 0)
        used = int(metadata.get("gemma4.expert_used_count") or 0)
    except (TypeError, ValueError):
        return 0
    if experts <= 0 or used <= 0:
        return 0
    lanes = int(_WMMA_PREFILL_MIN_LANES_PER_EXPERT) * experts
    return -(-lanes // used)


def _default_prompt_tokens(metadata: Mapping) -> tuple[int, str]:
    """Default padding target: the campaign gate's registered chain width.

    2048 matches the frozen ``--prompt 2048 --prefill 1024`` teacher-forced
    baselines, so the packet's chain length never depends on a verdict it has
    already seen; the derived route minimum only raises it for an artifact
    whose MoE gate sits above that width.
    """

    route_min = _moe_prefill_width(metadata)
    target = max(CAMPAIGN_GATE_CHAIN_TOKENS, route_min)
    source = "campaign-gate-chain"
    if route_min > CAMPAIGN_GATE_CHAIN_TOKENS:
        source = "route-min"
    return target, f"{source} (route min {route_min})"


def _corpus_forced_split(
    tokens: Sequence[int],
    prefill_len: int,
    decode_steps: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split the cycled prompt chain into the registered prefill and its band.

    Both arms prefill ``tokens[:prefill_len]`` and teacher-force the next
    ``decode_steps`` chain ids, so every scored row is a paired position on
    the frozen chain -- the campaign evaluator's row definition
    (``scripts/gemma4_teacher_forced_gate.py`` forces the frozen prompt ids
    into every arm) in the row band its registered recipe scores
    (``--prompt 2048 --prefill 1024``). Returns ``(prefill_ids, forced,
    teacher_row_ids)`` where ``teacher_row_ids`` labels row 0 (prefill-last)
    with the next chain token and decode row ``k`` with the token fed at that
    step.
    """

    ids = [int(token) for token in tokens]
    prefill = int(prefill_len)
    steps = int(decode_steps)
    if not ids:
        raise ValueError("prompt chain must be non-empty")
    if prefill <= 0 or steps <= 0:
        raise ValueError(
            f"prefill_len and decode_steps must be positive, got {prefill} "
            f"and {steps}"
        )
    if prefill + steps > len(ids):
        raise ValueError(
            f"prefill_len + decode_steps must land inside the prompt chain, "
            f"got {prefill} + {steps} for {len(ids)} tokens"
        )
    prefill_ids = ids[:prefill]
    forced = ids[prefill : prefill + steps]
    teacher_row_ids = [forced[0], *forced]
    return prefill_ids, forced, teacher_row_ids


def _trajectory_with_controls(
    session,
    *,
    prompt_ids: Sequence[int],
    forced_input_ids: Sequence[int] | None,
    teacher_row_ids: Sequence[int] | None,
    decode_steps: int,
    scenario_id: str,
    request_id: str,
    route_top_k: int,
    graph_bucket: str,
    rng_seed: int,
    route_env: Mapping[str, str | None],
    step_offset: int = 0,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Run prefill + decode under the arm's route pin and return live primitives.

    Same schedule shape as the template, but the binding packet runs *both*
    arms forced on the identical chain tail (``_corpus_forced_split``): rows
    are paired positions on the frozen chain, the campaign evaluator's row
    definition, rather than each arm's own sampled trajectory. Control and
    row-spec assembly is the shared contract with the template; only the
    route pin and the absence of a GDN context differ.

    ``teacher_row_ids`` is the strict chain for every row of a forced run
    (prefill emission first, then one per decode step). Row descriptors must
    carry the *teacher's* token, not the candidate's own emission: the gate
    requires ``strict.rows == candidate.rows`` so both captures describe the
    same aligned schedule. The template could use its own emissions there
    because its arms never diverged; the Gemma 4 MMQ-vs-grouped split does
    diverge, so a divergent candidate must still record the teacher's chain.
    """

    if forced_input_ids is None:
        if teacher_row_ids is not None:
            raise ValueError("teacher_row_ids is only valid with forced_input_ids")
    else:
        if teacher_row_ids is None:
            raise ValueError("forced runs require the teacher chain for row alignment")
        if len(teacher_row_ids) != len(forced_input_ids) + 1:
            raise ValueError(
                "teacher_row_ids must hold the prefill emission plus one token "
                f"per forced step, got {len(teacher_row_ids)} for "
                f"{len(forced_input_ids)} steps"
            )

    with _route_env(route_env):
        session.reset()
        result = session.prefill([int(token) for token in prompt_ids], return_logits=True)
        prompt_len = len(prompt_ids)
        logits_rows = [np.ascontiguousarray(result.logits, dtype=np.float32)]
        prefill_teacher = (
            int(result.token_id)
            if forced_input_ids is None
            else int(teacher_row_ids[0])
        )
        controls = [
            {
                "scenario_id": scenario_id,
                "scenario_step": step_offset + 0,
                "request_id": request_id,
                "input_token_id": int(prompt_ids[-1]),
                "position": prompt_len - 1,
                "context_length": prompt_len,
                "route_top_k": route_top_k,
                "graph_bucket": graph_bucket,
                "rng_seed": rng_seed,
            }
        ]
        row_specs = [
            {
                "scenario_step": step_offset + 0,
                "request_id": request_id,
                "teacher_step": 0,
                "category": "smoke",
                "shape": "prefill_last",
                "transition": "prefill_to_c1",
                "teacher_token_id": prefill_teacher,
            }
        ]
        if forced_input_ids is None:
            previous = int(result.token_id)
            for step in range(1, int(decode_steps) + 1):
                position = session.position
                probe = session.step(previous, return_logits=True)
                logits_rows.append(np.ascontiguousarray(probe.logits, dtype=np.float32))
                controls.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_step": step_offset + step,
                        "request_id": request_id,
                        "input_token_id": previous,
                        "position": int(position),
                        "context_length": int(position) + 1,
                        "route_top_k": route_top_k,
                        "graph_bucket": graph_bucket,
                        "rng_seed": rng_seed,
                    }
                )
                row_specs.append(
                    {
                        "scenario_step": step_offset + step,
                        "request_id": request_id,
                        "teacher_step": step,
                        "category": "smoke",
                        "shape": "c1",
                        "transition": "steady",
                        "teacher_token_id": int(probe.token_id),
                    }
                )
                previous = int(probe.token_id)
        else:
            for step, input_token_id in enumerate(forced_input_ids, start=1):
                position = session.position
                probe = session.step(int(input_token_id), return_logits=True)
                logits_rows.append(np.ascontiguousarray(probe.logits, dtype=np.float32))
                controls.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_step": step_offset + step,
                        "request_id": request_id,
                        "input_token_id": int(input_token_id),
                        "position": int(position),
                        "context_length": int(position) + 1,
                        "route_top_k": route_top_k,
                        "graph_bucket": graph_bucket,
                        "rng_seed": rng_seed,
                    }
                )
                row_specs.append(
                    {
                        "scenario_step": step_offset + step,
                        "request_id": request_id,
                        "teacher_step": step,
                        "category": "smoke",
                        "shape": "c1",
                        "transition": "steady",
                        # The teacher's token at this step, not this arm's: rows
                        # must match the strict capture's descriptors exactly.
                        "teacher_token_id": int(teacher_row_ids[step]),
                    }
                )
        logits = np.stack(logits_rows, axis=0)
        logits = np.reshape(logits, (int(logits.shape[0]), -1))
        return logits, controls, row_specs


def _gate_command(
    *,
    output_dir: Path,
    strict_capture: Path,
    production_capture: Path,
    production_fixture: Path,
    strict_fixture: Path,
    isolation_fixture: Path,
    repeat_capture: Path,
    isolation_capture: Path,
    task_path: Path,
    arithmetic_class: str,
    verdict_path: Path,
    bf16_logits: Path | None = None,
) -> list[str]:
    """Build this packet's ``execution_profile_gate.py`` invocation.

    ``bf16_logits`` attaches an aligned BF16 teacher cache via
    ``--bf16-logits``; without it the gate reports ``bf16_noninferiority``
    as ``unavailable`` and does not bind on it, which is the documented
    state until a teacher fixture exists for the packet (docs/TESTING.md).
    """

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "execution_profile_gate.py"),
        "--variant-manifest", str(output_dir / "production-variant-manifest.json"),
        "--strict-manifest", str(output_dir / "strict-variant-manifest.json"),
        "--strict-capture", str(strict_capture),
        "--candidate-capture", str(production_capture),
        "--expected-controls", str(production_fixture),
        "--strict-expected-controls", str(strict_fixture),
        "--comparison-controls", str(isolation_fixture),
        "--repeat-capture", str(repeat_capture),
        "--isolation-capture", str(isolation_capture),
        "--task-results", str(task_path),
        "--arithmetic-class", str(arithmetic_class),
        "--output", str(verdict_path),
    ]
    if bf16_logits is not None:
        cmd += ["--bf16-logits", str(bf16_logits)]
    return cmd


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default=GEMMA4_GGUF_BACKEND)
    parser.add_argument("--prompts", action="append", type=Path, required=True,
                        help="one or more prompt-suite files (merged, deduped by id)")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=None,
                        help="pad each prompt to this many tokens by cycling its "
                             "own text (default: the campaign gate's 2048-token "
                             "registered chain, raised to the derived MoE route "
                             "minimum when that is larger; 0 keeps prompts "
                             "as-is and certifies identity only)")
    parser.add_argument("--scenario-id", default=DEFAULT_SCENARIO_ID)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path,
                        help="exported as HIPENGINE_COMPILER_VERSION_FILE before "
                             "any kernel build so the JIT cache key is pinned")
    parser.add_argument("--arithmetic-class", default="T2",
                        choices=("T0", "T1", "T2", "T3"))
    parser.add_argument("--top-k", type=int, default=8,
                        help="routed experts used per token "
                             "(gemma4.expert_used_count on the shipped artifact)")
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--bf16-logits", type=Path, default=None,
                        help="aligned BF16 teacher cache (.npy, packet row "
                             "order) produced by scripts/gemma4_bf16_teacher.py; "
                             "omitted, the gate reports bf16_noninferiority "
                             "as unavailable")
    return parser


def main() -> int:
    args = _parser().parse_args()

    from hipengine.loading.gguf import scan_gguf

    if args.compiler_version_file is not None:
        if not args.compiler_version_file.is_file():
            raise SystemExit(
                f"--compiler-version-file not found: {args.compiler_version_file}"
            )
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    register_builtin_generators()
    if args.bf16_logits is not None and not Path(args.bf16_logits).is_file():
        raise SystemExit(f"--bf16-logits not found: {args.bf16_logits}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _prompt_rows(list(args.prompts), limit=int(args.limit))
    gguf_info = scan_gguf(args.model)
    tokenizer = Gemma4GGUFTokenizer.from_gguf_info(gguf_info)
    if args.prompt_tokens is None:
        prompt_tokens_target, width_source = _default_prompt_tokens(gguf_info.metadata)
    else:
        prompt_tokens_target = max(0, int(args.prompt_tokens))
        width_source = "explicit" if prompt_tokens_target else "disabled"
    prompt_tokens = {
        str(row["id"]): _prompt_token_ids(
            tokenizer, row, min_tokens=prompt_tokens_target
        )
        for row in rows
    }
    print(
        f"prompt width target={prompt_tokens_target} ({width_source}); "
        f"padded lengths="
        + ",".join(f"{pid}:{len(tokens)}" for pid, tokens in prompt_tokens.items()),
        flush=True,
    )
    if prompt_tokens_target == 0:
        print(
            "WARNING: --prompt-tokens 0 keeps prompts short, so both arms take "
            "the same prefill fallback and the packet certifies identity rather "
            "than the strict/production split",
            flush=True,
        )
    route_min = _moe_prefill_width(gguf_info.metadata)
    if 0 < prompt_tokens_target < route_min:
        print(
            f"WARNING: --prompt-tokens {prompt_tokens_target} is below the MoE "
            f"route minimum {route_min}; the MMQ/WMMA plans will not engage and "
            "the packet certifies identity only",
            flush=True,
        )
    prompt_formats = {
        "chat_template": sum(1 for row in rows if row.get("messages")),
        "raw": sum(1 for row in rows if not row.get("messages")),
    }
    max_sequence_length = (
        max(len(tokens) for tokens in prompt_tokens.values())
        + int(args.decode_steps)
        + 4
    )

    strict_plan = resolve_runtime_profile(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
    )
    production_plan = resolve_runtime_profile(
        model=GEMMA4_GGUF_MODEL,
        backend=GEMMA4_GGUF_BACKEND,
        quant=GEMMA4_GGUF_QUANT,
        profile=ExecutionProfile.PRODUCTION,
    )
    from hipengine.benchmark.execution_profiles import validate_variant_manifest

    strict_manifest = validate_variant_manifest(strict_plan.manifest)
    production_manifest = validate_variant_manifest(production_plan.manifest)
    for path, manifest in (
        (output_dir / "strict-variant-manifest.json", strict_manifest),
        (output_dir / "production-variant-manifest.json", production_manifest),
    ):
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    strict_segments: list[tuple[np.ndarray, list[dict], list[dict]]] = []
    production_segments: list[tuple[np.ndarray, list[dict], list[dict]]] = []
    repeat_segments: list[tuple[np.ndarray, list[dict], list[dict]]] = []
    isolation_segments: list[tuple[np.ndarray, list[dict], list[dict]]] = []
    fixture_records_by_prompt: dict[str, tuple[object, ...]] = {}
    teacher_by_prompt: dict[str, list[int]] = {}

    _llm, runner, resolution = _resolve_generator(args.model, max_sequence_length)
    target_arch = hip_target_arch_for_backend(str(args.backend))
    with Gemma4ResidentSession(
        runner, backend=str(args.backend), target_arch=target_arch
    ) as session:
        print(
            f"resolved backend={args.backend} arch={target_arch} "
            f"strict_plan={strict_plan.profile.value} "
            f"production_plan={production_plan.profile.value} "
            f"fell_back_to_strict={production_plan.fell_back_to_strict} "
            f"resolution={resolution.get('resolution')}",
            flush=True,
        )

        for prompt_index, prompt_row in enumerate(rows):
            prompt_id = prompt_row["id"]
            request_id = f"prompt-{prompt_id}"
            tokens = prompt_tokens[prompt_id]
            # Sequential prompts share one scenario, so each occupies its own
            # step band; otherwise every prompt's step 0 collides on slot 0.
            step_offset = prompt_index * (int(args.decode_steps) + 1)

            prefill_ids, forced, teacher_row_ids = _corpus_forced_split(
                tokens, CAMPAIGN_GATE_PREFILL_TOKENS, int(args.decode_steps)
            )
            strict_logits, strict_controls, strict_row_specs = _trajectory_with_controls(
                session,
                prompt_ids=prefill_ids,
                forced_input_ids=forced,
                teacher_row_ids=teacher_row_ids,
                decode_steps=int(args.decode_steps),
                scenario_id=args.scenario_id,
                request_id=request_id,
                route_top_k=int(args.top_k),
                graph_bucket="c1",
                rng_seed=int(args.rng_seed),
                route_env=_STRICT_ROUTE_ENV,
                step_offset=step_offset,
            )
            # Paired rows: both arms receive this identical chain tail.
            teacher = forced
            teacher_by_prompt[prompt_id] = teacher
            strict_segments.append((strict_logits, strict_controls, strict_row_specs))

            def _production(
                *,
                scenario_id: str,
                run_forced: Sequence[int],
            ) -> tuple[np.ndarray, list[dict], list[dict]]:
                return _trajectory_with_controls(
                    session,
                    prompt_ids=prefill_ids,
                    forced_input_ids=run_forced,
                    teacher_row_ids=teacher_row_ids,
                    decode_steps=int(args.decode_steps),
                    scenario_id=scenario_id,
                    request_id=request_id,
                    route_top_k=int(args.top_k),
                    graph_bucket="c1",
                    rng_seed=int(args.rng_seed),
                    route_env=_PRODUCTION_ROUTE_ENV,
                    step_offset=step_offset,
                )

            production_segments.append(
                _production(scenario_id=args.scenario_id, run_forced=teacher)
            )
            repeat_segments.append(
                _production(scenario_id=args.scenario_id, run_forced=teacher)
            )
            isolation_segments.append(
                _production(
                    scenario_id=args.scenario_id + ISOLATION_SCENARIO_SUFFIX,
                    run_forced=teacher,
                )
            )
            fixture_records_by_prompt[prompt_id] = schedule_c1_control_records(
                scenario_id=args.scenario_id,
                request_id=request_id,
                prompt_ids=prefill_ids,
                teacher_token_ids=teacher,
                route_top_k=int(args.top_k),
                graph_bucket="c1",
                rng_seed=int(args.rng_seed),
                step_offset=step_offset,
            )
            print(
                f"{prompt_id}: strict + production + repeat + isolation runs, "
                f"{len(teacher)} forced steps",
                flush=True,
            )

    strict_capture = _assemble_capture(
        output_dir=output_dir,
        run_id=f"{args.run_id}-strict",
        execution_profile=ExecutionProfile.STRICT.value,
        scenario_id=args.scenario_id,
        variant_manifest=strict_manifest,
        segments=strict_segments,
        repeat_index=0,
    )
    production_capture = _assemble_capture(
        output_dir=output_dir,
        run_id=f"{args.run_id}-production",
        execution_profile=ExecutionProfile.PRODUCTION.value,
        scenario_id=args.scenario_id,
        variant_manifest=production_manifest,
        segments=production_segments,
        repeat_index=0,
    )
    repeat_capture = _assemble_capture(
        output_dir=output_dir,
        run_id=f"{args.run_id}-production-repeat",
        execution_profile=ExecutionProfile.PRODUCTION.value,
        scenario_id=args.scenario_id,
        variant_manifest=production_manifest,
        segments=repeat_segments,
        repeat_index=1,
    )
    isolation_capture = _assemble_capture(
        output_dir=output_dir,
        run_id=f"{args.run_id}-isolation",
        execution_profile=ExecutionProfile.PRODUCTION.value,
        scenario_id=args.scenario_id + ISOLATION_SCENARIO_SUFFIX,
        variant_manifest=production_manifest,
        segments=isolation_segments,
        repeat_index=1,
    )
    strict_expected_records: tuple[object, ...] = tuple(
        record
        for prompt_id in [row["id"] for row in rows]
        for record in fixture_records_by_prompt[prompt_id]
    )
    strict_fixture = _write_fixture(
        output_dir=output_dir,
        name=f"{args.run_id}-strict-expected-controls",
        scenario_id=args.scenario_id,
        run_id=f"{args.run_id}-strict",
        records=strict_expected_records,
    )
    production_fixture = _write_fixture(
        output_dir=output_dir,
        name=f"{args.run_id}-production-expected-controls",
        scenario_id=args.scenario_id,
        run_id=f"{args.run_id}-production",
        records=strict_expected_records,
    )
    isolation_expected_records: tuple[object, ...] = tuple(
        record
        for prompt_index, prompt_id in enumerate(row["id"] for row in rows)
        for record in schedule_c1_control_records(
            scenario_id=args.scenario_id + ISOLATION_SCENARIO_SUFFIX,
            request_id=f"prompt-{prompt_id}",
            prompt_ids=_corpus_forced_split(
                prompt_tokens[prompt_id],
                CAMPAIGN_GATE_PREFILL_TOKENS,
                int(args.decode_steps),
            )[0],
            teacher_token_ids=teacher_by_prompt[prompt_id],
            route_top_k=int(args.top_k),
            graph_bucket="c1",
            rng_seed=int(args.rng_seed),
            step_offset=prompt_index * (int(args.decode_steps) + 1),
        )
    )
    isolation_fixture = _write_fixture(
        output_dir=output_dir,
        name=f"{args.run_id}-isolation-expected-controls",
        scenario_id=args.scenario_id + ISOLATION_SCENARIO_SUFFIX,
        run_id=f"{args.run_id}-isolation",
        records=isolation_expected_records,
    )
    greedy_aligned = bool(
        _selected_ids(strict_capture) == _selected_ids(production_capture)
    )

    task_results = {SMOKE_TASK_NAME: greedy_aligned}
    task_path = output_dir / "task-results.json"
    task_path.write_text(json.dumps(task_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env_note = {
        "scenario_id": args.scenario_id,
        "run_id": args.run_id,
        "backend": str(args.backend),
        "target_arch": target_arch,
        "limit": args.limit,
        "decode_steps": args.decode_steps,
        "route_env": {
            "strict": dict(_STRICT_ROUTE_ENV),
            "production": dict(_PRODUCTION_ROUTE_ENV),
        },
        "prompt_formats": prompt_formats,
        "prompt_tokens_target": int(prompt_tokens_target),
        "prompt_tokens_route_min": int(route_min),
        "prompt_tokens_width_source": width_source,
        "prompt_tokens_padded": {
            prompt_id: len(tokens) for prompt_id, tokens in prompt_tokens.items()
        },
        # The full padded ids, so a BF16 teacher (scripts/gemma4_bf16_teacher.py
        # prepare) replays exactly the rows this run scored instead of
        # re-deriving the padding and risking drift from it.
        "prompt_tokens_ids": {
            prompt_id: tokens for prompt_id, tokens in prompt_tokens.items()
        },
        "route_top_k": int(args.top_k),
        "arithmetic_class": args.arithmetic_class,
        "task_results": task_results,
        "smoke_only": True,
    }
    (output_dir / "smoke-env.json").write_text(
        json.dumps(env_note, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote captures/fixtures/manifests/task-results to {output_dir}; "
        f"greedy_aligned={greedy_aligned}",
        flush=True,
    )
    if args.skip_gate:
        return 0 if greedy_aligned else 1

    verdict_path = output_dir / "gate-verdict.json"
    gate_cmd = _gate_command(
        output_dir=output_dir,
        strict_capture=strict_capture,
        production_capture=production_capture,
        production_fixture=production_fixture,
        strict_fixture=strict_fixture,
        isolation_fixture=isolation_fixture,
        repeat_capture=repeat_capture,
        isolation_capture=isolation_capture,
        task_path=task_path,
        arithmetic_class=args.arithmetic_class,
        verdict_path=verdict_path,
        bf16_logits=(
            Path(args.bf16_logits).resolve() if args.bf16_logits is not None else None
        ),
    )
    print("invoking gate:", " ".join(gate_cmd), flush=True)
    gate_result = subprocess.run(gate_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if gate_result.stdout.strip():
        print("gate stdout:", gate_result.stdout.strip(), flush=True)
    if gate_result.stderr.strip():
        print("gate stderr:", gate_result.stderr.strip()[-800:], flush=True)
    if not verdict_path.is_file():
        print("gate produced no verdict file; rc=", gate_result.returncode, flush=True)
        return 1
    result = json.loads(verdict_path.read_text(encoding="utf-8"))
    print(
        f"gate verdict: execution_profile={result.get('execution_profile')} "
        f"status={result.get('decision', {}).get('status')} "
        f"automatic={result.get('decision', {}).get('eligible_for_automatic_admission')} "
        f"controls={result.get('control_semantics', {}).get('passed')} "
        f"determinism={result.get('determinism', {}).get('passed')} "
        f"isolation={result.get('isolation', {}).get('passed')} "
        f"tasks={result.get('task_quality', {}).get('passed')} "
        f"generated={result.get('generated_id_equality', {}).get('all_equal')}",
        flush=True,
    )
    return 0 if result.get("decision", {}).get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())