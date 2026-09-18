#!/usr/bin/env python3
"""Distribution gate for sampled MTP acceptance on real target logits.

Sampled speculative decoding is only usable in serving if a request that sets
``temperature > 0`` receives tokens distributed *exactly* as the autoregressive
sampler would emit them. That claim has two halves:

1. the sampler law itself — ``processed_distribution`` must be the same law
   ``select_token`` draws from, on real model rows;
2. the coupling — accepting a drafted token with ``min(1, p/q)`` and resampling
   from ``normalize(max(0, p - q))`` on rejection must reproduce ``p``.

This script proves both on rows the model actually produced. It runs a real
resident target session, collects the logits row at every decode step for
several prompts, and for every (row, sampler config) pair reports:

* the AR law check (the token ``select_token`` picks carries the weight
  ``processed_distribution`` assigns it);
* the induced first-token distribution under the coupling, computed in closed
  form by integrating the decision rule, compared against ``p`` by maximum
  absolute deviation, total variation, and both KL directions;
* the same comparison for the general coupling (a perturbed draft distribution,
  which is what a real draft model supplies);
* a Monte-Carlo cross-check that drives the implemented accept/resample walk on
  the real row with real draws and compares the emitted token histogram to ``p``.

Run on a host with ROCm and the target GGUF. The emitted JSON is the evidence
artifact; ``--output`` writes it.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hipengine.generation.sampling import (  # noqa: E402
    RowSamplingState,
    processed_distribution,
    select_token,
)
from hipengine.speculative.interfaces import TargetVerifyBatch  # noqa: E402
from hipengine.speculative.sampling import (  # noqa: E402
    SparseDistribution,
    acceptance_probability,
    residual_distribution,
    sampled_accept_from_distributions,
)

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
DEFAULT_PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, then explain its complexity.",
    "Summarize the trade-offs between paged attention and a contiguous KV cache in three sentences.",
    "Translate this sentence to Japanese and then back to English: The cache is warm but the kernel is not.",
)

# The closed-form identity is exact arithmetic over float64 arrays, so the gate
# is epsilon-tight rather than statistical. The Monte-Carlo arm is bounded by its
# own sampling error instead.
EXACT_TOLERANCE = 1e-9
KL_TOLERANCE = 1e-9
# Monte-Carlo histogram cells are judged against an exact Poisson tail at this
# probability, with a normal 5-sigma limit where the expected count is large.
MONTE_CARLO_TAIL = 1e-9
MONTE_CARLO_SIGMA = 5.0
# The Monte-Carlo arm rebuilds the residual on every draw, so it runs on the
# target's most likely tokens rather than the whole row. The exact identity above
# still covers the full support.
MONTE_CARLO_SUPPORT = 1024


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _params(**overrides: Any) -> SimpleNamespace:
    values = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logit_bias": (),
        "suppress_token_ids": (),
        "min_tokens": 0,
        "eos_token_id": None,
        "ignore_eos": False,
        "seed": 4242,
        "row_seeds": (),
        "stop_token_ids": (),
        "stop_token_sequences": (),
        "forced_tokens_pending": (),
        "forced_token_reason": None,
        "post_thinking_forced_tokens_pending": (),
        "post_thinking_forced_token_reason": None,
        "force_sequence_completion_token_sequences": (),
        "force_sequence_completion_reason": None,
        "json_object_close_forcing": False,
        "tool_call_constraint": None,
        "thinking_close_token_ids": (),
        "thinking_hard_token_cap": None,
        "thinking_soft_close_window": 0,
        "logprobs": False,
        "top_logprobs": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


SAMPLER_CONFIGS: dict[str, SimpleNamespace] = {
    "temperature_0.7": _params(temperature=0.7),
    "temperature_1.0": _params(temperature=1.0),
    "temperature_1.0_top_k_20": _params(temperature=1.0, top_k=20),
    "temperature_0.8_top_p_0.9_min_p_0.02": _params(
        temperature=0.8, top_p=0.9, min_p=0.02
    ),
    "temperature_1.0_repetition_penalty_1.15": _params(
        temperature=1.0, repetition_penalty=1.15
    ),
    "temperature_1.0_presence_penalty_0.5": _params(
        temperature=1.0, presence_penalty=0.5
    ),
}

# Draft distributions the gate couples against. A real draft model is a
# different network, so its law is close to p but not equal; these cases span
# "nearly identical" to "structurally different" so the residual path is
# exercised on every row.
DRAFT_CONFIGS: dict[str, SimpleNamespace] = {
    "draft_temperature_1.4": _params(temperature=1.4),
    "draft_temperature_0.5": _params(temperature=0.5),
    "draft_temperature_1.0_top_k_8": _params(temperature=1.0, top_k=8),
}


def _distribution_of(
    logits: np.ndarray,
    params: SimpleNamespace,
    state: RowSamplingState,
) -> SparseDistribution:
    token_ids, probabilities = processed_distribution(logits, params, state)
    return SparseDistribution.from_pairs(token_ids, probabilities)


def _id_sorted(distribution: SparseDistribution) -> tuple[np.ndarray, np.ndarray]:
    """Token ids and weights of a distribution, sorted ascending by id."""

    ids = np.asarray(distribution.token_ids, dtype=np.int64)
    weights = np.asarray(distribution.probabilities, dtype=np.float64)
    if ids.size == 0:
        return ids, weights
    order = np.argsort(ids, kind="stable")
    return ids[order], weights[order]


def _weights_at(
    ids: np.ndarray, weights: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Weights of ``ids``/``weights`` at ``query`` ids, zero outside the support."""

    out = np.zeros(query.size, dtype=np.float64)
    if ids.size == 0 or query.size == 0:
        return out
    position = np.searchsorted(ids, query)
    clipped = np.clip(position, 0, ids.size - 1)
    hit = ids[clipped] == query
    out[hit] = weights[clipped[hit]]
    return out


def _induced_first_token(
    target: SparseDistribution,
    draft: SparseDistribution,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the decision rule over the drafted token and the accept draw.

    The walk emits the drafted token with probability ``min(p, q)`` and otherwise
    draws from the residual, so the induced law is

        induced(e) = min(p(e), q(e)) + (1 - sum_t min(p(t), q(t))) * residual(e)

    which is what keeps a speculative cycle exactly ``p``-distributed. Written in
    closed form it is O(support) instead of O(support^2).
    """

    target_ids, target_weights = _id_sorted(target)
    draft_ids, draft_weights = _id_sorted(draft)
    try:
        residual = residual_distribution(target, draft)
    except ValueError:
        # p == q on the union support: every drafted token is accepted with
        # probability 1, so the rejection branch carries no mass.
        residual = None
    if residual is None:
        residual_ids = np.zeros(0, dtype=np.int64)
        residual_weights = np.zeros(0, dtype=np.float64)
    else:
        residual_ids, residual_weights = _id_sorted(residual)
    support = np.union1d(draft_ids, residual_ids)
    accepted = np.minimum(
        _weights_at(target_ids, target_weights, draft_ids), draft_weights
    )
    rejected_weight = 1.0 - float(np.sum(accepted))
    induced = np.zeros(support.size, dtype=np.float64)
    if draft_ids.size:
        induced[np.searchsorted(support, draft_ids)] = accepted
    if residual_ids.size:
        induced[np.searchsorted(support, residual_ids)] += (
            rejected_weight * residual_weights
        )
    return support, induced


def _distribution_metrics(
    induced_ids: np.ndarray,
    induced_weights: np.ndarray,
    target: SparseDistribution,
) -> dict[str, float]:
    target_ids, target_weights = _id_sorted(target)
    support = np.union1d(induced_ids, target_ids)
    p = _weights_at(target_ids, target_weights, support)
    q = _weights_at(induced_ids, induced_weights, support)
    max_abs = float(np.max(np.abs(p - q)))
    total_variation = float(0.5 * np.sum(np.abs(p - q)))
    floor = 1e-12
    p_floor = np.maximum(p, floor)
    q_floor = np.maximum(q, floor)
    forward_kl = float(np.sum(p * np.log(p_floor / q_floor)))
    reverse_kl = float(np.sum(q * np.log(q_floor / p_floor)))
    target_top = int(target.token_ids[int(np.argmax(target.probabilities))])
    induced_top = int(induced_ids[int(np.argmax(induced_weights))])
    return {
        "support": int(support.size),
        "max_abs_deviation": max_abs,
        "total_variation": total_variation,
        "kl_p_to_induced": forward_kl,
        "kl_induced_to_p": reverse_kl,
        "top1_agreement": 1.0 if target_top == induced_top else 0.0,
        "target_top1": target_top,
        "induced_top1": induced_top,
    }


def _chain_batch(token: int, *, request_id: int = 1) -> TargetVerifyBatch:
    return TargetVerifyBatch(
        request_ids=(request_id,),
        tokens=(0, int(token)),
        positions=(0, 1),
        row_to_request=(request_id, request_id),
        parent_rows=(-1, 0),
        root_rows=(0,),
        candidate_rows=(1,),
        draft_depths=(0, 1),
        active_mask=(True, True),
    )


def _poisson_count_limit(expected: float, *, tail: float) -> float:
    """Largest count a cell with this expected count may reach.

    Exact Poisson tail for small expectations (where the normal approximation
    would call a single stray draw a 15-sigma event), and a normal 5-sigma limit
    once the expected count is large enough for the approximation to hold.
    """

    if expected <= 0.0:
        return 0.0
    if expected >= 64.0:
        return expected + MONTE_CARLO_SIGMA * float(np.sqrt(expected))
    term = float(np.exp(-expected))
    cumulative = term
    count = 0
    while 1.0 - cumulative > tail and count < 100_000:
        count += 1
        term *= expected / count
        cumulative += term
    return float(count)


def _monte_carlo_check(
    target: SparseDistribution,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Drive the real accept/resample walk and compare the histogram to p.

    The walk is driven on the target's most likely ``MONTE_CARLO_SUPPORT`` tokens
    renormalized, because every draw rebuilds the residual over the support and a
    full 248k-token row would make the arm cost more than the exact check it is a
    cross-check of. The exact identity is established over the whole support above;
    this arm only has to catch a code-path defect in the walk itself.
    """

    if target.support_size > MONTE_CARLO_SUPPORT:
        order = np.argsort(
            -np.asarray(target.probabilities, dtype=np.float64), kind="stable"
        )[:MONTE_CARLO_SUPPORT]
        ids = np.asarray(target.token_ids, dtype=np.int64)[order]
        weights = np.asarray(target.probabilities, dtype=np.float64)[order]
        by_id = np.argsort(ids, kind="stable")
        target = SparseDistribution(
            tuple(int(token_id) for token_id in ids[by_id]),
            weights[by_id] / float(np.sum(weights)),
        )
    drafted = int(target.token_ids[int(np.argmax(target.probabilities))])
    draft = SparseDistribution.point_mass(drafted)
    batch = _chain_batch(drafted)
    rng = np.random.default_rng(seed)
    counts: dict[int, int] = {}
    for _ in range(draws):
        values = iter(float(rng.random()) for _ in range(3))

        def draw(values=values) -> float:
            return next(values)

        result = sampled_accept_from_distributions(
            batch,
            (target, target),
            (draft, draft),
            draws=draw,
            remaining_decode=(2,),
        )
        emitted = (
            result.accepted_tokens[0][0]
            if result.accepted_tokens[0]
            else result.next_tokens[0]
        )
        counts[int(emitted)] = counts.get(int(emitted), 0) + 1
    support = sorted(set(counts) | set(target.token_ids))
    p = np.asarray([target.probability(token_id) for token_id in support], dtype=np.float64)
    q = np.asarray([counts.get(token_id, 0) / draws for token_id in support], dtype=np.float64)
    total_variation = float(0.5 * np.sum(np.abs(p - q)))
    target_top = int(target.token_ids[int(np.argmax(target.probabilities))])
    observed_top = max(counts, key=lambda token_id: counts[token_id])
    # Calibrate the histogram against p per cell. The TV distance of a
    # multinomial histogram has no closed-form band for an arbitrary support
    # (measured: a 4-sigma TV band on a two-way split is exceeded by 1.5% of
    # seeds), so the criterion is per cell instead: a normal 5-sigma limit where
    # the expected count supports it, and an exact Poisson tail where it does not
    # (a 1,024-cell support has cells whose expected count is a hundredth of a
    # draw, where a single stray draw is 15 sigma). A defect in the walk - an
    # inverted accept test, a missing residual draw - moves whole percent of mass
    # and fails either form by orders of magnitude.
    expected = p * draws
    observed_counts = q * draws
    cell_limit = np.asarray(
        [_poisson_count_limit(float(value), tail=MONTE_CARLO_TAIL) for value in expected],
        dtype=np.float64,
    )
    cell_excess = observed_counts - cell_limit
    max_cell_z = float(np.max(cell_excess)) if cell_excess.size else 0.0
    tv_mean = 0.5 * float(np.sum(np.sqrt(2.0 * (p * (1.0 - p) / draws) / np.pi)))
    return {
        "draws": draws,
        "checked_support": int(target.support_size),
        "distinct_emitted_tokens": len(counts),
        "total_variation": total_variation,
        "tv_band_mean": tv_mean,
        "max_cell_excess": max_cell_z,
        "cell_count_limit_slack": 0.0,
        "top1_agreement": 1.0 if target_top == observed_top else 0.0,
        "target_top1": target_top,
        "observed_top1": observed_top,
        "accepted_rate": float(counts.get(drafted, 0) / draws),
        "target_drafted_probability": float(target.probability(drafted)),
    }


def _ar_law_check(
    logits: np.ndarray,
    params: SimpleNamespace,
    state: RowSamplingState,
) -> dict[str, float]:
    """The AR token's reported probability must be its weight in the law."""

    result = select_token(logits, params, state.clone())
    token_ids, probabilities = processed_distribution(logits, params, state)
    weights = {
        int(token): float(weight)
        for token, weight in zip(token_ids, probabilities, strict=True)
    }
    if result.forced or result.logprob is None:
        return {"checked": 0.0, "agreement": 1.0}
    weight = weights.get(int(result.token_id), 0.0)
    reported = float(np.exp(result.logprob))
    return {
        "checked": 1.0,
        "agreement": 1.0 if abs(weight - reported) <= 1e-9 else 0.0,
        "max_weight_deviation": abs(weight - reported),
        "support": len(token_ids),
        "weight_sum": float(np.sum(probabilities)),
    }


def _run_prompt(
    session: Any,
    tokenizer: Any,
    prompt: str,
    *,
    decode_steps: int,
    monte_carlo_draws: int,
    seed: int,
) -> dict[str, Any]:
    from scripts.gguf_mtp_bench import build_chat_prompt

    prompt_ids = build_chat_prompt(tokenizer, prompt, reasoning="off")
    prefill_started = time.time()
    prefill = session.prefill(prompt_ids, return_logits=True)
    print(
        f"[gate]   prefill {len(prompt_ids)} tokens in {time.time() - prefill_started:.1f}s",
        flush=True,
    )
    logits = np.asarray(prefill.logits, dtype=np.float32).reshape(-1)
    history = list(prompt_ids)
    generated: list[int] = []
    rows: list[dict[str, Any]] = []
    for step in range(decode_steps):
        state = RowSamplingState(
            prompt_tokens=tuple(history),
            generated_tokens=tuple(generated),
            seed=seed,
        )
        row_report: dict[str, Any] = {
            "step": step,
            "position": len(history) - 1,
            "vocab": int(logits.size),
            "logit_top1": int(np.argmax(logits)),
            "ar_law": {},
            "point_mass_coupling": {},
            "general_coupling": {},
            "monte_carlo": {},
        }
        greedy_token = int(np.argmax(logits))
        for name, params in SAMPLER_CONFIGS.items():
            target = _distribution_of(logits, params, state)
            row_report["ar_law"][name] = _ar_law_check(logits, params, state)
            # Point-mass coupling: what a greedy draft chain supplies. Sweep the
            # drafted token across the support so both the accept and the
            # residual branch run on real rows.
            ranked = sorted(
                range(target.support_size),
                key=lambda index: -float(target.probabilities[index]),
            )
            picks = [ranked[0]]
            if len(ranked) > 1:
                picks.append(ranked[1])
            if len(ranked) > 4:
                picks.append(ranked[-1])
            coupling: dict[str, Any] = {}
            for pick in picks:
                drafted = int(target.token_ids[pick])
                draft = SparseDistribution.point_mass(drafted)
                induced_ids, induced_weights = _induced_first_token(target, draft)
                metrics = _distribution_metrics(induced_ids, induced_weights, target)
                metrics["drafted_token"] = drafted
                metrics["drafted_probability"] = float(target.probability(drafted))
                coupling[f"drafted_rank_{picks.index(pick)}"] = metrics
            row_report["point_mass_coupling"][name] = coupling
            for draft_name, draft_params in DRAFT_CONFIGS.items():
                draft = _distribution_of(logits, draft_params, state)
                induced_ids, induced_weights = _induced_first_token(target, draft)
                metrics = _distribution_metrics(induced_ids, induced_weights, target)
                metrics["draft_support"] = draft.support_size
                row_report["general_coupling"][f"{name}|{draft_name}"] = metrics
            if step == 0:
                row_report["monte_carlo"][name] = _monte_carlo_check(
                    target,
                    draws=monte_carlo_draws,
                    seed=seed + step,
                )
        rows.append(row_report)
        generated.append(greedy_token)
        history.append(greedy_token)
        step_started = time.time()
        step_result = session.step(greedy_token, return_logits=True)
        logits = np.asarray(step_result.logits, dtype=np.float32).reshape(-1)
        print(
            f"[gate]   step {step + 1}/{decode_steps} checked in {time.time() - step_started:.1f}s",
            flush=True,
        )
    return {
        "prompt": prompt,
        "prompt_tokens": len(prompt_ids),
        "decode_steps": decode_steps,
        "generated_top1": generated,
        "rows": rows,
    }


def _aggregate(payload: dict[str, Any]) -> dict[str, Any]:
    exact_failures: list[dict[str, Any]] = []
    ar_failures: list[dict[str, Any]] = []
    monte_carlo_failures: list[dict[str, Any]] = []
    max_abs = 0.0
    max_tv = 0.0
    max_kl = 0.0
    top1_agreement = 1.0
    comparisons = 0
    ar_checks = 0
    ar_agreements = 0
    mc_rows = 0
    mc_max_tv = 0.0
    mc_top1_agreement = 1.0
    support_sizes: list[int] = []
    for prompt_report in payload["prompts"]:
        for row in prompt_report["rows"]:
            for name, metrics in row["ar_law"].items():
                if not metrics.get("checked"):
                    continue
                ar_checks += 1
                ar_agreements += int(metrics["agreement"] == 1.0)
                if metrics["agreement"] != 1.0:
                    ar_failures.append({"step": row["step"], "config": name, **metrics})
                support_sizes.append(int(metrics.get("support", 0)))
            for name, coupling in row["point_mass_coupling"].items():
                for label, metrics in coupling.items():
                    comparisons += 1
                    _record(
                        metrics,
                        label=f"{name}|{label}",
                        step=row["step"],
                        exact_failures=exact_failures,
                    )
                    max_abs = max(max_abs, metrics["max_abs_deviation"])
                    max_tv = max(max_tv, metrics["total_variation"])
                    max_kl = max(
                        max_kl,
                        metrics["kl_p_to_induced"],
                        metrics["kl_induced_to_p"],
                    )
                    top1_agreement = min(top1_agreement, metrics["top1_agreement"])
            for label, metrics in row["general_coupling"].items():
                comparisons += 1
                _record(
                    metrics,
                    label=label,
                    step=row["step"],
                    exact_failures=exact_failures,
                )
                max_abs = max(max_abs, metrics["max_abs_deviation"])
                max_tv = max(max_tv, metrics["total_variation"])
                max_kl = max(
                    max_kl,
                    metrics["kl_p_to_induced"],
                    metrics["kl_induced_to_p"],
                )
                top1_agreement = min(top1_agreement, metrics["top1_agreement"])
            for name, metrics in row["monte_carlo"].items():
                mc_rows += 1
                mc_max_tv = max(mc_max_tv, metrics["total_variation"])
                mc_top1_agreement = min(mc_top1_agreement, metrics["top1_agreement"])
                if (
                    metrics["max_cell_excess"] > 0.0
                    or metrics["top1_agreement"] != 1.0
                ):
                    monte_carlo_failures.append(
                        {"step": row["step"], "config": name, **metrics}
                    )
    gate = (
        not exact_failures
        and not ar_failures
        and not monte_carlo_failures
        and max_abs <= EXACT_TOLERANCE
        and max_tv <= EXACT_TOLERANCE
        and max_kl <= KL_TOLERANCE
        and top1_agreement == 1.0
    )
    return {
        "verdict": "pass" if gate else "fail",
        "exact_identity": {
            "comparisons": comparisons,
            "max_abs_deviation": max_abs,
            "max_total_variation": max_tv,
            "max_kl": max_kl,
            "top1_agreement": top1_agreement,
            "tolerance": EXACT_TOLERANCE,
            "failures": exact_failures[:10],
        },
        "ar_law": {
            "checks": ar_checks,
            "agreements": ar_agreements,
            "agreement_rate": (ar_agreements / ar_checks) if ar_checks else 1.0,
            "max_support": max(support_sizes) if support_sizes else 0,
            "mean_support": (sum(support_sizes) / len(support_sizes)) if support_sizes else 0.0,
            "failures": ar_failures[:10],
        },
        "monte_carlo": {
            "rows": mc_rows,
            "max_total_variation": mc_max_tv,
            "top1_agreement": mc_top1_agreement,
            "failures": monte_carlo_failures[:10],
        },
    }


def _record(
    metrics: dict[str, float],
    *,
    label: str,
    step: int,
    exact_failures: list[dict[str, Any]],
) -> None:
    if (
        metrics["max_abs_deviation"] > EXACT_TOLERANCE
        or metrics["total_variation"] > EXACT_TOLERANCE
        or metrics["kl_p_to_induced"] > KL_TOLERANCE
        or metrics["kl_induced_to_p"] > KL_TOLERANCE
        or metrics["top1_agreement"] != 1.0
    ):
        exact_failures.append({"step": step, "config": label, **metrics})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--backend",
        default="hip_gfx1151",
        help="resident session backend (the server passes this explicitly)",
    )
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--monte-carlo-draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--use-wmma-prefill", action="store_true")
    parser.add_argument("--use-gemv-decode", action="store_true")
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not _hip_available():
        raise SystemExit("sampled acceptance distribution gate requires ROCm")
    if args.decode_steps <= 0:
        raise SystemExit("--decode-steps must be positive")
    if args.monte_carlo_draws <= 0:
        raise SystemExit("--monte-carlo-draws must be positive")

    from hipengine.loading.gguf import GGUFReader
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    reader = GGUFReader(args.model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(reader.info)
    prompts = tuple(args.prompt) if args.prompt else DEFAULT_PROMPTS
    started = time.time()
    print(f"[gate] opening resident session (backend={args.backend})...", flush=True)
    session = Qwen35GGUFResidentSession(
        model_path=args.model,
        backend=str(args.backend),
        use_wmma_prefill=bool(args.use_wmma_prefill),
        use_gemv_decode=bool(args.use_gemv_decode),
        require_cached_build=bool(args.require_cached_build),
    )
    print(f"[gate] session ready in {time.time() - started:.1f}s", flush=True)
    payload: dict[str, Any] = {
        "title": "Sampled MTP acceptance distribution gate on real target logits",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "status": "measured",
        "model": args.model.name,
        "backend": "hip_gfx1151",
        "host": "zbook",
        "seed": args.seed,
        "decode_steps": args.decode_steps,
        "monte_carlo_draws": args.monte_carlo_draws,
        "sampler_configs": sorted(SAMPLER_CONFIGS),
        "draft_configs": sorted(DRAFT_CONFIGS),
        "prompts": [],
    }
    try:
        for prompt in prompts:
            print(f"[gate] prompt={prompt[:48]!r}...", flush=True)
            payload["prompts"].append(
                _run_prompt(
                    session,
                    tokenizer,
                    prompt,
                    decode_steps=args.decode_steps,
                    monte_carlo_draws=args.monte_carlo_draws,
                    seed=args.seed,
                )
            )
    finally:
        session.close()
    payload["summary"] = _aggregate(payload)
    payload["wall_seconds"] = time.time() - started
    print(json.dumps(payload["summary"], indent=1))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"[gate] wrote {args.output}")
    return 0 if payload["summary"]["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
