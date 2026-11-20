"""Accounting for internal speculative baselines (no runtime or GPU imports).

These legacy cycle timers exclude prefill and count every returned output. They
are not first-to-last-token timings: AR produces its first token during prefill,
whereas speculative cycles verify a lagged root. Keep that scope explicit rather
than comparing these rates with public streaming transition throughput.
"""
from __future__ import annotations

import math
import statistics


AGGREGATION = (
    "Primary: sum actual generated tokens / sum decode seconds across all prompts "
    "and repetitions, separately for speculative and per-repetition AR runs. "
    "Median-of-prompt rates are diagnostic. Deterministic repeats add timing "
    "samples, not independent correctness coverage."
)
TIMING_SCOPE = "legacy post-prefill cycle wall; numerator includes prompt-produced first token"


def measurement(*, method, prompt, prompt_len, output_ids, ar_ids, expected_tokens,
                seconds, ar_seconds, mean_acceptance, accepted_per_draft):
    for name, value in (("seconds", seconds), ("ar_seconds", ar_seconds)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    ids, reference = list(output_ids), list(ar_ids)
    agree = sum(a == b for a, b in zip(ids, reference))
    return {
        "id": prompt["id"], "category": prompt["category"], "prompt_len": prompt_len,
        f"{method}_tok_s": len(ids) / seconds, "ar_tok_s": len(reference) / ar_seconds,
        "generated_tokens": len(ids), "ar_generated_tokens": len(reference),
        "decode_seconds": seconds, "ar_seconds": ar_seconds,
        "mean_acceptance": mean_acceptance, "accepted_per_draft": accepted_per_draft,
        "ar_agreement": agree / max(len(ids), len(reference)) if ids or reference else None,
        "exact": ids == reference and len(ids) == expected_tokens,
    }


def aggregate(per_prompt, *, method):
    if not per_prompt or any(not reps for reps in per_prompt.values()):
        raise ValueError("cannot aggregate an empty suite or repetition list")
    results = []
    for pid, reps in per_prompt.items():
        med = statistics.median(e[f"{method}_tok_s"] for e in reps)
        ar_med = statistics.median(e["ar_tok_s"] for e in reps)
        results.append({
            "id": pid, "category": reps[0]["category"], "prompt_len": reps[0]["prompt_len"],
            f"{method}_tok_s_median": med, "ar_tok_s_median": ar_med,
            f"{method}_tok_s_reps": [e[f"{method}_tok_s"] for e in reps],
            "ar_tok_s_reps": [e["ar_tok_s"] for e in reps],
            **{f"{key}_reps": [e[key] for e in reps] for key in (
                "decode_seconds", "ar_seconds", "generated_tokens", "ar_generated_tokens",
                "mean_acceptance", "accepted_per_draft", "ar_agreement")},
            "mean_acceptance": statistics.mean(e["mean_acceptance"] for e in reps),
            "accepted_per_draft": statistics.mean(e["accepted_per_draft"] for e in reps),
            "exact_all_reps": all(e["exact"] for e in reps),
            "ratio_vs_ar": med / ar_med if ar_med else None,
        })
    samples = [e for reps in per_prompt.values() for e in reps]
    tokens = sum(e["generated_tokens"] for e in samples)
    ar_tokens = sum(e["ar_generated_tokens"] for e in samples)
    seconds = sum(e["decode_seconds"] for e in samples)
    ar_seconds = sum(e["ar_seconds"] for e in samples)
    rate, ar_rate = tokens / seconds, ar_tokens / ar_seconds
    med = statistics.median(r[f"{method}_tok_s_median"] for r in results)
    ar_med = statistics.median(r["ar_tok_s_median"] for r in results)
    return results, {
        f"{method}_tok_s_weighted": rate, "ar_tok_s_weighted": ar_rate,
        "ratio_vs_ar": rate / ar_rate if ar_rate else None,
        f"{method}_tok_s_median_of_prompts": med, "ar_tok_s_median_of_prompts": ar_med,
        "ratio_of_prompt_medians": med / ar_med if ar_med else None,
        "measured_generated_tokens": tokens, "ar_measured_generated_tokens": ar_tokens,
        "decode_seconds": seconds, "ar_seconds": ar_seconds,
        "prompt_count": len(results), "timing_sample_count": len(samples),
        "exact_all": all(r["exact_all_reps"] for r in results),
        "mean_acceptance": statistics.mean(e["mean_acceptance"] for e in samples),
    }
