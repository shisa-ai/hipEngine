#!/usr/bin/env python3
"""Run hipEngine MTP verifier economics over the llama.cpp MTP prompt suite.

The prompt texts are adapted from am17an's ad-hoc ``mtp-bench.py`` gist used
in llama.cpp MTP PR discussions.  Unlike the original OpenAI-compatible server
bench, this wrapper tokenizes each prompt for the local hipEngine model and
invokes ``scripts/mtp_verifier_economics.py`` per prompt so we can compare MTP
verifier economics across a broader prompt mix.

This is a diagnostic harness only: it does not touch runtime hot paths and its
JSON output always carries ``performance_claim=false``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")
DEFAULT_PROMPTS = REPO_ROOT / "benchmarks" / "fixtures" / "llamacpp_mtp_bench_prompts.json"
DEFAULT_RAW_ROOT = Path("/tmp/hipengine-mtp-llamacpp-prompt-suite-economics")

SUMMARY_FIELDS = (
    "cycle_cost_ar_tokens_mean",
    "observed_cycle_speedup_vs_ar_mean",
    "actual_decode_speedup_vs_ar_mean",
    "ar_decode_tok_s_mean",
    "mtp_decode_tok_s_mean",
    "avg_visible_tokens_per_cycle_mean",
    "avg_accepted_per_cycle_mean",
    "acceptance_rate_mean",
    "cycle_wall_ms_per_cycle_mean",
    "verify_ms_per_cycle_mean",
    "proposal_update_ms_per_cycle_mean",
    "proposal_snapshot_saves_mean",
    "proposal_snapshot_skips_mean",
    "proposal_snapshot_saves_per_cycle_mean",
    "proposal_snapshot_skips_per_cycle_mean",
    "ar_fallback_cycles_mean",
    "ar_fallback_tokens_mean",
    "ar_fallback_ms_per_cycle_mean",
    "ar_fallback_proposer_update_ms_per_cycle_mean",
    "confidence_threshold_mean",
    "confidence_ar_fallback_cycles_mean",
    "confidence_ar_fallback_tokens_mean",
    "ar_decode_ms_per_token_mean",
)

PROMPT_RENDER_MODES = ("raw", "qwen_chat_thinking_off", "qwen_chat_thinking_on")


@dataclass(frozen=True)
class EncodedPrompt:
    source_text: str
    rendered_text: str
    token_ids: list[int]


@dataclass(frozen=True)
class PromptEncoder:
    mode: str
    tokenizer: Any
    tokenization: str

    def encode(self, text: str) -> EncodedPrompt:
        if self.mode == "raw":
            ids = [int(x) for x in self.tokenizer.encode(text).ids]
            rendered = text
        elif self.mode in {"qwen_chat_thinking_off", "qwen_chat_thinking_on"}:
            enable_thinking = self.mode == "qwen_chat_thinking_on"
            messages = [{"role": "user", "content": text}]
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            ids = [int(x) for x in self.tokenizer.encode(rendered, add_special_tokens=False)]
        else:  # pragma: no cover - argparse choices should prevent this.
            raise ValueError(f"unknown prompt render mode: {self.mode}")
        if not ids:
            raise ValueError("prompt encoded to no tokens")
        return EncodedPrompt(source_text=text, rendered_text=str(rendered), token_ids=ids)


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    return statistics.fmean(vals) if vals else None


def _std(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0 if vals else None
    return statistics.stdev(vals)


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_prompt_budget_map(value: str | None) -> dict[str, int]:
    if not value:
        return {}
    text = str(value).strip()
    data: Any
    if text.startswith("{"):
        data = json.loads(text)
    else:
        path = Path(text)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = None

    if isinstance(data, dict):
        if isinstance(data.get("oracle_bound"), dict) and isinstance(data["oracle_bound"].get("choices"), dict):
            data = data["oracle_bound"]["choices"]
        elif isinstance(data.get("prompt_budget_map"), dict):
            data = data["prompt_budget_map"]
        elif isinstance(data.get("choices"), dict):
            data = data["choices"]
    if isinstance(data, dict):
        out = {str(name): int(budget) for name, budget in data.items()}
    else:
        out = {}
        for part in _split_csv(text):
            if "=" not in part:
                raise ValueError(
                    "--prompt-budget-map must be a JSON object, a JSON artifact "
                    "with oracle_bound.choices, or comma-separated name=budget pairs"
                )
            name, budget = part.split("=", 1)
            out[str(name).strip()] = int(str(budget).strip())
    if not out:
        raise ValueError("--prompt-budget-map resolved to an empty map")
    bad = {name: budget for name, budget in out.items() if int(budget) <= 0}
    if bad:
        raise ValueError(f"prompt budgets must be positive integers: {bad!r}")
    return out


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _load_prompt_suite(path: Path) -> dict[str, Any]:
    suite = json.loads(path.read_text(encoding="utf-8"))
    prompts = suite.get("prompts") or []
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path} contains no prompts")
    names: set[str] = set()
    for prompt in prompts:
        name = str(prompt.get("name") or "")
        text = str(prompt.get("prompt") or "")
        if not name or not text:
            raise ValueError(f"invalid prompt entry in {path}: {prompt!r}")
        if name in names:
            raise ValueError(f"duplicate prompt name in {path}: {name}")
        names.add(name)
    return suite


def _select_prompts(suite: dict[str, Any], *, names_csv: str | None, limit: int | None) -> list[dict[str, str]]:
    prompts = [{"name": str(p["name"]), "prompt": str(p["prompt"])} for p in suite["prompts"]]
    names = _split_csv(names_csv)
    if names:
        by_name = {p["name"]: p for p in prompts}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"unknown prompt name(s): {', '.join(missing)}")
        prompts = [by_name[name] for name in names]
    if limit is not None:
        prompts = prompts[: max(0, int(limit))]
    if not prompts:
        raise ValueError("prompt selection is empty")
    return prompts


def _load_raw_tokenizer(model: Path) -> Any:
    try:
        from tokenizers import Tokenizer
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("tokenizers is required to encode the prompt suite") from exc
    tokenizer_path = model / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def _load_hf_tokenizer(model: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("transformers is required for Qwen chat-template prompt rendering") from exc
    return AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)


def _load_prompt_encoder(model: Path, mode: str) -> PromptEncoder:
    if mode == "raw":
        return PromptEncoder(
            mode=mode,
            tokenizer=_load_raw_tokenizer(model),
            tokenization="raw prompt text encoded with model tokenizer.json",
        )
    if mode in {"qwen_chat_thinking_off", "qwen_chat_thinking_on"}:
        return PromptEncoder(
            mode=mode,
            tokenizer=_load_hf_tokenizer(model),
            tokenization=f"Qwen chat_template rendered with enable_thinking={mode == 'qwen_chat_thinking_on'}",
        )
    raise ValueError(f"unknown prompt render mode: {mode}")


def _candidate_budgets_for_prompt(args: argparse.Namespace, prompt_name: str) -> str:
    prompt_budget_map = getattr(args, "prompt_budget_map_resolved", {}) or {}
    if not prompt_budget_map:
        return str(args.candidate_budgets)
    if prompt_name not in prompt_budget_map:
        raise ValueError(f"--prompt-budget-map missing selected prompt: {prompt_name}")
    return str(int(prompt_budget_map[prompt_name]))


def _economics_command(
    args: argparse.Namespace,
    *,
    prompt_name: str,
    prompt_tokens_file: Path,
    prompt_raw_root: Path,
    out_path: Path,
) -> list[str]:
    candidate_budgets = _candidate_budgets_for_prompt(args, prompt_name)
    cmd = [
        sys.executable,
        "scripts/mtp_verifier_economics.py",
        "--model",
        str(args.model),
        "--prompt-tokens-file",
        str(prompt_tokens_file),
        "--decode-tokens",
        str(args.decode_tokens),
        "--candidate-budgets",
        str(candidate_budgets),
        "--runs",
        str(args.runs),
        "--proposal-impl",
        str(args.proposal_impl),
        "--backend",
        str(args.backend),
        "--hip-arch",
        str(args.hip_arch),
        "--chain-attn-mode",
        str(args.chain_attn_mode),
        "--graph-mode",
        str(args.graph_mode),
        "--raw-root",
        str(prompt_raw_root),
        "--out",
        str(out_path),
    ]
    if args.small_batch_decode_threshold is not None:
        cmd += ["--small-batch-decode-threshold", str(args.small_batch_decode_threshold)]
    if int(getattr(args, "active_budget_cap", 0)) > 0:
        cmd += ["--active-budget-cap", str(int(args.active_budget_cap))]
    if args.verify_gpu_accept is not None:
        cmd += ["--verify-gpu-accept", str(args.verify_gpu_accept)]
    if args.acceptance_diagnostics:
        cmd.append("--acceptance-diagnostics")
    if float(getattr(args, "confidence_threshold", 0.0) or 0.0) > 0.0:
        cmd += ["--confidence-threshold", str(float(args.confidence_threshold))]
    if int(getattr(args, "ar_fallback_zero_streak", 0)) > 0:
        cmd += [
            "--ar-fallback-zero-streak",
            str(int(args.ar_fallback_zero_streak)),
            "--ar-fallback-tokens",
            str(int(args.ar_fallback_tokens)),
        ]
        if bool(getattr(args, "ar_fallback_until_end", False)):
            cmd.append("--ar-fallback-until-end")
    if args.llama_target_cycle_cost is not None:
        cmd += ["--llama-target-cycle-cost", str(args.llama_target_cycle_cost)]
    return cmd


def _summary_for_economics(economics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for budget, payload in sorted((economics.get("by_budget") or {}).items(), key=lambda item: int(item[0])):
        aggregate = payload.get("aggregate") or {}
        out[str(budget)] = {
            "all_exact_ar_match": bool(aggregate.get("all_exact_ar_match")),
            **{field: aggregate.get(field) for field in SUMMARY_FIELDS},
            "accepted_lengths_by_run": aggregate.get("accepted_lengths_by_run"),
            "active_budgets_by_run": aggregate.get("active_budgets_by_run"),
            "acceptance_diagnostics_by_run": aggregate.get("acceptance_diagnostics_by_run"),
        }
    return out


def _aggregate_across_prompts(results: list[dict[str, Any]]) -> dict[str, Any]:
    budgets = sorted({budget for result in results for budget in result["by_budget"].keys()}, key=int)
    by_budget: dict[str, Any] = {}
    for budget in budgets:
        rows = [result["by_budget"][budget] for result in results if budget in result["by_budget"]]
        summary: dict[str, Any] = {
            "prompts": len(rows),
            "all_exact_ar_match": all(bool(row.get("all_exact_ar_match")) for row in rows),
        }
        for field in SUMMARY_FIELDS:
            vals = [row[field] for row in rows if row.get(field) is not None]
            summary[f"{field}_across_prompts_mean"] = _mean(vals)
            summary[f"{field}_across_prompts_std"] = _std(vals)
        for field in ("proposal_snapshot_saves_mean", "proposal_snapshot_skips_mean"):
            vals = [row[field] for row in rows if row.get(field) is not None]
            summary[f"{field}_across_prompts_sum"] = float(sum(vals)) if vals else None
        by_budget[budget] = summary
    return by_budget


def _aggregate_selected_policy(
    results: list[dict[str, Any]],
    prompt_budget_map: dict[str, int],
    *,
    decode_tokens: int,
) -> dict[str, Any]:
    if not prompt_budget_map:
        return {}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for result in results:
        name = str(result.get("name"))
        budget = str(int(prompt_budget_map[name]))
        row = (result.get("by_budget") or {}).get(budget)
        if row is None:
            missing.append(f"{name}:B{budget}")
            continue
        rows.append(row)
    if missing:
        raise ValueError(f"selected policy missing result rows: {', '.join(missing)}")
    summary: dict[str, Any] = {
        "prompts": len(rows),
        "all_exact_ar_match": all(bool(row.get("all_exact_ar_match")) for row in rows),
        "budget_choices": {str(name): int(budget) for name, budget in prompt_budget_map.items()},
    }
    for field in SUMMARY_FIELDS:
        vals = [row[field] for row in rows if row.get(field) is not None]
        summary[f"{field}_across_prompts_mean"] = _mean(vals)
        summary[f"{field}_across_prompts_std"] = _std(vals)
    for field in ("proposal_snapshot_saves_mean", "proposal_snapshot_skips_mean"):
        vals = [row[field] for row in rows if row.get(field) is not None]
        summary[f"{field}_across_prompts_sum"] = float(sum(vals)) if vals else None
    ar_seconds = [
        float(decode_tokens) / float(row["ar_decode_tok_s_mean"])
        for row in rows
        if row.get("ar_decode_tok_s_mean")
    ]
    mtp_seconds = [
        float(decode_tokens) / float(row["mtp_decode_tok_s_mean"])
        for row in rows
        if row.get("mtp_decode_tok_s_mean")
    ]
    if len(ar_seconds) == len(rows) and len(mtp_seconds) == len(rows) and sum(mtp_seconds) > 0.0:
        summary["actual_decode_speedup_vs_ar_total_time"] = float(sum(ar_seconds) / sum(mtp_seconds))
        summary["ar_decode_seconds_sum"] = float(sum(ar_seconds))
        summary["mtp_decode_seconds_sum"] = float(sum(mtp_seconds))
    return summary


def _run_prompt(args: argparse.Namespace, *, prompt: dict[str, str], encoder: PromptEncoder) -> dict[str, Any]:
    prompt_name = str(prompt["name"])
    prompt_dir = args.raw_root / _safe_name(prompt_name)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    encoded = encoder.encode(prompt["prompt"])
    token_ids = encoded.token_ids
    prompt_tokens_file = prompt_dir / "prompt-tokens.txt"
    prompt_tokens_file.write_text(",".join(str(token) for token in token_ids), encoding="utf-8")
    source_text_file = prompt_dir / "prompt-source.txt"
    source_text_file.write_text(encoded.source_text, encoding="utf-8")
    prompt_text_file = prompt_dir / "prompt.txt"
    prompt_text_file.write_text(encoded.rendered_text, encoding="utf-8")
    economics_out = prompt_dir / "economics.json"
    economics_log = prompt_dir / "economics.log"
    candidate_budgets = _candidate_budgets_for_prompt(args, prompt_name)
    cmd = _economics_command(
        args,
        prompt_name=prompt_name,
        prompt_tokens_file=prompt_tokens_file,
        prompt_raw_root=prompt_dir / "raw",
        out_path=economics_out,
    )

    result: dict[str, Any] = {
        "name": prompt_name,
        "selected_candidate_budgets": candidate_budgets,
        "prompt_render_mode": encoder.mode,
        "source_prompt_chars": len(encoded.source_text),
        "rendered_prompt_chars": len(encoded.rendered_text),
        "prompt_chars": len(encoded.rendered_text),
        "prompt_tokens": len(token_ids),
        "prompt_tokens_file": str(prompt_tokens_file),
        "prompt_source_file": str(source_text_file),
        "prompt_text_file": str(prompt_text_file),
        "economics_json": str(economics_out),
        "economics_log": str(economics_log),
        "command": " ".join(cmd),
    }
    if args.dry_run:
        result["dry_run"] = True
        result["by_budget"] = {}
        print(f"[dry-run] {prompt_name}: prompt_tokens={len(token_ids)}")
        return result

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    started = time.perf_counter()
    with economics_log.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, stdout=log_file, stderr=subprocess.STDOUT)
    result["subprocess_wall_seconds"] = time.perf_counter() - started
    if completed.returncode != 0:
        tail = economics_log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"economics failed for prompt={prompt_name!r} with exit {completed.returncode}; tail of {economics_log}:\n"
            + "\n".join(tail)
        )
    economics = json.loads(economics_out.read_text(encoding="utf-8"))
    result["by_budget"] = _summary_for_economics(economics)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-names", help="Comma-separated prompt names to run; default runs the full suite")
    parser.add_argument("--limit", type=int, help="Run only the first N selected prompts")
    parser.add_argument("--list-prompts", action="store_true", help="List prompt names and exit")
    parser.add_argument(
        "--prompt-render",
        choices=PROMPT_RENDER_MODES,
        default="raw",
        help="Prompt rendering before tokenization. raw preserves prior artifacts; qwen_chat_* uses the model chat_template.",
    )
    parser.add_argument("--decode-tokens", type=int, default=192)
    parser.add_argument("--candidate-budgets", default="3")
    parser.add_argument(
        "--active-budget-cap",
        type=int,
        default=0,
        help=(
            "Diagnostic fixed-shape adaptive-budget probe. Forwarded to child "
            "smokes to cap active drafted candidates while keeping verifier "
            "rows at --candidate-budgets."
        ),
    )
    parser.add_argument(
        "--prompt-budget-map",
        help=(
            "Diagnostic fixed-per-prompt policy. Accepts comma-separated "
            "name=budget pairs, a JSON object, or an artifact containing "
            "oracle_bound.choices. When set, each selected prompt runs only "
            "its mapped fixed budget."
        ),
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--proposal-impl", choices=("persistent_device", "persistent_device_b1", "reload_d2h"), default="persistent_device")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--hip-arch", default="gfx1151")
    parser.add_argument("--chain-attn-mode", choices=("c1_loop", "batched", "decode_batched"), default="batched")
    parser.add_argument("--graph-mode", choices=("off", "auto", "validate"), default="off")
    parser.add_argument("--small-batch-decode-threshold", type=int, default=7)
    parser.add_argument("--verify-gpu-accept", default=None)
    parser.add_argument(
        "--acceptance-diagnostics",
        action="store_true",
        help="Retain per-cycle MTP acceptance diagnostics from child smoke runs in the suite artifact.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=(
            "Opt-in MTP whole-cycle confidence gate. In persistent chain mode, "
            "child smokes route low-confidence depth-1 cycles through exact AR "
            "instead of running the verifier. 0 disables."
        ),
    )
    parser.add_argument(
        "--ar-fallback-zero-streak",
        type=int,
        default=0,
        help=(
            "Opt-in MTP policy diagnostic: after this many consecutive zero-accept "
            "cycles, have the child smoke skip the next --ar-fallback-tokens tokens "
            "through target AR. 0 disables."
        ),
    )
    parser.add_argument(
        "--ar-fallback-tokens",
        type=int,
        default=1,
        help="Number of target AR tokens to emit per --ar-fallback-zero-streak trigger.",
    )
    parser.add_argument(
        "--ar-fallback-until-end",
        action="store_true",
        help=(
            "When --ar-fallback-zero-streak triggers, finish remaining decode with "
            "plain target AR instead of resuming MTP."
        ),
    )
    parser.add_argument("--llama-target-cycle-cost", type=float, default=2.0)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / f"{date.today().isoformat()}-hipengine-mtp-llamacpp-prompt-suite-economics.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Tokenize prompts and print commands without running GPU economics")
    args = parser.parse_args()

    suite = _load_prompt_suite(args.prompts_file)
    prompts = _select_prompts(suite, names_csv=args.prompt_names, limit=args.limit)
    args.prompt_budget_map_resolved = _parse_prompt_budget_map(args.prompt_budget_map)
    if args.prompt_budget_map_resolved:
        selected_prompt_names = [str(prompt["name"]) for prompt in prompts]
        selected_names = set(selected_prompt_names)
        missing = sorted(selected_names.difference(args.prompt_budget_map_resolved))
        extra = sorted(set(args.prompt_budget_map_resolved).difference(selected_names))
        if missing:
            raise ValueError(f"--prompt-budget-map missing selected prompt(s): {', '.join(missing)}")
        if extra:
            print(f"[prompt-suite] ignoring map entries for unselected prompts: {', '.join(extra)}", flush=True)
        args.prompt_budget_map_resolved = {
            name: int(args.prompt_budget_map_resolved[name])
            for name in selected_prompt_names
        }
    if args.list_prompts:
        for prompt in prompts:
            print(f"{prompt['name']}\t{len(prompt['prompt'])} chars")
        return 0

    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder = _load_prompt_encoder(args.model, args.prompt_render)

    results: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts, 1):
        print(f"[prompt-suite] {idx}/{len(prompts)} {prompt['name']}", flush=True)
        result = _run_prompt(args, prompt=prompt, encoder=encoder)
        results.append(result)
        for budget, summary in (result.get("by_budget") or {}).items():
            c3 = summary.get("cycle_cost_ar_tokens_mean")
            visible = summary.get("avg_visible_tokens_per_cycle_mean")
            exact = summary.get("all_exact_ar_match")
            print(f"  B={budget} exact={exact} C={c3:.3f} visible/cycle={visible:.3f}", flush=True)

    artifact = {
        "schema": 1,
        "status": "diagnostic",
        "performance_claim": False,
        "date": date.today().isoformat(),
        "purpose": "hipEngine MTP verifier economics over the llama.cpp mtp-bench prompt suite.",
        "source_prompt_suite": suite.get("source"),
        "prompts_file": str(args.prompts_file),
        "model": str(args.model),
        "prompt_render_mode": str(args.prompt_render),
        "tokenization": encoder.tokenization,
        "decode_tokens": int(args.decode_tokens),
        "candidate_budgets": [int(x) for x in _split_csv(args.candidate_budgets)],
        "active_budget_cap": int(args.active_budget_cap),
        "prompt_budget_map": args.prompt_budget_map_resolved,
        "runs_per_prompt": int(args.runs),
        "proposal_impl": str(args.proposal_impl),
        "backend": str(args.backend),
        "hip_arch": str(args.hip_arch),
        "chain_attn_mode": str(args.chain_attn_mode),
        "graph_mode": str(args.graph_mode),
        "confidence_threshold": float(args.confidence_threshold),
        "small_batch_decode_threshold": int(args.small_batch_decode_threshold) if args.small_batch_decode_threshold is not None else None,
        "verify_gpu_accept": args.verify_gpu_accept,
        "acceptance_diagnostics": bool(args.acceptance_diagnostics),
        "ar_fallback_zero_streak": int(args.ar_fallback_zero_streak),
        "ar_fallback_tokens": int(args.ar_fallback_tokens),
        "ar_fallback_until_end": bool(args.ar_fallback_until_end),
        "dry_run": bool(args.dry_run),
        "results": results,
        "aggregate_by_budget": {} if args.dry_run else _aggregate_across_prompts(results),
        "selected_policy_aggregate": (
            {}
            if args.dry_run or not args.prompt_budget_map_resolved
            else _aggregate_selected_policy(
                results,
                args.prompt_budget_map_resolved,
                decode_tokens=int(args.decode_tokens),
            )
        ),
        "go_no_go_rule": {
            "beats_ar": "avg_visible_tokens_per_verify_cycle > cycle_cost_ar_tokens",
            "hits_1p5x": "avg_visible_tokens_per_verify_cycle / cycle_cost_ar_tokens >= 1.5",
            "final_confirmation": "Use >=3 runs per prompt/budget before promoting any performance claim.",
        },
    }
    args.out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
