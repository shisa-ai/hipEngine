#!/usr/bin/env python3
"""Cross-engine quality-gate comparator for the Qwen4Exp campaign.

Implements the G0-G4 characterization ladder of
docs/CROSS-ENGINE-PROTOCOL.md over the shared task battery (the committed
mtpbench suite plus the GDN category heldouts, 18 prompts). The hipEngine
strict profile is the G1/G2/G4 teacher; external engines are characterized,
never certified against hipEngine's internal calibrated envelope.

Battery text recipe: each prompt is rendered as ``[<id>]\\n<role>: <content>``
joined over messages with a trailing blank section, mirroring the canonical
fixture material recipe, with no chat template. The recipe string is recorded
in every artifact and must match across engines before scoring.

Modes:
  hipengine   in-process strict/production trajectories (teacher + self row)
  llamacpp    llama.cpp-family server, exact token ids, optional top-N logprobs
  gufo        OpenAI-completions server, raw text, no logprobs (G2 N/A)
  kyojin      OpenAI-completions server, raw text, return_token_ids + logprobs
  score       join engine artifacts against the strict-teacher artifact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import (  # noqa: E402
    _git_metadata,
    _host_metadata,
    _post_json,
    _terminate,
    _wait_for_health,
    sha256_path,
    token_ids_sha256,
)

BATTERY_SUITES = (
    ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
    ROOT / "benchmarks/prompts/gdn-prefill-category-heldouts.jsonl",
)
REQUIRED_CATEGORIES = frozenset({"code", "general_en", "general_ja", "mixed_ja_en"})
TEXT_RECIPE = "[<id>]\\n<role>: <content> sections, no chat template"
DEFAULT_STEPS = 32
DEFAULT_REPEATS = 2
TOP_K = 20


def battery_text(row: Mapping[str, Any]) -> str:
    sections = [
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in row["messages"]
    ]
    return f"[{row['id']}]\n" + "\n".join(sections) + "\n\n"


def load_battery(paths: Sequence[Path] = BATTERY_SUITES) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt_id = str(payload["id"])
            if prompt_id in seen:
                raise ValueError(f"duplicate prompt id {prompt_id!r} at {path}:{line_number}")
            seen.add(prompt_id)
            rows.append({**payload, "suite": str(path.resolve())})
    missing = REQUIRED_CATEGORIES - {str(row["category"]) for row in rows}
    if missing:
        raise ValueError(f"battery suites are missing categories: {sorted(missing)}")
    return rows


def battery_identity(paths: Sequence[Path] = BATTERY_SUITES) -> dict[str, Any]:
    return {
        "suites": [
            {"path": str(path.resolve()), "sha256": sha256_path(path)} for path in paths
        ],
        "text_recipe": TEXT_RECIPE,
        "steps": DEFAULT_STEPS,
    }


def _round_trip(tokenizer: Any, token_ids: Sequence[int]) -> bool:
    text = tokenizer.decode([int(token_id) for token_id in token_ids])
    return tokenizer.encode(text, add_special_tokens=False) == [int(t) for t in token_ids]


def _top20_from_logits(logits: np.ndarray, k: int = TOP_K) -> list[list[float]]:
    row = np.asarray(logits, dtype=np.float64).reshape(-1)
    order = np.argsort(-row)[:k]
    return [[int(index), float(row[index])] for index in order]


def _first_text_divergence(reference: str, candidate: str) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_tokenizer(model_root: Path) -> Any:
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.tokenization.gguf import Qwen4ExpGGUFTokenizer

    index = load_gguf_index(discover_gguf_files(model_root)[0])
    return Qwen4ExpGGUFTokenizer.from_gguf_info(index)


def _battery_prompt_records(tokenizer: Any, battery: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in battery:
        text = battery_text(row)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        records.append(
            {
                "id": str(row["id"]),
                "category": str(row["category"]),
                "suite": str(row["suite"]),
                "text": text,
                "prompt_tokens": len(token_ids),
                "prompt_token_ids": token_ids,
                "prompt_token_ids_sha256": token_ids_sha256(token_ids),
                "round_trip": _round_trip(tokenizer, token_ids),
            }
        )
    return records


def run_hipengine(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
    from hipengine.generation import register_builtin_generators
    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
    )
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models import resolve_model

    register_builtin_generators()
    battery = load_battery()
    tokenizer = _load_tokenizer(args.model_root)
    records = _battery_prompt_records(tokenizer, battery)
    if not all(record["round_trip"] for record in records):
        raise SystemExit("battery prompt failed tokenizer round-trip; refusing to run")

    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=args.quant or QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile(args.profile),
    )

    def factory() -> Qwen4ExpGGUFTextGenerator:
        return Qwen4ExpGGUFTextGenerator(
            model_path=args.model_root,
            weight_index=index,
            model_plugin=plugin,
            backend=QWEN4_EXP_BACKEND,
            max_sequence_length=int(args.max_sequence_length),
            prefill_chunk_size=int(args.prefill_chunk_size),
        )

    generator = resolved.construct_generator(factory)
    runner = generator.runner

    artifact: dict[str, Any] = {
        "schema": 1,
        "kind": "cross_engine_quality_run",
        "engine": args.engine_label,
        "surface": "hipengine_in_process",
        "profile": args.profile,
        "status": "running",
        "host": _host_metadata(),
        "source": _git_metadata(ROOT),
        "battery": battery_identity(),
        "protocol": {
            "steps": int(args.steps),
            "repeats": int(args.repeats),
            "temperature": 0.0,
            "chat_template": False,
            "free_trajectory": "fixed 32 greedy steps; EOS is fed back and not a stop",
            "topk_logprobs": TOP_K if args.capture_logprobs else None,
        },
        "prompts": [],
    }
    _write_json(args.output, artifact)
    try:
        for record in records:
            runs: list[dict[str, Any]] = []
            top20: list[list[list[float]]] | None = None
            for repetition in range(int(args.repeats)):
                runner.reset()
                result = runner.prefill([int(t) for t in record["prompt_token_ids"]])
                ids: list[int] = []
                positions: list[list[list[float]]] = []
                for index in range(int(args.steps)):
                    token = int(result.token_id)
                    ids.append(token)
                    if args.capture_logprobs and repetition == 0:
                        positions.append(_top20_from_logits(result.logits))
                    if index + 1 < int(args.steps):
                        result = runner.step(token)
                run = {"ids": ids, "ids_sha256": token_ids_sha256(ids)}
                if repetition == 0 and args.capture_logprobs:
                    run["top20"] = positions
                    top20 = positions
                runs.append(run)
            artifact["prompts"].append(
                {
                    "id": record["id"],
                    "category": record["category"],
                    "suite": record["suite"],
                    "prompt_tokens": record["prompt_tokens"],
                    "prompt_token_ids_sha256": record["prompt_token_ids_sha256"],
                    "text_sha256": hashlib.sha256(record["text"].encode("utf-8")).hexdigest(),
                    "round_trip": record["round_trip"],
                    "runs": runs,
                }
            )
            print(f"[{args.engine_label}] {record['id']} ids[0:4]={runs[0]['ids'][:4]}", flush=True)
        artifact["status"] = "completed"
    except Exception as exc:
        artifact["status"] = "failed"
        artifact["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _write_json(args.output, artifact)
    return artifact


def _server_artifact(
    args: argparse.Namespace, surface: str, extra_protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    battery = load_battery()
    tokenizer = _load_tokenizer(args.model_root)
    records = _battery_prompt_records(tokenizer, battery)
    command = [
        str(args.server_bin.resolve()),
        *args.server_arg,
        "--host",
        str(args.host),
        "--port",
        str(args.port),
    ]
    artifact: dict[str, Any] = {
        "schema": 1,
        "kind": "cross_engine_quality_run",
        "engine": args.engine_label,
        "surface": surface,
        "status": "running",
        "host": _host_metadata(),
        "source": _git_metadata(args.source_root) if args.source_root else None,
        "battery": battery_identity(),
        "server_binary": str(args.server_bin.resolve()),
        "server_binary_sha256": sha256_path(args.server_bin),
        "command": command,
        "protocol": {
            "steps": int(args.steps),
            "repeats": int(args.repeats),
            "temperature": 0.0,
            "chat_template": False,
            "ignore_eos": True,
            "free_trajectory": "fixed 32 greedy steps; EOS is fed back and not a stop",
            **extra_protocol,
        },
        "prompts": [],
    }
    return artifact, records, command


def _llamacpp_probs(response: Mapping[str, Any]) -> list[list[list[float]]] | None:
    """Per-position top-k logprobs from a llama.cpp /completion response.

    Returns [[ [token_id, logprob], ... ] per position] when the server emits
    them, else None. Parsing is deliberately tolerant across fork schemas:
    current llama.cpp serves `completion_probabilities[].top_logprobs` with
    `logprob` fields; older forks served `probs[].top_probs` with `prob`.
    """
    for container_key, entry_key in (("completion_probabilities", "top_logprobs"), ("probs", "top_probs")):
        container = response.get(container_key)
        if not isinstance(container, list):
            continue
        positions: list[list[list[float]]] = []
        for entry in container:
            if not isinstance(entry, Mapping):
                return None
            top = entry.get(entry_key)
            if not isinstance(top, list):
                return None
            positions.append(
                [
                    [
                        int(item["id"]),
                        float(item["logprob"])
                        if "logprob" in item
                        else float(np.log(max(float(item["prob"]), 1e-38))),
                    ]
                    for item in top
                    if isinstance(item, Mapping) and "id" in item and ("prob" in item or "logprob" in item)
                ]
            )
        if positions:
            return positions
    return None


def run_llamacpp(args: argparse.Namespace) -> dict[str, Any]:
    artifact, records, command = _server_artifact(
        args, "llamacpp_completion_server", {"ids_basis": "server_token_ids", "topk_logprobs": TOP_K}
    )
    _write_json(args.output, artifact)
    with (args.server_log).open("wb") as server_log:
        process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT, env=os.environ.copy())
        try:
            _wait_for_health(args.host, args.port, args.startup_timeout, process=process)
            for record in records:
                runs: list[dict[str, Any]] = []
                for repetition in range(int(args.repeats)):
                    payload = {
                        "prompt": [int(t) for t in record["prompt_token_ids"]],
                        "n_predict": int(args.steps),
                        "temperature": 0.0,
                        "top_k": 1,
                        "ignore_eos": True,
                        "return_tokens": True,
                        "n_probs": TOP_K,
                        "cache_prompt": False,
                        "stream": False,
                    }
                    started = time.perf_counter()
                    response = _post_json(args.host, args.port, "/completion", payload, args.request_timeout)
                    ids = [int(t) for t in response.get("tokens", [])]
                    if len(ids) != int(args.steps):
                        raise ValueError(
                            f"{args.engine_label} returned {len(ids)} tokens for {record['id']}; "
                            f"expected {args.steps}"
                        )
                    run: dict[str, Any] = {
                        "ids": ids,
                        "ids_sha256": token_ids_sha256(ids),
                        "client_wall_s": time.perf_counter() - started,
                    }
                    if repetition == 0:
                        probs = _llamacpp_probs(response)
                        if probs is not None:
                            run["top20"] = probs
                    runs.append(run)
                artifact["prompts"].append(
                    {
                        "id": record["id"],
                        "category": record["category"],
                        "suite": record["suite"],
                        "prompt_tokens": record["prompt_tokens"],
                        "prompt_token_ids_sha256": record["prompt_token_ids_sha256"],
                        "round_trip": record["round_trip"],
                        "runs": runs,
                    }
                )
                print(f"[{args.engine_label}] {record['id']} ids[0:4]={runs[0]['ids'][:4]}", flush=True)
            artifact["status"] = "completed"
        except Exception as exc:
            artifact["status"] = "failed"
            artifact["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _terminate(process)
            server_log.flush()
            artifact["server_returncode"] = process.returncode
            artifact["server_log"] = str(args.server_log)
            _write_json(args.output, artifact)
    return artifact


def _openai_completion_run(args: argparse.Namespace, adapter: str) -> dict[str, Any]:
    from scripts.qwen4exp_canonical_ar_bench import (
        OPENAI_COMPLETIONS_PATH,
        OPENAI_OPTIONAL_FIELDS,
        _openai_case_text,
        openai_completions_payload,
    )

    ids_basis = "server_return_token_ids" if adapter == "kyojin" else "text_retokenized"
    artifact, records, command = _server_artifact(
        args,
        f"openai_completions_{adapter}",
        {"ids_basis": ids_basis, "topk_logprobs": TOP_K if adapter == "kyojin" else None},
    )
    qualified = [record for record in records if record["round_trip"]]
    artifact["disqualified_prompts"] = [
        {"id": record["id"], "reason": "detokenize/retokenize round-trip mismatch"}
        for record in records
        if not record["round_trip"]
    ]
    _write_json(args.output, artifact)
    with (args.server_log).open("wb") as server_log:
        process = subprocess.Popen(
            command,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            cwd=args.server_cwd.resolve() if args.server_cwd else None,
        )
        try:
            _wait_for_health(args.host, args.port, args.startup_timeout, process=process)
            for record in qualified:
                runs: list[dict[str, Any]] = []
                for repetition in range(int(args.repeats)):
                    payload: dict[str, Any] = dict(
                        openai_completions_payload(adapter, record["text"], int(args.steps) - 1)
                    )
                    if adapter == "gufo":
                        payload["ignore_eos"] = True
                    started = time.perf_counter()
                    try:
                        response = _post_json(args.host, args.port, OPENAI_COMPLETIONS_PATH, payload, args.request_timeout)
                    except urllib.error.HTTPError as exc:
                        if exc.code != 400:
                            raise
                        body = exc.read().decode("utf-8", errors="replace")
                        if not any(field in body for field in OPENAI_OPTIONAL_FIELDS):
                            raise
                        minimal = {
                            key: value for key, value in payload.items()
                            if key in ("prompt", "max_tokens", "return_token_ids", "cache_prompt")
                        }
                        response = _post_json(args.host, args.port, OPENAI_COMPLETIONS_PATH, minimal, args.request_timeout)
                    ids: list[int] = []
                    text: str | None = None
                    raw_ids = response.get("token_ids")
                    if not isinstance(raw_ids, list):
                        choices = response.get("choices")
                        raw_ids = choices[0].get("token_ids") if isinstance(choices, list) and choices and isinstance(choices[0], Mapping) else None
                    if isinstance(raw_ids, list) and raw_ids:
                        ids = [int(t) for t in raw_ids]
                    else:
                        choices = response.get("choices")
                        if not isinstance(choices, list) or not choices:
                            raise ValueError(f"{args.engine_label} response omitted choices for {record['id']}")
                        text = str(choices[0].get("text") or "")
                    run: dict[str, Any] = {
                        "ids": ids,
                        "ids_sha256": token_ids_sha256(ids),
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None,
                        "text": text,
                        "client_wall_s": time.perf_counter() - started,
                    }
                    runs.append(run)
                artifact["prompts"].append(
                    {
                        "id": record["id"],
                        "category": record["category"],
                        "suite": record["suite"],
                        "prompt_tokens": record["prompt_tokens"],
                        "prompt_token_ids_sha256": record["prompt_token_ids_sha256"],
                        "round_trip": record["round_trip"],
                        "runs": runs,
                    }
                )
                print(f"[{args.engine_label}] {record['id']}", flush=True)
            artifact["status"] = "completed"
        except Exception as exc:
            artifact["status"] = "failed"
            artifact["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _terminate(process)
            server_log.flush()
            artifact["server_returncode"] = process.returncode
            artifact["server_log"] = str(args.server_log)
            _write_json(args.output, artifact)
    return artifact


def run_gufo(args: argparse.Namespace) -> dict[str, Any]:
    return _openai_completion_run(args, "gufo")


def run_kyojin(args: argparse.Namespace) -> dict[str, Any]:
    return _openai_completion_run(args, "kyojin")


def _first_divergence(reference: Sequence[int], candidate: Sequence[int]) -> int | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if int(left) != int(right):
            return index
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate))
    return None


def _truncated_kl(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> float:
    """KL over the union of two top-k lists, renormalized in float64.

    A token absent from one side's list carries that side's epsilon floor, not
    exp(0): absent tail mass is unknown, and assigning it the full weight of a
    present token would explode the divergence whenever the lists differ.
    """
    epsilon = 1e-12
    reference_p: dict[int, float] = {}
    candidate_p: dict[int, float] = {}
    for token_id, logprob in reference:
        reference_p[int(token_id)] = max(float(np.exp(float(logprob))), epsilon)
    for token_id, logprob in candidate:
        candidate_p[int(token_id)] = max(float(np.exp(float(logprob))), epsilon)
    for token_id in set(reference_p) | set(candidate_p):
        reference_p.setdefault(token_id, epsilon)
        candidate_p.setdefault(token_id, epsilon)
    reference_mass = sum(reference_p.values())
    candidate_mass = sum(candidate_p.values())
    divergence = 0.0
    for token_id in reference_p:
        p = reference_p[token_id] / reference_mass
        q = candidate_p[token_id] / candidate_mass
        if q > 0.0:
            divergence += float(q * (np.log(q) - np.log(p)))
    return divergence


def _aligned_topk_metrics(
    teacher_top20: Sequence[Sequence[Sequence[float]]],
    engine_top20: Sequence[Sequence[Sequence[float]]],
    teacher_ids: Sequence[int],
    engine_ids: Sequence[int],
) -> dict[str, Any] | None:
    aligned = 0
    top1_agreements = 0
    kl_values: list[float] = []
    for position in range(min(len(teacher_ids), len(engine_ids))):
        if int(teacher_ids[position]) != int(engine_ids[position]):
            break
        if position >= len(teacher_top20) or position >= len(engine_top20):
            break
        left = teacher_top20[position]
        right = engine_top20[position]
        if not left or not right:
            break
        aligned += 1
        if int(left[0][0]) == int(right[0][0]):
            top1_agreements += 1
        kl_values.append(_truncated_kl(left, right))
    if not kl_values:
        return {"aligned_positions": 0, "top1_agreement": None, "mean_kl": None}
    return {
        "aligned_positions": aligned,
        "top1_agreement": top1_agreements / aligned,
        "mean_kl": float(np.mean(kl_values)),
        "max_kl": float(np.max(kl_values)),
    }


def _engine_gate_verdicts(teacher: Mapping[str, Any], engine: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    teacher_prompts = {prompt["id"]: prompt for prompt in teacher["prompts"]}
    engine_prompts = {prompt["id"]: prompt for prompt in engine["prompts"]}
    missing = sorted(set(teacher_prompts) - set(engine_prompts))
    declared = {row.get("id") for row in engine.get("disqualified_prompts", [])}
    if [case_id for case_id in missing if case_id not in declared]:
        raise ValueError(
            f"engine {engine['engine']!r} is missing battery prompts {missing} without a declared reason"
        )
    g1_rows: dict[str, Any] = {}
    g2_positions: list[dict[str, Any]] = []
    g4_exact = 0
    for prompt_id in sorted(set(teacher_prompts) & set(engine_prompts)):
        teacher_run = teacher_prompts[prompt_id]["runs"][0]
        engine_first = engine_prompts[prompt_id]["runs"][0]
        teacher_ids = [int(t) for t in teacher_run["ids"]]
        engine_ids = [int(t) for t in engine_first["ids"]]
        ids_basis = engine.get("protocol", {}).get("ids_basis", "server_token_ids")
        topk = None
        if ids_basis == "text_retokenized":
            # Text-driven engines without a returned-id surface are compared on
            # decoded text: re-encoded ids are merge-unreliable at BPE
            # boundaries (the general_ja-p1024 lesson), so text is the honest
            # primary basis and ids stay auxiliary.
            if tokenizer is None:
                raise ValueError("scoring a text-basis engine requires --model-root")
            engine_text = engine_first.get("text")
            teacher_text = tokenizer.decode(teacher_ids)
            exact = engine_text is not None and engine_text == teacher_text
            first_divergence = None if exact else _first_text_divergence(teacher_text, engine_text or "")
        else:
            first_divergence = _first_divergence(teacher_ids, engine_ids)
            exact = first_divergence is None
            teacher_top = teacher_run.get("top20")
            engine_top = engine_first.get("top20")
            topk = (
                _aligned_topk_metrics(teacher_top, engine_top, teacher_ids, engine_ids)
                if teacher_top is not None and engine_top is not None
                else None
            )
        if topk is not None:
            g2_positions.append({"id": prompt_id, **topk})
        deterministic = all(
            run["ids_sha256"] == engine_first["ids_sha256"]
            or (run.get("text_sha256") is not None and run["text_sha256"] == engine_first.get("text_sha256"))
            for run in engine_prompts[prompt_id]["runs"][1:]
        )
        if exact:
            g4_exact += 1
        g1_rows[prompt_id] = {
            "exact": exact,
            "first_divergence": first_divergence,
            "comparison_basis": "decoded_text" if ids_basis == "text_retokenized" else "token_ids",
            "deterministic": deterministic,
        }
    aligned = [row for row in g2_positions if row["aligned_positions"] > 0]
    g2 = {
        "available": bool(g2_positions),
        "prompt_count": len(g2_positions),
        "top1_agreement_mean": float(np.mean([row["top1_agreement"] for row in aligned])) if aligned else None,
        "mean_kl": float(np.mean([row["mean_kl"] for row in aligned])) if aligned else None,
        "max_kl": max((row["max_kl"] for row in aligned), default=None),
        "per_prompt": g2_positions,
    }
    compared = len(g1_rows)
    return {
        "engine": engine["engine"],
        "surface": engine.get("surface"),
        "g0_conformance": {
            "battery_suites_match": engine.get("battery", {}).get("suites") == teacher.get("battery", {}).get("suites"),
            "text_recipe": engine.get("battery", {}).get("text_recipe"),
            "identity_present": bool(engine.get("server_binary_sha256") or engine.get("source")),
        },
        "g1_greedy_identity": {
            "prompt_count": compared,
            "exact_count": sum(1 for row in g1_rows.values() if row["exact"]),
            "first_divergences": {
                prompt_id: row["first_divergence"]
                for prompt_id, row in g1_rows.items()
                if not row["exact"]
            },
            "per_prompt": g1_rows,
        },
        "g2_distribution": g2 if g2["available"] else {"available": False, "reason": "engine exposes no logprobs"},
        "g3_determinism": {
            "prompt_count": compared,
            "all_repeats_exact": all(row["deterministic"] for row in g1_rows.values()),
        },
        "g4_task_battery": {
            "prompt_count": compared,
            "strict_exact_count": g4_exact,
            "verdict": "characterized" if compared == len(teacher_prompts) else "partial_coverage",
        },
    }


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    teacher = json.loads(Path(args.teacher).read_text(encoding="utf-8"))
    if teacher.get("status") != "completed":
        raise ValueError("teacher artifact is not completed")
    if teacher.get("profile") != "strict":
        raise ValueError("the G1/G2/G4 teacher must be a strict-profile hipEngine run")
    tokenizer = None
    if any(
        json.loads(Path(path).read_text(encoding="utf-8")).get("protocol", {}).get("ids_basis")
        == "text_retokenized"
        for path in args.engines
    ):
        tokenizer = _load_tokenizer(args.model_root)
    verdicts = []
    for path in args.engines:
        engine = json.loads(Path(path).read_text(encoding="utf-8"))
        if engine.get("status") != "completed":
            raise ValueError(f"engine artifact is not completed: {path}")
        verdicts.append(_engine_gate_verdicts(teacher, engine, tokenizer))
    artifact = {
        "schema": 1,
        "kind": "cross_engine_quality_score",
        "teacher": {
            "engine": teacher["engine"],
            "profile": teacher["profile"],
            "battery": teacher.get("battery"),
        },
        "engines": verdicts,
    }
    _write_json(args.output, artifact)
    return artifact


def _add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--engine-label", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    hip = subparsers.add_parser("hipengine", help="In-process strict/production battery run")
    hip.add_argument("--model-root", type=Path, required=True)
    hip.add_argument("--engine-label", required=True)
    hip.add_argument("--profile", choices=("strict", "production"), required=True)
    hip.add_argument("--quant")
    hip.add_argument("--max-sequence-length", type=int, default=8192)
    hip.add_argument("--prefill-chunk-size", type=int, default=1024)
    hip.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    hip.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    hip.add_argument("--capture-logprobs", action="store_true", default=True)
    hip.add_argument("--no-capture-logprobs", dest="capture_logprobs", action="store_false")
    hip.add_argument("--output", type=Path, required=True)

    for name, help_text in (
        ("llamacpp", "llama.cpp-family server battery run"),
        ("gufo", "gufo OpenAI-completions battery run"),
        ("kyojin", "kyojin OpenAI-completions battery run"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        _add_engine_arguments(sub)
        sub.add_argument("--server-bin", type=Path, required=True)
        sub.add_argument("--server-arg", action="append", default=[])
        sub.add_argument("--server-cwd", type=Path)

    score = subparsers.add_parser("score", help="Score engine artifacts against the strict teacher")
    score.add_argument("--teacher", type=Path, required=True)
    score.add_argument("--engines", nargs="+", type=Path, required=True)
    score.add_argument("--model-root", type=Path, help="GGUF root for the tokenizer; required when an engine is scored on the text basis")
    score.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "hipengine":
        artifact = run_hipengine(args)
    elif args.mode == "llamacpp":
        if not args.server_bin.is_file():
            raise SystemExit("server binary must exist")
        if args.server_log is None:
            raise SystemExit("--server-log is required")
        artifact = run_llamacpp(args)
    elif args.mode in ("gufo", "kyojin"):
        if not args.server_bin.is_file():
            raise SystemExit("server binary must exist")
        if args.server_log is None:
            raise SystemExit("--server-log is required")
        artifact = run_gufo(args) if args.mode == "gufo" else run_kyojin(args)
    else:
        artifact = run_score(args)
    print(json.dumps({"kind": artifact["kind"], "status": artifact.get("status", "completed"), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
