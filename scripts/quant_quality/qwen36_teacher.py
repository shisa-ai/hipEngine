#!/usr/bin/env python3
"""Portable BF16-teacher quantization-quality suite for Qwen3.6-35B-A3B.

Large ``.npy`` logit caches stay in ``--output-dir`` and are not repository
artifacts.  The compact JSON emitted by ``compare`` is suitable for
``benchmarks/results/``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.quant_quality.metrics import compare_logits, per_row_metrics


DEFAULT_BF16 = "/home/lhl/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
DEFAULT_PROMPTS = "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
PROTOCOL_ID = "qwen36-bf16-teacher-mtpbench-v1"
TEACHER_STEPS = 9
HELDOUT_PROMPT_IDS = frozenset(
    {
        "code_markdown_table",
        "general_en_explain",
        "general_ja_explain",
        "mixed_ja_en_review",
    }
)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text())
    if fixture.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"unsupported fixture protocol: {fixture.get('protocol_id')!r}")
    if int(fixture.get("teacher_steps", 0)) != TEACHER_STEPS:
        raise ValueError("fixture teacher-step count does not match this harness")
    return fixture


def _fixture_rows(fixture: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    groups: list[str] = []
    for prompt in fixture["prompts"]:
        teacher = prompt["teacher_token_ids"]
        if len(teacher) != TEACHER_STEPS:
            raise ValueError(f"prompt {prompt['id']} has the wrong teacher length")
        labels.extend(int(token) for token in teacher)
        groups.extend([str(prompt["category"])] * TEACHER_STEPS)
    return np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=str)


def _fixture_row_scopes(fixture: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    prompt_ids: list[str] = []
    splits: list[str] = []
    for prompt in fixture["prompts"]:
        prompt_id = str(prompt["id"])
        prompt_ids.extend([prompt_id] * TEACHER_STEPS)
        split = "heldout" if prompt_id in HELDOUT_PROMPT_IDS else "train"
        splits.extend([split] * TEACHER_STEPS)
    return np.asarray(prompt_ids, dtype=str), np.asarray(splits, dtype=str)


def _write_llama_input(path: Path, fixture: dict[str, Any]) -> None:
    with path.open("wb") as handle:
        handle.write(b"Q36Q")
        handle.write(struct.pack("<II", 1, len(fixture["prompts"])))
        for prompt in fixture["prompts"]:
            prompt_ids = np.asarray(prompt["prompt_token_ids"], dtype="<i4")
            teacher_ids = np.asarray(prompt["teacher_token_ids"], dtype="<i4")
            handle.write(struct.pack("<II", prompt_ids.size, teacher_ids.size))
            handle.write(prompt_ids.tobytes())
            handle.write(teacher_ids.tobytes())


def _cache_manifest(
    *,
    name: str,
    runtime: str,
    model_path: str,
    fixture_path: Path,
    logits_path: Path,
    shape: tuple[int, int],
    dtype: str,
    elapsed_seconds: float,
    model_sha256: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": 1,
        "kind": "quant_quality_full_logits_cache",
        "protocol_id": PROTOCOL_ID,
        "name": name,
        "runtime": runtime,
        "model_path": model_path,
        "model_bytes": Path(model_path).stat().st_size if Path(model_path).is_file() else None,
        "model_sha256": model_sha256,
        "fixture_path": str(fixture_path),
        "fixture_sha256": _sha256(fixture_path),
        "logits_path": str(logits_path),
        "logits_sha256": _sha256(logits_path),
        "shape": list(shape),
        "dtype": dtype,
        "elapsed_seconds_diagnostic": elapsed_seconds,
        "host": platform.node(),
    }
    if extra:
        payload.update(extra)
    return payload


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    return rendered, [int(token) for token in ids]


def capture_bf16(args: argparse.Namespace) -> int:
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration

    prompt_path = Path(args.prompts).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / "fixture.json"
    logits_path = output_dir / "bf16.npy"
    llama_input_path = output_dir / "llama_input.bin"
    manifest_path = output_dir / "bf16.manifest.json"

    source_rows = [json.loads(line) for line in prompt_path.read_text().splitlines() if line.strip()]
    if len(source_rows) != 10:
        raise ValueError(f"portable suite requires all 10 mtp-bench prompts; found {len(source_rows)}")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    rendered_prompts = []
    for row in source_rows:
        rendered, prompt_ids = _render_prompt(tokenizer, row["messages"])
        rendered_prompts.append((row, rendered, prompt_ids))

    print(f"loading BF16 reference from {args.model}", flush=True)
    started = time.perf_counter()
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    vocab_size = int(model.config.get_text_config().vocab_size)
    n_rows = len(rendered_prompts) * TEACHER_STEPS
    cache = np.lib.format.open_memmap(
        logits_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_rows, vocab_size),
    )

    fixture_prompts: list[dict[str, Any]] = []
    row_index = 0
    with torch.inference_mode():
        for prompt_index, (row, rendered, prompt_ids) in enumerate(rendered_prompts):
            current = torch.tensor([prompt_ids], dtype=torch.long)
            past = None
            teacher: list[int] = []
            for step in range(TEACHER_STEPS):
                output = model(input_ids=current, past_key_values=past, use_cache=True)
                logits = output.logits[0, -1].float().cpu().numpy()
                if logits.shape != (vocab_size,) or not np.isfinite(logits).all():
                    raise RuntimeError(f"non-finite or malformed BF16 logits at {row['id']} step {step}")
                cache[row_index] = logits.astype(np.float16)
                token_id = int(np.argmax(logits))
                teacher.append(token_id)
                row_index += 1
                past = output.past_key_values
                current = torch.tensor([[token_id]], dtype=torch.long)
                del output, logits
            fixture_prompts.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "split": "heldout" if row["id"] in HELDOUT_PROMPT_IDS else "train",
                    "messages": row["messages"],
                    "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "prompt_token_ids": prompt_ids,
                    "prompt_token_ids_sha256": _sha256_json(prompt_ids),
                    "teacher_token_ids": teacher,
                    "teacher_text": tokenizer.decode(teacher, skip_special_tokens=False),
                }
            )
            del past, current
            gc.collect()
            print(
                f"BF16 {prompt_index + 1:2d}/{len(rendered_prompts)} {row['id']}: {teacher}",
                flush=True,
            )
    cache.flush()
    del cache, model
    gc.collect()

    fixture = {
        "schema": 1,
        "kind": "quant_quality_teacher_fixture",
        "protocol_id": PROTOCOL_ID,
        "teacher_steps": TEACHER_STEPS,
        "reference_model": args.model,
        "reference_revision": Path(args.model).name,
        "prompt_source": str(prompt_path),
        "prompt_source_sha256": _sha256(prompt_path),
        "rendering": "official HF chat template; add_generation_prompt=true; enable_thinking=false",
        "reference_scoring_execution": "autoregressive prefill plus cached one-token steps",
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": vocab_size,
        "prompts": fixture_prompts,
    }
    _json_dump(fixture_path, fixture)
    _write_llama_input(llama_input_path, fixture)
    elapsed = time.perf_counter() - started
    manifest = _cache_manifest(
        name="Original BF16 HF",
        runtime=f"Transformers {transformers.__version__} / PyTorch {torch.__version__} CPU",
        model_path=args.model,
        fixture_path=fixture_path,
        logits_path=logits_path,
        shape=(n_rows, vocab_size),
        dtype="float16",
        elapsed_seconds=elapsed,
        extra={"role": "reference", "teacher_generation": "greedy argmax"},
    )
    _json_dump(manifest_path, manifest)
    print(json.dumps({"fixture": str(fixture_path), "manifest": str(manifest_path), "rows": n_rows}))
    return 0


def capture_transformers_paro(args: argparse.Namespace) -> int:
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText

    try:
        import paroquant.inference.backends.transformers.quantizer  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "ParoQuant registration failed; add the paroquant repository to PYTHONPATH"
        ) from exc

    fixture_path = Path(args.fixture).resolve()
    fixture = _load_fixture(fixture_path)
    labels, _ = _fixture_rows(fixture)
    vocab_size = int(fixture["vocab_size"])
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(labels.size, vocab_size),
    )
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    ).eval()
    model.to(args.device)

    row_index = 0
    with torch.inference_mode():
        for prompt_index, prompt in enumerate(fixture["prompts"]):
            prompt_ids = [int(token) for token in prompt["prompt_token_ids"]]
            teacher = [int(token) for token in prompt["teacher_token_ids"]]
            sequence = prompt_ids + teacher[:-1]
            output = model(
                input_ids=torch.tensor([sequence], dtype=torch.long, device=args.device),
                use_cache=False,
            )
            rows = output.logits[
                0,
                len(prompt_ids) - 1 : len(prompt_ids) - 1 + TEACHER_STEPS,
            ].float().cpu().numpy()
            if rows.shape != (TEACHER_STEPS, vocab_size) or not np.isfinite(rows).all():
                raise RuntimeError(f"non-finite or malformed PARO logits at {prompt['id']}")
            cache[row_index : row_index + TEACHER_STEPS] = rows
            row_index += TEACHER_STEPS
            cache.flush()
            print(
                f"Transformers PARO {prompt_index + 1:2d}/{len(fixture['prompts'])} {prompt['id']}",
                flush=True,
            )
            del output, rows
    cache.flush()
    del cache, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = _cache_manifest(
        name=args.name,
        runtime=f"ParoQuant Transformers {transformers.__version__} / PyTorch {torch.__version__} {args.device}",
        model_path=args.model,
        fixture_path=fixture_path,
        logits_path=output_path,
        shape=(labels.size, vocab_size),
        dtype="float32",
        elapsed_seconds=time.perf_counter() - started,
        model_sha256=args.model_sha256,
        extra={"role": "candidate", "teacher_forced": True},
    )
    _json_dump(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "rows": int(labels.size)}))
    return 0


def _capture_hipengine_common(args: argparse.Namespace, *, paro: bool) -> int:
    from hipengine.core.memory import copy_device_to_host, host_array_ptr
    from hipengine.runtime.prefill import PrefillConfig

    fixture_path = Path(args.fixture).resolve()
    fixture = _load_fixture(fixture_path)
    labels, _ = _fixture_rows(fixture)
    vocab_size = int(fixture["vocab_size"])
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(labels.size, vocab_size),
    )
    max_sequence_length = max(
        len(prompt["prompt_token_ids"]) + TEACHER_STEPS for prompt in fixture["prompts"]
    )
    prefill_config = PrefillConfig(
        linear_chunk_size=args.prefill_chunk_size,
        full_attn_query_chunk_size=args.prefill_chunk_size,
        full_attn_post_chunk_size=args.prefill_chunk_size,
        full_attn_rope_chunk_size=args.prefill_chunk_size,
        moe_chunk_size=args.prefill_chunk_size,
        auto_tune_chunk_sizes=False,
    )
    started = time.perf_counter()
    row_index = 0

    if paro:
        from hipengine.runtime.qwen35_paro_runner import (
            Qwen35ParoNextTokenRunner,
            Qwen35ParoResidentSession,
        )

        runner = Qwen35ParoNextTokenRunner(
            args.model,
            backend=args.backend,
            shared_expert_format="packed_paro_w4",
        )
        session: Any = Qwen35ParoResidentSession(
            runner,
            max_sequence_length=max_sequence_length,
            max_layers=0,
            prefill_config=prefill_config,
        )
        runtime_name = "hipEngine PARO packed W4"
    else:
        from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

        session = Qwen35GGUFResidentSession(
            args.model,
            backend=args.backend,
            max_sequence_length=max_sequence_length,
            prefill_config=prefill_config,
        )
        runtime_name = "hipEngine GGUF Q4_K_M"

    def copy_paro_logits() -> np.ndarray:
        session.runtime.device_synchronize()
        values = np.empty((vocab_size,), dtype=np.float32)
        copy_device_to_host(host_array_ptr(values), session.lm_logits, runtime=session.runtime)
        return values

    try:
        for prompt_index, prompt in enumerate(fixture["prompts"]):
            prompt_ids = [int(token) for token in prompt["prompt_token_ids"]]
            teacher = [int(token) for token in prompt["teacher_token_ids"]]
            session.reset()
            if paro:
                result = session.prefill_native(prompt_ids, sample=True)
                if result is None:
                    raise RuntimeError("PARO prefill returned no sampled row")
                values = copy_paro_logits()
                if int(np.argmax(values)) != int(result.token_id):
                    raise RuntimeError("PARO copied logits do not match the sampled token")
                cache[row_index] = values
            else:
                result = session.prefill(
                    prompt_ids,
                    use_bulk=True,
                    bulk_attention_mode="bulk",
                    return_logits=True,
                )
                if int(np.argmax(result.logits)) != int(result.token_id):
                    raise RuntimeError("GGUF returned logits do not match the sampled token")
                cache[row_index] = result.logits
            row_index += 1
            for step, token_id in enumerate(teacher[:-1]):
                if paro:
                    result = session.step(token_id, position=len(prompt_ids) + step, sample=True)
                    if result is None:
                        raise RuntimeError("PARO step returned no sampled row")
                    values = copy_paro_logits()
                    if int(np.argmax(values)) != int(result.token_id):
                        raise RuntimeError("PARO copied logits do not match the sampled token")
                    cache[row_index] = values
                else:
                    result = session.step(
                        token_id,
                        position=len(prompt_ids) + step,
                        return_logits=True,
                    )
                    if int(np.argmax(result.logits)) != int(result.token_id):
                        raise RuntimeError("GGUF returned logits do not match the sampled token")
                    cache[row_index] = result.logits
                row_index += 1
            cache.flush()
            print(
                f"{runtime_name} {prompt_index + 1:2d}/{len(fixture['prompts'])} {prompt['id']}",
                flush=True,
            )
    finally:
        session.close()
    cache.flush()
    del cache

    elapsed = time.perf_counter() - started
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = _cache_manifest(
        name=args.name,
        runtime=f"{runtime_name} ({args.backend})",
        model_path=args.model,
        fixture_path=fixture_path,
        logits_path=output_path,
        shape=(labels.size, vocab_size),
        dtype="float32",
        elapsed_seconds=elapsed,
        model_sha256=args.model_sha256,
        extra={"role": "candidate", "teacher_forced": True},
    )
    _json_dump(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "rows": int(labels.size)}))
    return 0


def register_raw(args: argparse.Namespace) -> int:
    fixture_path = Path(args.fixture).resolve()
    fixture = _load_fixture(fixture_path)
    labels, _ = _fixture_rows(fixture)
    shape = (int(labels.size), int(fixture["vocab_size"]))
    raw_path = Path(args.raw).resolve()
    raw_dtype = np.dtype(args.raw_dtype)
    expected = int(np.prod(shape)) * raw_dtype.itemsize
    if raw_path.stat().st_size != expected:
        raise ValueError(f"raw logits size is {raw_path.stat().st_size}, expected {expected}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw = np.memmap(raw_path, mode="r", dtype=raw_dtype, shape=shape)
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=raw_dtype, shape=shape)
    output[:] = raw
    output.flush()
    del raw, output
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = _cache_manifest(
        name=args.name,
        runtime=args.runtime,
        model_path=args.model,
        fixture_path=fixture_path,
        logits_path=output_path,
        shape=shape,
        dtype=raw_dtype.name,
        elapsed_seconds=time.perf_counter() - started,
        model_sha256=args.model_sha256,
        extra={"role": "candidate", "teacher_forced": True, "raw_capture": str(raw_path)},
    )
    _json_dump(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "shape": list(shape)}))
    return 0


def compare(args: argparse.Namespace) -> int:
    fixture_path = Path(args.fixture).resolve()
    fixture = _load_fixture(fixture_path)
    labels, groups = _fixture_rows(fixture)
    reference_manifest = json.loads(Path(args.reference_manifest).read_text())
    candidate_manifest = json.loads(Path(args.candidate_manifest).read_text())
    fixture_sha = _sha256(fixture_path)
    for manifest in (reference_manifest, candidate_manifest):
        if manifest.get("fixture_sha256") != fixture_sha:
            raise ValueError(f"manifest {manifest.get('name')} does not match the fixture")
    if reference_manifest.get("shape") != candidate_manifest.get("shape"):
        raise ValueError("reference and candidate manifest shapes differ")

    reference = np.load(reference_manifest["logits_path"], mmap_mode="r", allow_pickle=False)
    candidate = np.load(candidate_manifest["logits_path"], mmap_mode="r", allow_pickle=False)
    metrics = compare_logits(reference, candidate, labels, groups=groups, top_k=args.top_k)
    prompt_ids, splits = _fixture_row_scopes(fixture)
    metrics["by_split"] = {
        split: compare_logits(
            reference[splits == split],
            candidate[splits == split],
            labels[splits == split],
            top_k=args.top_k,
        )
        for split in sorted(np.unique(splits))
    }
    metrics["by_prompt"] = {
        prompt_id: compare_logits(
            reference[prompt_ids == prompt_id],
            candidate[prompt_ids == prompt_id],
            labels[prompt_ids == prompt_id],
            top_k=args.top_k,
        )
        for prompt_id in [str(prompt["id"]) for prompt in fixture["prompts"]]
    }
    payload = {
        "schema": 1,
        "kind": "qwen36_quant_quality_teacher_comparison",
        "protocol_id": PROTOCOL_ID,
        "scope": "portable ten-prompt BF16-teacher trajectory; not canonical held-out-corpus PPL",
        "fixture_sha256": fixture_sha,
        "prompt_source_sha256": fixture["prompt_source_sha256"],
        "teacher_steps_per_prompt": TEACHER_STEPS,
        "prompt_count": len(fixture["prompts"]),
        "reference": reference_manifest,
        "candidate": candidate_manifest,
        "metrics": metrics,
    }
    output_path = Path(args.output)
    _json_dump(output_path, payload)
    print(json.dumps(metrics, indent=2))
    return 0


def _load_aligned_cache(manifest_path: str, fixture_sha256: str) -> tuple[dict[str, Any], np.ndarray]:
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("fixture_sha256") != fixture_sha256:
        raise ValueError(f"manifest {manifest.get('name')} does not match the fixture")
    logits = np.load(manifest["logits_path"], mmap_mode="r", allow_pickle=False)
    return manifest, logits


def noninferiority(args: argparse.Namespace) -> int:
    fixture_path = Path(args.fixture).resolve()
    fixture = _load_fixture(fixture_path)
    fixture_sha = _sha256(fixture_path)
    labels, groups = _fixture_rows(fixture)
    prompt_ids, _ = _fixture_row_scopes(fixture)
    reference_manifest, reference = _load_aligned_cache(args.reference_manifest, fixture_sha)
    baseline_manifest, baseline = _load_aligned_cache(args.baseline_manifest, fixture_sha)
    candidate_manifest, candidate = _load_aligned_cache(args.candidate_manifest, fixture_sha)
    if reference.shape != baseline.shape or reference.shape != candidate.shape:
        raise ValueError("reference, baseline, and candidate logits shapes differ")

    baseline_rows = per_row_metrics(reference, baseline, labels, top_k=args.top_k)
    candidate_rows = per_row_metrics(reference, candidate, labels, top_k=args.top_k)
    ordered_prompt_ids = [str(prompt["id"]) for prompt in fixture["prompts"]]
    prompt_indices = [np.flatnonzero(prompt_ids == prompt_id) for prompt_id in ordered_prompt_ids]
    if any(index.size != TEACHER_STEPS for index in prompt_indices):
        raise ValueError("paired bootstrap requires exactly nine rows per prompt")

    def prompt_means(rows: dict[str, np.ndarray], key: str) -> np.ndarray:
        return np.asarray([rows[key][index].mean() for index in prompt_indices], dtype=np.float64)

    baseline_kl = prompt_means(baseline_rows, "kl_nats")
    candidate_kl = prompt_means(candidate_rows, "kl_nats")
    baseline_top1 = prompt_means(baseline_rows, "top1_equal")
    candidate_top1 = prompt_means(candidate_rows, "top1_equal")
    ref_nll = prompt_means(candidate_rows, "reference_teacher_nll_nats")
    baseline_nll = prompt_means(baseline_rows, "teacher_nll_nats")
    candidate_nll = prompt_means(candidate_rows, "teacher_nll_nats")

    rng = np.random.default_rng(args.bootstrap_seed)
    samples = rng.integers(0, len(prompt_indices), size=(args.bootstrap_samples, len(prompt_indices)))
    kl_delta = (candidate_kl[samples] - baseline_kl[samples]).mean(axis=1)
    top1_delta_pp = 100.0 * (candidate_top1[samples] - baseline_top1[samples]).mean(axis=1)
    reference_sample_nll = ref_nll[samples].mean(axis=1)
    baseline_ppl_ratio = np.exp(baseline_nll[samples].mean(axis=1) - reference_sample_nll)
    candidate_ppl_ratio = np.exp(candidate_nll[samples].mean(axis=1) - reference_sample_nll)
    ppl_ratio_delta = candidate_ppl_ratio - baseline_ppl_ratio

    def interval(values: np.ndarray) -> dict[str, float]:
        low, median, high = np.percentile(values, (2.5, 50.0, 97.5))
        return {"low_95": float(low), "median": float(median), "high_95": float(high)}

    categories: dict[str, Any] = {}
    category_veto = False
    for category in sorted(np.unique(groups)):
        mask = groups == category
        baseline_summary = compare_logits(reference[mask], baseline[mask], labels[mask], top_k=args.top_k)
        candidate_summary = compare_logits(reference[mask], candidate[mask], labels[mask], top_k=args.top_k)
        delta = {
            "mean_kl_nats": candidate_summary["mean_kl_nats"] - baseline_summary["mean_kl_nats"],
            "top1_agreement_pp": candidate_summary["top1_agreement_pct"] - baseline_summary["top1_agreement_pct"],
        }
        veto = bool(
            delta["mean_kl_nats"] > args.mean_kl_margin
            or delta["top1_agreement_pp"] < -args.top1_margin_pp
        )
        category_veto |= veto
        categories[str(category)] = {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "candidate_minus_baseline": delta,
            "point_margin_veto": veto,
        }

    intervals = {
        "mean_kl_candidate_minus_baseline_nats": interval(kl_delta),
        "top1_candidate_minus_baseline_pp": interval(top1_delta_pp),
        "teacher_ppl_ratio_candidate_minus_baseline": interval(ppl_ratio_delta),
    }
    gates = {
        "mean_kl_noninferior": intervals["mean_kl_candidate_minus_baseline_nats"]["high_95"] <= args.mean_kl_margin,
        "top1_noninferior": intervals["top1_candidate_minus_baseline_pp"]["low_95"] >= -args.top1_margin_pp,
        "teacher_ppl_ratio_noninferior_diagnostic": intervals["teacher_ppl_ratio_candidate_minus_baseline"]["high_95"] <= args.ppl_ratio_margin,
        "category_point_margin_veto": category_veto,
    }
    gates["portable_q4_equivalent"] = bool(
        gates["mean_kl_noninferior"]
        and gates["top1_noninferior"]
        and gates["teacher_ppl_ratio_noninferior_diagnostic"]
        and not category_veto
    )
    payload = {
        "schema": 1,
        "kind": "qwen36_quant_quality_paired_prompt_bootstrap",
        "protocol_id": PROTOCOL_ID,
        "fixture_sha256": fixture_sha,
        "bootstrap": {
            "unit": "prompt block (nine teacher rows)",
            "prompt_count": len(prompt_indices),
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
        },
        "margins": {
            "mean_kl_candidate_minus_q4_max_nats": args.mean_kl_margin,
            "top1_candidate_minus_q4_min_pp": -args.top1_margin_pp,
            "teacher_ppl_ratio_candidate_minus_q4_max": args.ppl_ratio_margin,
            "teacher_ppl_note": "portable BF16-teacher trajectory diagnostic; not canonical held-out-corpus PPL",
        },
        "reference": reference_manifest,
        "q4_baseline": baseline_manifest,
        "candidate": candidate_manifest,
        "confidence_intervals": intervals,
        "categories": categories,
        "gates": gates,
        "verdict": "q4-equivalent" if gates["portable_q4_equivalent"] else "quality-traded",
    }
    _json_dump(Path(args.output), payload)
    print(json.dumps({"confidence_intervals": intervals, "gates": gates, "verdict": payload["verdict"]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bf16 = sub.add_parser("capture-bf16", help="render prompts and capture BF16 teacher logits")
    bf16.add_argument("--model", default=DEFAULT_BF16)
    bf16.add_argument("--prompts", default=DEFAULT_PROMPTS)
    bf16.add_argument("--output-dir", required=True)
    bf16.add_argument("--threads", type=int, default=16)
    bf16.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    bf16.set_defaults(func=capture_bf16)

    transformers_paro = sub.add_parser("capture-transformers-paro")
    transformers_paro.add_argument("--model", required=True)
    transformers_paro.add_argument("--fixture", required=True)
    transformers_paro.add_argument("--output", required=True)
    transformers_paro.add_argument("--name", default="PARO full8192 packed / Transformers")
    transformers_paro.add_argument("--model-sha256")
    transformers_paro.add_argument("--device", default="cuda:0")
    transformers_paro.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    transformers_paro.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    transformers_paro.set_defaults(func=capture_transformers_paro)

    for command, paro, default_name in (
        ("capture-hipengine-gguf", False, "GGUF UD-Q4_K_M / hipEngine"),
        ("capture-hipengine-paro", True, "PARO full8192 packed / hipEngine"),
    ):
        capture = sub.add_parser(command)
        capture.add_argument("--model", required=True)
        capture.add_argument("--fixture", required=True)
        capture.add_argument("--output", required=True)
        capture.add_argument("--name", default=default_name)
        capture.add_argument("--backend", default="hip_gfx1151")
        capture.add_argument(
            "--prefill-chunk-size",
            type=int,
            default=0,
            help="manual chunk size; 0 keeps the certified short-prompt unchunked path",
        )
        capture.add_argument("--model-sha256")
        capture.set_defaults(func=lambda ns, use_paro=paro: _capture_hipengine_common(ns, paro=use_paro))

    raw = sub.add_parser("register-raw", help="convert a llama-logits raw file into a cache")
    raw.add_argument("--fixture", required=True)
    raw.add_argument("--raw", required=True)
    raw.add_argument("--raw-dtype", default="float32", choices=("float16", "float32"))
    raw.add_argument("--output", required=True)
    raw.add_argument("--name", required=True)
    raw.add_argument("--runtime", required=True)
    raw.add_argument("--model", required=True)
    raw.add_argument("--model-sha256")
    raw.set_defaults(func=register_raw)

    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--fixture", required=True)
    compare_parser.add_argument("--reference-manifest", required=True)
    compare_parser.add_argument("--candidate-manifest", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--top-k", type=int, default=5)
    compare_parser.set_defaults(func=compare)

    bootstrap = sub.add_parser("noninferiority", help="paired prompt-block bootstrap vs Q4_K_M")
    bootstrap.add_argument("--fixture", required=True)
    bootstrap.add_argument("--reference-manifest", required=True)
    bootstrap.add_argument("--baseline-manifest", required=True, help="Q4_K_M baseline cache manifest")
    bootstrap.add_argument("--candidate-manifest", required=True)
    bootstrap.add_argument("--output", required=True)
    bootstrap.add_argument("--top-k", type=int, default=5)
    bootstrap.add_argument("--bootstrap-samples", type=int, default=10_000)
    bootstrap.add_argument("--bootstrap-seed", type=int, default=1234)
    bootstrap.add_argument("--mean-kl-margin", type=float, default=0.005)
    bootstrap.add_argument("--top1-margin-pp", type=float, default=2.0)
    bootstrap.add_argument("--ppl-ratio-margin", type=float, default=0.01)
    bootstrap.set_defaults(func=noninferiority)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
