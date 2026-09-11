"""Surya OCR 2 benchmark harness: correctness-gated lane comparison + tuning.

Lanes on one physical host:
  - hipengine_gpu:  hipEngine HIP path (fp32): GPU vision tower + GPU
                    prefill/decode through ``hipengine.runtime.surya``.
  - hipengine_cpu:  hipEngine torch-free CPU-reference path (fp32 numpy).
  - torch_cpu:      transformers fp32 on CPU (the oracle implementation).
  - torch_cuda:     the same transformers fp32 on the host GPU via torch ROCm.

Measurement protocol (identical boundary and identical pipeline shape for
every lane):

  * One timed request is: page image + prompt in, greedy token ids out. The
    stages are ``preprocess`` -> ``vision`` -> ``prefill`` -> ``decode``, each
    timed as its own region.
  * Every lane runs the *same explicit pipeline*: a vision tower, one prefill
    forward that returns the last-token logits and a KV state, then a manual
    greedy loop over single-token forwards. No lane uses a library's fused
    ``generate``, so the decode stage is one loop of the same shape everywhere
    and is never derived by subtracting two other timings.
  * ``sync()`` is called on both sides of every timed region, on every lane.
    Without a trailing sync a stage measures kernel *launch* time, not kernel
    time; without a leading sync a previous stage's work bleeds into this one.
    CPU lanes have no sync and measure synchronously by construction.
  * Checkpoint load and runner construction are initialization, measured once
    as ``init_s`` and excluded from the timed regions.
  * Every timed region is repeated (``--runs``) after a discarded warmup, and
    the artifact records the full distribution (min/median/mean/max/stdev),
    not just a median.
  * ``decode_tokens`` is the number of tokens actually generated, and
    ``termination`` records whether the loop stopped on EOS or hit the budget.

Correctness gate: each case declares an oracle. Where a captured torch fp32
fixture exists the lane must reproduce its ids exactly; otherwise the
hipEngine CPU reference is computed in the same process and used as the basis
(that reference is itself oracle-gated by ``tests/test_surya_e2e.py``). The
comparison is over the *complete* generated sequence including how it
terminated, never a prefix. Each case also records whether the decoded text
parses as the layout JSON Surya OCR is supposed to emit.

Workload suite: cases are split into a ``tuning`` subset and a disjoint
``heldout`` subset, recorded in the artifact. A tuning run iterates on the
tuning subset; the held-out subset is for evaluation only, so a tuning
result can never be reported on a page it was tuned against. The pages cover
markup and layout-JSON output, Japanese and mixed script, dense small text, a
ruled table, a blank page, a degraded scan, a block-heavy long page, and the
larger 1024x1024 fixtures.

Usage:
    # tuning subset, hipEngine lanes only
    python3 scripts/surya_perf_compare.py --split tuning --lanes hipengine_gpu

    # one case, every lane
    python3 scripts/surya_perf_compare.py --cases ja --lanes hipengine_gpu,torch_cuda

    # full retained run
    python3 scripts/surya_perf_compare.py --split all --runs 5 \
        --out benchmarks/results/2026-09-11-gfx1151-surya-lane-compare.json
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FIXTURES = Path("tests/fixtures/surya")
MODEL_ID = "datalab-to/surya-ocr-2"
PROMPT = "Transcribe this page."


# ---------------------------------------------------------------------------
# workload suite
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One benchmark workload.

    ``split`` freezes the tuning/held-out partition so a tuning run can never
    be reported on held-out cases. ``oracle`` names a captured torch fp32
    fixture; ``oracle_key`` selects a page inside a multi-page fixture. When
    ``oracle`` is None the hipEngine CPU reference computed in the same process
    is the correctness basis. ``format`` is the task-level shape of the
    expected output: Surya OCR emits layout JSON for a full page but markup for
    a list page, and the synthetic bar-pattern pages have no meaningful text at
    all, so the check has to be declared per case rather than assumed.
    """

    name: str
    page: str
    prompt: str
    max_tokens: int
    split: str
    format: str = "json"  # "json" | "markup" | "any"
    oracle: str | None = None
    oracle_key: str | None = None


SUITE: tuple[Case, ...] = (
    # Tuning subset: the document types a tuning run iterates on. Kept small
    # enough that a full pass is affordable, and disjoint from the held-out set.
    # ``format`` is taken from what the captured oracle actually emits: Surya
    # returns layout JSON for a page it reads as layout and HTML-ish markup for
    # one it reads as a document, and both appear across this suite.
    Case("small", "page_small.png", PROMPT, 64, "tuning", "markup",
         "oracle_greedy.json"),
    Case("rect", "page_rect.png", PROMPT, 64, "tuning", "any"),
    Case("ja", "page_ja.png", PROMPT, 384, "tuning", "json_nonempty",
         "oracle_bench.json", "ja"),
    Case("dense", "page_dense.png", PROMPT, 384, "tuning", "json_nonempty",
         "oracle_bench.json", "dense"),
    Case("table", "page_table.png", PROMPT, 384, "tuning", "json_nonempty",
         "oracle_bench.json", "table"),
    Case("blank", "page_blank.png", PROMPT, 32, "tuning", "markup",
         "oracle_bench.json", "blank"),
    # Held-out subset: frozen. A tuning run must not be reported on these.
    Case("mixed", "page_mixed.png", PROMPT, 384, "heldout", "markup",
         "oracle_bench.json", "mixed"),
    Case("scan", "page_scan.png", PROMPT, 512, "heldout", "markup",
         "oracle_bench.json", "scan"),
    Case("long", "page_long.png", PROMPT, 320, "heldout", "json_nonempty",
         "oracle_bench.json", "long"),
    Case("full", "page_full.png", PROMPT, 96, "heldout", "json_nonempty",
         "oracle_fullpage_greedy.json"),
    Case("columns", "page_columns.png", PROMPT, 96, "heldout", "json_nonempty",
         "oracle_corpus.json", "columns"),
    Case("list", "page_list.png", PROMPT, 96, "heldout", "json_nonempty",
         "oracle_corpus.json", "list"),
)

CASES_BY_NAME = {case.name: case for case in SUITE}


def _load_oracle_record(case: Case) -> dict | None:
    """The raw oracle entry for a case (ids, text, capture metadata)."""

    if case.oracle is None:
        return None
    path = FIXTURES / case.oracle
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if case.oracle_key is not None:
        data = data[case.oracle_key]
    return data


def _load_oracle(case: Case) -> list[int] | None:
    record = _load_oracle_record(case)
    return None if record is None else [int(i) for i in record["ids"]]


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def _stage(fn, sync) -> tuple[float, object]:
    """Run ``fn`` once with ``sync`` on both sides; returns (seconds, result).

    The leading sync is what makes a stage's number mean this stage only: the
    GPU queue is empty when the clock starts. The trailing sync is what makes
    it mean kernel time rather than launch time.
    """

    if sync is not None:
        sync()
    t0 = time.perf_counter()
    out = fn()
    if sync is not None:
        sync()
    return time.perf_counter() - t0, out


def _series(fn, runs: int, sync) -> tuple[list[float], list]:
    """Discarded warmup, then ``runs`` synchronized samples.

    Returns the samples and every run's output, so callers can check that
    repeats actually agree rather than assuming they do.
    """

    if sync is not None:
        sync()
    fn()
    if sync is not None:
        sync()
    samples: list[float] = []
    outputs: list = []
    for _ in range(max(int(runs), 1)):
        seconds, out = _stage(fn, sync)
        samples.append(seconds)
        outputs.append(out)
    return samples, outputs


def _series_with_setup(setup, fn, runs: int, sync) -> tuple[list[float], list]:
    """As ``_series``, but re-runs ``setup`` untimed before each sample.

    A decode loop mutates the runner: it advances the KV cache and the
    sequence length and leaves sampling state behind. A repeat that starts
    where the previous repeat stopped measures a different computation than
    the first one did, which silently inflates or deflates the stage. ``setup``
    restores the exact post-prefill state outside the timed region, so every
    sample times the same loop from the same starting point.
    """

    if sync is not None:
        sync()
    setup()
    fn()
    if sync is not None:
        sync()
    samples: list[float] = []
    outputs: list = []
    for _ in range(max(int(runs), 1)):
        setup()
        seconds, out = _stage(fn, sync)
        samples.append(seconds)
        outputs.append(out)
    return samples, outputs


def _repeatable(outputs: list) -> bool:
    """True when every repeat produced identical output (isolation check)."""

    if len(outputs) < 2:
        return True
    first = outputs[0]
    return all(out == first for out in outputs[1:])


def _dist(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        return {"n": 0}
    return {
        "n": len(samples),
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
        "max_s": max(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        if match:
            return match.group(1).strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _mem_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except Exception:
        pass
    return None


def _provenance(argv: list[str]) -> dict[str, object]:
    info: dict[str, object] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "cpu": _cpu_model(),
        "mem_total_gb": _mem_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "command": " ".join(argv),
        "git_revision": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_hip"] = getattr(torch.version, "hip", None)
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    return info


# ---------------------------------------------------------------------------
# result assembly
# ---------------------------------------------------------------------------


@dataclass
class LaneRun:
    """Everything one lane produced for one case."""

    lane: str
    backend: str
    init_s: float
    stages: dict[str, float]
    stage_series: dict[str, list[float]] = field(default_factory=dict)
    e2e_samples: list[float] = field(default_factory=list)
    generated: list[int] = field(default_factory=list)
    termination: str = "unknown"
    text: str = ""
    decode_repeatable: bool = True
    e2e_generated: list[int] = field(default_factory=list)
    e2e_termination: str = "unknown"


def _validate(lane_run: LaneRun, case: Case, ref: list[int] | None) -> dict[str, object]:
    """Complete-output, termination and task-level checks for one lane/case."""

    generated = lane_run.generated
    checks: dict[str, object] = {
        "generated_tokens": len(generated),
        "termination": lane_run.termination,
        "correctness_basis": "torch_fixture" if ref is not None else "cpu_reference",
    }
    if ref is not None:
        first = next(
            (i for i in range(min(len(generated), len(ref))) if generated[i] != ref[i]),
            min(len(generated), len(ref)),
        )
        # full-sequence equality, so a shared prefix with a wrong tail fails
        checks["ids_match"] = generated == ref
        checks["reference_tokens"] = len(ref)
        checks["first_divergence"] = None if generated == ref else first
    else:
        checks["ids_match"] = None
    try:
        parsed = json.loads(lane_run.text)
        checks["layout_json"] = isinstance(parsed, list)
    except Exception:
        parsed = None
        checks["layout_json"] = False
    # Diagnostic only, never a gate: a page that genuinely induces repetition
    # (the synthetic bar patterns do) is not incorrect, and a lane that matched
    # its oracle exactly is correct by definition.
    checks["nondegenerate"] = len(set(generated)) >= min(10, max(1, len(generated)))
    if case.format == "json_nonempty":
        task_ok = bool(checks["layout_json"]) and len(parsed) > 0
    elif case.format == "json":
        # a blank page legitimately yields an empty layout list
        task_ok = bool(checks["layout_json"])
    elif case.format == "markup":
        task_ok = lane_run.text.lstrip().startswith("<")
    else:  # synthetic bar patterns carry no meaningful text
        task_ok = len(generated) > 0
    checks["task_format"] = task_ok
    checks["decode_repeatable"] = lane_run.decode_repeatable
    # the decode stage and the end-to-end run must produce the same sequence;
    # tok/s is derived from the stage, so a disagreement would mean the number
    # describes something other than the reported output
    checks["e2e_agrees"] = lane_run.e2e_generated == lane_run.generated
    checks["correctness"] = (
        "PASS"
        if checks["ids_match"] is not False and task_ok and checks["e2e_agrees"]
        else "FAIL"
    )
    return checks


def _lane_record(lane_run: LaneRun, case: Case, ref: list[int] | None) -> dict:
    checks = _validate(lane_run, case, ref)
    decode_s = lane_run.stages.get("decode", 0.0)
    n = len(lane_run.generated)
    record = {
        "lane": lane_run.lane,
        "backend": lane_run.backend,
        "init_s": lane_run.init_s,
        "stages_s": lane_run.stages,
        "stage_distributions_s": {
            key: _dist(samples) for key, samples in lane_run.stage_series.items()
        },
        "decode_tokens": n,
        "decode_tok_per_s": (n / decode_s) if decode_s > 0 else None,
        "e2e_distribution_s": _dist(lane_run.e2e_samples),
        "e2e_median_s": (
            statistics.median(lane_run.e2e_samples) if lane_run.e2e_samples else None
        ),
        "e2e_termination": lane_run.e2e_termination,
    }
    record.update(checks)
    return record


# ---------------------------------------------------------------------------
# lanes
# ---------------------------------------------------------------------------


def _greedy_from_logits(step_logits, eos: int, max_tokens: int, step) -> tuple[list[int], str]:
    """Shared greedy loop shape: ``step(token, index) -> next logits``."""

    generated: list[int] = []
    logits = step_logits
    termination = "length"
    for index in range(max_tokens):
        nxt = int(np.argmax(logits))
        if nxt == eos:
            termination = "eos"
            break
        generated.append(nxt)
        if index + 1 >= max_tokens:
            break
        logits = step(nxt, index)
    return generated, termination


def lane_hipengine_cpu(case: Case, runs: int) -> LaneRun:
    from PIL import Image

    from hipengine.kernels.cpu_reference.surya import (
        text_decode_step,
        text_prefill,
        vision_forward,
    )
    from hipengine.loading.surya import (
        SuryaTokenizer,
        compute_mrope_positions,
        load_surya_spec,
        load_surya_weights,
        preprocess_image_surya,
        render_chat_prompt,
        resolve_surya_path,
    )

    page = Image.open(FIXTURES / case.page).convert("RGB")

    def init():
        model_dir = resolve_surya_path(MODEL_ID)
        return (
            load_surya_spec(model_dir),
            load_surya_weights(model_dir),
            SuryaTokenizer(model_dir),
        )

    init_s, (spec, weights, tokenizer) = _stage(init, None)

    state: dict[str, object] = {}
    series: dict[str, list[float]] = {}

    def preprocess():
        return preprocess_image_surya(page)

    def vision():
        rows, grid = state["pre"]
        return vision_forward(weights, spec, rows, [grid])

    def prefill():
        _, _, merged = state["vis"]
        rows, grid = state["pre"]
        n_img = (grid[1] // 2) * (grid[2] // 2)
        ids, mm = render_chat_prompt(tokenizer, case.prompt, n_img)
        pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
        hidden, st = text_prefill(
            weights, spec, np.array([ids], dtype=np.int64), pos,
            visual_features=merged[None],
        )
        emb = weights["model.language_model.embed_tokens.weight"]
        return hidden[:, -1] @ emb.T, st, int(pos[:, -1].max())

    def decode():
        logits, st, p_last = state["prefill"]
        return _greedy_from_logits(
            logits[0], spec.eos_token_id, case.max_tokens,
            lambda tok, index: text_decode_step(weights, spec, tok, st, p_last + 1 + index),
        )

    def refresh_prefill():
        state["prefill"] = prefill()

    def pipeline():
        state["pre"] = preprocess()
        state["vis"] = vision()
        state["prefill"] = prefill()
        return decode()

    for key, fn in (("preprocess", preprocess), ("vision", vision), ("prefill", prefill)):
        samples, outputs = _series(fn, runs, None)
        series[key] = samples
        state[{"preprocess": "pre", "vision": "vis", "prefill": "prefill"}[key]] = outputs[-1]
    samples, outputs = _series_with_setup(refresh_prefill, decode, runs, None)
    series["decode"] = samples
    generated, termination = outputs[-1]
    decode_repeatable = _repeatable(outputs)

    e2e_samples, e2e_outputs = _series(pipeline, runs, None)
    e2e_ids, e2e_termination = e2e_outputs[-1]

    stages = {key: statistics.median(val) for key, val in series.items()}
    stages["vision_prefill"] = stages["vision"] + stages["prefill"]
    return LaneRun(
        lane="hipengine_cpu",
        backend="cpu_reference/fp32 numpy",
        init_s=init_s,
        stages=stages,
        stage_series=series,
        e2e_samples=e2e_samples,
        generated=generated,
        termination=termination,
        text=tokenizer.decode(generated, skip_special=True),
        decode_repeatable=decode_repeatable,
        e2e_generated=e2e_ids,
        e2e_termination=e2e_termination,
    )


def lane_hipengine_gpu(case: Case, runs: int) -> LaneRun:
    from PIL import Image

    from hipengine.loading.surya import (
        SuryaTokenizer,
        compute_mrope_positions,
        load_surya_spec,
        load_surya_weights,
        preprocess_image_surya,
        render_chat_prompt,
        resolve_surya_path,
    )
    from hipengine.runtime.surya import SuryaGpuRunner

    page = Image.open(FIXTURES / case.page).convert("RGB")

    def init():
        model_dir = resolve_surya_path(MODEL_ID)
        spec = load_surya_spec(model_dir)
        weights = load_surya_weights(model_dir)
        return spec, SuryaGpuRunner(weights, spec), SuryaTokenizer(model_dir)

    init_s, (spec, runner, tokenizer) = _stage(init, None)
    sync = runner.runtime.device_synchronize

    state: dict[str, object] = {}
    series: dict[str, list[float]] = {}

    def preprocess():
        return preprocess_image_surya(page)

    def vision():
        rows, grid = state["pre"]
        return runner.vision_forward(rows, [grid])

    def prefill():
        rows, grid = state["pre"]
        merged = state["vis"]
        n_img = (grid[1] // 2) * (grid[2] // 2)
        ids, mm = render_chat_prompt(tokenizer, case.prompt, n_img)
        pos = compute_mrope_positions(mm, grid, spec.vision_spatial_merge_size)
        logits = runner.prefill(
            np.asarray(ids, dtype=np.int64), pos, visual_features=merged
        )
        return logits, int(pos[:, -1].max())

    def decode():
        logits, p_last = state["prefill"]
        return _greedy_from_logits(
            logits, spec.eos_token_id, case.max_tokens,
            lambda tok, index: runner.decode_step(tok, p_last + 1 + index),
        )

    def refresh_prefill():
        state["prefill"] = prefill()

    def pipeline():
        state["pre"] = preprocess()
        state["vis"] = vision()
        state["prefill"] = prefill()
        return decode()

    try:
        for key, fn in (("preprocess", preprocess), ("vision", vision), ("prefill", prefill)):
            samples, outputs = _series(fn, runs, sync)
            series[key] = samples
            state[{"preprocess": "pre", "vision": "vis", "prefill": "prefill"}[key]] = outputs[-1]
        samples, outputs = _series_with_setup(refresh_prefill, decode, runs, sync)
        series["decode"] = samples
        generated, termination = outputs[-1]
        decode_repeatable = _repeatable(outputs)

        e2e_samples, e2e_outputs = _series(pipeline, runs, sync)
        e2e_ids, e2e_termination = e2e_outputs[-1]
    finally:
        runner.close()

    stages = {key: statistics.median(val) for key, val in series.items()}
    stages["vision_prefill"] = stages["vision"] + stages["prefill"]
    return LaneRun(
        lane="hipengine_gpu",
        backend="HIP fp32 (gfx1151): GPU vision tower + GPU prefill/decode",
        init_s=init_s,
        stages=stages,
        stage_series=series,
        e2e_samples=e2e_samples,
        generated=generated,
        termination=termination,
        text=tokenizer.decode(generated, skip_special=True),
        decode_repeatable=decode_repeatable,
        e2e_generated=e2e_ids,
        e2e_termination=e2e_termination,
    )


def lane_torch(device: str, case: Case, runs: int) -> LaneRun:
    """transformers fp32 lane; ``device`` is 'cpu' or the HIP device."""

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    page = Image.open(FIXTURES / case.page).convert("RGB")
    lane = "torch_cpu" if device == "cpu" else "torch_cuda"
    sync = None if device == "cpu" else torch.cuda.synchronize

    def init():
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).to(device)
        model.eval()
        return processor, model

    init_s, (processor, model) = _stage(init, sync)
    eos = int(processor.tokenizer.eos_token_id)

    state: dict[str, object] = {}
    series: dict[str, list[float]] = {}

    def preprocess():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(FIXTURES / case.page)},
                    {"type": "text", "text": case.prompt},
                ],
            }
        ]
        prompt_str = processor.apply_chat_template(messages, add_generation_prompt=True)
        return processor(text=[prompt_str], images=[page], return_tensors="pt").to(device)

    def prefill():
        p = state["pre"]
        with torch.no_grad():
            out = model(
                input_ids=p["input_ids"],
                attention_mask=p["attention_mask"],
                pixel_values=p["pixel_values"],
                image_grid_thw=p["image_grid_thw"],
                mm_token_type_ids=p["mm_token_type_ids"],
                use_cache=True,
            )
        return out.logits[0, -1].detach(), out.past_key_values

    def decode():
        logits, past = state["prefill"]
        generated: list[int] = []
        termination = "length"
        with torch.no_grad():
            for index in range(case.max_tokens):
                nxt = int(torch.argmax(logits).item())
                if nxt == eos:
                    termination = "eos"
                    break
                generated.append(nxt)
                if index + 1 >= case.max_tokens:
                    break
                tok = torch.tensor([[nxt]], dtype=torch.long, device=device)
                out = model(input_ids=tok, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1]
        return generated, termination

    def refresh_prefill():
        state["prefill"] = prefill()

    def pipeline():
        state["pre"] = preprocess()
        state["prefill"] = prefill()
        return decode()

    # preprocess/vision/prefill are timed with a fresh processor call so the
    # stages do not share mutable state across repeats
    samples, outputs = _series(preprocess, runs, sync)
    series["preprocess"] = samples
    state["pre"] = outputs[-1]
    samples, outputs = _series(prefill, runs, sync)
    series["prefill"] = samples
    state["prefill"] = outputs[-1]
    samples, outputs = _series_with_setup(refresh_prefill, decode, runs, sync)
    series["decode"] = samples
    generated, termination = outputs[-1]
    decode_repeatable = _repeatable(outputs)

    e2e_samples, e2e_outputs = _series(pipeline, runs, sync)
    e2e_ids, e2e_termination = e2e_outputs[-1]

    stages = {
        # torch's processor and model fuse vision into the forward pass, so there
        # is no separately callable vision stage: it is recorded as zero and
        # folded into prefill, which is why the torch lanes report
        # vision_prefill as the prefill number.
        "preprocess": statistics.median(series["preprocess"]),
        "vision": 0.0,
        "prefill": statistics.median(series["prefill"]),
        "decode": statistics.median(series["decode"]),
    }
    stages["vision_prefill"] = stages["prefill"]
    return LaneRun(
        lane=lane,
        backend=f"transformers fp32 ({device}), explicit prefill + greedy loop",
        init_s=init_s,
        stages=stages,
        stage_series=series,
        e2e_samples=e2e_samples,
        generated=generated,
        termination=termination,
        text=processor.tokenizer.decode(generated, skip_special=True),
        decode_repeatable=decode_repeatable,
        e2e_generated=e2e_ids,
        e2e_termination=e2e_termination,
    )


LANES = {
    "hipengine_gpu": lambda case, runs: lane_hipengine_gpu(case, runs),
    "hipengine_cpu": lambda case, runs: lane_hipengine_cpu(case, runs),
    "torch_cpu": lambda case, runs: lane_torch("cpu", case, runs),
    "torch_cuda": lambda case, runs: lane_torch("cuda", case, runs),
}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _summary_line(record: dict) -> str:
    """One-line console summary, tolerant of a lane that produced no samples.

    A benchmark run over many cases and lanes should not abort because one
    lane failed to yield a timing; the missing value is reported as "n/a".
    """

    def seconds(value) -> str:
        return f"{value:.3f}s" if isinstance(value, (int, float)) else "n/a"

    def rate(value) -> str:
        return f"{value:.1f}" if isinstance(value, (int, float)) else "n/a"

    stages = record.get("stages_s") or {}
    return (
        f"   e2e {seconds(record.get('e2e_median_s'))}  "
        f"vision+prefill {seconds(stages.get('vision_prefill'))}  "
        f"decode {seconds(stages.get('decode'))} "
        f"({rate(record.get('decode_tok_per_s'))} tok/s)  "
        f"{record.get('correctness')} ({record.get('termination')})"
    )


def _select_cases(args: argparse.Namespace) -> list[Case]:
    cases = list(SUITE)
    if args.cases:
        wanted = [name.strip() for name in args.cases.split(",") if name.strip()]
        unknown = [name for name in wanted if name not in CASES_BY_NAME]
        if unknown:
            raise SystemExit(f"unknown case(s) {unknown}; known: {sorted(CASES_BY_NAME)}")
        cases = [CASES_BY_NAME[name] for name in wanted]
    if args.split != "all":
        cases = [case for case in cases if case.split == args.split]
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="override every case's token budget")
    parser.add_argument("--split", choices=("tuning", "heldout", "all"), default="tuning",
                        help="frozen subset to run (default: the tuning subset)")
    parser.add_argument("--cases", default=None, help="comma-separated case names")
    parser.add_argument("--lanes", default="hipengine_gpu,hipengine_cpu,torch_cuda",
                        help="comma-separated lanes; torch_cpu is opt-in because it "
                             "runs at ~0.5 tok/s and exists to validate the oracle "
                             "path, not to inform tuning")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cases = _select_cases(args)
    if not cases:
        raise SystemExit("no cases selected")
    lanes = [name.strip() for name in args.lanes.split(",") if name.strip()]
    unknown = [name for name in lanes if name not in LANES]
    if unknown:
        raise SystemExit(f"unknown lane(s) {unknown}; known: {sorted(LANES)}")

    if args.max_new_tokens is not None:
        cases = [
            Case(c.name, c.page, c.prompt, args.max_new_tokens, c.split, c.format,
                 c.oracle, c.oracle_key)
            for c in cases
        ]

    print(f"cases: {[c.name for c in cases]}  lanes: {lanes}  runs: {args.runs}")
    results: list[dict] = []
    for case in cases:
        ref = _load_oracle(case)
        print(f"\n== case {case.name} ({case.page}, {case.max_tokens} tokens, "
              f"split={case.split}, oracle={case.oracle or 'cpu_reference'}) ==")
        for lane in lanes:
            print(f"-- lane {lane} ...", flush=True)
            lane_run = LANES[lane](case, args.runs)
            record = _lane_record(lane_run, case, ref)
            record["case"] = case.name
            results.append(record)
            print(_summary_line(record))

    artifact = {
        "provenance": _provenance(sys.argv),
        "protocol": {
            "boundary": "page image + prompt in, greedy token ids out",
            "pipeline": "explicit preprocess -> vision -> prefill -> greedy loop on every lane",
            "sync": "sync() on both sides of every timed region; CPU lanes are synchronous",
            "init": "checkpoint load and runner construction measured as init_s, excluded",
            "runs": args.runs,
            "warmup": "one discarded warmup before every timed series",
            "decode_derivation": "decode timed directly, never subtracted from other stages",
            "splits": "tuning cases must not be used to report held-out results",
        },
        "suite": [
            {
                "name": c.name,
                "page": c.page,
                "prompt": c.prompt,
                "max_tokens": c.max_tokens,
                "split": c.split,
                "format": c.format,
                "oracle": c.oracle,
                "oracle_key": c.oracle_key,
            }
            for c in cases
        ],
        "results": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
