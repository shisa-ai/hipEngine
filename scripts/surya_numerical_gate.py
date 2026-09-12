"""Teacher-forced full-vocabulary numerical gate for the Surya OCR lane.

The Surya lane's arithmetic gate was "reproduce the oracle's greedy token ids".
That is all-or-nothing: it cannot qualify a reassociation (a different GEMM
tiling, a fused norm) because it reports a flip without saying how much drift
caused it, and it cannot report a route that is close but not bit-identical.

This module measures the project's declared production envelope instead, on
teacher-forced full-vocabulary rows, per `docs/EXECUTION-PROFILES.md` section 6:

* mean / p95 / p99 / max row KL of the candidate against the teacher;
* top-1 agreement overall and in every declared scope (one scope per page);
* the BF16-relative comparison: how far the candidate drifts from a
  full-precision teacher relative to how far a BF16 teacher already drifts.

The teacher's own greedy chain is forced into every arm, so a flip cannot
cascade into incomparable contexts. The vision features stay arm-specific,
because the arithmetic under test is the vision-attention reassociation itself.

Pure math lives here and is unit-tested; the GPU/torch arms are constructed by
the CLI. Run it with::

    python3 scripts/surya_numerical_gate.py --arms dense,tiled --teacher dense

"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# ---------------------------------------------------------------------------
# declared envelope (docs/EXECUTION-PROFILES.md section 6.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductionEnvelope:
    """Binding automatic-admission limits. Not tuning targets."""

    mean_kl: float = 1e-3
    p95_kl: float = 5e-3
    p99_kl: float = 2e-2
    max_kl: float = 5e-2
    top1: float = 0.99
    per_scope_top1: float = 0.97
    review_kl: float = 2e-2
    """Rows above this need explicit diagnosis even when ``max_kl`` passes."""


ENVELOPE = ProductionEnvelope()


# ---------------------------------------------------------------------------
# pure math
# ---------------------------------------------------------------------------


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax in float64, max-shifted for stability."""

    x = np.asarray(logits, dtype=np.float64)
    shifted = x - x.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def kl_divergence(teacher_logits: np.ndarray, candidate_logits: np.ndarray) -> np.ndarray:
    """Per-row ``KL(p_teacher || p_candidate)`` with full vocabulary.

    ``teacher_logits`` and ``candidate_logits`` are ``(rows, vocab)``.
    """

    teacher = np.asarray(teacher_logits, dtype=np.float64)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    if teacher.shape != candidate.shape:
        raise ValueError(
            f"logit shapes differ: teacher {teacher.shape} vs candidate {candidate.shape}"
        )
    if teacher.ndim != 2:
        raise ValueError(f"logits must be (rows, vocab); got {teacher.shape}")
    log_p = log_softmax(teacher)
    log_q = log_softmax(candidate)
    return (np.exp(log_p) * (log_p - log_q)).sum(axis=-1)


def percentile(values: np.ndarray, q: float) -> float:
    """Linear-interpolated percentile; matches ``numpy.percentile`` default."""

    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


@dataclass
class KlReport:
    """One scope's measured drift."""

    scope: str
    n_rows: int
    mean_kl: float
    p95_kl: float
    p99_kl: float
    max_kl: float
    top1_agreement: float
    top1_flips: int
    rows_over_review: int
    finite: bool

    def as_dict(self) -> dict:
        return {
            "scope": self.scope,
            "n_rows": self.n_rows,
            "mean_kl": self.mean_kl,
            "p95_kl": self.p95_kl,
            "p99_kl": self.p99_kl,
            "max_kl": self.max_kl,
            "top1_agreement": self.top1_agreement,
            "top1_flips": self.top1_flips,
            "rows_over_review": self.rows_over_review,
            "finite": self.finite,
        }


def summarize(
    kl: np.ndarray,
    teacher_top1: np.ndarray,
    candidate_top1: np.ndarray,
    *,
    scope: str,
    review_kl: float = ENVELOPE.review_kl,
) -> KlReport:
    """Aggregate per-row KL and top-1 agreement into a report."""

    kl = np.asarray(kl, dtype=np.float64)
    teacher_top1 = np.asarray(teacher_top1)
    candidate_top1 = np.asarray(candidate_top1)
    if kl.ndim != 1:
        raise ValueError(f"kl must be per-row; got shape {kl.shape}")
    if teacher_top1.shape != kl.shape or candidate_top1.shape != kl.shape:
        raise ValueError("top-1 arrays must have one entry per row")
    finite = bool(np.isfinite(kl).all())
    # A non-finite KL is a hard failure; fold it to +inf so max_kl reports it
    # rather than letting nan poison every aggregate. Percentiles of an array
    # containing inf are nan, so report inf directly for the tails.
    safe = np.where(np.isfinite(kl), kl, np.inf)
    flips = int((teacher_top1 != candidate_top1).sum())
    n = int(kl.shape[0])
    if n and finite:
        p95 = percentile(safe, 95)
        p99 = percentile(safe, 99)
    else:
        p95 = p99 = math.inf if n else 0.0
    return KlReport(
        scope=scope,
        n_rows=n,
        mean_kl=float(safe.mean()) if n else 0.0,
        p95_kl=p95,
        p99_kl=p99,
        max_kl=float(safe.max()) if n else 0.0,
        top1_agreement=(1.0 - flips / n) if n else 1.0,
        top1_flips=flips,
        rows_over_review=int((safe > review_kl).sum()),
        finite=finite,
    )


def evaluate_envelope(
    report: KlReport, envelope: ProductionEnvelope = ENVELOPE
) -> tuple[bool, list[str]]:
    """Apply the declared envelope, returning ``(passed, failures)``."""

    failures: list[str] = []
    if not report.finite:
        failures.append("non-finite KL")
    if report.mean_kl > envelope.mean_kl:
        failures.append(f"mean_kl {report.mean_kl:.3e} > {envelope.mean_kl:.3e}")
    if report.p95_kl > envelope.p95_kl:
        failures.append(f"p95_kl {report.p95_kl:.3e} > {envelope.p95_kl:.3e}")
    if report.p99_kl > envelope.p99_kl:
        failures.append(f"p99_kl {report.p99_kl:.3e} > {envelope.p99_kl:.3e}")
    if report.max_kl > envelope.max_kl:
        failures.append(f"max_kl {report.max_kl:.3e} > {envelope.max_kl:.3e}")
    if report.top1_agreement < envelope.top1:
        failures.append(
            f"top1 {report.top1_agreement:.4f} < {envelope.top1:.4f}"
        )
    if report.rows_over_review:
        failures.append(
            f"{report.rows_over_review} rows above review KL {envelope.review_kl:.0e}"
        )
    return (not failures, failures)


def merge_reports(reports: list[KlReport], scope: str = "global") -> KlReport:
    """Combine scopes for the global row.

    The tails are taken as the worst per-scope tail rather than recomputed from
    raw rows, which the caller does not retain; that is conservative (never
    optimistic) and the per-scope rows carry the exact numbers.
    """

    total = sum(r.n_rows for r in reports)
    if not reports:
        return summarize(np.zeros(0), np.zeros(0), np.zeros(0), scope=scope)
    flips = sum(r.top1_flips for r in reports)
    return KlReport(
        scope=scope,
        n_rows=total,
        mean_kl=(
            sum(r.mean_kl * r.n_rows for r in reports) / total if total else 0.0
        ),
        p95_kl=max(r.p95_kl for r in reports),
        p99_kl=max(r.p99_kl for r in reports),
        max_kl=max(r.max_kl for r in reports),
        top1_agreement=(1.0 - flips / total) if total else 1.0,
        top1_flips=flips,
        rows_over_review=sum(r.rows_over_review for r in reports),
        finite=all(r.finite for r in reports),
    )


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


class Arm(Protocol):
    """A model route that can be pre-filled and stepped one token at a time."""

    name: str

    def prepare(self, prompt: str, page_path: Path) -> "PreparedPrompt": ...

    def start(self, prepared: "PreparedPrompt") -> np.ndarray: ...

    def step(self, token_id: int, position: int) -> np.ndarray: ...


@dataclass
class PreparedPrompt:
    payload: Any
    first_position: int


def generate_chain(
    arm: Arm, prepared: PreparedPrompt, *, max_tokens: int, eos_token_id: int
) -> tuple[list[int], np.ndarray]:
    """Greedy-decode with ``arm``; returns ``(chain, logits_rows)``.

    ``logits_rows[i]`` is the distribution that produced ``chain[i]``, so the
    returned rows are already teacher-forced on the chain.
    """

    logits = arm.start(prepared)
    rows: list[np.ndarray] = [logits]
    chain: list[int] = []
    for step in range(max_tokens):
        nxt = int(np.argmax(logits))
        if nxt == eos_token_id:
            break
        chain.append(nxt)
        logits = arm.step(nxt, prepared.first_position + 1 + step)
        rows.append(logits)
    return chain, np.stack(rows[: len(chain)])


def force_chain(arm: Arm, prepared: PreparedPrompt, chain: list[int]) -> np.ndarray:
    """Force ``chain`` into ``arm`` and return one logits row per chain token."""

    logits = arm.start(prepared)
    rows: list[np.ndarray] = [logits]
    for step, token in enumerate(chain):
        if step == len(chain) - 1:
            break
        logits = arm.step(token, prepared.first_position + 1 + step)
        rows.append(logits)
    return np.stack(rows)


# ---------------------------------------------------------------------------
# gate driver
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One page: the scope a row belongs to."""

    name: str
    page_path: Path
    prompt: str
    max_tokens: int


@dataclass
class ReviewRow:
    """Diagnostics for a row above the review KL bar.

    A large KL does not by itself mean the route is wrong: a row where the
    teacher is nearly undecided can have a big KL while the argmax is stable.
    `docs/EXECUTION-PROFILES.md` requires top-k overlap and the strict
    logit-margin next to any such row, so they are recorded here rather than
    left as an unexplained tail.
    """

    scope: str
    index: int
    kl: float
    teacher_top1: int
    candidate_top1: int
    flipped: bool
    teacher_prob_margin: float
    teacher_logit_margin: float
    top5_overlap: int

    def as_dict(self) -> dict:
        return {
            "scope": self.scope,
            "index": self.index,
            "kl": self.kl,
            "teacher_top1": self.teacher_top1,
            "candidate_top1": self.candidate_top1,
            "flipped": self.flipped,
            "teacher_prob_margin": self.teacher_prob_margin,
            "teacher_logit_margin": self.teacher_logit_margin,
            "top5_overlap": self.top5_overlap,
        }


def _top_k(logits: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-np.asarray(logits, dtype=np.float64), axis=-1)[:, :k]


def review_rows(
    kl: np.ndarray,
    teacher_logits: np.ndarray,
    candidate_logits: np.ndarray,
    *,
    scope: str,
    review_kl: float = ENVELOPE.review_kl,
    limit: int = 32,
) -> list[ReviewRow]:
    """Top-k overlap and margin for the rows above ``review_kl``, worst first."""

    kl = np.asarray(kl, dtype=np.float64)
    teacher_logits = np.asarray(teacher_logits, dtype=np.float64)
    candidate_logits = np.asarray(candidate_logits, dtype=np.float64)
    over = np.flatnonzero(np.where(np.isfinite(kl), kl, np.inf) > review_kl)
    # Sort on the folded values so a non-finite row ranks worst, not last.
    safe = np.where(np.isfinite(kl), kl, np.inf)
    order = over[np.argsort(-safe[over])]
    log_p = log_softmax(teacher_logits)
    teacher_top = _top_k(teacher_logits, 5)
    candidate_top = _top_k(candidate_logits, 5)
    rows: list[ReviewRow] = []
    for index in order[:limit]:
        index = int(index)
        probs = np.exp(log_p[index])
        best_two = np.sort(probs)[-2:]
        rows.append(
            ReviewRow(
                scope=scope,
                index=index,
                kl=float(kl[index]),
                teacher_top1=int(teacher_top[index, 0]),
                candidate_top1=int(candidate_top[index, 0]),
                flipped=bool(teacher_top[index, 0] != candidate_top[index, 0]),
                teacher_prob_margin=float(best_two[1] - best_two[0]),
                teacher_logit_margin=float(
                    np.sort(teacher_logits[index])[-1] - np.sort(teacher_logits[index])[-2]
                ),
                top5_overlap=int(
                    len(set(teacher_top[index].tolist()) & set(candidate_top[index].tolist()))
                ),
            )
        )
    return rows


@dataclass
class ArmComparison:
    arm: str
    teacher: str
    scopes: list[KlReport] = field(default_factory=list)
    review_rows: list[ReviewRow] = field(default_factory=list)

    @property
    def global_report(self) -> KlReport:
        return merge_reports(self.scopes)

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "teacher": self.teacher,
            "global": self.global_report.as_dict(),
            "scopes": [r.as_dict() for r in self.scopes],
            "review_rows": [r.as_dict() for r in self.review_rows],
        }


def run_comparison(
    *,
    teacher: Arm,
    arms: list[Arm],
    cases: list[Case],
    eos_token_id: int,
) -> list[ArmComparison]:
    """Teacher-force the teacher's own chain into every arm, page by page."""

    comparisons = {arm.name: ArmComparison(arm=arm.name, teacher=teacher.name) for arm in arms}
    for case in cases:
        prepared = {arm.name: arm.prepare(case.prompt, case.page_path) for arm in [teacher, *arms]}
        chain, teacher_rows = generate_chain(
            teacher, prepared[teacher.name], max_tokens=case.max_tokens,
            eos_token_id=eos_token_id,
        )
        if not chain:
            continue
        teacher_top1 = teacher_rows.argmax(axis=-1)
        for arm in arms:
            if arm.name == teacher.name:
                candidate_rows = teacher_rows
            else:
                candidate_rows = force_chain(arm, prepared[arm.name], chain)
            kl = kl_divergence(teacher_rows, candidate_rows)
            comparisons[arm.name].scopes.append(
                summarize(
                    kl,
                    teacher_top1,
                    candidate_rows.argmax(axis=-1),
                    scope=case.name,
                )
            )
            comparisons[arm.name].review_rows.extend(
                review_rows(
                    kl, teacher_rows, candidate_rows, scope=case.name
                )
            )
    return [comparisons[arm.name] for arm in arms]


def verdict(comparison: ArmComparison, envelope: ProductionEnvelope = ENVELOPE) -> tuple[bool, list[str]]:
    """Apply the envelope to every scope and the global row."""

    failures: list[str] = []
    for report in [*comparison.scopes, comparison.global_report]:
        passed, scope_failures = evaluate_envelope(report, envelope)
        if not passed:
            failures.extend(f"{report.scope}: {f}" for f in scope_failures)
        if report.scope != "global" and report.top1_agreement < envelope.per_scope_top1:
            failures.append(
                f"{report.scope}: top1 {report.top1_agreement:.4f} < "
                f"{envelope.per_scope_top1:.4f}"
            )
    return (not failures, failures)


# ---------------------------------------------------------------------------
# arms built from the model checkpoints
# ---------------------------------------------------------------------------

# The full-page transcription protocol is the only prompt whose output is a
# page transcription, so it is the only one worth gating numerically.
DEFAULT_CASES = ("ja", "mixed", "dense", "table", "scan", "long")


class HipEngineArm:
    """A hipEngine fp32 route, selected by its vision-attention budget."""

    def __init__(self, name: str, runner, tokenizer, spec, prompt_fn) -> None:
        self.name = name
        self.runner = runner
        self.tokenizer = tokenizer
        self.spec = spec
        self._prompt_fn = prompt_fn

    def prepare(self, prompt: str, page_path: Path) -> PreparedPrompt:
        from PIL import Image

        from hipengine.loading.surya import (
            compute_mrope_positions,
            preprocess_image_surya,
            render_chat_prompt,
        )

        page = Image.open(page_path).convert("RGB")
        pixel_rows, grid = preprocess_image_surya(page)
        n_image_tokens = (grid[1] // 2) * (grid[2] // 2)
        input_ids, mm = render_chat_prompt(self.tokenizer, prompt, n_image_tokens)
        positions = compute_mrope_positions(
            mm, grid, self.spec.vision_spatial_merge_size
        )
        self.runner.check_vision_capacity([grid])
        merged = self.runner.vision_forward(pixel_rows, [grid])
        ids = np.asarray(input_ids, dtype=np.int64).reshape(1, -1)
        first_position = int(np.asarray(positions).reshape(3, -1)[:, -1].max())
        return PreparedPrompt(payload=(ids, positions, merged), first_position=first_position)

    def start(self, prepared: PreparedPrompt) -> np.ndarray:
        ids, positions, merged = prepared.payload
        return self.runner.prefill(ids[0], positions, visual_features=merged)

    def step(self, token_id: int, position: int) -> np.ndarray:
        return self.runner.decode_step(token_id, position)


class TorchArm:
    """A transformers fp32/bf16 route; the full-precision reference."""

    def __init__(self, name: str, model, processor, device: str, dtype) -> None:
        self.name = name
        self.model = model
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self._past = None

    def prepare(self, prompt: str, page_path: Path) -> PreparedPrompt:
        from PIL import Image

        page = Image.open(page_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(page_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[rendered], images=[page], return_tensors="pt"
        ).to(self.device)
        return PreparedPrompt(payload=inputs, first_position=0)

    def start(self, prepared: PreparedPrompt) -> np.ndarray:
        import torch

        with torch.no_grad():
            out = self.model(**prepared.payload, use_cache=True)
        self._past = out.past_key_values
        return out.logits[0, -1].to(torch.float32).cpu().numpy()

    def step(self, token_id: int, position: int) -> np.ndarray:
        import torch

        token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=token, past_key_values=self._past, use_cache=True)
        self._past = out.past_key_values
        return out.logits[0, -1].to(torch.float32).cpu().numpy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cases(names: list[str], fixtures: Path) -> list[Case]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from surya_bench_pages import acceptance_page  # noqa: PLC0415

    from hipengine.generation.surya_protocol import FULL_PAGE_HTML_PROMPT  # noqa: PLC0415

    budgets = {
        "ja": 600, "mixed": 600, "dense": 1500, "table": 640,
        "blank": 64, "scan": 640, "long": 2600,
    }
    return [
        Case(
            name=name,
            page_path=fixtures / acceptance_page(name),
            prompt=FULL_PAGE_HTML_PROMPT,
            max_tokens=budgets[name],
        )
        for name in names
    ]


def _build_hip_arm(name: str, budget: int | None):
    from hipengine.loading.surya import (
        SuryaTokenizer,
        load_surya_spec,
        load_surya_weights,
        resolve_surya_path,
    )
    from hipengine.runtime.surya import DEFAULT_MAX_SEQ, SuryaGpuRunner

    model_dir = resolve_surya_path("datalab-to/surya-ocr-2")
    spec = load_surya_spec(model_dir)
    weights = load_surya_weights(model_dir)
    tokenizer = SuryaTokenizer(model_dir)
    runner = SuryaGpuRunner(
        weights, spec, max_seq=DEFAULT_MAX_SEQ, max_vision_scratch_bytes=budget
    )
    return HipEngineArm(name, runner, tokenizer, spec, None)


def _host_identity() -> dict:
    cpu = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"cpu": cpu, "platform": platform.platform()}


def _print_comparison(comparison: ArmComparison, envelope: ProductionEnvelope) -> bool:
    passed, failures = verdict(comparison, envelope)
    report = comparison.global_report
    print(
        f"{comparison.arm:12s} vs {comparison.teacher:10s} "
        f"rows={report.n_rows:5d} mean={report.mean_kl:.3e} "
        f"p95={report.p95_kl:.3e} p99={report.p99_kl:.3e} "
        f"max={report.max_kl:.3e} top1={report.top1_agreement:.5f} "
        f"passed={passed}"
    )
    for failure in failures:
        print(f"    FAIL {failure}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument(
        "--arms", default="dense,tiled",
        help="hipEngine arms: dense (no vision budget) and/or tiled (budgeted)",
    )
    parser.add_argument("--teacher", default="dense")
    parser.add_argument(
        "--tiled-budget", type=int, default=None,
        help=("vision score-tile budget for the 'tiled' arm. The production "
              "default (512 MiB) is a no-op on these page sizes, so pass a "
              "smaller budget to actually exercise the query-row reassociation"),
    )
    parser.add_argument(
        "--torch-arms", default="",
        help=(
            "torch arms, e.g. torch_fp32,torch_bf16. torch_fp32 is used as the "
            "full-precision teacher for the BF16-relative table"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures/surya"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", action="store_true", help="print the artifact as JSON")
    args = parser.parse_args()

    case_names = [c.strip() for c in args.cases.split(",") if c.strip()]
    cases = _build_cases(case_names, args.fixtures)

    from hipengine.runtime.surya import DEFAULT_MAX_VISION_SCRATCH_BYTES

    tiled_budget = (
        DEFAULT_MAX_VISION_SCRATCH_BYTES if args.tiled_budget is None
        else args.tiled_budget
    )
    arm_specs = [a.strip() for a in args.arms.split(",") if a.strip()]
    hip_arms = []
    for name in arm_specs:
        if name == "dense":
            hip_arms.append(_build_hip_arm("dense", None))
        elif name == "tiled":
            hip_arms.append(_build_hip_arm("tiled", tiled_budget))
        else:
            raise SystemExit(f"unknown hipEngine arm {name!r}")

    arms_by_name: dict[str, Arm] = {arm.name: arm for arm in hip_arms}
    if args.teacher not in arms_by_name:
        raise SystemExit(
            f"teacher {args.teacher!r} must be one of the hipEngine arms "
            f"{sorted(arms_by_name)}"
        )
    teacher: Arm = arms_by_name[args.teacher]

    torch_names = [a.strip() for a in args.torch_arms.split(",") if a.strip()]
    if torch_names:
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        processor = AutoProcessor.from_pretrained("datalab-to/surya-ocr-2")
        for name in torch_names:
            dtype = torch.bfloat16 if name.endswith("bf16") else torch.float32
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                "datalab-to/surya-ocr-2", dtype=dtype
            ).to(args.device)
            model.eval()
            arms_by_name[name] = TorchArm(name, model, processor, args.device, dtype)

    artifact: dict = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "datalab-to/surya-ocr-2",
        "hardware": _host_identity(),
        "teacher": args.teacher,
        "tiled_budget_bytes": tiled_budget,
        "protocol": (
            "teacher-forced full-vocabulary rows; the teacher's own greedy chain "
            "is forced into every arm and vision features stay arm-specific"
        ),
        "bf16_relative_note": (
            "Diagnostic, not a binding gate: this compares two different "
            "implementations (hipEngine GEMM routes vs torch), where the "
            "profile's tail bars are calibrated for production-versus-strict "
            "within one implementation. The binding production comparison is "
            "`comparisons`. Rows above the review bar carry top-5 overlap, the "
            "teacher margin, and the flip flag."
        ),
        "envelope": {
            "mean_kl": ENVELOPE.mean_kl,
            "p95_kl": ENVELOPE.p95_kl,
            "p99_kl": ENVELOPE.p99_kl,
            "max_kl": ENVELOPE.max_kl,
            "top1": ENVELOPE.top1,
            "per_scope_top1": ENVELOPE.per_scope_top1,
            "review_kl": ENVELOPE.review_kl,
        },
        "cases": case_names,
        "comparisons": [],
        "bf16_relative": [],
    }
    ok = True

    comparison_arms = [arms_by_name[name] for name in arm_specs]
    for comparison in run_comparison(
        teacher=teacher, arms=comparison_arms, cases=cases, eos_token_id=2
    ):
        passed = _print_comparison(comparison, ENVELOPE)
        ok = ok and passed
        entry = comparison.as_dict()
        entry["passed"] = passed
        _, entry["failures"] = verdict(comparison, ENVELOPE)
        artifact["comparisons"].append(entry)

    # BF16-relative: how far each arm drifts from the full-precision torch
    # teacher, next to how far a BF16 teacher already drifts from it.
    if "torch_fp32" in arms_by_name:
        bf16_teacher = arms_by_name["torch_fp32"]
        others = [
            arms_by_name[name] for name in [*arm_specs, *torch_names]
            if name != "torch_fp32"
        ]
        print("\nBF16-relative (teacher = torch_fp32):")
        for comparison in run_comparison(
            teacher=bf16_teacher, arms=others, cases=cases, eos_token_id=2
        ):
            passed = _print_comparison(comparison, ENVELOPE)
            entry = comparison.as_dict()
            entry["passed"] = passed
            _, entry["failures"] = verdict(comparison, ENVELOPE)
            artifact["bf16_relative"].append(entry)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=1))
        print(f"\nwrote {args.out}")
    if args.json:
        print(json.dumps(artifact, indent=1))

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
