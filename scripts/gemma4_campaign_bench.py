"""G0 measurement harness for the Gemma 4 26B-A4B optimization campaign.

Separates the phases ``docs/campaigns/GEMMA4-26B-A4B-OPTIMIZATION.md`` requires
for its primary timing rows: load, warmup, prefill (including the last-row
output head, which lives inside ``Gemma4Runner.forward``), first-token latency,
decode, and one untouched public ``LLM.generate`` wall measurement. The
campaign metric is ``decode_tps``: (output_tokens - 1) decode forwards over
their summed time, printed as ``decode_tps=<median>``.

The instrumented path drives the same ``Gemma4Runner`` methods, in the same
order, as ``Gemma4GGUFGenerator.generate_detailed`` (see
``hipengine/generation/gemma4_gguf.py``); the public wall row runs that
generator untouched with the exact same token ids, so a divergence between the
two is reported rather than smoothed over.

Timing boundaries, stated once and recorded in every artifact:

- ``prefill_s``: ``forward(prompt)`` plus an explicit device synchronize.
  Greedy sampling of the first token is excluded.
- ``first_sample_s``: the host finite-logit check and greedy argmax that produce token one.
- ``first_token_s``: request start to token one available (prefill + sample).
- ``decode_s``: the output-1 subsequent single-token forwards, each with its
  finite-logit check, greedy sample and an explicit device synchronize.
- ``wall_s``: instrumented request start to last token, phases above combined.
- ``public_wall_s``: an untouched ``generate_detailed`` call on the same ids.

Fixed-length campaign rows ignore EOS by construction and say so; natural-EOS
rows are a different workload and are never mixed into these numbers.

Unit tests for the accounting live in
``tests/test_unit_gemma4_campaign_bench.py`` and run without a GPU.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# Run against THIS checkout even when the venv's editable install points at a
# different worktree (observed in this repo before: the shared environment
# shadowed imports and measured the wrong code).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The campaign's GGUF. This was a hard-coded path under /mnt/nvme1 until that
# mount moved, which broke every script here at import time with a
# FileNotFoundError that named the model rather than the missing mount. Resolve
# instead: an explicit HIPENGINE_GEMMA4_ARTIFACT wins, then the known locations.
# A missing artifact still fails loudly, but at the point of use and naming every
# path that was tried.
ARTIFACT_ENV = "HIPENGINE_GEMMA4_ARTIFACT"
_ARTIFACT_CANDIDATES = (
    Path("/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"),
    Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"),
)


def resolve_artifact(explicit: str | Path | None = None) -> Path:
    """Return the campaign GGUF path, preferring one that exists.

    Never raises: an import of this module must not depend on a model being
    present, so a box without the artifact still imports and fails at the point
    of use, where the caller can say which paths were tried.
    """

    override = os.environ.get(ARTIFACT_ENV, "").strip()
    candidates = [Path(override)] if override else []
    if explicit is not None:
        candidates.append(Path(explicit))
    candidates.extend(_ARTIFACT_CANDIDATES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_ARTIFACT = resolve_artifact()
DEFAULT_OUT = Path("/tmp/gemma4_campaign_bench.json")
DEFAULT_CONTEXT = 8192
DEFAULT_EXPECT_GPU = "RX 7900 XTX"

# Frozen natural corpus for primary timing rows. Cycled to reach the exact
# prompt width; the exact token ids used by every sample are saved in the
# artifact so both engines in a paired comparison see the same sequence.
CORPUS: tuple[str, ...] = (
    "The capital of France is Paris, and the city has been a center of art "
    "and science for centuries.",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose "
    "and oxygen inside the chloroplasts of green plants.",
    "def quicksort(items):\n    if len(items) <= 1:\n        return items\n"
    "    pivot = items[len(items) // 2]\n    left = [x for x in items if x < pivot]\n"
    "    right = [x for x in items if x > pivot]\n"
    "    return quicksort(left) + [pivot] + quicksort(right)",
    "京都の寺社は季節ごとに異なる表情を見せ、春の桜と秋の紅葉では観光客の数も変わる。",
    "The Treaty of Westphalia in 1648 is often cited as the moment that "
    "established the modern system of sovereign states in Europe.",
    "A cache hit ratio of ninety-nine percent still misses one request in a "
    "hundred, so the slow path has to stay correct and has to stay fast.",
    "Ocean currents distribute heat around the planet; the Gulf Stream alone "
    "carries water warmer than the air above much of northern Europe.",
    "Historians disagree about whether the industrial revolution began as a "
    "slow structural shift or as a genuinely sudden break with the past.",
)


PROBE_CORPUS_SEED = 20260927

# Word pool for the shuffled and enumerative probe fragments. Ordinary English,
# so the model's grammar is not the source of uncertainty - only the choice is.
_PROBE_WORDS: tuple[str, ...] = (
    "amber", "anchor", "apron", "basalt", "beacon", "bramble", "cinder",
    "cobalt", "copper", "cotton", "dagger", "dahlia", "ember", "fennel",
    "flint", "gable", "granite", "harbor", "hazel", "indigo", "ivory",
    "juniper", "kestrel", "lantern", "lichen", "marble", "meadow", "nectar",
    "nickel", "obsidian", "orchard", "pebble", "pepper", "quartz", "quill",
    "raven", "ridge", "saffron", "sandal", "sequoia", "shale", "sorrel",
    "tallow", "thistle", "timber", "umber", "velvet", "walnut", "willow",
    "yarrow", "zephyr",
)


def probe_corpus(count: int = 96, *, seed: int = PROBE_CORPUS_SEED) -> tuple[str, ...]:
    """Deterministic, never-repeating fragments for margin-aware numerical probing.

    The frozen campaign corpus is eight sentences cycled to the target length, so
    a 1024-token chain repeats each of them many times and the model is close to
    certain about every continuation: top-1 margins are large everywhere and a
    top-1 bar cannot see a reordering-class divergence. This corpus is generated
    from a fixed seed instead, and mixes material with no predictable continuation
    (hex digests, id-like digit groups, random codes) with material where only the
    wording is open (shuffled word lists, interchangeable enumerations). One chain
    therefore carries rows across the whole margin range rather than only the
    near-one-hot end.

    It is seeded so both arms of a paired comparison see byte-identical ids, which
    is what makes the comparison paired at all.
    """

    import random

    rng = random.Random(int(seed))
    letters = "abcdefghijklmnopqrstuvwxyz"
    hexits = "0123456789abcdef"
    fragments: list[str] = []
    for index in range(int(count)):
        kind = index % 6
        if kind == 0:
            digest = "".join(rng.choice(hexits) for _ in range(56))
            fragments.append(f"Record {index:04d} digest {digest} closes the entry.")
        elif kind == 1:
            groups = " ".join(f"{rng.randrange(10000):04d}" for _ in range(13))
            fragments.append(f"Sequence {index:04d} reads {groups} and stops there.")
        elif kind == 2:
            picks = [rng.choice(_PROBE_WORDS) for _ in range(16)]
            fragments.append(f"Order {index:04d}: " + ", ".join(picks) + ".")
        elif kind == 3:
            picks = [rng.choice(_PROBE_WORDS) for _ in range(11)]
            fragments.append(
                "The candidate labels are "
                + ", ".join(picks[:-1])
                + " and "
                + picks[-1]
                + f", listed under case {index:04d}."
            )
        elif kind == 4:
            codes = " ".join(
                "".join(rng.choice(letters + "0123456789") for _ in range(5))
                for _ in range(9)
            )
            fragments.append(f"Codes {index:04d}: {codes}.")
        else:
            mixed = "".join(
                rng.choice(letters + "0123456789 .,;:-") for _ in range(110)
            )
            fragments.append(f"Payload {index:04d} begins: {mixed}")
    return tuple(fragments)


def exact_prompt_ids(
    tokenize: Callable[[str], Sequence[int]],
    target: int,
    *,
    corpus: Sequence[str] = CORPUS,
    require_single_pass: bool = False,
) -> list[int]:
    """Tokenize the frozen corpus, cycled, and cut to exactly ``target`` ids."""

    target = int(target)
    if target <= 0:
        raise ValueError(f"target must be positive, got {target}")
    corpus = tuple(corpus)
    if not corpus:
        raise ValueError("corpus must not be empty")
    ids: list[int] = []
    while len(ids) < target:
        cycle_start = len(ids)
        index = 0
        while len(ids) < target and index < len(corpus):
            ids.extend(int(token) for token in tokenize(corpus[index]))
            index += 1
        if len(ids) == cycle_start:
            raise ValueError("corpus produced no tokens")
        if require_single_pass and len(ids) < target:
            raise ValueError(
                f"corpus supplies only {len(ids)} of {target} ids; a single-pass "
                "chain must not cycle, or repeated context makes every "
                "continuation predictable again"
            )
    return ids[:target]


def run_instrumented(
    runner: Any,
    prompt_ids: Sequence[int],
    max_tokens: int,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sync: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """One fixed-length greedy request through ``runner``, with phase timing.

    The call order mirrors ``Gemma4GGUFGenerator.generate_detailed`` exactly:
    reset, one prefill ``forward`` carrying the whole prompt, greedy sample of
    token one from the prefill logits, then ``max_tokens - 1`` single-token
    forwards each followed by a greedy sample. EOS is never consulted; every
    campaign timing row is explicitly fixed-length.
    """

    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    max_tokens = int(max_tokens)
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    capacity = getattr(runner, "capacity", None)
    if capacity is not None and len(prompt_ids) + max_tokens > capacity:
        raise ValueError(
            f"prompt ({len(prompt_ids)} tokens) plus max_tokens ({max_tokens}) "
            f"exceeds runner capacity {capacity}"
        )
    if sync is None:
        from hipengine.core.hip import get_hip_runtime

        device_synchronize = get_hip_runtime().device_synchronize
        sync = device_synchronize

    import numpy as np

    runner.reset()
    sync_count = 0

    def timed_sync() -> None:
        nonlocal sync_count
        sync()
        sync_count += 1

    timed_sync()
    t0 = clock()
    logits = runner.forward(list(prompt_ids))
    timed_sync()
    t1 = clock()
    prefill_s = t1 - t0
    if not np.all(np.isfinite(logits)):
        raise ValueError("prefill must return finite logits")

    token_id = runner.next_token(logits)
    t_first = clock()
    first_token_s = t_first - t0
    first_sample_s = t_first - t1
    generated: list[int] = [int(token_id)]

    decode_s = 0.0
    for _ in range(1, max_tokens):
        step_start = clock()
        logits = runner.forward([generated[-1]])
        if not np.all(np.isfinite(logits)):
            raise ValueError("decode must return finite logits")
        token_id = runner.next_token(logits)
        timed_sync()
        step_end = clock()
        decode_s += step_end - step_start
        generated.append(int(token_id))

    wall_s = clock() - t0
    if len(generated) != max_tokens:
        raise RuntimeError(
            f"accounting error: generated {len(generated)} tokens, expected {max_tokens}"
        )
    return {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "decode_forwards": max_tokens - 1,
        "prefill_s": prefill_s,
        "first_sample_s": first_sample_s,
        "first_token_s": first_token_s,
        "decode_s": decode_s,
        "wall_s": wall_s,
        "syncs": sync_count,
        "finish_reason": "length",
        "eos_ignored": True,
        "finite_logits": True,
        "generated_token_ids": generated,
    }


def validate_generation_rows(
    warmups: Sequence[dict[str, Any]], samples: Sequence[dict[str, Any]],
    public: dict[str, Any], outputs: int,
) -> None:
    """Require fixed-shape repeatability and public-route identity, not cross-engine parity."""
    expected = samples[0]["generated_token_ids"]
    rows = [*warmups, *samples, public]
    for row in rows:
        ids = row.get("generated_token_ids")
        if (not isinstance(ids, list) or len(ids) != outputs
                or any(type(token) is not int or token < 0 for token in ids)
                or ids != expected):
            raise ValueError("generation output count, token IDs, repeatability or public parity failed")
    wall = public["public_wall_s"]
    if not math.isfinite(wall) or wall <= 0:
        raise ValueError("generation public wall must be finite and positive")


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(list(values)))


def _p95_nearest_rank(values: Sequence[float]) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return float(ordered[rank - 1])


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate instrumented samples into the campaign's reported statistics."""

    records = list(records)
    if not records:
        raise ValueError("summarize needs at least one sample")
    shapes = {
        (r["prompt_tokens"], r["generated_tokens"], r["decode_forwards"])
        for r in records
    }
    if len(shapes) != 1:
        raise ValueError(f"samples disagree on shape: {sorted(shapes)}")
    prompt_tokens, generated_tokens, decode_forwards = next(iter(shapes))
    for record in records:
        for field in ("prefill_s", "first_token_s", "wall_s", "first_sample_s", "decode_s"):
            value = record[field]
            zero_allowed = field == "first_sample_s" or (field == "decode_s" and decode_forwards == 0)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or (value == 0 and not zero_allowed)):
                raise ValueError(f"{field} must be a finite {'nonnegative' if zero_allowed else 'positive'} duration")

    decode_tps_values = [
        r["decode_forwards"] / r["decode_s"]
        for r in records
        if r["decode_s"] > 0 and r["decode_forwards"] > 0
    ]
    if decode_tps_values:
        decode_tps = _median(decode_tps_values)
        decode_min = min(decode_tps_values)
        decode_max = max(decode_tps_values)
        decode_p95 = _p95_nearest_rank(decode_tps_values)
        decode_stdev = (
            float(statistics.stdev(decode_tps_values))
            if len(decode_tps_values) >= 2
            else None
        )
    else:
        decode_tps = decode_min = decode_max = decode_p95 = decode_stdev = None

    prefill_tps_values = [
        r["prompt_tokens"] / r["prefill_s"]
        for r in records
        if r["prefill_s"] > 0
    ]
    return {
        "samples": len(records),
        "decode_tps": decode_tps,
        "decode_tps_min": decode_min,
        "decode_tps_max": decode_max,
        "decode_tps_p95": decode_p95,
        "decode_tps_stdev": decode_stdev,
        "prefill_s": _median([r["prefill_s"] for r in records]),
        "prefill_tps": _median(prefill_tps_values) if prefill_tps_values else None,
        "first_token_s": _median([r["first_token_s"] for r in records]),
        "first_sample_s": _median([r["first_sample_s"] for r in records]),
        "decode_s": _median([r["decode_s"] for r in records]),
        "wall_s": _median([r["wall_s"] for r in records]),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "decode_forwards_per_sample": decode_forwards,
    }


def _sha256_head(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def _run(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or completed.stderr or "").strip()


def _gpu_name() -> str:
    """Name of logical HIP device 0, honoring visibility masks."""

    try:
        from hipengine.core.hip import get_hip_runtime

        get_hip_runtime()  # load libamdhip64 first
        library = ctypes.CDLL("libamdhip64.so")
        library.hipDeviceGetName.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.hipDeviceGetName.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(256)
        result = library.hipDeviceGetName(buffer, len(buffer), 0)
        if result != 0:
            return f"hipDeviceGetName failed ({result})"
        return buffer.value.decode("utf-8", errors="replace")
    except Exception as error:  # pragma: no cover - diagnostic best effort
        return f"unavailable: {error}"


def _provenance(artifact: Path) -> dict[str, Any]:
    commit = _run(["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(_REPO_ROOT), "status", "--porcelain"])
    env_keys = sorted(
        key
        for key in os.environ
        if key.startswith(("HIP", "HSA", "ROCR", "AMD", "GPU_", "HCC", "PYTORCH", "TORCH"))
    )
    clock_report = _run(["rocm-smi", "--showclock", "--showpower", "--showtemp"])
    return {
        "host": platform.node(),
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "git_dirty_files": dirty.splitlines()[:40],
        "gpu_name_device0": _gpu_name(),
        "artifact": {
            "path": str(artifact),
            "bytes": artifact.stat().st_size if artifact.exists() else None,
            "mtime": datetime.fromtimestamp(artifact.stat().st_mtime, timezone.utc).isoformat()
            if artifact.exists()
            else None,
            "head_sha256_64MiB": _sha256_head(artifact) if artifact.exists() else None,
        },
        "env": {key: os.environ.get(key) for key in env_keys},
        "hipcc": (_run(["hipcc", "--version"]) or "").splitlines()[:2],
        "rocm_smi_clock_power": clock_report.splitlines()[:40],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _resolve_generator(artifact: Path, context: int) -> tuple[Any, Any, dict[str, Any]]:
    """Build the public LLM surface, resolve the generator, load weights."""

    import hipengine

    resolve_start = time.perf_counter()
    # Set capacity through the public factory contract. The generator may be
    # wrapped by SubmitPollTextGenerator: assigning context_length on that
    # wrapper shadows the inner field and leaves the runner at its default, so
    # the factory argument is what actually sizes the runner's KV cache. The
    # explicit assignment below is kept so the wrapper and the inner generator
    # cannot disagree about the capacity the run was measured at.
    llm = hipengine.LLM(model=str(artifact), max_sequence_length=int(context))
    generator = llm._get_text_generator()
    resolve_s = time.perf_counter() - resolve_start
    generator.context_length = int(context)
    load_start = time.perf_counter()
    runner = generator._ensure_runner()
    load_s = time.perf_counter() - load_start
    resolution = resolution_row(llm, generator)
    return llm, runner, {
        "resolve_s": resolve_s,
        "load_s": load_s,
        "internal_load_s": getattr(generator, "_load_seconds", None),
        "resolution": resolution,
        "context_length": int(getattr(generator, "context_length", context)),
        "runner_capacity": int(getattr(runner, "capacity", 0)),
        "max_block": int(getattr(runner, "max_block", 0)),
    }


def _public_wall(llm: Any, prompt_ids: Sequence[int], max_tokens: int) -> dict[str, Any]:
    """One untouched public generate call on the exact same token ids."""

    from hipengine.llm import SamplingParams

    params = SamplingParams(max_tokens=int(max_tokens), temperature=0.0, ignore_eos=True)
    started = time.perf_counter()
    output = llm.generate_detailed(list(prompt_ids), params)[0]
    wall_s = time.perf_counter() - started
    return {
        "public_wall_s": wall_s,
        "public_tps": (int(max_tokens) / wall_s if wall_s > 0 else None),
        "generated_tokens": len(output.generated_token_ids or ()),
        "finish_reason": getattr(output.finish_details, "reason", None),
        "sampler_mode": getattr(output.finish_details, "sampler_mode", None),
        "finite_logits": None,  # public API exposes IDs, not every intermediate logit row
        "generated_token_ids": list(output.generated_token_ids or ()),
    }


def memory_row(samples: Sequence[tuple[str, int, int]]) -> dict:
    """Build the artifact memory block from labeled ``(label, free, total)`` rows.

    Every row must agree on total device memory; a mismatch means the device
    changed mid-run (or a sample tuple was malformed), which would make the
    per-label used-bytes comparisons meaningless.
    """
    if not samples:
        raise ValueError("memory_row needs at least one (label, free, total) sample")
    labels: list[str] = []
    used: dict[str, int] = {}
    total: int | None = None
    for sample in samples:
        if len(sample) != 3:
            raise ValueError(
                "each memory sample must be (label, free_bytes, total_bytes)"
            )
        label, free_bytes, total_bytes = sample
        if total is None:
            total = int(total_bytes)
        elif int(total_bytes) != total:
            raise ValueError(f"inconsistent device total: {total_bytes} != {total}")
        used[str(label)] = total - int(free_bytes)
        labels.append(str(label))
    return {
        "total_bytes": total,
        "used_bytes": used,
        "peak_used_bytes": max(used.values()),
        "labels": labels,
    }


def resolution_row(llm: Any, generator: Any) -> dict[str, Any]:
    """JSON-safe resolution labels for the artifact's ``loading`` row.

    Backend and quant resolve to enums (unwrapped via ``.value``); a registered
    execution profile resolves to a ``ResolvedRuntimeProfile`` whose profile
    name lives under ``.profile``. Embedding that object raw crashes
    ``json.dumps`` when the artifact is written.
    """
    from hipengine.execution_profiles import ResolvedRuntimeProfile

    resolution: dict[str, Any] = {}
    for attribute in ("_resolved_backend", "_resolved_quant", "_resolved_execution_profile"):
        value = getattr(llm, attribute, None)
        if isinstance(value, ResolvedRuntimeProfile):
            resolution[attribute] = value.profile.value
            resolution[attribute + "_manifest"] = {
                "manifest_sha256": value.manifest_sha256,
                "fell_back_to_strict": value.fell_back_to_strict,
            }
        else:
            resolution[attribute] = getattr(value, "value", value)
    resolution["generator_type"] = type(generator).__name__
    return resolution


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--prompt", type=int, required=True, help="exact prompt tokens")
    parser.add_argument("--output", type=int, required=True, help="exact output tokens")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1, help="full-shape warmup requests")
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--label", default="campaign")
    parser.add_argument("--expect-gpu", default=DEFAULT_EXPECT_GPU)
    args = parser.parse_args(argv)

    if args.prompt < 1 or args.output < 1:
        parser.error("--prompt and --output must be positive")
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if args.prompt + args.output > args.context:
        parser.error(
            f"prompt ({args.prompt}) + output ({args.output}) exceeds context {args.context}"
        )

    print(f"[gemma4_campaign_bench] label={args.label}", flush=True)
    provenance = _provenance(args.artifact)
    gpu_name = provenance["gpu_name_device0"]
    print(f"[gemma4_campaign_bench] device0={gpu_name}", flush=True)
    if args.expect_gpu and args.expect_gpu not in gpu_name:
        print(
            f"ERROR: logical HIP device 0 is {gpu_name!r}, expected it to contain "
            f"{args.expect_gpu!r}; check ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES "
            "before measuring",
            file=sys.stderr,
        )
        return 2

    from hipengine.core.hip import get_hip_runtime

    get_hip_runtime().device_synchronize()
    mem_samples: list[tuple[str, int, int]] = [
        ("before_load", *get_hip_runtime().mem_get_info())
    ]

    llm, runner, loading = _resolve_generator(args.artifact, args.context)
    generator = llm._get_text_generator()
    print(
        f"[gemma4_campaign_bench] loaded in {loading['load_s']:.1f}s "
        f"(resolve {loading['resolve_s']:.1f}s, context {loading['context_length']}, "
        f"max_block {loading['max_block']})",
        flush=True,
    )
    mem_samples.append(("after_load", *get_hip_runtime().mem_get_info()))

    prompt_ids = exact_prompt_ids(generator.tokenize, args.prompt)
    if len(prompt_ids) != args.prompt:
        raise RuntimeError(
            f"prompt accounting error: {len(prompt_ids)} ids, expected {args.prompt}"
        )

    warmups = []
    for index in range(args.warmup):
        record = run_instrumented(runner, prompt_ids, args.output)
        record["role"] = "warmup"
        record["index"] = index
        warmups.append(record)
        print(
            f"[gemma4_campaign_bench] warmup {index}: "
            f"prefill {record['prefill_s']:.2f}s, decode {record['decode_s']:.2f}s",
            flush=True,
        )

    samples = []
    for index in range(args.samples):
        record = run_instrumented(runner, prompt_ids, args.output)
        record["role"] = "sample"
        record["index"] = index
        samples.append(record)
        sample_tps = (
            record["decode_forwards"] / record["decode_s"] if record["decode_s"] > 0 else 0.0
        )
        print(
            f"[gemma4_campaign_bench] sample {index}: prefill {record['prefill_s']:.2f}s "
            f"({record['prompt_tokens'] / max(record['prefill_s'], 1e-9):.0f} tok/s), "
            f"first token {record['first_token_s']:.2f}s, "
            f"decode {record['decode_s']:.2f}s ({sample_tps:.2f} tok/s), "
            f"wall {record['wall_s']:.2f}s",
            flush=True,
        )

    stats = summarize(samples)

    get_hip_runtime().device_synchronize()
    public = _public_wall(llm, prompt_ids, args.output)
    instrumented_ids = samples[0]["generated_token_ids"]
    public_ids = public["generated_token_ids"]
    parity_index = next(
        (i for i, (a, b) in enumerate(zip(instrumented_ids, public_ids)) if a != b),
        None if len(instrumented_ids) == len(public_ids) else min(len(instrumented_ids), len(public_ids)),
    )
    public["public_path_parity"] = parity_index is None
    public["public_path_first_divergence"] = parity_index
    public["public_generated_equals_expected"] = len(public_ids) == args.output
    validate_generation_rows(warmups, samples, public, args.output)
    print(
        f"[gemma4_campaign_bench] public wall {public['public_wall_s']:.2f}s "
        f"({public['public_tps']:.2f} tok/s incl. prefill), "
        f"path parity={public['public_path_parity']}",
        flush=True,
    )

    mem_samples.append(("after_runs", *get_hip_runtime().mem_get_info()))

    artifact = {
        "schema": 1,
        "status": "ok",
        "correctness": {"instrumented_finite_logits": True, "public_finite_logits": None,
                        "all_replays_and_public_ids_equal": True},
        "label": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "workload": {
            "prompt_tokens": args.prompt,
            "output_tokens": args.output,
            "samples": args.samples,
            "warmups": args.warmup,
            "context": args.context,
            "decode_forwards_per_sample": args.output - 1,
            "eos": "ignored (fixed-length row)",
        },
        "prompt_token_ids": prompt_ids,
        "prompt_ids_sha256": hashlib.sha256(
            b"".join(int(t).to_bytes(4, "little") for t in prompt_ids)
        ).hexdigest(),
        "provenance": provenance,
        "loading": loading,
        "memory": memory_row(mem_samples),
        "warmups": warmups,
        "samples": samples,
        "stats": stats,
        "public": public,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")

    metric_name = f"decode_tps_{args.prompt}p{args.output}o"
    decode_value = stats["decode_tps"]
    if decode_value is None:
        if args.output > 1:
            print("ERROR: no measurable decode phase", file=sys.stderr)
            return 1
        # Single-output rows (the campaign's 8191+1 capacity row) are legal
        # correctness rows with no decode phase; they report no metric.
        print("decode_tps=none")
        print(f"{metric_name}=none")
        print(f"artifact={args.out}")
        return 0
    print(f"decode_tps={decode_value:.4f}")
    print(f"{metric_name}={decode_value:.4f}")
    print(f"artifact={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())