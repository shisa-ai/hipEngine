#!/usr/bin/env python3
"""Time ordinary single-stream Inkling 2 GGUF decode at several prompt depths in one process.

For each prompt depth this streams a fixed number of output tokens after a prompt of exactly
that many tokens and reports, for every repetition:

``first_token_s``
    Request start to the first non-empty text delta: prompt processing plus the first token.
``decode_s``, ``decode_tok_s``
    That first delta to stream close, divided by the tokens generated in between.
``output_ids_sha256``, ``diagnostics``
    What was generated and which path ran. Two runs are compared on these before their timings.

``--counts`` wraps the HIP runtime's synchronizing and copy calls for the decode window and
reports how many ran and how long the host spent inside each. Time inside those calls is GPU
work plus launch latency as seen from the host; time outside them is host work and launch
submission. It does not measure GPU-busy time.

Greedy, one stream, prefix cache off, drafting off. This is the in-process control for
optimization units. The matched Loom protocol is ``scripts/inkling2_ordinary_pairs.py`` against
live servers.

    python scripts/inkling2_decode_bench.py MODEL --backend hip_gfx1100 \\
        --execution-profile fast --prompt-tokens 47 575 2057 --reps 5 --counts --output out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import statistics
import sys
import time
from contextlib import closing
from pathlib import Path

from hipengine import LLM, SamplingParams
from hipengine.core.hip import HipRuntime

# The same opening as scripts/inkling2_gguf_decode_trace.py, so both tools render the same
# prompt IDs for a requested depth.
_SEED = ("Explain why the day gets longer in spring, using plain language and a few practical examples. "
         "Distinguish what an observer sees from the motion of the Earth. ")

COUNTED = ("stream_synchronize", "event_synchronize", "device_synchronize", "stream_wait_event",
           "event_record", "event_query", "graph_launch")
COPIED = ("memcpy", "memcpy_async")
PROTOCOL = ("one greedy stream, prefix cache off, drafting off; first_token_s is request start to the first "
            "non-empty text delta; decode is that delta to stream close over the tokens generated in between")


def prompt_ids(seed, count):
    """Repeat ``seed`` and cut it to exactly ``count`` tokens."""
    if count < 1:
        raise ValueError("prompt length must be positive")
    seed = tuple(seed)
    if not seed:
        raise ValueError("seed must contain at least one token")
    return (seed * ((count + len(seed) - 1) // len(seed)))[:count]


class CallCounter:
    """Count calls to methods of ``cls`` between ``start()`` and ``stop()``.

    Copy calls are split by direction and also sum their bytes. Methods are replaced on the
    class, so every holder of an instance is counted; ``stop()`` restores the originals.
    """

    def __init__(self, cls, names=COUNTED, copies=COPIED):
        self._cls = cls
        self._names = tuple(names)
        self._copies = tuple(copies)
        self._originals = {}
        self.tally = {}

    def _record(self, key, seconds, nbytes=0):
        entry = self.tally.setdefault(key, {"calls": 0, "seconds": 0.0, "bytes": 0})
        entry["calls"] += 1
        entry["seconds"] += seconds
        entry["bytes"] += nbytes

    def start(self):
        if self._originals:
            raise RuntimeError("counter already started")
        for name in self._names + self._copies:
            split = name in self._copies
            original = getattr(self._cls, name)
            self._originals[name] = original

            def wrapped(*args, _name=name, _original=original, _split=split, **kwargs):
                started = time.perf_counter()
                try:
                    return _original(*args, **kwargs)
                finally:
                    seconds = time.perf_counter() - started
                    if _split:
                        # copy(self, dst, src, nbytes, kind[, stream]): the instance is the first
                        # positional argument unless the call came straight from the class. The
                        # byte count is third after it unless it was passed by keyword.
                        base = 1 if args and isinstance(args[0], self._cls) else 0
                        nbytes = args[base + 2] if len(args) > base + 2 else int(kwargs.get("nbytes", 0))
                        kind = kwargs.get("kind", args[base + 3] if len(args) > base + 3 else None)
                        self._record(f"{_name}:{_direction(kind)}", seconds, int(nbytes))
                    else:
                        self._record(_name, seconds)

            setattr(self._cls, name, wrapped)

    def stop(self):
        for name, original in self._originals.items():
            setattr(self._cls, name, original)
        self._originals.clear()
        return self.tally


def _direction(kind):
    return (getattr(kind, "name", None) or str(kind)).lower()


def install_token_counter(generator):
    """Return ``[count]``, advanced once per token the text generator hands its stream.

    The stream yields a chunk only for non-empty text and carries token IDs only on its last
    chunk, so neither the chunk count nor the IDs on the first delta give the token count.
    Counting where the generator yields does.
    """
    counter = [0]
    original = generator._iterate_row

    def counting(*args, **kwargs):
        for token in original(*args, **kwargs):
            counter[0] += 1
            yield token

    generator._iterate_row = counting
    return counter


def stream_once(llm, ids, sampling, *, tokens, counts=False):
    """Stream one request and return its timings, generated IDs and diagnostics.

    ``tokens`` is the counter from ``install_token_counter``. It is reset here and read when
    the first non-empty text delta arrives; the tokens generated before that point came out of
    prompt processing, so they are not decode steps.
    """
    counter = CallCounter(HipRuntime) if counts else None
    tokens[0] = 0
    started = time.perf_counter()
    first_at = None
    first_count = 0
    output_ids = None
    diagnostics = None
    finish = None
    try:
        with closing(llm.stream_detailed(ids, sampling)) as stream:
            for chunk in stream:
                seen = chunk.generated_token_ids
                if first_at is None and chunk.text:
                    first_at = time.perf_counter()
                    first_count = tokens[0]
                    if counter is not None:
                        counter.start()
                if seen is not None:
                    output_ids = tuple(int(token) for token in seen)
                if chunk.telemetry is not None and chunk.telemetry.diagnostics:
                    diagnostics = dict(chunk.telemetry.diagnostics)
                if chunk.finish_details is not None:
                    finish = chunk.finish_details.reason
        ended = time.perf_counter()
    finally:
        calls = counter.stop() if counter is not None and counter._originals else None
    if first_at is None or not output_ids:
        raise RuntimeError("no text delta or no generated token IDs")
    intervals = len(output_ids) - first_count
    if intervals < 1:
        raise RuntimeError("no decode step after the first text delta")
    decode_s = ended - first_at
    return {
        "prompt_tokens": len(ids),
        "first_token_s": first_at - started,
        "decode_s": decode_s,
        "decode_intervals": intervals,
        "decode_tok_s": intervals / decode_s,
        "output_ids_sha256": hashlib.sha256(json.dumps(output_ids, separators=(",", ":")).encode()).hexdigest(),
        "output_ids": list(output_ids),
        "finish": finish,
        "diagnostics": diagnostics,
        "calls": calls,
    }


def summarize(samples):
    """Reduce the repetitions of one depth to medians and an output-identity check."""
    digests = {sample["output_ids_sha256"] for sample in samples}

    def spread(key):
        values = [sample[key] for sample in samples]
        return {"median": statistics.median(values), "min": min(values), "max": max(values)}

    calls = {}
    for sample in samples:
        for name, entry in (sample.get("calls") or {}).items():
            total = calls.setdefault(name, {"calls": 0, "seconds": 0.0, "bytes": 0})
            for field in total:
                total[field] += entry[field]
    for entry in calls.values():
        for field in entry:
            entry[field] /= len(samples)
    seconds_median = statistics.median(sample["decode_s"] for sample in samples)
    intervals = samples[0]["decode_intervals"]
    per_token = {name: {field: value / intervals for field, value in entry.items()} for name, entry in calls.items()}
    return {
        "reps": len(samples),
        "first_token_s": spread("first_token_s"),
        "decode_tok_s": spread("decode_tok_s"),
        "decode_intervals": intervals,
        "decode_ms_per_token": seconds_median * 1e3 / intervals,
        "identical_output": len(digests) == 1,
        "output_ids_sha256": sorted(digests),
        "diagnostics": samples[-1]["diagnostics"],
        "calls_per_request": calls,
        "calls_per_token": per_token,
    }


def environment():
    """Record the host and every hipEngine variable that could change what ran."""
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hipengine_env": {name: value for name, value in sorted(os.environ.items())
                          if name.startswith("HIPENGINE_")},
        "rocm_path": os.environ.get("ROCM_PATH", ""),
    }


def provenance(model, args):
    """Canonical artifact provenance, with the repo and model identity other harnesses record.

    The compiler version is read from ``HIPENGINE_COMPILER_VERSION_FILE`` when that file
    exists. Without it no compiler is invoked: ``hipcc --version`` has hung on this host.
    """
    from hipengine.benchmark.provenance import collect_artifact_provenance

    version_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
    hipcc_version = None
    if version_file and Path(version_file).is_file():
        hipcc_version = Path(version_file).read_text(encoding="utf-8")
    return collect_artifact_provenance(
        repo_root=Path(__file__).resolve().parent.parent,
        configured_backend=args.backend,
        model_path=model,
        quant=args.quant,
        kv_dtype="fp32",
        command=[sys.argv[0], *sys.argv[1:]],
        environment={name: value for name, value in sorted(os.environ.items())
                     if name.startswith("HIPENGINE_")},
        timing_protocol=PROTOCOL,
        warmups=args.warmup,
        repetitions=args.reps,
        hipcc_version=hipcc_version,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--execution-profile", default=None,
                        help="profile to request; omit for the shipped default")
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[47, 575, 2057])
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--counts", action="store_true", help="count HIP calls inside the decode window")
    parser.add_argument("--label", default="")
    parser.add_argument("--quant", default=None, help="quantization label recorded in the provenance block")
    parser.add_argument("--output", type=Path, default=None, help="write the summary JSON here")
    args = parser.parse_args(argv)
    if args.output_tokens < 2 or args.reps < 1 or args.warmup < 0 or min(args.prompt_tokens) < 1:
        parser.error("need at least two output tokens, one repetition, a non-negative warmup and positive prompts")
    if max(args.prompt_tokens) + args.output_tokens > args.max_sequence_length:
        parser.error("max-sequence-length must cover the longest prompt plus the output tokens")

    # Keep the spelling the caller used: a GGUF reached through a symlink whose target has
    # another suffix must still be recognised as a GGUF model.
    model = Path(os.path.abspath(args.model))
    if not model.is_file():
        parser.error(f"model file not found: {model}")
    profile = None if args.execution_profile in (None, "", "default") else args.execution_profile
    llm = LLM(model=str(model), backend=args.backend, execution_profile=profile,
              max_sequence_length=args.max_sequence_length, prefix_cache="off", speculative_mtp_serving="off")
    try:
        tokens = install_token_counter(llm._get_text_generator())
        seed = llm.tokenize(_SEED)
        sampling = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
        depths = []
        first_output_ids = {}
        repetitions = []
        for count in args.prompt_tokens:
            ids = prompt_ids(seed, count)
            for _ in range(args.warmup):
                stream_once(llm, ids, sampling, tokens=tokens)
            samples = [stream_once(llm, ids, sampling, tokens=tokens, counts=args.counts) for _ in range(args.reps)]
            depths.append({"prompt_tokens": count, **summarize(samples)})
            first_output_ids[str(count)] = samples[0]["output_ids"]
            repetitions.extend({key: value for key, value in sample.items()
                                if key not in ("output_ids", "diagnostics", "calls")} for sample in samples)
            print("measured prompt_tokens=%d" % count, file=sys.stderr, flush=True)
        result = {
            "tool": "scripts/inkling2_decode_bench.py",
            "label": args.label,
            "model": {"path": str(model), "bytes": model.stat().st_size},
            "backend": args.backend,
            "requested_profile": profile,
            "resolved_profile": getattr(llm, "resolved_execution_profile", None),
            "manifest_sha256": getattr(llm, "execution_profile_manifest_sha256", None),
            "fell_back_to_strict": getattr(llm, "execution_profile_fell_back_to_strict", None),
            "output_tokens": args.output_tokens,
            "max_sequence_length": args.max_sequence_length,
            "warmup": args.warmup,
            "timing_protocol": PROTOCOL,
            "environment": environment(),
            "provenance": provenance(model, args),
            "depths": depths,
            "first_output_ids": first_output_ids,
            "repetitions": repetitions,
        }
        text = json.dumps(result, indent=1)
        if args.output is not None:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)
    finally:
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
