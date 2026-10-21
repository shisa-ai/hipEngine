#!/usr/bin/env python3
"""Replay Gufo's ordered HTTP depth recipe without modifying its source tree.

The reviewed source checkout supplies the prompt generator, calibration, prefix
reply and retry rules. Record every request and prefix reply, including failed
attempts. --repetitions 1 preserves the publication's sample count; higher
counts are a separately labelled extension. Output remains diagnostic until
its G0–G4 characterization is bound to these exact rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEPTHS = (0, 4096, 8192, 12288, 16384, 32768, 65536, 131072)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_paths(source: Path, output: Path) -> None:
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("output must be outside the read-only reference checkout")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; preserve previous measurements")


def quality_columns() -> dict[str, Any]:
    return {
        "G0": {"status": "pending", "reason": "bind full identities, tokenization and host-state checks"},
        "G1": {"status": "unavailable", "reason": "HTTP observations expose text hashes, not native output IDs"},
        "G2": {"status": "not_applicable", "reason": "Gufo HTTP exposes no logits/logprob surface"},
        "G3": {"status": "pending", "reason": "requires identical-input repeats and cross-launch checks"},
        "G4": {"status": "pending", "reason": "bind the same-pin task battery and workload scope"},
    }


def checked_sample(observation, *, depth: int, outputs: int, speculative: bool) -> dict[str, Any]:
    counts = {name: getattr(observation, name) for name in (
        "prompt_tokens", "cached_prompt_tokens", "prefill_tokens", "completion_tokens",
        "draft_tokens", "draft_accepted_tokens",
    )}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("native token counts must be non-negative integers")
    if (counts["completion_tokens"] != outputs
            or counts["prompt_tokens"] != counts["cached_prompt_tokens"] + counts["prefill_tokens"]
            or abs(counts["cached_prompt_tokens"] - depth) > max(32, int(depth * .005))
            or abs(counts["prefill_tokens"] - 2048) > 32
            or counts["draft_accepted_tokens"] > counts["draft_tokens"]):
        raise ValueError("native counts disagree with the ordered depth protocol")
    for name in ("prefill_ms", "decode_ms"):
        value = getattr(observation, name)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not speculative and counts["draft_tokens"]:
        raise ValueError("AR control executed draft work")
    if speculative and not counts["draft_tokens"]:
        raise ValueError("MTP measurement has no native speculative-work witness")
    return {
        "observation": observation.public(),
        "rates": {
            "prefill_new_tok_s": counts["prefill_tokens"] * 1000 / observation.prefill_ms,
            "decode_output_tok_s": outputs * 1000 / observation.decode_ms,
            "decode_transition_tok_s": (outputs - 1) * 1000 / observation.decode_ms,
        },
        "output_identity": "visible_text_sha256_not_native_ids",
    }


def ordered_depths(upstream, session, url, tokenizer, *, depths, repetitions, task, speculative,
                   save=None):
    """Keep one tokenizer and resident session across the complete ordered history."""
    session.scope = {"phase": "warmup"}
    session.request(url, upstream.synthetic_text(8888, tokenizer.words_for(2048)), 16)
    rows = []
    for depth in depths:
        for repetition in range(repetitions):
            session.scope = {"phase": "depth", "depth": depth, "repetition": repetition, "task": task}
            sample = upstream._measure_depth(
                session, url, tokenizer, depth=depth, prompt_tokens=2048,
                output_tokens=128, fraction=.005, seed=1, repetition=repetition, task=task,
            )
            row = {**session.scope, **checked_sample(sample, depth=depth, outputs=128, speculative=speculative)}
            rows.append(row)
            if save is not None:
                save(rows)
            print(f"{task} d{depth} repetition {repetition}: native counts and route checked", flush=True)
    return rows


def host_state():
    from scripts.qwen4exp_canonical_ar_bench import _host_metadata
    state = _host_metadata()
    state["gpu_temperature_millidegrees_c"] = {}
    for path in Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*/temp*_input"):
        try:
            state["gpu_temperature_millidegrees_c"][str(path)] = int(path.read_text())
        except (OSError, ValueError):
            pass
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="First target GGUF shard")
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--arms", default="ar,mtp-prose,mtp-repetition")
    parser.add_argument("--depths", default=",".join(map(str, DEPTHS)),
                        help="Ordered depths; a subset is labelled a diagnostic, never an exact replay")
    parser.add_argument("--context", type=int, default=133760)
    parser.add_argument("--probe-only", action="store_true",
                        help="Run short-prompt allocation probes at the requested capacity, then exit")
    args = parser.parse_args(argv)
    validate_paths(args.source_root, args.output_dir)
    arms = args.arms.split(",")
    depths = [int(value) for value in args.depths.split(",")]
    if (args.repetitions < 1 or args.context < max(depths, default=0) + 2176
            or not depths or depths != sorted(set(depths)) or any(d < 0 for d in depths)
            or not arms or len(set(arms)) != len(arms) or set(arms) - {"ar", "mtp-prose", "mtp-repetition"}):
        parser.error("require ordered unique depths, positive repetitions, sufficient context and unique named arms")
    source = args.source_root.resolve()
    config_path = source / "docs/models/qwen3.8-flash-next/artifacts/bench.json"
    # Import the reviewed harness without writing bytecode into its checkout.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "tools"))
    from gufo.model_bench import llm as upstream
    from gufo.model_bench.config import load_config
    from gufo.model_bench.servers import HipMemory
    from scripts.qwen4exp_canonical_ar_bench import _git_metadata

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(source, "qwen3.8-flash-next", config_path)
    config.artifacts_override = args.output_dir
    config.files = {"gguf": {"default": args.model.resolve()}, "mtp": {"default": args.sidecar.resolve()}}
    source_identity = _git_metadata(source)
    identity = {
        "source": source_identity, "binary_sha256": sha256(args.binary),
        "benchmark_sources": {str(p.relative_to(source)): sha256(p) for p in (
            config_path, source / "tools/gufo/model_bench/llm.py", source / "tools/gufo/serving_bench.py")},
        "harness_sha256": sha256(Path(__file__)),
        "command": [str(Path(__file__).relative_to(ROOT)), *(sys.argv[1:] if argv is None else argv)],
    }
    journal_path = args.output_dir / "requests.jsonl"

    class RecordedSession(upstream.Session):
        scope = None

        arm = None

        def record(self, kind, inputs, output):
            row = {"kind": kind, "arm": self.arm, "scope": self.scope, "inputs": inputs, "output": output}
            with journal_path.open("a") as journal:
                journal.write(json.dumps(row, ensure_ascii=False) + "\n")

        def request(self, base_url, prompt, max_tokens, **kwargs):
            inputs = {"prompt": prompt, "max_tokens": max_tokens, **kwargs}
            try:
                result = super().request(base_url, prompt, max_tokens, **kwargs)
            except BaseException as exc:
                self.record("request_failure", inputs, {"error": str(exc)})
                raise
            self.record("request", inputs, result.public())
            return result

        def chat_text(self, base_url, messages, max_tokens, extra_body=None):
            inputs = {"messages": messages, "max_tokens": max_tokens, "extra_body": extra_body}
            try:
                result = super().chat_text(base_url, messages, max_tokens, extra_body)
            except BaseException as exc:
                self.record("prefix_failure", inputs, {"error": str(exc)})
                raise
            self.record("prefix_reply", inputs, {"text": result})
            return result

    session = RecordedSession(
        config, "gufo", gufo_binary=args.binary.resolve(), reference_binary="unused",
        source={"revision": source_identity["head"], "dirty": not source_identity["tracked_clean"]},
        fingerprint=None, log_dir=args.output_dir, document="BENCHMARKS", todo_only=False,
    )
    memory = HipMemory.for_binary(args.binary)
    report = {"schema": 1, "status": "running", "performance_claim": False, "identity": identity,
              "protocol": "capacity_probe" if args.probe_only else (
                  "published_ordered_depths" if tuple(depths) == DEPTHS and args.repetitions == 1
                  and args.context == 133760 else "declared_depth_extension"),
              "quality": quality_columns(), "arms": {}}
    report_path = args.output_dir / "replay.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for arm in arms:
            session.arm = arm
            mode = "ar" if arm == "ar" else "mtp"
            task = "repetition" if arm == "mtp-repetition" else "prose"
            table = config.table("single-ar" if mode == "ar" else "single-mtp")
            block = {"host_before": host_state(), "rows": [], "probe": None,
                     "idle_device_gib": None if memory is None else memory.used_gib()}
            report["arms"][arm] = block
            # Prove allocation with a short request, then reopen a fresh server so
            # the capacity check cannot change the published prefix history.
            probe = session.server(table, mode=mode, context=args.context, sessions=1, tag=f"{arm}-probe")
            block["server_command"] = probe.command
            try:
                with probe:
                    session.scope = {"phase": "capacity_probe", "arm": arm, "context": args.context}
                    sample = session.request(probe.base_url, upstream.synthetic_text(101, 512), 4)
                    if sample.completion_tokens != 4:
                        raise ValueError("capacity probe did not complete four output tokens")
                    block["probe"] = sample.public()
            finally:
                probe.stop()
            save()
            if not args.probe_only:
                server = session.server(table, mode=mode, context=args.context, sessions=1, tag=arm)
                block["server_command"] = server.command
                try:
                    with server:
                        session.scope = {"phase": "calibration", "arm": arm}
                        tokenizer = upstream.Tokenizer(session, server.base_url, table.variant)
                        block["calibration"] = {"overhead": tokenizer.overhead, "initial_ratio": tokenizer.ratio}
                        def progress(rows):
                            block["rows"] = rows
                            save()
                        ordered_depths(upstream, session, server.base_url, tokenizer, depths=depths,
                                       repetitions=args.repetitions, task=task, speculative=mode != "ar", save=progress)
                        block["calibration"]["final_ratio"] = tokenizer.ratio
                finally:
                    server.stop()
            block["host_after"] = host_state()
            save()
        report["status"] = "capacity_probe_passed" if args.probe_only else "measured_diagnostic"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report["source_after"] = _git_metadata(source)
        report["request_journal_sha256"] = sha256(journal_path) if journal_path.exists() else None
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
