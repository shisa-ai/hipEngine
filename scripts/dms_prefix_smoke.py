"""Check DMS exact, extended, and branching cache reuse through public requests."""
import argparse
import dataclasses
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--codec", default="int8_per_token_head")
    parser.add_argument("--lengths", default="127,128,300")
    parser.add_argument("--branch-at", type=int, default=270)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--mode", choices=("ar", "mtp", "transition"), default="transition")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. ")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.temperature and args.mode != "ar":
        parser.error("sampled requests use --mode ar")
    from hipengine import LLM, SamplingParams
    from hipengine.core.memory import memory_stats
    lengths = sorted(set(int(n) for n in args.lengths.split(",")))
    rows = []
    result = {"completed": False, "rows": rows, "configuration": vars(args)}
    try:
        for cache in ("off", "radix"):
            llm = LLM(args.model, backend=args.backend, execution_profile="production",
                      max_sequence_length=max(lengths) + 32, max_active_requests=1,
                      prefix_cache=cache, kv_storage=args.codec, kv_scale_dtype="fp32",
                      dms_metadata_path=args.metadata, engine_command_timeout_seconds=1800)
            try:
                source = llm.tokenize(args.prompt * (max(lengths) + 1))[:max(lengths)]
                scenarios = []
                previous = 0
                for length in lengths:
                    assert len(source[:length]) == length
                    scenarios.extend(((f"extend-{length}", source[:length], previous, 8),
                                      (f"repeat-{length}", source[:length], length, 12)))
                    previous = length
                if max(lengths) >= args.branch_at + 30:
                    branch = source[:args.branch_at] + llm.tokenize("Explain compiler intermediate representations. " * 20)[:30]
                    shared = max([0, *(length for length in lengths if length <= args.branch_at)])
                    scenarios.append(("branch", branch, shared, 12))
                    scenarios.append(("return-source", source, len(source), 12))
                for index, (name, ids, reused, count) in enumerate(scenarios):
                    mtp = args.mode == "mtp" or args.mode == "transition" and index > 0
                    params = SamplingParams(max_tokens=count, temperature=args.temperature,
                                            seed=42, ignore_eos=not mtp)
                    method = llm.generate_speculative_mtp_detailed if mtp else llm.generate_detailed
                    out = method([ids], params)[0]
                    telemetry = dataclasses.asdict(out.telemetry)
                    prefix = telemetry["diagnostics"]["prefix_cache"]
                    expected = reused if cache == "radix" else 0
                    if name == "branch" and cache == "radix":
                        # Only natural prefill boundaries are snapshotted; an
                        # extension can skip 256 while reusing an earlier end.
                        allowed = {*range(256, args.branch_at + 1, 256),
                                   *(length for length in lengths if length <= args.branch_at)}
                        assert prefix["reused_tokens"] in allowed, (name, prefix)
                        expected = prefix["reused_tokens"]
                    else:
                        assert prefix["reused_tokens"] == expected, (name, prefix)
                    if expected == len(ids):
                        assert telemetry["timing"]["prefill_chunk_count"] == 0
                        assert telemetry["diagnostics"]["dms"]["operations"]["streaming_pack_calls"] == 0
                    if mtp:
                        assert telemetry["timing"].get("mtp_cycles_count", 0) > 0, telemetry
                    row = {"cache": cache, "scenario": name, "length": len(ids), "mtp": mtp,
                           "max_tokens": count, "ids": list(out.generated_token_ids), "telemetry": telemetry}
                    rows.append(row)
                    print(json.dumps({key: row[key] for key in ("cache", "scenario", "mtp", "ids")}
                                     | {"reused_tokens": prefix["reused_tokens"]}), flush=True)
                    if cache == "radix":
                        baseline = next(r for r in rows if r["cache"] == "off" and r["scenario"] == name)
                        assert row["ids"] == baseline["ids"], (name, row["ids"], baseline["ids"])
            finally:
                llm.close()
                result[f"memory_after_{cache}_close"] = memory_stats()
            assert memory_stats()["current_allocated_bytes"] == 0, memory_stats()
        result["completed"] = True
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()
