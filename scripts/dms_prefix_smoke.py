"""Exercise full-prompt DMS cache reuse through LLM.generate_detailed."""
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
    parser.add_argument("--lengths", default="127,256,300")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from hipengine import LLM, SamplingParams
    from hipengine.core.memory import memory_stats
    lengths = [int(n) for n in args.lengths.split(",")]
    rows = []
    result = {"completed": False, "rows": rows, "configuration": vars(args)}
    try:
        for cache in ("off", "radix"):
            llm = LLM(args.model, backend=args.backend, execution_profile="production",
                      max_sequence_length=max(lengths) + 32, max_active_requests=1,
                      prefix_cache=cache, kv_storage=args.codec, kv_scale_dtype="fp32",
                      dms_metadata_path=args.metadata, engine_command_timeout_seconds=1800)
            try:
                source = llm.tokenize("The quick brown fox jumps over the lazy dog. " * (max(lengths) // 8 + 8))
                for length in lengths:
                    ids = source[:length]
                    assert len(ids) == length
                    for repeat, count in enumerate((4, 8)):
                        params = SamplingParams(max_tokens=count, temperature=args.temperature,
                                                seed=42, ignore_eos=True)
                        out = llm.generate_detailed([ids], params)[0]
                        telemetry = dataclasses.asdict(out.telemetry)
                        prefix = telemetry["diagnostics"]["prefix_cache"]
                        expected = length if cache == "radix" and repeat else 0
                        assert prefix["reused_tokens"] == expected, prefix
                        if expected:
                            assert telemetry["timing"]["prefill_chunk_count"] == 0
                            assert prefix["cache_resident_bytes"] > 0
                            assert telemetry["diagnostics"]["dms"]["operations"]["streaming_pack_calls"] == 0
                        row = {"cache": cache, "length": length, "max_tokens": count,
                               "ids": list(out.generated_token_ids), "telemetry": telemetry}
                        rows.append(row)
                        print(json.dumps({"cache": cache, "length": length, "repeat": repeat,
                                          "reused_tokens": expected, "ids": row["ids"]}), flush=True)
                        if cache == "radix":
                            baseline = next(r for r in rows if r["cache"] == "off" and
                                            r["length"] == length and r["max_tokens"] == count)
                            assert row["ids"] == baseline["ids"], (row["ids"], baseline["ids"])
            finally:
                llm.close()
                result[f"memory_after_{cache}_close"] = memory_stats()
            assert memory_stats()["current_allocated_bytes"] == 0
        result["completed"] = True
    finally:
        args.output.write_text(json.dumps(result, default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()
