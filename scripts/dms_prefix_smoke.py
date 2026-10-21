"""Check public DMS cache reuse against uncached, matched-partition execution.

Fresh whole-prompt output is a diagnostic, not the restoration oracle: changing
prefill chunk widths can change arithmetic. The matched arm replays the actual
per-layer-kind partitions recorded by the cached arm without restoring state.
"""
import argparse
import dataclasses
import json
from pathlib import Path
from unittest.mock import patch


def compose_partition(prefix, reused, suffix, total):
    """Join a restored prefix and computed suffix without duplicating layers."""
    result = [tuple(span) for span in prefix if span[1] <= reused]
    result.extend((a + reused, b + reused) for a, b in suffix)
    cursor = 0
    for start, end in result:
        if start != cursor or end <= start:
            raise ValueError(f"non-contiguous partition: {result}")
        cursor = end
    if cursor != total:
        raise ValueError(f"partition covers {cursor}, expected {total}")
    return tuple(result)


def validate_reuse_count(*, cache, scenario, actual, expected, lengths, branch_at):
    """Check reuse without requiring an evicted source snapshot to survive."""
    if cache != "radix":
        assert actual == 0, (scenario, actual)
    elif scenario == "branch":
        allowed = {*range(256, branch_at + 1, 256),
                   *(length for length in lengths if length <= branch_at)}
        assert actual in allowed, (scenario, actual)
    elif scenario == "return-source":
        # The host byte budget may evict the full source after a branch. The
        # caller still requires a matching recorded parent for positive reuse
        # and an exact matched-partition output, including any recomputed tail.
        assert 0 <= actual <= max(lengths), (scenario, actual)
    else:
        assert actual == expected, (scenario, actual, expected)
    return actual


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
    from hipengine.runtime import qwen35_gguf_runner as runtime
    lengths = sorted(set(int(n) for n in args.lengths.split(",")))
    rows, plans, prompt_ids = [], {}, {}
    result = {"completed": False, "rows": rows, "configuration": vars(args)}
    original_ranges = runtime._chunk_ranges
    cache, name, ids, observed = "", "", (), {}

    def ranges(total, chunk_size, *, min_chunk_size=1):
        if cache == "matched":
            partition = plans[name][min_chunk_size]
            assert total == len(ids), (total, len(ids))
            assert all(b - a <= chunk_size for a, b in partition), (partition, chunk_size)
            return partition
        spans = original_ranges(total, chunk_size, min_chunk_size=min_chunk_size)
        if cache == "radix":
            record = (len(ids) - total, spans)
            # Each recurrent/attention layer repeats its kind's partition.
            assert min_chunk_size not in observed or observed[min_chunk_size] == record
            observed[min_chunk_size] = record
        return spans

    try:
        with patch.object(runtime, "_chunk_ranges", ranges):
            for cache in ("off", "radix", "matched"):
                llm = LLM(args.model, backend=args.backend, execution_profile="production",
                          max_sequence_length=max(lengths) + 32, max_active_requests=1,
                          prefix_cache="off" if cache == "matched" else cache,
                          kv_storage=args.codec, kv_scale_dtype="fp32",
                          dms_metadata_path=args.metadata, engine_command_timeout_seconds=1800)
                try:
                    source = llm.tokenize(args.prompt * (max(lengths) + 1))[:max(lengths)]
                    scenarios, previous = [], 0
                    for length in lengths:
                        assert len(source[:length]) == length
                        scenarios.extend(((f"extend-{length}", source[:length], previous, 8),
                                          (f"repeat-{length}", source[:length], length, 12)))
                        previous = length
                    if max(lengths) >= args.branch_at + 30:
                        branch = source[:args.branch_at] + llm.tokenize("Explain compiler intermediate representations. " * 20)[:30]
                        scenarios.extend((("branch", branch, 0, 12),
                                          ("return-source", source, len(source), 12)))
                    for index, (name, ids, reused, count) in enumerate(scenarios):
                        observed = {}
                        mtp = args.mode == "mtp" or args.mode == "transition" and index > 0
                        params = SamplingParams(max_tokens=count, temperature=args.temperature,
                                                seed=42, ignore_eos=not mtp)
                        method = llm.generate_speculative_mtp_detailed if mtp else llm.generate_detailed
                        out = method([ids], params)[0]
                        telemetry = dataclasses.asdict(out.telemetry)
                        prefix = telemetry["diagnostics"]["prefix_cache"]
                        expected = validate_reuse_count(
                            cache=cache, scenario=name, actual=prefix["reused_tokens"],
                            expected=reused, lengths=lengths, branch_at=args.branch_at,
                        )
                        if expected == len(ids):
                            assert telemetry["timing"]["prefill_chunk_count"] == 0
                            assert telemetry["diagnostics"]["dms"]["operations"]["streaming_pack_calls"] == 0
                        if mtp:
                            assert telemetry["timing"].get("mtp_cycles_count", 0) > 0, telemetry
                        if cache == "radix":
                            parent = next((key for key, tokens in reversed(list(prompt_ids.items()))
                                           if expected and len(tokens) >= expected
                                           and tuple(tokens[:expected]) == tuple(ids[:expected])
                                           and all(expected in [b for a, b in plan]
                                                   for plan in plans[key].values())), None)
                            assert not expected or parent is not None, (name, expected)
                            inherited = plans[parent] if parent is not None else {}
                            kinds = set(inherited) | set(observed)
                            plans[name] = {}
                            for kind in kinds:
                                offset, suffix = observed.get(kind, (len(ids), ()))
                                assert offset == expected, (name, offset, expected)
                                plans[name][kind] = compose_partition(inherited.get(kind, ()), expected, suffix, len(ids))
                            assert plans[name], name
                            prompt_ids[name] = tuple(ids)
                        row = {"cache": cache, "scenario": name, "length": len(ids), "mtp": mtp,
                               "max_tokens": count, "ids": list(out.generated_token_ids), "telemetry": telemetry}
                        rows.append(row)
                        if cache == "radix":
                            baseline = next(r for r in rows if r["cache"] == "off" and r["scenario"] == name)
                            row["natural_ids_equal"] = row["ids"] == baseline["ids"]
                            row["partitions"] = plans[name]
                        if cache == "matched":
                            cached = next(r for r in rows if r["cache"] == "radix" and r["scenario"] == name)
                            assert row["ids"] == cached["ids"], (name, row["ids"], cached["ids"])
                            row["matched_partition_exact"] = True
                        print(json.dumps({key: row[key] for key in ("cache", "scenario", "mtp", "ids")}
                                         | {"reused_tokens": prefix["reused_tokens"]}), flush=True)
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
