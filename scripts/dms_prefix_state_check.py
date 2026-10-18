"""Verify DMS prefix restore against an uncached, matched-partition control.

This checks state restoration, not latency or whole-prompt numerical promotion.
The ordinary uncached arm is recorded separately: changing prefill chunk widths
can change arithmetic even when restoration is byte-exact.
"""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--codec", default="int8_per_token_head")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from hipengine import LLM, SamplingParams
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import qwen35_gguf_runner as runtime
    from hipengine.runtime.dms_prefill_cache import read_prefix
    from hipengine.runtime.dms_prefix import _buffers

    records = {}
    arm, length = "", 0
    original_ranges = runtime._chunk_ranges
    original_prefill = runtime.Qwen35GGUFResidentSession.prefill
    original_finalize = runtime._ExternalDMSDevicePrefillCollector.finalize

    def digest(array):
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

    def ranges(total, chunk_size, *, min_chunk_size=1):
        if arm == "matched" and total == 513:
            return ((0, 255), (255, 257), (257, 513))
        return original_ranges(total, chunk_size, min_chunk_size=min_chunk_size)

    def finalize(collector, *positional, **kwargs):
        shape = (collector.source.config.num_layers, collector.token_count, collector.num_kv_heads)
        scores = read_prefix(collector._logits, int(np.prod(shape)) * 4)
        decisions = original_finalize(collector, *positional, **kwargs)
        records[(arm, length)].update(scores=digest(scores), decisions=digest(decisions))
        return decisions

    def prefill(session, *positional, **kwargs):
        kwargs["return_logits"] = True
        result = original_prefill(session, *positional, **kwargs)
        if session._dms_source is not None:
            state = [digest(read_prefix(buffer, buffer.nbytes)) for buffer in _buffers(session, False)]
            records[(arm, length)].update(logits=digest(result.logits), recurrent=state)
        return result

    result = {"completed": False, "configuration": vars(args), "rows": []}
    try:
        with patch.object(runtime, "_chunk_ranges", ranges), \
             patch.object(runtime.Qwen35GGUFResidentSession, "prefill", prefill), \
             patch.object(runtime._ExternalDMSDevicePrefillCollector, "finalize", finalize):
            for arm, cache in (("natural", "off"), ("matched", "off"), ("cached", "radix")):
                llm = LLM(args.model, backend=args.backend, execution_profile="production",
                          max_sequence_length=545, max_active_requests=1, prefix_cache=cache,
                          kv_storage=args.codec, kv_scale_dtype="fp32", dms_metadata_path=args.metadata,
                          engine_command_timeout_seconds=1800)
                try:
                    source = llm.tokenize("Explain how 日本語 and English technical documentation can "
                                          "describe the same algorithm. " * 514)
                    for length in ((255, 257, 513) if arm == "cached" else (513,)):
                        record = records[(arm, length)] = {"arm": arm, "length": length}
                        result["rows"].append(record)
                        out = llm.generate_detailed([source[:length]], SamplingParams(
                            max_tokens=8, temperature=0, ignore_eos=True))[0]
                        record["ids"] = list(out.generated_token_ids)
                        record["reused_tokens"] = out.telemetry.diagnostics["prefix_cache"]["reused_tokens"]
                        print(json.dumps(record), flush=True)
                finally:
                    llm.close()
                assert memory_stats()["current_allocated_bytes"] == 0, memory_stats()
        for field in ("logits", "scores", "decisions", "recurrent", "ids"):
            assert records[("matched", 513)][field] == records[("cached", 513)][field], field
        assert records[("cached", 513)]["reused_tokens"] == 257
        result["natural_ids_equal"] = records[("natural", 513)]["ids"] == records[("cached", 513)]["ids"]
        result["matched_partition_exact"] = True
        result["completed"] = True
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()
