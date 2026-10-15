"""Check public DMS MTP failure recovery, cache eviction, and device cleanup.

Requires a HIP device and a model with a DMS sidecar and NextN weights. Injects
one predictor failure and constrains the cache to one entry; not a benchmark.
"""
import argparse
import dataclasses
import json
from pathlib import Path
from unittest.mock import patch


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
    from hipengine.runtime.dms_prefix import DMSPrefixCache
    from hipengine.runtime import qwen35_gguf_runner as runtime

    original = DMSPrefixCache.__init__

    def one_entry(self, *positional, **kwargs):
        original(self, *positional, **kwargs)
        self.max_entries = 1

    result = {"completed": False, "configuration": vars(args), "rows": []}
    llm = None
    try:
        llm = LLM(args.model, backend=args.backend, execution_profile="production",
                  max_sequence_length=340, max_active_requests=1, prefix_cache="radix",
                  kv_storage=args.codec, kv_scale_dtype="fp32", dms_metadata_path=args.metadata,
                  engine_command_timeout_seconds=1800)
        source = llm.tokenize("The quick brown fox jumps over the lazy dog. " * 100)
        params = SamplingParams(max_tokens=8, temperature=0)

        def generate(name, length, reused):
            out = llm.generate_speculative_mtp_detailed([source[:length]], params)[0]
            telemetry = dataclasses.asdict(out.telemetry)
            result["rows"].append({"scenario": name, "ids": list(out.generated_token_ids),
                                   "telemetry": telemetry})
            assert telemetry["diagnostics"]["prefix_cache"]["reused_tokens"] == reused
            assert telemetry["timing"].get("mtp_cycles_count", 0) > 0
            return out.generated_token_ids

        with patch.object(DMSPrefixCache, "__init__", one_entry):
            first = generate("initial", 127, 0)
        message = "injected DMS cache prefill failure"
        with patch.object(runtime._ExternalDMSDevicePrefillCollector, "finalize",
                          side_effect=RuntimeError(message)):
            try:
                llm.generate_speculative_mtp_detailed([source[:300]], params)
            except Exception as exc:
                result["expected_failure"] = str(exc)
                assert message in str(exc)
            else:
                raise AssertionError("injected failure did not reach caller")
        assert generate("recover", 127, 127) == first
        second = generate("extend", 300, 127)
        assert generate("repeat", 300, 300) == second
        assert generate("evicted", 127, 0) == first
        llm.close()
        llm = None
        result["memory_after_close"] = memory_stats()
        assert memory_stats()["current_allocated_bytes"] == 0, memory_stats()
        result["completed"] = True
    finally:
        if llm is not None:
            llm.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, default=str, indent=2) + "\n")
    print("DMS failure recovery, eviction, and cleanup passed", flush=True)


if __name__ == "__main__":
    main()
