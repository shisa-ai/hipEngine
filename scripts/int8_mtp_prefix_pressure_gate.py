#!/usr/bin/env python3
"""Drive the prefix cache to eviction with MTP requests and check the parity.

Run against two servers that serve the same model, the same capacity and the
same KV storage, one with prefix reuse enabled and one with it off:

    HIPENGINE_PREFIX_CACHE=radix .venv/bin/python -m hipengine.server \
        --model <gguf> --backend hip_gfx1151 --kv-storage int8_per_token_head \
        --max-active-requests 4 --port 8098
    HIPENGINE_PREFIX_CACHE=off .venv/bin/python -m hipengine.server \
        --model <gguf> --backend hip_gfx1151 --kv-storage int8_per_token_head \
        --max-active-requests 4 --port 8099

    .venv/bin/python scripts/int8_mtp_prefix_pressure_gate.py \
        --base-url http://127.0.0.1:8098 \
        --no-reuse-base-url http://127.0.0.1:8099 \
        --json /tmp/int8-mtp-prefix-pressure.json

Why the workload is shaped this way. The retained prefix working set is bounded
by the engine capacity -- `_prefix_retained_limit` defaults to the request
capacity, one durable boundary per active request -- and `_trim_prefix_snapshots`
evicts oldest-first. So seeding `2 * capacity` distinct prompts deterministically
evicts the oldest `capacity` of them, and the first-seeded prompt is a guaranteed
miss while the last-seeded one is a guaranteed hit. Seeding sequentially keeps
that order deterministic; the final pair is issued concurrently so a real
overlapping MTP request is live while the cache is under pressure.

The two miss causes are distinct and both are reported: `fallback_reason`
`"miss"` is "no target hit", while `"provider_checkpoint_unavailable"` is "the
target hit but its provider checkpoint was not usable". This gate asserts the
first and records the second when it appears.

Parity claims: the same request sequence runs in all four configurations
(reuse on/off x MTP on/off) and every probe must return the same generated ids.
Prefix reuse changes which tokens are prefilled, so identical ids across the
reuse axis are the direct evidence that a reused provider position is not
corrupted.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

BLOCK_TOKENS = 256
MISS_REASONS = ("miss", "provider_checkpoint_unavailable")


def _prefix_cache(metadata):
    return metadata["diagnostics"]["prefix_cache"]


def check_reporting(metadata, *, expect_hit, mode, prompt_tokens, label):
    """Assert the request diagnostics describe the hit, miss or disabled state.

    A disabled cache reports ``cache_off`` and never looks up, so the miss
    assertions below belong to the radix configuration only.
    """

    cache = _prefix_cache(metadata)
    reused = int(cache["reused_tokens"])
    if mode == "off":
        assert not cache["hit"] and reused == 0, (label, cache)
        assert int(cache["matched_tokens"]) == 0, (label, cache)
        assert cache["fallback_reason"] == "cache_off", (label, cache)
        assert int(cache["cache_resident_entries"]) == 0, (label, cache)
        return cache
    assert bool(cache["lookup"]) is True, (label, cache)
    if expect_hit:
        assert cache["hit"] and reused >= BLOCK_TOKENS, (label, cache)
    else:
        assert not cache["hit"] and reused == 0, (label, cache)
        assert int(cache["matched_tokens"]) == 0, (label, cache)
        assert cache["fallback_reason"] == "miss", (label, cache)
    assert int(cache["matched_tokens"]) >= reused, (label, cache)
    assert reused % BLOCK_TOKENS == 0, (label, cache)
    assert int(cache["executed_prefill_tokens"]) == prompt_tokens - reused, (label, cache)
    assert int(cache["avoided_prefill_tokens"]) == reused, (label, cache)
    return cache


def run_configuration(args, base_url, *, reuse, prefixes, suffix, report):
    """Seed `2 * capacity` distinct prompts, then probe the eviction extremes."""

    label = "reuse" if reuse else "no-reuse"
    rows = {"mode": None, "probes": {}, "seeds": []}
    report["configurations"][label] = rows
    with httpx.Client(base_url=base_url, timeout=600) as client:
        readiness = client.get("/ready")
        readiness.raise_for_status()
        payload = readiness.json()
        rows["ready"] = payload.get("prefix_cache")
        expected_mode = "radix" if reuse else "off"
        assert payload["prefix_cache"]["mode"] == expected_mode, payload["prefix_cache"]
        rows["mode"] = expected_mode
        capability = payload["model"]["kv_capability"]
        assert capability["effective_kv_storage"] == "int8_per_token_head", capability
        if capability.get("diagnostic_override"):
            assert args.allow_kv_diagnostic_override, capability

        def request(tokens, speculative, max_tokens):
            response = client.post(
                "/v1/completions",
                json={
                    "model": args.model,
                    "prompt": tokens,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "speculative_mtp": speculative,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["hipengine"]

        def seed(prompt):
            return request(prompt, True, 8)

        # Sequential: the trim is oldest-first, so the eviction order has to be
        # the submission order for the extremes below to be deterministic.
        for prompt in prefixes[:-2]:
            metadata = seed(prompt)
            rows["seeds"].append(int(_prefix_cache(metadata)["cache_resident_entries"]))
        # Concurrent tail: a real overlapping MTP request is live while the
        # cache evicts, and neither request can evict the other's boundary
        # because they are the two newest.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for metadata in pool.map(seed, prefixes[-2:]):
                rows["seeds"].append(int(_prefix_cache(metadata)["cache_resident_entries"]))

        probes = {}
        # Each arm observes a genuine outcome. A miss re-populates a boundary,
        # so the two arms of the evicted probe must use different prompts: the
        # second request on the same prompt would legitimately hit. The resident
        # probe can reuse one prompt because a hit refreshes the boundary it just
        # used rather than adding one.
        plan = (
            ("evicted_mtp", 0, True),
            ("evicted_ar", 1, False),
            ("resident_mtp", len(prefixes) - 1, True),
            ("resident_ar", len(prefixes) - 1, False),
        )
        for name, index, speculative in plan:
            # A probe must extend the seeded boundary: reusing an exact full
            # prompt is a distinct outcome (`full_prompt_boundary_requires_suffix`),
            # not the hit this gate is measuring.
            prompt = prefixes[index] + suffix
            expect_hit = reuse and name.startswith("resident")
            metadata = request(prompt, speculative, 24)
            cache = check_reporting(
                metadata,
                expect_hit=expect_hit,
                mode=expected_mode,
                prompt_tokens=len(prompt),
                label=f"{label}:{name}",
            )
            cycles = 0
            if speculative:
                cycles = int(metadata.get("timing", {}).get("mtp_cycles_count", 0))
                assert cycles > 0, metadata
            probes[name] = {
                "index": index,
                "expected": "hit" if expect_hit else ("miss" if reuse else "no_cache"),
                "speculative": speculative,
                "fallback_reason": cache["fallback_reason"],
                "reused_tokens": int(cache["reused_tokens"]),
                "matched_tokens": int(cache["matched_tokens"]),
                "cache_resident_entries": int(cache["cache_resident_entries"]),
                "cycles": cycles,
                "ids": metadata["generated_token_ids"],
            }
        rows["probes"] = probes
        assert probes["resident_mtp"]["ids"] == probes["resident_ar"]["ids"], {
            "stage": f"{label}:mtp-vs-ar-over-reused-kv",
            "mtp": probes["resident_mtp"]["ids"],
            "ar": probes["resident_ar"]["ids"],
        }
        if reuse:
            # The evicted probe must miss on a cache that is demonstrably still
            # working: the newest boundary served a hit in the same run.
            assert probes["evicted_mtp"]["expected"] == "miss"
            assert probes["resident_mtp"]["expected"] == "hit"
            assert probes["resident_ar"]["expected"] == "hit"
            assert probes["resident_mtp"]["cache_resident_entries"] > 0
            assert max(rows["seeds"]) <= args.capacity + 1, rows["seeds"]
        return rows


def run(args):
    report = {"passed": False, "configurations": {}, "parity": {}}
    try:
        prompt_file = (
            Path(__file__).resolve().parents[1] / "benchmarks/prompts/mtpbench-code-general-ja.jsonl"
        )
        texts = [json.loads(line)["messages"][0]["content"] for line in prompt_file.read_text().splitlines()]
        with httpx.Client(base_url=args.base_url, timeout=600) as client:
            def encode(text):
                response = client.post("/v1/hipengine/tokenize", json={"text": text})
                response.raise_for_status()
                return response.json()["token_ids"]

            prompts = []
            for text in texts:
                seed = encode(text)
                prompts.append((seed * ((args.prefix_tokens + len(seed) - 1) // len(seed)))[: args.prefix_tokens])
            suffix = encode(" Now summarize the main invariant in one short paragraph.")
            needed = 2 * args.capacity
            assert len(prompts) >= needed, (len(prompts), needed)
            prefixes = prompts[:needed]
            report["prefix_tokens"] = args.prefix_tokens
            report["suffix_tokens"] = len(suffix)
            report["capacity"] = args.capacity
            report["prompt_count"] = len(prefixes)

            run_configuration(
                args, args.base_url, reuse=True, prefixes=prefixes, suffix=suffix, report=report
            )
            run_configuration(
                args,
                args.no_reuse_base_url,
                reuse=False,
                prefixes=prefixes,
                suffix=suffix,
                report=report,
            )

        reuse = report["configurations"]["reuse"]
        off = report["configurations"]["no-reuse"]
        for name in ("evicted_mtp", "evicted_ar", "resident_mtp", "resident_ar"):
            assert reuse["probes"][name]["ids"] == off["probes"][name]["ids"], {
                "stage": f"reuse-parity:{name}",
                "reuse": reuse["probes"][name]["ids"],
                "no_reuse": off["probes"][name]["ids"],
            }
            report["parity"][name] = reuse["probes"][name]["ids"]
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "configurations"}, ensure_ascii=False))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--no-reuse-base-url", default="http://127.0.0.1:8099")
    parser.add_argument("--model", default="int8-mtp")
    parser.add_argument(
        "--capacity",
        type=int,
        default=4,
        help="The servers' --max-active-requests; the retained working set is one entry per request.",
    )
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--allow-kv-diagnostic-override", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
