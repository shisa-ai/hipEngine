#!/usr/bin/env python3
"""Gate Laguna public blocking/streaming generation against direct eager c=1."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/home/lhl/models/gguf/laguna-s-2.1-Q4_K_M.gguf")
DEFAULT_TEMPLATE = ROOT / "tests/fixtures/laguna_poolside_v1_template.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--case", default="oracle_no_thinking")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser


def _progress(label: str, started: float) -> None:
    print(f"{label} elapsed_s={time.perf_counter() - started:.3f}", file=sys.stderr, flush=True)


def run_gate(
    model: str | Path,
    *,
    template_path: Path,
    case_name: str,
    backend: str,
    max_tokens: int,
) -> dict[str, object]:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    case = next(item for item in template["cases"] if item["name"] == case_name)
    prompt_ids = tuple(int(token) for token in case["token_ids"])
    params = SamplingParams(max_tokens=max_tokens)
    tracked_before = memory_stats()
    llm = LLM(str(model), backend=backend)
    started = time.perf_counter()
    result: dict[str, object] = {}
    try:
        tokenized = llm.tokenize(case["rendered"])
        _progress("tokenizer-ready", started)

        blocking_started = time.perf_counter()
        blocking = llm.generate_detailed(prompt_ids, params)[0]
        blocking_seconds = time.perf_counter() - blocking_started
        _progress("blocking-complete", started)

        stream_started = time.perf_counter()
        chunks = list(llm.stream_detailed(prompt_ids, params))
        stream_seconds = time.perf_counter() - stream_started
        _progress("stream-complete", started)
        if not chunks:
            raise RuntimeError("Laguna public stream produced no terminal chunk")
        terminal = chunks[-1]

        wrapper = llm._text_generator
        inner = None if wrapper is None else getattr(wrapper, "inner", None)
        weights = None if inner is None else getattr(inner, "resident_weights", None)
        if weights is None:
            raise RuntimeError("Laguna public generator did not retain resident weights")
        direct_started = time.perf_counter()
        direct_session = LagunaGGUFResidentSession(
            resident_weights=weights,
            context_length=4_096,
            backend=llm.resolved_backend,
        )
        try:
            tokenizer = getattr(inner, "tokenizer")
            direct_ids = direct_session.generate_greedy(
                prompt_ids,
                max_new_tokens=max_tokens,
                stop_token_ids=tokenizer.stop_token_ids,
            )
        finally:
            direct_session.close()
        direct_seconds = time.perf_counter() - direct_started
        _progress("direct-complete", started)

        blocking_ids = tuple(blocking.generated_token_ids or ())
        stream_ids = tuple(terminal.generated_token_ids or ())
        stream_text = "".join(chunk.text for chunk in chunks)
        result = {
            "schema": 1,
            "command": [sys.executable, *sys.argv],
            "model": str(Path(model).resolve()),
            "backend": llm.resolved_backend,
            "quant": llm.resolved_quant,
            "case": case_name,
            "prompt_tokens": len(prompt_ids),
            "max_tokens": max_tokens,
            "tokenizer_matches_frozen_ids": tokenized == prompt_ids,
            "blocking_seconds": blocking_seconds,
            "stream_seconds": stream_seconds,
            "direct_seconds": direct_seconds,
            "blocking_ids": list(blocking_ids),
            "stream_ids": list(stream_ids),
            "direct_ids": list(direct_ids),
            "blocking_matches_direct": blocking_ids == direct_ids,
            "stream_matches_direct": stream_ids == direct_ids,
            "stream_text_matches_blocking": stream_text == blocking.text,
            "eot_markup_absent": (
                "</assistant>" not in blocking.text and "</assistant>" not in stream_text
            ),
            "blocking_finish": (
                None if blocking.finish_details is None else blocking.finish_details.to_json_dict()
            ),
            "stream_finish": (
                None if terminal.finish_details is None else terminal.finish_details.to_json_dict()
            ),
            "stream_chunk_count": len(chunks),
            "generator_metadata": getattr(inner, "last_batch_generation", None),
            "tracked_before": tracked_before,
        }
    finally:
        llm.close()
    tracked_after = memory_stats()
    result["tracked_after"] = tracked_after
    result["tracked_returned_to_baseline"] = (
        tracked_after["current_allocated_bytes"] == tracked_before["current_allocated_bytes"]
        and tracked_after["active_allocations"] == tracked_before["active_allocations"]
    )
    result["pass"] = all(
        bool(result[name])
        for name in (
            "tokenizer_matches_frozen_ids",
            "blocking_matches_direct",
            "stream_matches_direct",
            "stream_text_matches_blocking",
            "eot_markup_absent",
            "tracked_returned_to_baseline",
        )
    )
    return result


def main() -> int:
    args = _parser().parse_args()
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    result = run_gate(
        args.model,
        template_path=args.template,
        case_name=args.case,
        backend=args.backend,
        max_tokens=args.max_tokens,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
