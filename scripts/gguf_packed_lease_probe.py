#!/usr/bin/env python3
"""Measure the packed workspace lease against the geometry the loop requests.

The eager packed-execution KV workspace is leased once when the GGUF server
binds its engine-loop configuration, before any request layout exists. Its slot
term must therefore cover every layout the serving loop can build, and its
per-slot term must cover the longest context a request can reach. This probe
measures both by driving the real in-process server and reading the pool's own
accounting, so a sizing change can be compared before and after on the same
host without a client.

It reports, per capacity:

* ``pool.stats`` current/pinned/free pages, the workspace lease length, and the
  sampled HIP device usage from ``kv_pool_memory_snapshot()``,
* every ``_packed_verify_union_geometry`` request and the union it resolved to,
* every ``_ensure_packed_verify_workspace`` request with the state it allocated
  (and any fail-closed refusal), and
* the route and usage of a short prompt, an explicit ``speculative_mtp`` request,
  and a long prompt that must still prefill.

Usage::

    HIPENGINE_GGUF_AUTO_CONTEXT=0 python scripts/gguf_packed_lease_probe.py \
        --capacity 1 --max-sequence-length 8192 --out /tmp/lease-c1.json

Retained evidence for the capacity term:
``benchmarks/results/2026-09-19-gfx1151-qwen38-packed-workspace-lease-capacity.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hipengine.runtime.qwen35_gguf_runner as gguf_runner  # noqa: E402

DEFAULT_MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"

_observed: dict[str, list[dict[str, Any]]] = {
    "geometry": [],
    "workspace": [],
    "allocate": [],
}


def _install_probes() -> None:
    """Record the geometry, workspace, and state allocations the loop requests."""

    original_geometry = gguf_runner.Qwen35GGUFResidentSession._packed_verify_union_geometry

    def geometry_probe(self, *, slot_count, rows, max_sequence_length):
        result = original_geometry(
            self,
            slot_count=slot_count,
            rows=rows,
            max_sequence_length=max_sequence_length,
        )
        _observed["geometry"].append(
            {
                "slot_count": int(slot_count),
                "rows": int(rows),
                "max_sequence_length": int(max_sequence_length),
                "union_slots": int(result[0]),
                "union_rows": int(result[1]),
                "union_max_sequence_length": int(result[2]),
                "union_segments": int(result[3]),
                "session_max_batch_size": int(getattr(self, "max_batch_size", 0) or 0),
            }
        )
        return result

    gguf_runner.Qwen35GGUFResidentSession._packed_verify_union_geometry = geometry_probe

    original_ensure = gguf_runner.Qwen35GGUFResidentSession._ensure_packed_verify_workspace

    def ensure_probe(self, *, slot_count, rows, max_sequence_length, **kwargs):
        before = getattr(self, "_packed_verify_state", None)
        try:
            state, scratch = original_ensure(
                self,
                slot_count=slot_count,
                rows=rows,
                max_sequence_length=max_sequence_length,
                **kwargs,
            )
        except BaseException as exc:  # noqa: BLE001 - record the refusal geometry
            _observed["workspace"].append(
                {
                    "slot_count": int(slot_count),
                    "rows": int(rows),
                    "max_sequence_length": int(max_sequence_length),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        _observed["workspace"].append(
            {
                "slot_count": int(slot_count),
                "rows": int(rows),
                "max_sequence_length": int(max_sequence_length),
                "ok": True,
                "state_slot_count": int(state.slot_count),
                "state_max_sequence_length": int(state.max_sequence_length),
                "state_total_positions": int(state.total_positions),
                "state_backing": str(getattr(state, "kv_backing_kind", "unknown")),
                "state_reallocated": before is None
                or int(before.slot_count) < int(state.slot_count)
                or int(before.max_sequence_length) < int(state.max_sequence_length),
            }
        )
        return state, scratch

    gguf_runner.Qwen35GGUFResidentSession._ensure_packed_verify_workspace = ensure_probe

    original_allocate = gguf_runner._GGUFPackedTargetState.allocate

    def allocate_probe(runner, **kwargs):
        state = original_allocate(runner, **kwargs)
        _observed["allocate"].append(
            {
                "slot_count": int(state.slot_count),
                "max_sequence_length": int(state.max_sequence_length),
                "total_positions": int(state.total_positions),
                "kv_backing_kind": str(getattr(state, "kv_backing_kind", "unknown")),
            }
        )
        return state

    gguf_runner._GGUFPackedTargetState.allocate = staticmethod(allocate_probe)


def _source_digest() -> dict[str, str]:
    """Identify the tree this measurement ran against."""

    digest: dict[str, str] = {}
    for name, relative in (
        ("qwen35_gguf.py", "hipengine/generation/qwen35_gguf.py"),
        ("qwen35_gguf_runner.py", "hipengine/runtime/qwen35_gguf_runner.py"),
    ):
        path = REPO_ROOT / relative
        digest[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    digest["head"] = subprocess.check_output(
        ("git", "rev-parse", "--short", "HEAD"), cwd=REPO_ROOT, text=True
    ).strip()
    digest["dirty"] = subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=REPO_ROOT,
        text=True,
    ).strip()[:200]
    return digest


def _resident_runner(llm):
    generator = llm._get_text_generator()
    runner = getattr(generator, "_runner", None)
    if runner is None:
        runner = getattr(getattr(generator, "_driver", None), "_runner", None)
    if runner is None:
        raise RuntimeError("could not reach the resident model runner")
    return runner


def _pool_summary(runner) -> dict[str, Any]:
    snapshot = runner.kv_pool_memory_snapshot()
    pool = snapshot.get("dynamic_pool") or {}
    observability = runner.observability_snapshot()
    model_runner = observability.get("model_runner") or {}
    routes = observability.get("routes") or {}
    current_pages = pool.get("current_pages")
    return {
        "capacity": int(runner.capacity),
        "current_pages": current_pages,
        "pinned_pages": pool.get("pinned_pages"),
        "free_pages": pool.get("free_pages"),
        "refcounted_pages": pool.get("refcounted_pages"),
        "current_bytes": pool.get("current_bytes"),
        "page_bytes": (
            None
            if not current_pages
            else int(pool.get("current_bytes", 0)) // int(current_pages)
        ),
        "workspace_lease_pages": snapshot.get("packed_workspace_lease_pages"),
        "packed_workspace_backing": snapshot.get("packed_workspace_backing"),
        "hip_used_current_bytes": snapshot.get("hip_used_current_bytes"),
        "hip_used_peak_sampled_bytes": snapshot.get("hip_used_peak_sampled_bytes"),
        "max_context_tokens": model_runner.get("max_context_tokens"),
        "route_counts": {
            key: value for key, value in (routes.get("counts") or {}).items() if value
        },
        "fallback_reasons": dict(routes.get("fallback_reasons") or {}),
    }


def _long_prompt_text(target_tokens: int) -> str:
    """Build a long, realistic-shaped prompt without depending on a tokenizer."""

    paragraph = (
        "The resident scheduler admits a request only when its packed group can "
        "open the KV planes it needs, and the workspace lease is the eager floor "
        "for those planes. "
    )
    repeats = max(1, int(target_tokens // 20))
    return "Summarize the following engineering notes in three sentences.\n\n" + (
        paragraph * repeats
    )


def _request(client, *, model: str, prompt: str, max_tokens: int, mtp: bool | None):
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "top_p": 1.0,
    }
    if mtp is not None:
        payload["speculative_mtp"] = bool(mtp)
    response = client.post("/v1/completions", json=payload)
    body = response.json()
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {body}")
    usage = body.get("usage") or {}
    return {
        "status": response.status_code,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "route": (body.get("speculative_mtp") or {}).get("route"),
        "speculative_mtp": body.get("speculative_mtp"),
        "text_head": (body.get("choices") or [{}])[0].get("text", "")[:80],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--quant", default="q4_k_m")
    parser.add_argument("--max-sequence-length", type=int, default=8192)
    parser.add_argument("--candidate-budget", type=int, default=2)
    parser.add_argument("--long-prompt-tokens", type=int, default=3530)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from fastapi.testclient import TestClient

    from hipengine import LLM
    from hipengine.server.api import ServerConfig, create_app

    _install_probes()
    served_model = "qwen38-lease-probe"
    report: dict[str, Any] = {
        "kind": "packed_workspace_lease_probe",
        "source_digest": _source_digest(),
        "capacity": int(args.capacity),
        "candidate_budget": int(args.candidate_budget),
        "max_sequence_length": int(args.max_sequence_length),
        "long_prompt_target_tokens": int(args.long_prompt_tokens),
        "backend": str(args.backend),
        "quant": str(args.quant),
        "model": str(args.model),
    }
    llm = LLM(
        args.model,
        backend=args.backend,
        execution_profile="production",
        max_active_requests=int(args.capacity),
        max_sequence_length=int(args.max_sequence_length),
        kv_storage="bf16",
        speculative_candidate_budget=int(args.candidate_budget),
        speculative_mtp_serving="auto",
    )
    app = create_app(
        ServerConfig(
            model=str(args.model),
            backend=str(args.backend),
            quant=str(args.quant),
            served_model_name=served_model,
            eager_load=False,
            metrics="prometheus",
            max_context_tokens=int(args.max_sequence_length),
            max_active_requests=int(args.capacity),
            speculative_mtp_serving="auto",
            speculative_candidate_budget=int(args.candidate_budget),
            shutdown_grace_seconds=5.0,
        ),
        llm=llm,
    )
    runner = _resident_runner(llm)
    report["after_configure"] = _pool_summary(runner)
    report["sessions"] = [
        {
            "max_batch_size": int(getattr(lease.session, "max_batch_size", 0) or 0),
            "scratch_max_positions": int(
                getattr(getattr(lease.session, "scratch", None), "max_positions", 0) or 0
            ),
        }
        for lease in getattr(runner, "_available", ())
    ]

    with TestClient(app) as client:
        report["short_auto"] = _request(
            client,
            model=served_model,
            prompt="Write one short sentence about the sea.",
            max_tokens=int(args.max_tokens),
            mtp=None,
        )
        report["after_short_auto"] = _pool_summary(runner)
        report["short_explicit"] = _request(
            client,
            model=served_model,
            prompt="Write one short sentence about the sea.",
            max_tokens=int(args.max_tokens),
            mtp=True,
        )
        report["after_short_explicit"] = _pool_summary(runner)
        report["long_explicit"] = _request(
            client,
            model=served_model,
            prompt=_long_prompt_text(int(args.long_prompt_tokens)),
            max_tokens=8,
            mtp=True,
        )
        report["after_long"] = _pool_summary(runner)

    geometries = _observed["geometry"]
    report["geometry_summary"] = {
        "calls": len(geometries),
        "max_slot_count_requested": max(
            (row["slot_count"] for row in geometries), default=None
        ),
        "max_union_slots": max((row["union_slots"] for row in geometries), default=None),
        "max_rows_requested": max((row["rows"] for row in geometries), default=None),
        "max_union_rows": max((row["union_rows"] for row in geometries), default=None),
        "max_union_max_sequence_length": max(
            (row["union_max_sequence_length"] for row in geometries), default=None
        ),
        "distinct_slot_counts": sorted({row["slot_count"] for row in geometries}),
        "distinct_union_slots": sorted({row["union_slots"] for row in geometries}),
        "session_max_batch_sizes": sorted(
            {row["session_max_batch_size"] for row in geometries}
        ),
    }
    report["workspace_calls"] = _observed["workspace"]
    report["allocations"] = _observed["allocate"]
    report["geometry_calls"] = geometries
    report["failures"] = [row for row in _observed["workspace"] if not row["ok"]]

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
