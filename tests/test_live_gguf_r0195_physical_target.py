"""R0195 gfx1100 physical-target rowtile device attribution + exact-output gate.

Public ``LLM.generate_detailed()`` with ``speculative_mtp=True`` on a nextn-bearing
GGUF artifact must, on gfx1100:

- production: pad physical groups to the admitted rows6 rowtile
  (``selected_rows == 6``, ``reason == "padded_to_row_multiple"``) with a bound
  ``pad_counts`` rows plan;
- strict: use no rowtiles (``reason == "no_padding_policy"``, empty rows plan);
- produce bit-identical ``generated_token_ids`` between the padded production route
  and the unpadded strict route (the padded accept/commit path is exact);
- refuse an unsupported candidate depth with a named capability miss rather than a
  silent downgrade.

Each case runs in its own subprocess: the candidate budget is fixed at adapter
construction, and accumulating engines in one process exhausts VRAM. The test is
guarded on HIP availability and the local model so no-ROCm / model-less runners skip.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_MODEL = Path(os.environ.get("R0195_TEST_MODEL", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
_BACKEND = "hip_gfx1100"
_PROMPTS = [
    "The capital of France is",
    "List three prime numbers:",
    "Explain gravity in one sentence.",
    "Write a haiku about the sea.",
]
_MAX_TOKENS = 24


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime is not available"),
    pytest.mark.skipif(not _MODEL.exists(), reason=f"local GGUF fixture not found: {_MODEL}"),
]


def _find_adapter(root):
    from collections import deque

    seen = set()
    queue = deque([(root, 0)])
    while queue:
        obj, depth = queue.popleft()
        if obj is None or id(obj) in seen or depth > 9:
            continue
        seen.add(id(obj))
        if callable(getattr(obj, "physical_width_contract", None)):
            return obj
        children = {}
        try:
            children.update(vars(obj))
        except TypeError:
            pass
        for name in ("_runner", "_inner", "_text_generator", "_adapter", "_generator",
                     "generator", "instance", "_loop", "_engine", "_mtp2_adapter",
                     "_resident", "_session", "runner", "_impl", "_owner"):
            try:
                child = getattr(obj, name, None)
            except Exception:
                child = None
            if child is not None:
                children[name] = child
        for child in children.values():
            if isinstance(child, (str, bytes, int, float, bool, type(None))):
                continue
            if isinstance(child, (list, tuple)):
                for item in child[:8]:
                    queue.append((item, depth + 1))
            elif isinstance(child, dict):
                for item in list(child.values())[:8]:
                    queue.append((item, depth + 1))
            else:
                queue.append((child, depth + 1))
    return None


def _worker(budget: int, profile: str) -> None:
    from hipengine import LLM, SamplingParams

    engine = LLM(
        str(_MODEL), backend=_BACKEND, execution_profile=profile,
        speculative_candidate_budget=budget,
        max_active_requests=len(_PROMPTS), max_sequence_length=512,
    )
    sampling = SamplingParams(
        max_tokens=_MAX_TOKENS, temperature=0.0, top_p=1.0,
        ignore_eos=True, speculative_mtp=True,
    )
    result: dict = {"budget": budget, "profile": profile}
    try:
        outputs = engine.generate_detailed(_PROMPTS, sampling)
        result["token_ids"] = [list(o.generated_token_ids or ()) for o in outputs]
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    adapter = _find_adapter(engine)
    if adapter is not None:
        contract = adapter.physical_width_contract()
        policy = contract.get("resolved_target_policy") or {}
        result["last_target_shape"] = contract.get("last_target_shape")
        result["rows_plan"] = policy.get("rows")
        result["last_partition"] = contract.get("last_partition")
    print("R0195_WORKER_RESULT " + json.dumps(result, default=str), flush=True)


def _run_case(budget: int, profile: str) -> dict:
    env = dict(os.environ, R0195_WORKER=f"{budget}:{profile}",
               R0195_TEST_MODEL=str(_MODEL))
    proc = subprocess.run([sys.executable, __file__], env=env,
                          capture_output=True, text=True, timeout=900)
    line = next((l for l in proc.stdout.splitlines()
                 if l.startswith("R0195_WORKER_RESULT ")), None)
    if line is None:
        raise AssertionError(
            f"R0195 worker crashed for {profile} C{budget}: {proc.stderr[-800:]}"
        )
    return json.loads(line[len("R0195_WORKER_RESULT "):])


def test_production_pads_to_rowtile_and_matches_strict_exactly() -> None:
    prod = _run_case(2, "production")
    assert prod["status"] == "ok", prod.get("error")
    shape = prod["last_target_shape"]
    assert shape["selected_rows"] == 6, shape
    assert shape["reason"] == "padded_to_row_multiple", shape
    assert shape["pad_rows"] > 0, shape
    assert prod["rows_plan"]["pad_counts"] == [6], prod["rows_plan"]

    strict = _run_case(2, "strict")
    assert strict["status"] == "ok", strict.get("error")
    sshape = strict["last_target_shape"]
    assert sshape["reason"] == "no_padding_policy", sshape
    assert sshape["pad_rows"] == 0, sshape
    assert strict["rows_plan"]["pad_counts"] == [], strict["rows_plan"]
    assert strict["rows_plan"]["exact_counts"] == [], strict["rows_plan"]

    # The padded production accept/commit path must be exact vs the unpadded strict.
    assert prod["token_ids"] == strict["token_ids"], (
        "padded production output diverged from unpadded strict output"
    )
    # Offset/selector partition contract is attributed on both routes.
    for case in (prod, strict):
        partition = case["last_partition"]
        assert partition["resolved_max_requests"] >= 1, partition
        assert len(partition["static_max_realized_group_rows"]) == \
            partition["resolved_max_requests"], partition


def test_unsupported_candidate_depth_fails_with_named_capability() -> None:
    result = _run_case(4, "production")
    assert result["status"] == "error", result
    assert "mtp_candidate_depth_unsupported" in result["error"], result["error"]


if __name__ == "__main__":
    spec = os.environ.get("R0195_WORKER")
    if spec:
        b, p = spec.split(":")
        _worker(int(b), p)