"""R0195 gfx1100 physical-target rowtile device attribution + exact-output gate.

Public ``LLM.generate_detailed()`` with ``speculative_mtp=True`` on a nextn-bearing
GGUF artifact must, on gfx1100:

- production: pad physical groups to the admitted rows6 rowtile
  (``selected_rows == 6``, ``reason == "padded_to_row_multiple"``) with a bound
  ``pad_counts`` rows plan;
- strict: use no rowtiles (``reason == "no_padding_policy"``, empty rows plan);
- produce bit-identical ``generated_token_ids`` between the padded production route
  and the unpadded strict route (the padded accept/commit path is exact), and
  bit-identical committed conv/recurrent device state (the canonical scratch, not
  the padded verification scratch);
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
    "Describe the water cycle briefly.",
    "What is the speed of light?",
    "Name a programming language.",
    "Summarize photosynthesis in a sentence.",
]
_MAX_TOKENS = 24
# Highly predictable continuations so the NextN draft proposes the full candidate
# budget for every request; used to reach the exact_row_count shape (logical_rows 8).
_PREDICTABLE_PROMPTS = [
    "Count from 1 to 20: 1, 2, 3,",
    "The alphabet: a, b, c,",
]


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


def _find_committed_state(engine):
    """Return the batch owner's committed canonical scratch, or None.

    The packed ``verify_state`` is a verification scratch holding the full padded
    physical group (including unaccepted candidate rows); the accept/commit path
    (``prepare_external_verify_state_commit``) copies only the selected rows into
    the session's canonical scratch. That committed conv/recurrent state is the
    exactness target, not the scratch.
    """
    owner = engine._text_generator._runner._resident_batch_owner
    rss = getattr(owner, "_resident_state_storage", None)
    if rss is None and callable(getattr(owner, "_state_storage", None)):
        rss = owner._state_storage()
    canon = getattr(rss, "canonical", None)
    return getattr(canon, "scratch", None)


def _hash_committed_state(state) -> dict:
    """SHA-256 the raw conv/recurrent bytes of every linear-attention layer.

    Raw-byte hashing is dtype-agnostic; identical committed state hashes identically
    regardless of fp16/fp32 recurrent storage.
    """
    import hashlib

    import numpy as np

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    runtime = get_hip_runtime()
    runtime.device_synchronize()
    convs = state.layer_conv_states
    recs = state.layer_recurrent_states
    hashes: dict = {}
    for layer_id in range(len(recs)):
        rec = recs[layer_id]
        conv = convs[layer_id] if layer_id < len(convs) else None
        if rec is None:
            continue
        entry = {}
        for name, buf in (("conv", conv), ("recurrent", rec)):
            if buf is None:
                continue
            nbytes = int(buf.nbytes)
            host = np.empty(nbytes, dtype=np.uint8)
            copy_device_to_host(host_array_ptr(host),
                                DeviceBuffer(int(buf.ptr), nbytes), nbytes,
                                runtime=runtime)
            entry[name] = hashlib.sha256(host.tobytes()).hexdigest()
        hashes[str(layer_id)] = entry
    return hashes


def _worker(budget: int, profile: str, num_prompts: int = 4,
            prompt_set: str = "default") -> None:
    from hipengine import LLM, SamplingParams

    base = _PREDICTABLE_PROMPTS if prompt_set == "predictable" else _PROMPTS
    prompts = base[:num_prompts]
    engine = LLM(
        str(_MODEL), backend=_BACKEND, execution_profile=profile,
        speculative_candidate_budget=budget,
        max_active_requests=num_prompts, max_sequence_length=512,
    )
    sampling = SamplingParams(
        max_tokens=_MAX_TOKENS, temperature=0.0, top_p=1.0,
        ignore_eos=True, speculative_mtp=True,
    )
    result: dict = {"budget": budget, "profile": profile}
    try:
        outputs = engine.generate_detailed(prompts, sampling)
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
    state = _find_committed_state(engine)
    if state is not None:
        result["committed_state_hashes"] = _hash_committed_state(state)
    print("R0195_WORKER_RESULT " + json.dumps(result, default=str), flush=True)


def _run_case(budget: int, profile: str, num_prompts: int = 4,
              prompt_set: str = "default") -> dict:
    env = dict(os.environ,
               R0195_WORKER=f"{budget}:{profile}:{num_prompts}:{prompt_set}",
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


@pytest.mark.parametrize("budget", [1, 2])
def test_production_pads_to_rowtile_and_matches_strict_exactly(budget: int) -> None:
    prod = _run_case(budget, "production")
    assert prod["status"] == "ok", prod.get("error")
    shape = prod["last_target_shape"]
    assert shape["selected_rows"] == 6, shape
    assert shape["reason"] == "padded_to_row_multiple", shape
    assert shape["pad_rows"] > 0, shape
    assert prod["rows_plan"]["pad_counts"] == [6], prod["rows_plan"]

    strict = _run_case(budget, "strict")
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
    # Exact device-state parity: the committed conv/recurrent state (canonical
    # scratch) must be bit-identical, not just the emitted token IDs. The packed
    # verify_state scratch legitimately differs (it holds the padded physical group).
    assert prod.get("committed_state_hashes"), "no committed state hashes captured (production)"
    assert strict.get("committed_state_hashes"), "no committed state hashes captured (strict)"
    assert prod["committed_state_hashes"] == strict["committed_state_hashes"], (
        "committed conv/recurrent device state diverged between padded production "
        "and unpadded strict"
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


def test_wider_group_realizes_eight_request_capacity() -> None:
    # 8 prompts + max_active_requests=8 lifts owner capacity so the width-8
    # physical cell is admissible; the realized verify group is wider than the
    # 2-request default and attributes a valid row-multiple reason.
    result = _run_case(3, "production", num_prompts=8)
    assert result["status"] == "ok", result.get("error")
    partition = result["last_partition"]
    assert partition["owner_capacity"] == 8, partition
    assert partition["physical_max_requests"] == 8, partition
    assert partition["resolved_max_requests"] == 8, partition
    shape = result["last_target_shape"]
    assert shape["request_count"] >= 2, shape
    assert shape["reason"] in {
        "aligned_row_multiple", "padded_to_row_multiple", "exact_row_count",
    }, shape
    assert shape["selected_rows"] >= shape["logical_rows"], shape
    assert len(result["token_ids"]) == 8, result
    assert all(len(t) == _MAX_TOKENS for t in result["token_ids"]), result


def test_exact_row_count_shape_attributed_without_padding() -> None:
    # Predictable continuations let the draft propose the full C3 budget for both
    # requests, so logical_rows = 2 + 2*3 = 8 lands in the exact row-count set
    # [8,28,32] and is admitted with no padding (the exact_row_count shape).
    result = _run_case(3, "production", num_prompts=2, prompt_set="predictable")
    assert result["status"] == "ok", result.get("error")
    shape = result["last_target_shape"]
    assert shape["reason"] == "exact_row_count", shape
    assert shape["logical_rows"] == 8, shape
    assert shape["selected_rows"] == 8, shape
    assert shape["pad_rows"] == 0, shape
    assert shape["request_count"] == 2, shape


if __name__ == "__main__":
    spec = os.environ.get("R0195_WORKER")
    if spec:
        parts = spec.split(":")
        b, p = int(parts[0]), parts[1]
        n = int(parts[2]) if len(parts) > 2 else 4
        ps = parts[3] if len(parts) > 3 else "default"
        _worker(b, p, n, ps)