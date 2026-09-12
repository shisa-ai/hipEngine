"""The packed-AR prefill chunk boundary is a physical constant, and it is testable without a GPU.

`_plan_packed_ar_prefill_chunks` is pure Python, so the wave size at which a wave stops prefilling
together can be asserted directly. This guards the explanation for the measured non-monotone grouped
admission row (C7 426.692 tok/s > C8 397.655 tok/s in
`benchmarks/results/2026-08-30-w7900-q4km-c1c8-hipengine-prefill-row-grouped.json`).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gguf_packed_prefill_slab_probe.py"
CANONICAL_PROMPT_TOKENS = 36


def _probe_module():
    spec = importlib.util.spec_from_file_location("slab_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _planner():
    from hipengine.runtime.qwen35_gguf_runner import _plan_packed_ar_prefill_chunks

    return _plan_packed_ar_prefill_chunks


def test_script_imports_without_the_rocm_runtime():
    """The HIP import must stay inside main so --help and tests work anywhere."""
    module = _probe_module()
    assert hasattr(module, "_plan_shape")
    assert callable(module._hip_available)


def test_help_is_callable_as_a_subprocess():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "--prompt-tokens" in proc.stdout


def test_width_8_is_the_first_wave_to_split_on_a_256_row_slab():
    """36 tokens per lane: widths 1-7 fit one chunk, width 8 (288 rows) does not.

    This is the shape that makes the grouped admission row non-monotone. If a future change raises
    the resident slab past 288 rows, this test must be updated *with* the measurement that shows the
    C8 admission row moved - the slab size is not free memory, it is BF16 activation scratch.
    """
    module = _probe_module()
    pool = (760, 4087, 369, 220, 16, 17)
    plans = [
        module._plan_shape(
            _planner(),
            lanes=width,
            prompt_tokens=CANONICAL_PROMPT_TOKENS,
            token_pool=pool,
            row_capacity=256,
        )
        for width in range(1, 9)
    ]
    assert [p["single_chunk"] for p in plans] == [True, True, True, True, True, True, True, False]
    assert plans[7]["total_rows"] == 288
    assert plans[7]["chunk_count"] == 2
    # Slot fairness: every unfinished slot must appear in each round, so a split round still
    # touches all 8 lanes rather than silently dropping the tail.
    assert all(slots == 8 for slots in plans[7]["slots_per_chunk"])


def test_a_slab_that_fits_the_wave_keeps_every_width_in_one_chunk():
    module = _probe_module()
    pool = (760, 4087, 369, 220, 16, 17)
    plans = [
        module._plan_shape(
            _planner(),
            lanes=width,
            prompt_tokens=CANONICAL_PROMPT_TOKENS,
            token_pool=pool,
            row_capacity=288,
        )
        for width in range(1, 9)
    ]
    assert all(p["single_chunk"] for p in plans)


def test_planner_fails_closed_when_the_slab_cannot_hold_one_row_per_slot():
    """A capacity below the lane count must refuse, not prefill a subset of the wave."""
    planner = _planner()
    prompts = tuple(tuple(range(12)) for _ in range(8))
    try:
        planner(prompts, row_capacity=4)
    except ValueError as exc:
        assert "cannot represent all 8 active slots" in str(exc)
    else:  # pragma: no cover - the fail-closed branch is the contract
        raise AssertionError("planner silently accepted an impossible wave")
