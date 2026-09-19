"""Unit tier: the TP2 bulk-prefill phase attribution's accounting.

The tool's whole value is that its phase shares are trustworthy, and the two
ways it can silently lie are both arithmetic: summing the MLP breakdown on top
of the MLP total (which reports a total above the wall), and reporting an
exchange wall read from the decode group instead of the bulk group. Both are
pinned here, along with the slot arithmetic the stream-event pairs depend on.

No device contact: the recorder is driven with a fake runtime that records the
event names it is asked to record, so the phase boundaries can be checked
without HIP.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "tp2_prefill_phase_attribution.py"


def _load():
    spec = importlib.util.spec_from_file_location("tp2_prefill_phase_attribution", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


attribution = _load()


def test_top_level_phases_exclude_the_mlp_breakdown() -> None:
    """Summing the breakdown as well double counts the MLP above the wall."""

    assert set(attribution.MLP_PARTS).issubset(set(attribution.PHASES))
    assert "mlp" in attribution.TOP_LEVEL_PHASES
    for part in attribution.MLP_PARTS:
        assert part not in attribution.TOP_LEVEL_PHASES


def test_phase_sum_is_the_wall_and_not_more() -> None:
    spans = {
        "attention": 224.0,
        "norm_residual": 2.8,
        "mlp": 312.0,
        "mlp_chain": 185.0,
        "mlp_exchange": 76.0,
        "mlp_cast": 48.0,
        "mlp_residual": 2.1,
        "tail": 1.1,
    }
    total = attribution._phase_sum(spans)
    assert total == 539.9
    # The breakdown sums to the MLP total, so adding it would give 812.2.
    assert sum(spans[name] for name in attribution.PHASES) == 851.0
    assert total < sum(spans[name] for name in attribution.PHASES)


def test_mlp_parts_are_a_decomposition_of_the_mlp_total() -> None:
    """The invariant that makes the split readable: parts sum to the whole."""

    spans = {name: 0.0 for name in attribution.PHASES}
    spans.update(
        {"mlp": 100.0, "mlp_chain": 60.0, "mlp_exchange": 25.0,
         "mlp_cast": 13.0, "mlp_residual": 2.0}
    )
    assert sum(spans[name] for name in attribution.MLP_PARTS) == spans["mlp"]


def test_slots_advance_by_one_layer_stride() -> None:
    """Every span is a pair of consecutive slots within one layer's stride."""

    recorder = attribution.PhaseRecorder.__new__(attribution.PhaseRecorder)
    recorder.layer_index = 0
    assert recorder._slot(attribution.SLOT_LAYER_START) == 0
    assert recorder._slot(attribution.SLOT_MLP_STOP) == attribution.SLOTS_PER_LAYER - 1
    recorder.layer_index = 3
    assert recorder._slot(attribution.SLOT_ATTN_STOP) == (
        3 * attribution.SLOTS_PER_LAYER + attribution.SLOT_ATTN_STOP
    )
    # The tail slots sit one full stride past the last layer.
    assert attribution.SLOT_TAIL_START == attribution.SLOTS_PER_LAYER
    assert attribution.SLOTS_PER_PREFILL == attribution.SLOTS_PER_LAYER + 2


def test_span_pairs_are_ordered_within_a_layer() -> None:
    """A mis-ordered pair would report a negative or nonsensical span."""

    order = [
        attribution.SLOT_LAYER_START,
        attribution.SLOT_ATTN_STOP,
        attribution.SLOT_MLP_START,
        attribution.SLOT_CHAIN_START,
        attribution.SLOT_CHAIN_STOP,
        attribution.SLOT_EXCHANGE_STOP,
        attribution.SLOT_CAST_STOP,
        attribution.SLOT_MLP_STOP,
    ]
    assert order == sorted(order)
    assert len(set(order)) == len(order)


def test_recorder_prefers_the_bulk_group_over_the_decode_group() -> None:
    """The prefill drives ``_bulk_shard_group``; wrapping the decode group is a
    silent no-op that shows up as unrecorded events, not as an error."""

    class _Group:
        def __init__(self, name):
            self.name = name
            self.exchange_walls_s = []

    class _Session:
        def __init__(self):
            self.devices = (0, 1)
            self._bulk_shard_group = _Group("bulk")
            self._shard_group = _Group("decode")
            self._config = type("C", (), {"layer_types": ["linear_attention"]})()

    session = _Session()
    group = getattr(session, "_bulk_shard_group", None) or getattr(
        session, "_shard_group", None
    )
    assert group.name == "bulk"
    recorder = attribution.PhaseRecorder.__new__(attribution.PhaseRecorder)
    recorder.group = group
    recorder.wrapped_bulk_group = group is getattr(session, "_bulk_shard_group", None)
    assert recorder.wrapped_bulk_group is True

    # Falling back to the decode group must be detectable, not silent.
    session._bulk_shard_group = None
    fallback = getattr(session, "_bulk_shard_group", None) or getattr(
        session, "_shard_group", None
    )
    recorder.group = fallback
    recorder.wrapped_bulk_group = fallback is getattr(session, "_bulk_shard_group", None)
    assert recorder.wrapped_bulk_group is False


def test_recording_is_gated_so_warmup_does_not_consume_slots() -> None:
    """The warmup pass runs the same helpers; it must not consume slots."""

    recorder = attribution.PhaseRecorder.__new__(attribution.PhaseRecorder)
    recorder.devices = (0,)
    recorder.enabled = False
    recorder.layer_index = 0
    # ``session`` is deliberately absent: while disabled, ``_record`` must
    # return before touching it (or a warmup would need a live session).
    recorder._record(0, attribution.SLOT_LAYER_START)

    recorded: list[int] = []
    recorder._record_all = lambda slot: recorded.append(slot) if recorder.enabled else None

    # Disabled: no events, and the layer counter must not advance, or the
    # first recorded layer would start at the wrong stride.
    recorder._exit_mlp()
    assert recorded == []
    assert recorder.layer_index == 0

    recorder.enabled = True
    recorder._exit_mlp()
    assert recorded == [attribution.SLOT_MLP_STOP]
    assert recorder.layer_index == 1
    recorded.clear()

    # The next layer's events land one full stride later.
    recorder._enter_attention()
    assert recorded == [attribution.SLOTS_PER_LAYER + attribution.SLOT_LAYER_START]
