from types import SimpleNamespace

import pytest

from scripts.qwen4exp_chunk_workspace import PREFILL_FIELDS, capture_workspace, use_workspace


def runners():
    runtime = SimpleNamespace(device_synchronize=lambda: None)
    resident = object()

    def make(chunk):
        fields = {name: object() for name in PREFILL_FIELDS}
        fields["prefill_chunk_size"] = chunk
        return SimpleNamespace(
            **fields, runtime=runtime, resident=resident, closed=False,
            max_sequence_length=4352, backend="hip_gfx1151",
            state=object(), attention_states=object(), index_states=object(),
            moe_graph_cache=object(), layer_graph_cache=object(),
            _q8_mmq_buffers=(), _q8_mmq_weight_sidecars=None)

    return make(2048), make(1024)


def test_borrow_changes_only_prefill_and_restores_original_owners():
    active, donor = runners()
    original = vars(active).copy()
    workspace = capture_workspace(donor)
    with pytest.raises(TypeError):
        workspace.values["state"] = object()
    with use_workspace(active, workspace):
        assert active.prefill_chunk_size == 1024
        for field in PREFILL_FIELDS:
            assert getattr(active, field) is getattr(donor, field)
        for field in ("state", "attention_states", "index_states",
                      "moe_graph_cache", "layer_graph_cache"):
            assert getattr(active, field) is original[field]
    assert vars(active) == original


def test_borrow_restores_even_if_body_or_final_sync_fails():
    active, donor = runners()
    original = vars(active).copy()
    with pytest.raises(RuntimeError, match="body"):
        with use_workspace(active, capture_workspace(donor)):
            raise RuntimeError("body")
    assert vars(active) == original
    calls = []

    def sync():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("sync")

    active.runtime.device_synchronize = sync
    with pytest.raises(RuntimeError, match="sync"):
        with use_workspace(active, capture_workspace(donor)):
            pass
    assert vars(active) == original


@pytest.mark.parametrize("field,value", [
    ("runtime", object()), ("resident", object()), ("max_sequence_length", 2048),
    ("backend", "other"), ("closed", True), ("_q8_mmq_buffers", (object(),)),
])
def test_incompatible_owner_rejects_before_mutation(field, value):
    active, donor = runners()
    original = vars(active).copy()
    workspace = capture_workspace(donor)
    setattr(donor, field, value)
    with pytest.raises(ValueError):
        with use_workspace(active, workspace):
            pytest.fail("invalid workspace admitted")
    assert vars(active) == original
