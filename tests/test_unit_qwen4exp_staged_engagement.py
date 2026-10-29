"""Staged engagement follows final dispatch, including cached selections."""
import pytest
from hipengine.runtime import gguf_linear as linear
from scripts import execution_profile_q8_wmma_prefill_layers_gate as gate


@pytest.mark.parametrize("route", ["staged_f32", "staged_f32_composed_vs_exact", "staged_f32_over_wmma"])
def test_staged_routes_have_final_dispatch_evidence(route):
    assert gate.ROUTE_ENGAGEMENT_SELECTORS[route] == "staged_f32"


def test_final_dispatch_reports_both_arms_without_selector_execution():
    arms = {}
    for arm, variant in [("teacher", "coltile8_rowbatch4_f32_f32_out"),
                         ("candidate", linear._Q8_STAGED_F32_VARIANT)]:
        records = []
        with linear.wmma_prefill_dispatch_recorder(records):
            # The final-dispatch hook is called after cache lookup; no selector
            # invocation or environment setting is needed to identify its key.
            for _ in range(2):
                linear._record_gguf_linear_dispatch(
                    backend="hip_gfx1151", layer="linear", quant="gguf_q8_0",
                    variant=variant, abi="raw", rows=512, in_features=256,
                    out_features=128, activation_dtype="float32", output_dtype="float32",
                )
        arms[arm] = records
        assert len(records) == 2
        assert all(r["engaged"] == (arm == "candidate") for r in records)
    summary = gate._summarise_engagement(arms, route_selector="staged_f32")
    assert summary["candidate_route_engaged"]
    assert not summary["teacher_route_engaged"]


def test_real_launcher_records_cached_staged_key(monkeypatch):
    from types import SimpleNamespace

    class Cached(dict):
        def get(self, key, default=None):
            return ("raw", lambda *a, **kw: None, "hip_gfx1151", "linear",
                    "gguf_q8_0", linear._Q8_STAGED_F32_VARIANT)

    weight = SimpleNamespace(
        spec=SimpleNamespace(layout="raw_gguf", quant_key="gguf_q8_0",
                             slot_path="layers.3.attn_qkv.weight"),
        backend="hip_gfx1151",
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=4096)),
    )
    monkeypatch.setattr(linear, "_DISPATCH_RESOLVE_CACHE", Cached())
    def forbidden(*a, **kw):
        pytest.fail("cache hit must not re-run staged selector")
    monkeypatch.setattr(linear, "_raw_k_staged_f32_dispatch", forbidden)
    records = []
    # Stop after the actual final-dispatch hook, before any synthetic pointer
    # could reach a device launcher. The hook is what this regression observes.
    original = linear._record_gguf_linear_dispatch
    class Recorded(Exception):
        pass
    def stop(**kwargs):
        original(**kwargs)
        raise Recorded
    monkeypatch.setattr(linear, "_record_gguf_linear_dispatch", stop)
    with linear.wmma_prefill_dispatch_recorder(records):
        with pytest.raises(Recorded):
            linear.launch_gguf_linear(weight, 0, 0, 512, 256, 128, libraries={})
    staged = [r for r in records if r["selector"] == "staged_f32"]
    assert len(staged) == 1
    assert staged[0]["engaged"] is True
