from types import SimpleNamespace
import pytest


def weight(raw=100, k=2560, n=10240):
    return SimpleNamespace(
        backend="hip_gfx1151",
        spec=SimpleNamespace(quant_key="gguf_q8_0", source=SimpleNamespace(shape=(n,k))),
        allocation=lambda name: SimpleNamespace(tensor=SimpleNamespace(ptr=raw)))


def test_owner_reuses_and_closes(monkeypatch):
    from hipengine.runtime import gguf_q8_mmq_sidecars as owner
    packed,freed = [],[]
    monkeypatch.setattr(owner,"malloc",lambda size,**kw: SimpleNamespace(ptr=1000000,nbytes=size))
    monkeypatch.setattr(owner,"free",lambda buf,**kw: freed.append(buf.ptr))
    monkeypatch.setattr(owner,"resolve",lambda **kw: lambda *args,**kwargs: packed.append(args))
    runtime = SimpleNamespace(device_synchronize=lambda: None)
    sidecars = owner.Q8MMQWeightSidecars(runtime=runtime,library=object())
    sidecars.prepare(weight())
    sidecars.prepare(weight())
    assert len(packed) == 1
    assert sidecars.mapping[(100,2560,10240)][0] == 1000000
    assert sidecars.nbytes == 31129600
    sidecars.close()
    sidecars.close()
    assert freed == [1000000]
    assert not sidecars.mapping
    with pytest.raises(RuntimeError,match="closed"):
        sidecars.prepare(weight())


def test_owner_pack_failure_does_not_publish(monkeypatch):
    from hipengine.runtime import gguf_q8_mmq_sidecars as owner
    freed = []
    monkeypatch.setattr(owner,"malloc",lambda size,**kw: SimpleNamespace(ptr=1000000,nbytes=size))
    monkeypatch.setattr(owner,"free",lambda buf,**kw: freed.append(buf.ptr))
    def fail(*args,**kwargs):
        raise RuntimeError("pack failed")
    monkeypatch.setattr(owner,"resolve",lambda **kw: fail)
    sidecars = owner.Q8MMQWeightSidecars(
        runtime=SimpleNamespace(device_synchronize=lambda: None),library=object())
    with pytest.raises(RuntimeError,match="pack failed"):
        sidecars.prepare(weight())
    assert not sidecars.mapping
    assert freed == [1000000]


@pytest.mark.parametrize("mapping,uses_packed", [
    ({(100,2560,10240):(600000000,31129600)}, True),
    ({(100,2560,320):(600000000,31129600)}, False),
    ({}, False),
])
def test_session_keeps_repair_raw(monkeypatch,mapping,uses_packed):
    from hipengine.runtime import gguf_linear as linear
    from hipengine.kernels.hip_gfx1100.quant.gguf_q8_0_mmq_prefill import QWEN4EXP_Q8_MMQ_PREFILL_POLICY
    calls = []
    monkeypatch.setattr(linear,"gguf_q8_0_mmq128_quantize_f32_d4x3",lambda *a,**k: None)
    monkeypatch.setattr(linear,"gguf_q8_0_mmq128_sparse_exact_correct_f32",
                        lambda *a,**k: calls.append(("repair",a)))
    monkeypatch.setattr(linear,"resolve",lambda **kw: lambda *a,**k: calls.append(("packed",a)))
    runtime = SimpleNamespace(memset_async=lambda *a: None)
    with linear.q8_mmq_prefill_session(
        workspace_ptr=100000000,workspace_nbytes=10000000,
        risk_count_ptr=200000000,risk_count_nbytes=4,
        risk_indices_ptr=300000000,risk_indices_nbytes=30000000,
        policy=QWEN4EXP_Q8_MMQ_PREFILL_POLICY,library=object(),prepacked_weights=mapping):
        linear._launch_raw_mmq_d4x3_f32(
            lambda *a,**k: calls.append(("raw",a)),weight(),400000000,500000000,
            512,2560,10240,{"runtime":runtime})
    assert calls[0][0] == ("packed" if uses_packed else "raw")
    assert calls[0][1][1] == (600000000 if uses_packed else 100)
    assert calls[1][0] == "repair" and calls[1][1][1] == 100
    assert linear._q8_mmq_prefill_session.get() is None


def test_runner_admission_scope_and_reuse(monkeypatch):
    from hipengine.runtime.qwen4_exp_runner import Qwen4ExpGGUFResidentModelRunner
    from hipengine.runtime import gguf_q8_mmq_sidecars as owner
    prepared = []
    class FakeOwner:
        def __init__(self,**kwargs): pass
        def prepare(self,w): prepared.append(w)
        def close(self): pass
    monkeypatch.setattr(owner,"Q8MMQWeightSidecars",FakeOwner)
    runner = SimpleNamespace(
        _q8_mmq_weight_sidecars=None,_q8_mmq_library=object(),runtime=object(),
        gdn_bindings={0:SimpleNamespace(mixer=SimpleNamespace(projections={
            "attn_qkv":weight(),"ssm_out":weight(raw=200,k=6144,n=2560)}))})
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK","0")
    Qwen4ExpGGUFResidentModelRunner._configure_q8_mmq_weight_sidecars(runner)
    assert not prepared
    monkeypatch.setenv("HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK","1")
    Qwen4ExpGGUFResidentModelRunner._configure_q8_mmq_weight_sidecars(runner)
    Qwen4ExpGGUFResidentModelRunner._configure_q8_mmq_weight_sidecars(runner)
    assert len(prepared) == 2


def test_ab_mode_changes_only_prepack_flag():
    from scripts.qwen4exp_halo_box_campaign_ab import _apply_mode
    env = {"HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL":"1","unrelated":"keep"}
    for mode,value in (("before","0"),("after","1"),("before","0")):
        _apply_mode(mode,environment=env,route_package="q8-mmq-prepack")
        assert env == {"HIPENGINE_QWEN4_EXP_Q8_MMQ_PREFILL":"1","unrelated":"keep",
                       "HIPENGINE_QWEN4_EXP_Q8_MMQ_PREPACK":value}
