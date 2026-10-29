"""Layer scratch owns only the two residual buffers outside its children."""

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen4_exp_runner as module


@pytest.mark.parametrize("kind", ["qsa", "gdn"])
@pytest.mark.parametrize("rows", [1, 17, 65])
@pytest.mark.parametrize("fail_output", [False, True])
def test_layer_scratch_has_no_unused_float_moe_buffer(monkeypatch, kind, rows, fail_output):
    allocated, freed, children = [], [], []

    def child(**kwargs):
        owner = SimpleNamespace(closes=0)
        def close():
            owner.closes += 1
        owner.close = close
        children.append(owner)
        return owner

    for name in ("Qwen4ExpGRScratch", "Qwen4ExpMoEScratch",
                 "Qwen4ExpQSAScratch", "Qwen4ExpGDNScratch"):
        monkeypatch.setattr(getattr(module, name), "allocate", child)

    def malloc(nbytes, **kwargs):
        if fail_output and len(allocated) == 1:
            raise MemoryError("output allocation")
        buffer = SimpleNamespace(nbytes=nbytes)
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(module, "malloc", malloc)
    monkeypatch.setattr(module, "free", lambda buffer, **kwargs: freed.append(buffer))
    cls = getattr(module, "Qwen4Exp" + kind.upper() + "LayerScratch")
    kwargs = dict(rows=rows, branches=4, hidden=256, low_rank=16,
                  ffn=64, experts=8, top_k=2, runtime=object())
    if kind == "qsa":
        kwargs.update(query_heads=2, kv_heads=1, head_dim=128)
    else:
        kwargs.update(qkv_width=128, core_width=64, scalar_width=4)
    if fail_output:
        with pytest.raises(MemoryError, match="output allocation"):
            cls.allocate(**kwargs)
    else:
        scratch = cls.allocate(**kwargs)
        assert not hasattr(scratch, "moe_f32")
        assert [buffer.nbytes for buffer in allocated] == [rows * 4 * 256 * 2] * 2
        scratch.close()
        scratch.close()
    assert [id(buffer) for buffer in freed] == [id(buffer) for buffer in reversed(allocated)]
    assert len(children) == 4
    assert all(owner.closes == 1 for owner in children)
