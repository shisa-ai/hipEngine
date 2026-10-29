"""Package-selected head-major layout with resource and capability fallbacks."""

import inspect

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as runner


@pytest.mark.parametrize("legacy", ["0", "1"])
def test_retired_head_major_boolean_cannot_override_package_or_gapped_gather(monkeypatch, legacy):
    monkeypatch.setenv("HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV", legacy)
    monkeypatch.delenv(runner._GGUF_GAPPED_GATHER_ENV, raising=False)
    assert runner._gguf_aotriton_head_major_kv_enabled("hip_gfx1151") is True
    assert runner._gguf_aotriton_head_major_kv_enabled("hip_gfx1100") is False
    monkeypatch.setattr(runner, "resolve", lambda **kw: object())
    assert runner._gguf_gapped_slot_local_fast_route_available(
        backend="hip_gfx1100", context_tokens=1224, kv_width=1024,
    )
    source = inspect.getsource(runner)
    assert '"HIPENGINE_GGUF_AOTRITON_HEAD_MAJOR_KV"' not in source


@pytest.mark.parametrize("tokens", [37, 511, 512, 65792, 65793])
@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_head_major_sizing_and_allocator_agree_on_resource_admission(monkeypatch, tokens, backend):
    monkeypatch.delenv(runner._GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS_ENV, raising=False)
    monkeypatch.delenv(runner._GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_BYTES_ENV, raising=False)
    allocated = []
    runtime = object()

    def allocate(nbytes, **kwargs):
        assert kwargs == {"runtime": runtime}
        result = DeviceBuffer(0x1000 + len(allocated) * 0x1000, nbytes)
        allocated.append(result)
        return result

    monkeypatch.setattr(runner, "malloc", allocate)
    kwargs = {"backend": backend, "capacity_tokens": tokens, "kv_width": 128}
    planned = runner._gguf_aotriton_head_major_kv_scratch_nbytes(**kwargs)
    pair = runner._try_allocate_gguf_aotriton_head_major_kv_scratch(**kwargs, runtime=runtime)
    expected = tokens * 128 * 2 * 2 if backend == "hip_gfx1151" and tokens <= 65792 else 0
    assert planned == expected
    if expected:
        assert pair == tuple(allocated)
        assert [buf.nbytes for buf in allocated] == [expected // 2] * 2
    else:
        assert pair is None and allocated == []


@pytest.mark.parametrize("cap_env,cap", [
    ("_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_TOKENS_ENV", "36"),
    ("_GGUF_AOTRITON_HEAD_MAJOR_KV_MAX_BYTES_ENV", "18943"),
])
def test_head_major_resource_caps_remain_effective(monkeypatch, cap_env, cap):
    monkeypatch.setenv(getattr(runner, cap_env), cap)
    assert runner._gguf_aotriton_head_major_kv_scratch_nbytes(
        backend="hip_gfx1151", capacity_tokens=37, kv_width=128,
    ) == 0
