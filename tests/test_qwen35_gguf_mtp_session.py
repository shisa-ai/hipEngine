from __future__ import annotations

from types import SimpleNamespace

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_mtp as mtp_module
from hipengine.runtime.qwen35_gguf_mtp import Qwen35GGUFMTPDecodeSession


def test_mtp_prompt_admission_bulk_prefills_target_then_catches_up_shifted_draft(
    monkeypatch,
) -> None:
    allocations = iter((DeviceBuffer(0x1000, 8), DeviceBuffer(0x2000, 24)))
    freed: list[int] = []
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: next(allocations))
    monkeypatch.setattr(
        mtp_module,
        "free",
        lambda buffer, *, runtime: freed.append(int(buffer.ptr)),
    )

    target_calls: list[tuple[tuple[int, ...], dict[str, object]]] = []

    class Target:
        runner = SimpleNamespace(hidden_size=4)
        runtime = SimpleNamespace(memset=lambda *_args: None)

        def prefill(self, prompt, **kwargs):
            target_calls.append((tuple(prompt), kwargs))
            return SimpleNamespace(token_id=91)

        def step(self, *_args, **_kwargs):
            raise AssertionError("bulk MTP admission must not serial-prefill the target")

    draft_calls: list[tuple[int, int, int, int]] = []

    class Executor:
        def run_step(self, request_id, token_id, position, target_hidden, **_kwargs):
            draft_calls.append(
                (
                    int(request_id),
                    int(token_id),
                    int(position),
                    int(target_hidden.ptr),
                )
            )

    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = Target()
    decoder.draft_provider = SimpleNamespace(executor=Executor())

    result = decoder._prefill_target_and_draft(
        (11, 22, 33),
        request_id=7,
        use_bulk=True,
    )

    assert result.token_id == 91
    assert target_calls == [
        (
            (11, 22, 33),
            {
                "use_bulk": True,
                "return_logits": False,
                "capture_target_hidden_rows": DeviceBuffer(0x2000, 24),
            },
        )
    ]
    assert draft_calls == [
        (7, 11, 0, 0x1000),
        (7, 22, 1, 0x2000),
        (7, 33, 2, 0x2008),
    ]
    assert freed == [0x2000, 0x1000]


def test_mtp_prompt_admission_preserves_target_default_bulk_selector(monkeypatch) -> None:
    allocations = iter((DeviceBuffer(0x1000, 8), DeviceBuffer(0x2000, 8)))
    monkeypatch.setattr(mtp_module, "malloc", lambda _nbytes, *, runtime: next(allocations))
    monkeypatch.setattr(mtp_module, "free", lambda _buffer, *, runtime: None)
    target_calls: list[object] = []

    class Target:
        runner = SimpleNamespace(hidden_size=4)
        runtime = SimpleNamespace(memset=lambda *_args: None)

        def prefill(self, _prompt, **kwargs):
            target_calls.append(kwargs["use_bulk"])
            return SimpleNamespace(token_id=91)

    decoder = Qwen35GGUFMTPDecodeSession.__new__(Qwen35GGUFMTPDecodeSession)
    decoder.target = Target()
    decoder.draft_provider = SimpleNamespace(
        executor=SimpleNamespace(run_step=lambda *_args, **_kwargs: None)
    )

    decoder._prefill_target_and_draft((11,), request_id=7, use_bulk=None)

    assert target_calls == [None]
