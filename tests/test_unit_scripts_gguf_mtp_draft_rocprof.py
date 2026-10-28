from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from scripts import gguf_mtp_draft_rocprof as profiler


@pytest.mark.parametrize("retired", [None, "0", "false", "1", "true"])
def test_draft_profiler_reports_shared_dual_without_retired_env_reader(
    monkeypatch, tmp_path, retired: str | None,
) -> None:
    from hipengine.core import memory
    from hipengine.runtime import qwen35_gguf_runner as target
    from hipengine.speculative import mtp_resident_draft as draft

    if retired is None:
        monkeypatch.delenv("HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_RESIDENT_MTP_DRAFT_Q8_SHARED_DUAL", retired)

    class Session:
        runtime = object()
        position = 0

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Draft:
        num_kv_heads = 1
        qk_head_dim = 4

        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    empty = np.empty((0, 4), dtype=np.float32)
    monkeypatch.setattr(target, "Qwen35GGUFResidentSession", Session)
    monkeypatch.setattr(draft, "Qwen35GGUFResidentMTPDraftRunner", Draft)
    monkeypatch.setattr(memory, "malloc", lambda *args, **kwargs: object())
    monkeypatch.setattr(memory, "free", lambda *args, **kwargs: None)
    monkeypatch.setattr(profiler, "_Roctx", object)
    monkeypatch.setattr(profiler, "_load_mtp_inputs", lambda *args: (
        {"output.weight": (np.zeros((8, 4), dtype=np.float32),)}, empty, empty, empty,
    ))
    monkeypatch.setattr(profiler, "_bulk_prefill_with_hidden_trace", lambda *args: (0, empty, empty))
    monkeypatch.setattr(profiler, "llama_cpp_mtp_catchup_rows", lambda *args: ([], empty))
    output = tmp_path / "child.json"
    monkeypatch.setattr(sys, "argv", [
        "gguf_mtp_draft_rocprof.py", "--child", "--steps", "0", "--warmup", "0",
        "--child-json", str(output),
    ])
    assert profiler.main() == 0
    assert json.loads(output.read_text())["q8_shared_dual"] is True
    assert not hasattr(profiler, "_resident_q8_shared_dual_enabled")
