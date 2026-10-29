from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

from scripts import gguf_mtp_draft_rocprof as profiler


@pytest.mark.parametrize("sync,event", [(False, False), (True, False), (False, True), (True, True)])
def test_draft_profiler_preserves_independent_timing_mode_arguments(monkeypatch, sync, event):
    class ReachedChild(Exception):
        pass

    captured = []

    def child(args):
        captured.append(args)
        raise ReachedChild

    monkeypatch.setattr(profiler, "_run_child", child)
    argv = ["gguf_mtp_draft_rocprof.py", "--child", "--record-stage-timings"]
    if sync:
        argv.append("--sync-stage-timings")
    if event:
        argv.append("--gpu-event-stage-timings")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ReachedChild):
        profiler.main()
    args, = captured
    assert args.sync_stage_timings is sync
    assert args.gpu_event_stage_timings is event
    assert args.record_stage_timings is True


@pytest.mark.parametrize("sync,event", [(False, False), (True, False), (False, True), (True, True)])
def test_draft_profiler_preserves_explicit_row_comparison(monkeypatch, sync, event):
    captured = []

    def child(args):
        profiler._apply_route_env(args)
        captured.append(args)
        return 0

    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_REPACK", "1")
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_TOP1_STAGE1_THREADS", "128")
    monkeypatch.setenv("HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE", "x8")
    monkeypatch.setenv("HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A", "0")
    monkeypatch.setattr(profiler, "_run_child", child)
    argv = ["gguf_mtp_draft_rocprof.py", "--child", "--record-stage-timings",
            "--q6-top1-dp4a", "--q6-top1-stage1-shape", "row"]
    if sync:
        argv.append("--sync-stage-timings")
    if event:
        argv.append("--gpu-event-stage-timings")
    monkeypatch.setattr(sys, "argv", argv)
    assert profiler.main() == 0
    args, = captured
    assert args.q6_top1_stage1_shape == "row"
    assert args.sync_stage_timings is sync
    assert args.gpu_event_stage_timings is event
    assert os.environ["HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE"] == "row"
    assert os.environ["HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A"] == "1"


@pytest.mark.parametrize("retired", [None, "0", "false", "1", "true"])
@pytest.mark.parametrize("sync,event", [(False, False), (True, False), (False, True), (True, True)])
def test_draft_profiler_reports_shared_dual_without_retired_env_reader(
    monkeypatch, tmp_path, retired: str | None, sync: bool, event: bool,
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

    constructed = []

    class Draft:
        num_kv_heads = 1
        qk_head_dim = 4

        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)

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
    argv = [
        "gguf_mtp_draft_rocprof.py", "--child", "--steps", "0", "--warmup", "0",
        "--child-json", str(output), "--record-stage-timings",
    ]
    if sync:
        argv.append("--sync-stage-timings")
    if event:
        argv.append("--gpu-event-stage-timings")
    monkeypatch.setattr(sys, "argv", argv)
    assert profiler.main() == 0
    options, = constructed
    assert options["sync_stage_timings"] is sync
    assert options["gpu_event_stage_timings"] is event
    assert json.loads(output.read_text())["q8_shared_dual"] is True
    assert not hasattr(profiler, "_resident_q8_shared_dual_enabled")
