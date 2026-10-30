from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("shape", ["row", "pack8_scalehoist", "pack8_llama", "pack16", "x8_dscale"])
@pytest.mark.parametrize("sync,event", [(False, False), (True, False), (False, True), (True, True)])
def test_draft_profiler_preserves_explicit_head_comparison(monkeypatch, shape, sync, event):
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
            "--q6-top1-dp4a", "--q6-top1-stage1-shape", shape]
    if sync:
        argv.append("--sync-stage-timings")
    if event:
        argv.append("--gpu-event-stage-timings")
    monkeypatch.setattr(sys, "argv", argv)
    assert profiler.main() == 0
    args, = captured
    assert args.q6_top1_stage1_shape == shape
    assert args.sync_stage_timings is sync
    assert args.gpu_event_stage_timings is event
    assert os.environ["HIPENGINE_GGUF_Q6_TOP1_STAGE1_SHAPE"] == shape
    assert os.environ["HIPENGINE_RESIDENT_MTP_DRAFT_Q6_TOP1_DP4A"] == "1"


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ambient", [None, "0", "1"])
def test_draft_profiler_router_comparison_publishes_only_positive_request(monkeypatch, enabled, ambient):
    name = "HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL"
    if ambient is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, ambient)
    captured = []

    def child(args):
        profiler._apply_route_env(args)
        captured.append(args.router_row_parallel)
        return 0

    monkeypatch.setattr(profiler, "_run_child", child)
    argv = ["gguf_mtp_draft_rocprof.py", "--child"]
    if enabled:
        argv.append("--router-row-parallel")
    monkeypatch.setattr(sys, "argv", argv)
    assert profiler.main() == 0
    assert captured == [enabled]
    assert os.environ.get(name) == ("1" if enabled else ambient)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ambient", [None, "0", "1"])
def test_draft_profiler_preserves_explicit_silu_down_comparison(monkeypatch, enabled, ambient):
    name = "HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED"
    if ambient is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, ambient)

    def child(args):
        assert args.selected_silu_down_fused is enabled
        profiler._apply_route_env(args)
        return 0

    monkeypatch.setattr(profiler, "_run_child", child)
    argv = ["gguf_mtp_draft_rocprof.py", "--child"]
    if enabled:
        argv.append("--selected-silu-down-fused")
    monkeypatch.setattr(sys, "argv", argv)
    assert profiler.main() == 0
    assert os.environ.get(name) == ("1" if enabled else ambient)


@pytest.mark.parametrize(
    "field,option,control_env",
    [
        ("selected_silu_down_fused", "--selected-silu-down-fused", "HIPENGINE_RESIDENT_MTP_DRAFT_SELECTED_SILU_DOWN_FUSED"),
        ("router_row_parallel", "--router-row-parallel", "HIPENGINE_RESIDENT_MTP_DRAFT_ROUTER_ROW_PARALLEL"),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_draft_profiler_artifact_labels_comparisons_honestly(
    monkeypatch, tmp_path, field, option, control_env, enabled
):
    name = control_env
    monkeypatch.delenv(name, raising=False)
    compiler_version_file = tmp_path / "hipcc.txt"
    compiler_version_file.write_text("hipcc\n", encoding="utf-8")
    output = tmp_path / "artifact.json"
    args = SimpleNamespace(
        model=tmp_path / "model.gguf",
        rocprofv3="/fake/rocprofv3",
        raw_root=tmp_path / "raw",
        compiler_version_file=compiler_version_file,
        roctx_sdk=tmp_path / "roctx",
        prompt_ids="1",
        steps=1,
        warmup=0,
        max_seq=8,
        draft_n_max=1,
        top_k=1,
        vocab_cap=0,
        q6_top1_stage1_threads=128,
        q6_top1_stage1_shape="pack8",
        selected_down_x8_repack="off",
        q6_top1_dp4a=False,
        dense_q8_dp4a=False,
        dense_q8_dp4a_stages="",
        selected_silu_down_fused=False,
        router_row_parallel=False,
        record_stage_timings=False,
        sync_stage_timings=False,
        gpu_event_stage_timings=False,
        skip_warmbuild=True,
        out=output,
        top=1,
    )
    setattr(args, field, enabled)
    monkeypatch.setattr(
        profiler, "_prepare_roctx_override", lambda sdk: tmp_path / "roctx"
    )
    monkeypatch.setattr(profiler, "_roctx_sdk_dep_paths", lambda sdk: [])
    commands = []

    def run(command, **kwargs):
        commands.append(list(command))
        if "--child" in command:
            child_json = Path(command[command.index("--child-json") + 1])
            child_json.write_text(
                json.dumps(
                    {
                        "host_ms": ["1.0"],
                        field: option in command,
                    }
                ),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(profiler.subprocess, "run", run)
    monkeypatch.setattr(
        profiler, "_single_file", lambda root, pattern: root / "trace.csv"
    )
    monkeypatch.setattr(
        profiler, "_read_marker_windows", lambda path, prefix: [(0, 1.0, 2.0)]
    )
    monkeypatch.setattr(profiler, "_read_kernels", lambda path: [])
    monkeypatch.setattr(
        profiler, "_summarize_draft", lambda *args, **kwargs: {"total_ms": 1.0}
    )
    monkeypatch.setattr(profiler, "_print_summary", lambda summary: None)
    monkeypatch.setattr(profiler, "_print_stage_timings", lambda child: None)

    assert profiler._run_parent(args) == 0

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact[field] is enabled
    assert artifact["status"] == "diagnostic_retained"
    assert artifact["performance_claim"] is False
    assert "host/GPU split" in artifact["purpose"]
    assert artifact["child"][field] is enabled
    rocprof_command, = commands
    assert (option in rocprof_command) is enabled


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
