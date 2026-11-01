"""RED/GREEN for the DE-3 route probe's diagnostic lifetime.

``scripts/qwen4exp_de3_route_probe.py`` turns on two diagnostic environment
switches (``HIPENGINE_KERNEL_CENSUS``, ``HIPENGINE_GGUF_PACK8_GEMV_WIDE``) and
swaps ``gguf_linear._pack8_gemv_decode_threads`` for a counting wrapper.  Both
are process-global, so every exit path - success, setup failure, capture
failure, report failure, and the closed-session early return - has to put the
original values (or their absence) back and close the session it opened.

These cases drive ``main`` on the CPU with fake modules: no model, GPU, or HIP
runtime is involved.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys

import pytest

from hipengine.kernels import launch_census
from hipengine.loading import gguf as gguf_loading
from hipengine.runtime import gguf_linear
from hipengine.runtime import prefill as prefill_module
from hipengine.runtime import qwen35_gguf_runner
from hipengine.tokenization import gguf as tokenization_gguf
from scripts import gguf_mtp_bench

_CENSUS_ENV = "HIPENGINE_KERNEL_CENSUS"
_WIDE_ENV = "HIPENGINE_GGUF_PACK8_GEMV_WIDE"

_PROBE_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "qwen4exp_de3_route_probe.py"
)


def _load_probe():
    spec = importlib.util.spec_from_file_location("_de3_route_probe_under_test", _PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Start every case from "neither switch is set"."""

    monkeypatch.delenv(_CENSUS_ENV, raising=False)
    monkeypatch.delenv(_WIDE_ENV, raising=False)


class _Result:
    def __init__(self, token_id: int) -> None:
        self.token_id = token_id


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runner_present: bool = True,
    reset_error: BaseException | None = None,
    prompt_error: BaseException | None = None,
    prefill_error: BaseException | None = None,
    snapshot_error: BaseException | None = None,
) -> dict:
    """Replace every module ``main`` reaches with a CPU fake.

    The returned ``record`` collects the calls that prove the diagnostic was
    live while the session ran and torn down afterwards.
    """

    record: dict = {
        "constructed": 0,
        "closed": 0,
        "resets": 0,
        "census_resets": 0,
        "census_snapshots": 0,
        "prefills": 0,
        "steps": 0,
        "predicate_calls": 0,
    }

    def sentinel_predicate(**kwargs):
        record["predicate_calls"] += 1
        return 0

    monkeypatch.setattr(gguf_linear, "_pack8_gemv_decode_threads", sentinel_predicate)
    record["sentinel"] = sentinel_predicate

    def fake_reset() -> None:
        record["census_resets"] += 1
        if reset_error is not None:
            raise reset_error

    def fake_snapshot() -> dict:
        record["census_snapshots"] += 1
        if snapshot_error is not None:
            raise snapshot_error
        return {"rows": [{"quant": "q8_0", "symbol": "gguf_k_pack8_gemv_out_kernel"}]}

    monkeypatch.setattr(launch_census, "reset", fake_reset)
    monkeypatch.setattr(launch_census, "snapshot", fake_snapshot)

    class FakePrefillConfig:
        def __init__(self, **kwargs) -> None:
            record["prefill_config"] = kwargs

    def fake_scan_gguf(path) -> dict:
        record["scanned_model"] = str(path)
        return {"path": str(path)}

    class FakeTokenizer:
        @classmethod
        def from_gguf_info(cls, info):
            record["tokenizer_info"] = info
            return cls()

    def fake_build_chat_prompt(tokenizer, prompt) -> list[int]:
        record["prompt"] = prompt
        if prompt_error is not None:
            raise prompt_error
        return [1, 2, 3]

    class FakeRunner:
        backend = "fake_backend"

    class FakeSession:
        def __init__(self, model, **kwargs) -> None:
            record["constructed"] += 1
            record["model"] = model
            record["session_kwargs"] = kwargs
            self.runner = FakeRunner() if runner_present else None

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            self.close()

        def close(self) -> None:
            record["closed"] += 1

        def reset(self) -> None:
            record["resets"] += 1

        def prefill(self, tokens, **kwargs) -> _Result:
            record["prefills"] += 1
            record["prefill_tokens"] = list(tokens)
            record["prefill_kwargs"] = kwargs
            record["env_during_run"] = (os.environ.get(_CENSUS_ENV), os.environ.get(_WIDE_ENV))
            patched = gguf_linear._pack8_gemv_decode_threads
            record["predicate_patched"] = patched is not sentinel_predicate
            record["predicate_value"] = patched(
                rows=1,
                out_features=320,
                variant="pack8_gemv_f32_f32_out",
                threads=0,
            )
            if prefill_error is not None:
                raise prefill_error
            return _Result(7)

        def step(self, token, **kwargs) -> _Result:
            record["steps"] += 1
            return _Result(int(token) + 1)

    monkeypatch.setattr(prefill_module, "PrefillConfig", FakePrefillConfig)
    monkeypatch.setattr(gguf_loading, "scan_gguf", fake_scan_gguf)
    monkeypatch.setattr(tokenization_gguf, "Qwen35GGUFTokenizer", FakeTokenizer)
    monkeypatch.setattr(gguf_mtp_bench, "build_chat_prompt", fake_build_chat_prompt)
    monkeypatch.setattr(qwen35_gguf_runner, "Qwen35GGUFResidentSession", FakeSession)
    return record


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, *, json_out: bool = True
):
    argv = [
        "qwen4exp_de3_route_probe",
        "--model",
        str(tmp_path / "fake.gguf"),
        "--decode-steps",
        "2",
    ]
    out = tmp_path / "probe.json"
    if json_out:
        argv += ["--json-out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    return probe.main(), out


def _assert_env_restored(record: dict, *, census: str | None, wide: str | None) -> None:
    """The switches must be back to their pre-probe values, or gone."""

    assert os.environ.get(_CENSUS_ENV) == census
    assert os.environ.get(_WIDE_ENV) == wide


def _assert_predicate_restored(record: dict) -> None:
    assert gguf_linear._pack8_gemv_decode_threads is record["sentinel"]


def test_success_restores_absent_env_and_predicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    record = _install_fakes(monkeypatch)

    code, out = _run(monkeypatch, tmp_path)

    assert code == 0
    _assert_env_restored(record, census=None, wide=None)
    _assert_predicate_restored(record)
    assert record["env_during_run"] == ("1", "1")
    assert record["predicate_patched"] is True
    assert record["predicate_value"] == 0
    assert record["closed"] == 1
    assert record["prefills"] == 1
    assert record["steps"] == 2

    # Default geometry is unchanged: prompt 3 + decode 2 + 8 headroom.
    assert record["prefill_config"] == {"attn_aotriton_min_tokens": 0}
    assert record["session_kwargs"]["backend"] == "auto"
    assert record["session_kwargs"]["max_sequence_length"] == 3 + 2 + 8
    assert record["prefill_kwargs"] == {
        "use_bulk": True,
        "bulk_attention_mode": "bulk",
        "return_logits": True,
        "capture_hidden_seed_fp32": False,
    }

    printed = capsys.readouterr().out
    assert "prompt tokens: 3" in printed
    assert "backend: fake_backend" in printed
    assert "predicate returned a wider block 0 times" in printed
    assert "=== census: pack8_gemv launches ===" in printed
    assert f"wrote {out}" in printed

    payload = json.loads(out.read_text())
    assert set(payload) == {"decisions", "eligible", "census"}
    assert payload["decisions"] == {"(1, 320, 'pack8_gemv_f32_f32_out', 0)": 1}
    assert payload["eligible"] == {"(320, 'pack8_gemv_f32_f32_out')": 1}
    assert payload["census"] == {
        "rows": [{"quant": "q8_0", "symbol": "gguf_k_pack8_gemv_out_kernel"}]
    }


def test_success_restores_preexisting_env_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv(_CENSUS_ENV, "0")
    monkeypatch.setenv(_WIDE_ENV, "off")
    record = _install_fakes(monkeypatch)

    code, _ = _run(monkeypatch, tmp_path)

    assert code == 0
    _assert_env_restored(record, census="0", wide="off")
    _assert_predicate_restored(record)
    assert record["env_during_run"] == ("1", "1")
    assert record["closed"] == 1


def test_setup_failure_before_patch_restores_env_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    record = _install_fakes(monkeypatch, reset_error=RuntimeError("census reset exploded"))

    with pytest.raises(RuntimeError, match="census reset exploded"):
        _run(monkeypatch, tmp_path)

    _assert_env_restored(record, census=None, wide=None)
    _assert_predicate_restored(record)
    assert record["constructed"] == 0
    assert record["closed"] == 0


def test_setup_failure_after_patch_restores_env_and_predicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    record = _install_fakes(monkeypatch, prompt_error=RuntimeError("prompt build exploded"))

    with pytest.raises(RuntimeError, match="prompt build exploded"):
        _run(monkeypatch, tmp_path)

    _assert_env_restored(record, census=None, wide=None)
    _assert_predicate_restored(record)
    assert record["constructed"] == 0
    assert record["closed"] == 0


def test_closed_session_returns_2_and_restores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture
) -> None:
    record = _install_fakes(monkeypatch, runner_present=False)

    code, out = _run(monkeypatch, tmp_path)

    assert code == 2
    assert "session closed during setup" in capsys.readouterr().out
    _assert_env_restored(record, census=None, wide=None)
    _assert_predicate_restored(record)
    assert record["closed"] == 1
    assert record["prefills"] == 0
    assert not out.exists()


def test_capture_failure_closes_session_and_restores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    record = _install_fakes(monkeypatch, prefill_error=RuntimeError("prefill exploded"))

    with pytest.raises(RuntimeError, match="prefill exploded"):
        _run(monkeypatch, tmp_path)

    _assert_env_restored(record, census=None, wide=None)
    _assert_predicate_restored(record)
    assert record["closed"] == 1
    assert record["steps"] == 0


def test_report_failure_in_snapshot_closes_session_and_restores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    record = _install_fakes(monkeypatch, snapshot_error=RuntimeError("census snapshot exploded"))

    with pytest.raises(RuntimeError, match="census snapshot exploded"):
        _run(monkeypatch, tmp_path)

    _assert_env_restored(record, census=None, wide=None)
    _assert_predicate_restored(record)
    assert record["closed"] == 1
    assert record["steps"] == 2


def test_report_failure_in_json_write_closes_session_and_restores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    record = _install_fakes(monkeypatch)

    # A missing parent directory makes the report write itself fail.
    argv = [
        "qwen4exp_de3_route_probe",
        "--model",
        str(tmp_path / "fake.gguf"),
        "--json-out",
        str(tmp_path / "missing" / "probe.json"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(OSError):
        probe.main()

    _assert_env_restored(record, census=None, wide=None)
    _assert_predicate_restored(record)
    assert record["closed"] == 1
    assert record["census_snapshots"] == 1


def test_help_leaves_diagnostics_untouched(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    record = _install_fakes(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["qwen4exp_de3_route_probe", "--help"])

    with pytest.raises(SystemExit) as exc:
        probe.main()

    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out
    _assert_env_restored(record, census=None, wide=None)
    _assert_predicate_restored(record)
    assert record["constructed"] == 0
    assert record["census_resets"] == 0
