"""Tests for the AR-only control shim (no GPU, no harness import)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import types

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "gguf_c1c8_ar_only_control.py"


def _load():
    spec = importlib.util.spec_from_file_location("ar_only_ctl", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_import_has_no_side_effects():
    # Importing must not parse argv or call the harness: the shim runs main() only under __main__.
    module = _load()
    assert callable(module.main)
    assert callable(module.build_argv)


def test_build_argv_pins_explicit_mode_and_no_expected_mtp_widths():
    module = _load()
    argv = module.build_argv("/m/x.gguf", "/tmp/o.json", 3, 24)
    joined = " ".join(argv)
    assert "--mtp-request-mode explicit" in joined
    assert "--expected-mtp-widths none" in joined
    assert "--widths 3" in joined
    assert "--max-tokens 24" in joined
    assert "--model /m/x.gguf" in joined
    assert "--output /tmp/o.json" in joined


def test_control_refuses_when_hook_is_missing():
    module = _load()
    fake = types.SimpleNamespace(ARMS=("ar", "mtp"))
    try:
        module.install_ar_only_control(fake)
    except SystemExit as exc:
        assert "_request_mtp_value" in str(exc)
    else:
        raise AssertionError("expected SystemExit when the harness hook is absent")


def test_control_refuses_when_harness_already_declines_to_speculate():
    # If the harness would not speculate anyway, an AR-only run proves nothing; fail closed.
    module = _load()
    fake = types.SimpleNamespace(
        ARMS=("ar", "mtp"),
        _request_mtp_value=lambda **kw: False,
    )
    try:
        module.install_ar_only_control(fake)
    except SystemExit as exc:
        assert "meaningless" in str(exc)
    else:
        raise AssertionError("expected SystemExit when speculation was already off")


def test_control_patches_every_arm_to_decline(capsys):
    module = _load()
    calls = []

    def original(*, arm: str, request_mode: str) -> bool:
        calls.append((arm, request_mode))
        return arm == "mtp" and request_mode == "explicit"

    fake = types.SimpleNamespace(ARMS=("ar", "mtp"), _request_mtp_value=original)
    module.install_ar_only_control(fake)
    assert fake._request_mtp_value(arm="mtp", request_mode="explicit") is False
    assert fake._request_mtp_value(arm="ar", request_mode="automatic") is False
    out = capsys.readouterr().out
    assert "[CONTROL" in out and "original explicit MTP value=True" in out
    # The probe must ask the same question the harness asks, not a different one.
    assert calls == [("mtp", "explicit")]


def test_control_passes_unknown_kwargs_through_to_false():
    # The harness may call the hook positionally in future; the replacement must not TypeError.
    module = _load()
    fake = types.SimpleNamespace(
        ARMS=("ar", "mtp"), _request_mtp_value=lambda *a, **k: True
    )
    module.install_ar_only_control(fake)
    assert fake._request_mtp_value("mtp", "explicit") is False
