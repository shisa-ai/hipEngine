"""Unit tests for hipEngine server log styling.

The binding contract is that styling is presentation-only: stripping ANSI from a
styled line must return the exact unstyled text, and a server started without a
supporting terminal must produce byte-identical output to plain uvicorn logging.
"""

from __future__ import annotations

import io
import logging
import sys

import pytest
from uvicorn.logging import DefaultFormatter

from hipengine.server.__main__ import build_parser
from hipengine.server.log_style import (
    COLOR_MODES,
    HipEngineLogFormatter,
    build_log_config,
    colors_enabled,
    label_color,
    resolve_color_mode,
    strip_styles,
    style_log_message,
    value_color,
)

REQUEST_INFO_LINE = (
    "REQUEST_INFO: endpoint=/v1/chat/completions stream=true model=qwen3.8-27b "
    "tokens_in=49 tokens_out=51 prefill_ms=340.7 prefill_tok_s=143.8 ttft_ms=363.2 "
    "decode_ms=5124.1 decode_tok_s=9.95 wall_ms=5487.3 kv_request_alloc_mib=16.0 "
    "kv_pool_gib=66.00 kv_pool_pages=4224 kv_pool_pinned_pages=4096 kv_pool_grows=0"
)
REQUEST_FAILED_LINE = (
    "REQUEST_FAILED: POST /v1/chat/completions status=400 code=invalid_request "
    "param=max_tokens message=must be >= 1"
)
MODEL_LOAD_LINE = "MODEL_LOAD: loading model=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf size=15.93 GiB"
MODEL_LOAD_PROGRESS_LINE = (
    "MODEL_LOAD: [#####-----------------------]  19.3% approx 3.08 GiB/15.93 GiB "
    "VRAM 5.48 GiB/124.00 GiB"
)
UNHANDLED_ERROR_LINE = (
    "UNHANDLED_ERROR: POST /v1/chat/completions status=500 code=internal_error "
    "exception=TimeoutError"
)


class _Stream:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class _BrokenStream:
    def isatty(self) -> bool:
        raise ValueError("I/O operation on closed file")


def _record(msg: str, args: tuple[object, ...] = (), **extra: object) -> logging.LogRecord:
    level = int(extra.pop("level", logging.INFO))
    record = logging.LogRecord(
        name="uvicorn.error",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )
    record.__dict__.update(extra)
    return record


def _formatter(**kwargs: object) -> HipEngineLogFormatter:
    return HipEngineLogFormatter(fmt="%(levelprefix)s %(message)s", **kwargs)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "auto"),
        ("", "auto"),
        ("auto", "auto"),
        ("AUTO", "auto"),
        ("always", "always"),
        ("ALWAYS", "always"),
        ("on", "always"),
        ("true", "always"),
        ("1", "always"),
        (True, "always"),
        ("never", "never"),
        ("off", "never"),
        ("false", "never"),
        ("0", "never"),
        (False, "never"),
        ("bogus", "auto"),
        ("  always  ", "always"),
    ],
)
def test_resolve_color_mode_normalizes_flag_and_env_values(value, expected) -> None:
    assert resolve_color_mode(value) == expected


def test_resolve_color_mode_keeps_an_explicit_default_for_unset_values() -> None:
    assert resolve_color_mode(None, default="never") == "never"
    assert resolve_color_mode("", default="never") == "never"


@pytest.mark.parametrize(
    ("mode", "tty", "environ", "expected"),
    [
        ("never", True, {"TERM": "xterm-256color"}, False),
        ("always", False, {"NO_COLOR": "1"}, True),
        ("auto", True, {"TERM": "xterm-256color"}, True),
        ("auto", False, {"TERM": "xterm-256color"}, False),
        ("auto", True, {"TERM": "xterm-256color", "NO_COLOR": ""}, False),
        ("auto", False, {"TERM": "xterm-256color", "FORCE_COLOR": "1"}, True),
        ("auto", False, {"TERM": "xterm-256color", "FORCE_COLOR": "0"}, False),
        ("auto", True, {"TERM": "dumb"}, False),
        ("auto", True, {}, True),
    ],
)
def test_colors_enabled_honors_mode_no_color_force_color_and_tty(
    mode, tty, environ, expected
) -> None:
    assert colors_enabled(mode, stream=_Stream(tty), environ=environ) is expected


def test_colors_enabled_handles_a_stream_without_a_working_isatty() -> None:
    assert colors_enabled("auto", stream=_BrokenStream(), environ={"TERM": "xterm"}) is False
    assert colors_enabled("auto", stream=None, environ={"TERM": "xterm"}) is (
        bool(sys.stderr.isatty())
    )


@pytest.mark.parametrize(
    "message",
    [
        REQUEST_INFO_LINE,
        REQUEST_FAILED_LINE,
        UNHANDLED_ERROR_LINE,
        MODEL_LOAD_LINE,
        MODEL_LOAD_PROGRESS_LINE,
        "EFFECTIVE_MTP: serving=pending engine_supported=unknown candidate_budget=requested:None",
        "KVCache: storage=bf16 scale=fp16 slots=1 requested_kv=16.00 GiB pages=4224",
        "Config: served_model=qwen3.8-27b eager_load=False",
        "RuntimeError: engine closed after 30.0 s",
        "hipEngine ready",
        "INFO: 127.0.0.1:52452 - \"POST /v1/chat/completions HTTP/1.1\" 200 OK",
        "50% of the budget is used",
        "a=b=c nested=1 UPPER=2",
        "[DONE]",
    ],
)
def test_styling_never_changes_the_message_text(message: str) -> None:
    styled = style_log_message(message)

    assert strip_styles(styled) == message


def test_styling_marks_the_label_and_dims_field_keys() -> None:
    styled = style_log_message(REQUEST_INFO_LINE)

    assert "\x1b[" in styled
    assert styled.startswith("\x1b[1m\x1b[96mREQUEST_INFO:\x1b[0m")
    assert "\x1b[2mendpoint\x1b[0m=\x1b[94m/v1/chat/completions\x1b[0m" in styled
    assert "\x1b[2mtokens_in\x1b[0m=\x1b[1m\x1b[96m49\x1b[0m" in styled


def test_styling_follows_the_level_for_an_unlisted_label() -> None:
    warning = style_log_message(REQUEST_FAILED_LINE, level=logging.WARNING)
    error = style_log_message(UNHANDLED_ERROR_LINE, level=logging.ERROR)

    assert warning.startswith("\x1b[1m\x1b[93mREQUEST_FAILED:\x1b[0m")
    assert error.startswith("\x1b[1m\x1b[91mUNHANDLED_ERROR:\x1b[0m")
    assert strip_styles(error) == UNHANDLED_ERROR_LINE


def test_label_color_prefers_a_known_label_over_the_level() -> None:
    assert label_color("REQUEST_INFO", level=logging.ERROR) == "bright_cyan"
    assert label_color("UNLISTED_LABEL", level=logging.CRITICAL) == "bright_red"
    assert label_color("UNLISTED_LABEL", level=logging.WARNING) == "bright_yellow"
    assert label_color("UNLISTED_LABEL", level=logging.INFO) == "cyan"
    assert label_color("UNLISTED_LABEL") == "cyan"


def test_styling_highlights_a_progress_bar_fill_and_dim_remainder() -> None:
    styled = style_log_message(MODEL_LOAD_PROGRESS_LINE)

    assert "[\x1b[92m#####\x1b[0m\x1b[2m-----------------------\x1b[0m]" in styled
    assert strip_styles(styled) == MODEL_LOAD_PROGRESS_LINE


@pytest.mark.parametrize(
    "value, expected",
    [
        ("49", "bright_cyan"),
        ("340.7", "bright_cyan"),
        ("-1", "bright_cyan"),
        ("true", "bright_green"),
        ("True", "bright_green"),
        ("enabled", "bright_green"),
        ("false", "bright_yellow"),
        ("False", "bright_yellow"),
        ("off", "bright_yellow"),
        ("None", "bright_black"),
        ("pending", "bright_black"),
        ("requested:None", "bright_black"),
        ("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf", "bright_blue"),
        ("bf16", "bright_white"),
        ("qwen3.8-27b", "bright_white"),
    ],
)
def test_value_color_reads_the_value_rather_than_the_key(value: str, expected: str) -> None:
    assert value_color(value) == expected


def test_styling_bolds_a_measurement_and_dims_its_unit() -> None:
    line = "MODEL_LOAD: loading model=/models/gguf/x.gguf size=15.93 GiB"
    styled = style_log_message(line)

    assert "\x1b[2msize\x1b[0m=\x1b[1m\x1b[96m15.93\x1b[0m\x1b[2m GiB\x1b[0m" in styled
    assert strip_styles(styled) == line


def test_styling_colors_a_bare_measurement_in_a_progress_line() -> None:
    styled = style_log_message(MODEL_LOAD_PROGRESS_LINE)

    assert "\x1b[1m\x1b[96m19.3\x1b[0m\x1b[2m%\x1b[0m" in styled
    assert "\x1b[1m\x1b[96m3.08\x1b[0m\x1b[2m GiB\x1b[0m" in styled


def test_styling_keeps_a_unit_word_out_of_a_non_numeric_value() -> None:
    line = "KVCache: storage=bf16 pages=4224 pages_pinned=4096"
    styled = style_log_message(line)

    assert "\x1b[2mstorage\x1b[0m=\x1b[97mbf16\x1b[0m" in styled
    assert "\x1b[2mpages\x1b[0m=\x1b[1m\x1b[96m4224\x1b[0m" in styled
    assert strip_styles(styled) == line


def test_styling_colors_a_known_mixed_case_label_and_leaves_prose_alone() -> None:
    known = style_log_message("KVCache: storage=bf16")
    prose = style_log_message("RuntimeError: engine closed after 30.0 s")

    assert known.startswith("\x1b[1m\x1b[95mKVCache:\x1b[0m")
    assert prose.startswith("RuntimeError: ")
    assert strip_styles(prose) == "RuntimeError: engine closed after 30.0 s"


def test_styling_ignores_brackets_that_are_not_progress_bars() -> None:
    for message in ("[DONE]", "retry [5] times", "[######]", "[----]"):
        assert strip_styles(style_log_message(message)) == message


def test_styling_leaves_an_unknown_label_on_the_default_color() -> None:
    styled = style_log_message("SOME_NEW_LABEL: value=1")

    assert styled.startswith("\x1b[1m\x1b[36mSOME_NEW_LABEL:\x1b[0m")


def test_formatter_styles_records_and_keeps_the_plain_text() -> None:
    styled = _formatter(use_colors=True).format(_record(REQUEST_INFO_LINE))
    plain = _formatter(use_colors=False).format(_record(REQUEST_INFO_LINE))

    assert "\x1b[96mREQUEST_INFO:\x1b[0m" in styled
    assert strip_styles(styled) == plain
    assert strip_styles(styled) == f"INFO:     {REQUEST_INFO_LINE}"


def test_formatter_styles_a_parameterized_record_after_substitution() -> None:
    styled = _formatter(use_colors=True).format(
        _record(
            "REQUEST_FAILED: %s status=%d code=%s",
            ("POST /v1/models", 400, "invalid"),
            level=logging.WARNING,
        )
    )

    assert strip_styles(styled) == (
        "WARNING:  REQUEST_FAILED: POST /v1/models status=400 code=invalid"
    )
    assert "\x1b[93mREQUEST_FAILED:\x1b[0m" in styled
    assert "\x1b[2mstatus\x1b[0m=\x1b[1m\x1b[96m400\x1b[0m" in styled


def test_formatter_colors_a_warning_label_yellow_and_an_error_label_red() -> None:
    warning = _formatter(use_colors=True).format(
        _record(REQUEST_FAILED_LINE, level=logging.WARNING)
    )
    error = _formatter(use_colors=True).format(_record(UNHANDLED_ERROR_LINE, level=logging.ERROR))

    assert "\x1b[93mREQUEST_FAILED:\x1b[0m" in warning
    assert "\x1b[91mUNHANDLED_ERROR:\x1b[0m" in error


def test_formatter_leaves_uvicorn_color_message_records_to_uvicorn() -> None:
    record = _record(
        "Uvicorn running on %s://%s:%d (Press CTRL+C to quit)",
        ("http", "127.0.0.1", 8000),
        color_message="Uvicorn running on \x1b[1m%s://%s:%d\x1b[0m (Press CTRL+C to quit)",
    )

    styled = _formatter(use_colors=True).format(record)
    plain = DefaultFormatter(fmt="%(levelprefix)s %(message)s", use_colors=True).format(
        _record(
            "Uvicorn running on %s://%s:%d (Press CTRL+C to quit)",
            ("http", "127.0.0.1", 8000),
            color_message="Uvicorn running on \x1b[1m%s://%s:%d\x1b[0m (Press CTRL+C to quit)",
        )
    )

    assert styled == plain
    assert "8000" in styled
    assert "%s" not in styled


def test_formatter_output_matches_uvicorn_when_colors_are_disabled() -> None:
    record = _record(REQUEST_INFO_LINE)
    ours = _formatter(use_colors=False).format(record)
    uvicorn = DefaultFormatter(fmt="%(levelprefix)s %(message)s", use_colors=False).format(
        _record(REQUEST_INFO_LINE)
    )

    assert ours == uvicorn
    assert "\x1b[" not in ours


@pytest.mark.parametrize("mode", COLOR_MODES)
def test_build_log_config_applies_the_color_mode(mode: str) -> None:
    config = build_log_config(mode=mode, stream=_Stream(tty=False), environ={})

    assert config["formatters"]["default"]["()"] == (
        "hipengine.server.log_style.HipEngineLogFormatter"
    )
    assert config["formatters"]["default"]["fmt"] == "%(levelprefix)s %(message)s"
    assert config["formatters"]["default"]["use_colors"] is (mode == "always")
    assert config["formatters"]["access"]["use_colors"] is (mode == "always")


def test_build_log_config_keeps_uvicorn_defaults_and_does_not_mutate_them() -> None:
    from uvicorn.config import LOGGING_CONFIG

    before = repr(LOGGING_CONFIG)
    config = build_log_config(mode="always")

    assert repr(LOGGING_CONFIG) == before
    assert config["formatters"]["access"]["()"] == "uvicorn.logging.AccessFormatter"
    assert config["handlers"]["default"]["stream"] == "ext://sys.stderr"
    assert config["loggers"]["uvicorn"]["handlers"] == ["default"]


def test_build_log_config_accepts_a_mode_and_styles_a_tee_style_stream() -> None:
    piped = build_log_config(mode="auto", stream=io.StringIO(), environ={})
    forced = build_log_config(mode="always", stream=io.StringIO(), environ={})

    assert piped["formatters"]["default"]["use_colors"] is False
    assert forced["formatters"]["default"]["use_colors"] is True


def test_log_color_flag_env_and_typo_handling(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_LOG_COLOR", raising=False)
    assert build_parser().parse_args(["--model", "fake-path"]).log_color == "auto"

    monkeypatch.setenv("HIPENGINE_LOG_COLOR", "always")
    assert build_parser().parse_args(["--model", "fake-path"]).log_color == "always"

    monkeypatch.setenv("HIPENGINE_LOG_COLOR", "bogus")
    args = build_parser().parse_args(["--model", "fake-path"])
    assert args.log_color == "bogus"
    config = build_log_config(mode=args.log_color, stream=_Stream(tty=False), environ={})
    assert config["formatters"]["default"]["use_colors"] is False

    assert build_parser().parse_args(["--model", "fake-path", "--log-color", "never"]).log_color == (
        "never"
    )


def test_log_color_flag_rejects_an_unknown_value() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "fake-path", "--log-color", "sometimes"])
