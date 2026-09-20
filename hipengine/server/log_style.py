"""Optional ANSI styling for hipEngine server logs.

The server writes structured, greppable lines such as
``REQUEST_INFO: endpoint=/v1/chat/completions tokens_in=49 ...``. This module
styles the parts a reader scans for - the level prefix, the leading label, and
the ``key=`` of each field - without changing a single character of the text, so
``grep``, ``cut``, and log shippers keep working on styled output.

Styling is off unless the target stream looks like a supporting terminal:

- ``--log-color auto`` (default) styles only when the stream is a TTY, honoring
  ``NO_COLOR`` and ``FORCE_COLOR``.
- ``--log-color always`` styles even when the output is redirected, which is what
  a ``| tee`` pane needs.
- ``--log-color never`` never styles.

Text is never rewritten: :func:`strip_styles` on a styled message returns the
unstyled message byte for byte.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from typing import Any, Mapping

from uvicorn.logging import ColourizedFormatter

COLOR_MODES = ("auto", "always", "never")

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"

_FOREGROUNDS: dict[str, str] = {
    "black": "\x1b[30m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "bright_black": "\x1b[90m",
    "bright_red": "\x1b[91m",
    "bright_green": "\x1b[92m",
    "bright_yellow": "\x1b[93m",
    "bright_blue": "\x1b[94m",
    "bright_magenta": "\x1b[95m",
    "bright_cyan": "\x1b[96m",
    "bright_white": "\x1b[97m",
}

# Colors for the leading ``LABEL:`` of a structured line. Labels that are not
# listed take their color from the record's level, so a new label - and a new
# failure label such as ``UNHANDLED_ERROR`` - is already legible without editing
# this map.
_LABEL_COLORS: dict[str, str] = {
    "REQUEST_INFO": "bright_cyan",
    "MODEL_LOAD": "bright_blue",
    "EFFECTIVE_MTP": "bright_magenta",
    "LOAD_TIMING": "blue",
    "STARTUP_MEMORY": "blue",
    "STARTUP_MEMORY_SAMPLE": "blue",
    "STARTUP_SCRATCH_PROBE": "blue",
    "WARMUP": "blue",
    "WARMUP_CHAT": "blue",
    "DEBUG_PAYLOAD": "bright_black",
}
_DEFAULT_LABEL_COLOR = "cyan"
_WARNING_LABEL_COLOR = "bright_yellow"
_ERROR_LABEL_COLOR = "bright_red"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_LABEL_RE = re.compile(r"^([A-Z][A-Z0-9_]{2,}):")
_FIELD_RE = re.compile(r"(?<![\w=])([a-z][a-z0-9_]*)=(\S*)")
_PROGRESS_RE = re.compile(r"\[([#=]+)(-*)\]")

_ALWAYS_VALUES = frozenset({"always", "on", "true", "yes", "1", "force"})
_NEVER_VALUES = frozenset({"never", "off", "false", "no", "0", "none"})


def resolve_color_mode(value: Any, *, default: str = "auto") -> str:
    """Normalize a requested color mode from a flag or environment value.

    Anything unrecognized resolves to ``auto`` rather than raising, so a typo in
    an environment variable cannot take a server down.
    """

    if value is None:
        return default
    if isinstance(value, bool):
        return "always" if value else "never"
    text = str(value).strip().lower()
    if not text:
        return default
    if text in _ALWAYS_VALUES:
        return "always"
    if text in _NEVER_VALUES:
        return "never"
    return "auto"


def colors_enabled(
    mode: Any = "auto",
    *,
    stream: Any = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether ANSI styling should be emitted for ``stream``.

    ``auto`` styles a TTY only, and respects the ``NO_COLOR`` convention and the
    ``FORCE_COLOR`` override. ``stream`` defaults to ``sys.stderr`` because that
    is where the server's own log handler writes.
    """

    resolved = resolve_color_mode(mode)
    if resolved == "never":
        return False
    if resolved == "always":
        return True
    environment = os.environ if environ is None else environ
    if "NO_COLOR" in environment:
        return False
    force_color = str(environment.get("FORCE_COLOR", "")).strip().lower()
    if force_color and force_color not in _NEVER_VALUES:
        return True
    if str(environment.get("TERM", "")).strip().lower() == "dumb":
        return False
    target = _default_stream() if stream is None else stream
    try:
        return bool(target.isatty())
    except (AttributeError, ValueError):
        return False


def _default_stream() -> Any:
    import sys

    return sys.stderr


def paint(text: str, color: str | None = None, *, bold: bool = False, dim: bool = False) -> str:
    """Wrap ``text`` in ANSI styling, or return it unchanged for no styling."""

    prefix = ""
    if bold:
        prefix += BOLD
    if dim:
        prefix += DIM
    if color is not None:
        prefix += _FOREGROUNDS.get(str(color), "")
    if not prefix:
        return str(text)
    return f"{prefix}{text}{RESET}"


def strip_styles(text: str) -> str:
    """Remove ANSI styling, recovering the original message text."""

    return _ANSI_RE.sub("", str(text))


def label_color(label: str, *, level: int | None = None) -> str:
    """Return the color for a structured-line label.

    A known label keeps its own color. Everything else follows the record's
    level, so failures are red and warnings yellow even for a label this module
    has never seen.
    """

    known = _LABEL_COLORS.get(str(label))
    if known is not None:
        return known
    if level is not None:
        if level >= logging.ERROR:
            return _ERROR_LABEL_COLOR
        if level >= logging.WARNING:
            return _WARNING_LABEL_COLOR
    return _DEFAULT_LABEL_COLOR


def style_log_message(message: str, *, level: int | None = None) -> str:
    """Style one already-formatted log message without changing its text."""

    text = str(message)
    label_match = _LABEL_RE.match(text)
    if label_match is not None:
        label = label_match.group(1)
        colored = paint(f"{label}:", label_color(label, level=level), bold=True)
        text = colored + text[label_match.end() :]
    text = _FIELD_RE.sub(
        lambda match: f"{DIM}{match.group(1)}{RESET}={match.group(2)}", text
    )
    return _PROGRESS_RE.sub(
        lambda match: (
            f"[{paint(match.group(1), 'bright_green')}"
            f"{paint(match.group(2), dim=True)}]"
        ),
        text,
    )


class HipEngineLogFormatter(ColourizedFormatter):
    """uvicorn's level-prefix colors plus hipEngine structured-line styling.

    uvicorn's own records carry a ``color_message`` and keep uvicorn's formatting
    untouched; hipEngine's own lines are styled here, once, instead of at every
    call site.
    """

    def formatMessage(self, record: logging.LogRecord) -> str:
        if not self.use_colors or "color_message" in record.__dict__:
            return super().formatMessage(record)
        recordcopy = copy.copy(record)
        # uvicorn's formatter copies the record and reads its pre-formatted
        # ``message`` field, so the styled text has to be written there; leaving
        # ``msg`` and ``args`` alone keeps uvicorn's own paths intact.
        recordcopy.__dict__["message"] = style_log_message(
            record.getMessage(), level=record.levelno
        )
        return super().formatMessage(recordcopy)


def build_log_config(
    *,
    mode: Any = "auto",
    stream: Any = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return uvicorn's logging config with hipEngine styling applied.

    The access formatter keeps uvicorn's implementation, which colors status
    codes and bolds the request line, but takes this server's color decision
    instead of its own TTY detection.
    """

    from uvicorn.config import LOGGING_CONFIG

    config = copy.deepcopy(LOGGING_CONFIG)
    enabled = colors_enabled(mode, stream=stream, environ=environ)
    config["formatters"]["default"]["()"] = "hipengine.server.log_style.HipEngineLogFormatter"
    config["formatters"]["default"]["use_colors"] = enabled
    config["formatters"]["access"]["use_colors"] = enabled
    return config


__all__ = [
    "COLOR_MODES",
    "HipEngineLogFormatter",
    "build_log_config",
    "colors_enabled",
    "label_color",
    "paint",
    "resolve_color_mode",
    "strip_styles",
    "style_log_message",
]
