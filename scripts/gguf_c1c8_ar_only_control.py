#!/usr/bin/env python3
"""Run the C1-C8 server harness with speculation removed from *both* arms.

Why this exists
---------------
The published C1-C8 harness always runs both arms, and in `automatic` serving mode the engine
speculates as soon as group rows >= 2. A width-1-vs-width-2 kernel diff taken from the harness
therefore mixes MTP verifier kernels into the wider run, and a row-scaling reading built on it is
wrong (measured that way on 2026-08-30: the width-2 run's extra kernels were verifier shapes, not
extra rows). Pinning speculation off leaves the protocol, prompts, widths, sampling and timing
mechanics untouched and removes speculation from both runs.

Why it patches `_request_mtp_value` instead of `ARMS`
----------------------------------------------------
Setting `ARMS = ("ar",)` looks obvious and breaks the harness: `run()` reads
`measured["mtp"]["rows"]` unconditionally for the AR-vs-MTP content-exactness comparison, so
removing the arm raises `KeyError` and no packet is written. Both arms therefore keep running, and
both request no speculation. The server stays in `opt_in` mode under `--mtp-request-mode explicit`,
so the nominal MTP arm degenerates to plain AR. That doubles the AR work in every trace
proportionally at both widths, which leaves per-launch means and cross-width ratios - what a kernel
diff actually reads - untouched.

Usage
-----
    .venv/bin/python scripts/gguf_c1c8_ar_only_control.py \
        --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --width 1 \
        --output /tmp/he-rows/ar-1.json --max-tokens 24

Under `rocprofv3`, also export the pair docs/KERNELS.md trap 5 requires
(`HIPENGINE_REQUIRE_CACHED_BUILD=1` **and**
`HIPENGINE_COMPILER_VERSION_FILE=<file>`), or the profiled
process spawns `hipcc --version` and hangs with the GPU idle.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

HARNESS = "scripts/gguf_mtp_c1c8_server_bench.py"


def load_harness(repo: pathlib.Path):
    """Import the C1-C8 harness as a module without running its argparse main()."""
    path = (repo / HARNESS).resolve()
    if not path.is_file():
        raise SystemExit(f"harness not found: {path}")
    spec = importlib.util.spec_from_file_location("ar_only_c1c8", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load harness: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_argv(model: str, output: str, width: int, max_tokens: int) -> list[str]:
    """The harness argv this control runs: one width, explicit MTP mode, no expected MTP widths."""
    return [
        HARNESS,
        "--model",
        model,
        "--output",
        output,
        "--widths",
        str(width),
        "--mtp-request-mode",
        "explicit",
        "--expected-mtp-widths",
        "none",
        "--max-tokens",
        str(max_tokens),
    ]


def install_ar_only_control(harness) -> None:
    """Force every arm to request no speculation, and prove the patch was needed.

    The control is only meaningful if the harness *would* have speculated: assert that the original
    `_request_mtp_value` says True for the MTP arm in explicit mode. If it already says False, the
    engine was never going to speculate and this shim would be silently measuring nothing.
    """
    original = getattr(harness, "_request_mtp_value", None)
    if not callable(original):
        raise SystemExit(
            "harness has no _request_mtp_value to patch; the harness changed and this control "
            "must be re-read, not trusted"
        )
    would_speculate = original(arm="mtp", request_mode="explicit")
    if not would_speculate:
        raise SystemExit(
            "control is meaningless: the harness itself returns no speculation for the MTP arm in "
            "explicit mode, so an AR-only run proves nothing about speculation"
        )

    # Accept positional too: the harness calls it with keywords today, and a keyword-only stub
    # would TypeError if that ever changes, which is a worse failure than measuring nothing.
    harness._request_mtp_value = lambda *_a, **_k: False
    print(
        f"[CONTROL arms={tuple(harness.ARMS)} _request_mtp_value patched (original explicit "
        f"MTP value={would_speculate})]",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--repo", default=".", type=pathlib.Path)
    args = parser.parse_args(argv)

    harness = load_harness(args.repo)
    install_ar_only_control(harness)
    sys.argv = build_argv(args.model, args.output, args.width, args.max_tokens)
    return int(harness.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
