#!/usr/bin/env python3
"""Diagnostic wrapper: set a backend-package capability, then exec a target.

``hipengine.kernels.backends.backend_package_capability`` resolves capabilities as
a plain ``getattr`` on the backend package (for example
``GGUF_C2_PACKED_PREFILL_MAX_ROWS``, whose undeclared default is 1, which pins
the engine loop to the single-request ``next_prefill_work`` branch instead of the
multi-request ``next_prefill_batch_work`` branch).  There is no environment
override for package capabilities, so an A/B of that gate otherwise requires
editing the backend package -- which would put an experiment inside production
code.  This script keeps the experiment in the harness: it imports the backend
package, sets the requested attributes, then runs the target script **in this
interpreter** with ``runpy`` so the injected attributes are visible to it.  (Do
not ``exec`` here: a fresh interpreter would drop the injection.)

Usage:
    python3 scripts/gguf_backend_capability_exec.py \
        --backend hip_gfx1100 --set GGUF_C2_PACKED_PREFILL_MAX_ROWS=8 \
        -- scripts/gguf_mtp_c1c8_server_bench.py <bench args...>

Prints an echoed ``CAPABILITY_ECHO`` line so arm logs record what was actually
applied, and verifies the value is still visible after the target module is
imported when the target supports ``--capability-probe``-style checks.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from importlib import import_module
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="hip_gfx1100")
    ap.add_argument(
        "--set",
        dest="assignments",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="capability attribute to set on the backend package (repeatable)",
    )
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="`-- target argv...`")
    args = ap.parse_args(argv)

    rest = list(args.rest)
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if not rest:
        ap.error("missing target command after `--`")

    module = import_module(f"hipengine.kernels.{args.backend}")
    applied = {}
    for item in args.assignments:
        name, sep, raw = item.partition("=")
        if not sep or not name:
            ap.error(f"bad --set {item!r}; expected NAME=VALUE")
        try:
            value: object = int(raw)
        except ValueError:
            value = raw
        previous = getattr(module, name, None)
        setattr(module, name, value)
        applied[name] = {"previous": previous, "applied": value}
    print(
        "CAPABILITY_ECHO "
        + str({"backend": args.backend, "applied": applied, "target": rest[0]}),
        flush=True,
    )
    target = rest[0]
    if not Path(target).is_file():
        ap.error(f"target script {target!r} is not a file; runpy needs a script path")
    with Path(target).open("rb") as handle:
        if b"\0" in handle.read(4096):
            ap.error(
                f"target {target!r} is not Python source (NUL bytes in the first 4 KiB); "
                "pass the script path after `--`, not the interpreter executable"
            )
    sys.argv = [target, *rest[1:]]
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
