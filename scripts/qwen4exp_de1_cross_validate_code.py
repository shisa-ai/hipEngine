#!/usr/bin/env python3
"""Cross-validate two generated `reconcile` implementations.

The DE-1 task-materiality question is whether the candidate route degrades task
behaviour versus strict. Both arms generated a `reconcile` plus their own
assert-based tests, so the strongest available check is cross-validation: run
each implementation against the UNION of both arms' test cases. A test that
passes for the implementation that wrote it but fails for the other is evidence
of divergent behaviour; the union removes the "my own tests are weaker" bias.

Usage:
    python3 scripts/qwen4exp_de1_cross_validate_code.py --strict PATH --candidate PATH
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import traceback
from pathlib import Path
from typing import Any, Callable


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_tests(mod) -> dict[str, Callable[[], Any]]:
    out = {}
    for name, obj in vars(mod).items():
        if name.startswith("test_") and callable(obj):
            try:
                sig = inspect.signature(obj)
            except (TypeError, ValueError):
                continue
            if len(sig.parameters) == 0:
                out[name] = obj
    return out


def run_test(fn: Callable[[], Any]) -> tuple[bool, str]:
    try:
        fn()
        return True, ""
    except AssertionError as exc:
        tb = traceback.format_exc().strip().splitlines()
        return False, f"AssertionError: {exc}" if str(exc) else (tb[-1] if tb else "AssertionError")
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()

    strict_mod = load_module(args.strict, "arm_strict")
    cand_mod = load_module(args.candidate, "arm_candidate")

    if not hasattr(strict_mod, "reconcile") or not hasattr(cand_mod, "reconcile"):
        raise SystemExit("both arms must define reconcile()")

    strict_tests = collect_tests(strict_mod)
    cand_tests = collect_tests(cand_mod)
    union = {**strict_tests, **cand_tests}

    print(f"strict tests:    {len(strict_tests)} {sorted(strict_tests)}")
    print(f"candidate tests: {len(cand_tests)} {sorted(cand_tests)}")
    print(f"union:           {len(union)}")
    print()

    impls = {"strict": strict_mod, "candidate": cand_mod}
    header = f'{"test":44s} {"owner":10s} {"strict":>8s} {"candidate":>10s}'
    print(header)
    print("-" * len(header))

    failures: list[tuple[str, str, str]] = []
    disagreements: list[str] = []
    for tname in sorted(union):
        owner = "strict" if tname in strict_tests else "candidate"
        # The test closes over its own module's globals, so run it once per
        # implementation by temporarily swapping the module-level `reconcile`.
        results = {}
        for impl_name, mod in impls.items():
            original = mod.reconcile
            # Bind the test's globals to the implementation under test.
            fn = union[tname]
            saved_globals = fn.__globals__.get("reconcile")
            fn.__globals__["reconcile"] = original
            try:
                ok, msg = run_test(fn)
            finally:
                if saved_globals is None:
                    fn.__globals__.pop("reconcile", None)
                else:
                    fn.__globals__["reconcile"] = saved_globals
            results[impl_name] = (ok, msg)
        s_ok = results["strict"][0]
        c_ok = results["candidate"][0]
        print(f'{tname:44s} {owner:10s} {str(s_ok):>8s} {str(c_ok):>10s}')
        if not s_ok:
            failures.append((tname, "strict", results["strict"][1]))
        if not c_ok:
            failures.append((tname, "candidate", results["candidate"][1]))
        if s_ok != c_ok:
            disagreements.append(tname)

    print()
    print(f"failures:      {len(failures)}")
    for tname, who, msg in failures:
        print(f"  {who:10s} {tname}: {msg}")
    print(f"disagreements: {len(disagreements)} {disagreements}")

    verdict = "pass" if not disagreements else "review"
    print(f"\nverdict: {verdict}")
    return 0 if verdict == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
