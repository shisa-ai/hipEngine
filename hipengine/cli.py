"""Top-level hipEngine command-line interface."""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import sys


_BENCHMARKS = {
    "paro": ("scripts.qwen35_paro_bench", "Qwen/PARO resident single-request benchmark"),
    "gguf": ("scripts.qwen35_gguf_bench", "Qwen GGUF resident benchmark"),
    "sweep": ("scripts.qwen35_readme_sweep", "README-style repeated workload sweep"),
    "batch-serial": ("scripts.qwen35_batch_serial_bench", "Diagnostic c>N scheduler serial bridge benchmark"),
    "c-sweep": ("scripts.qwen35_batch_c_sweep", "Qwen/PARO c=1/2/4/8 concurrency diagnostic sweep"),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the ``hipengine`` console command."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0
    if args[0] in {"-V", "--version", "version"}:
        print(_version())
        return 0

    command, rest = args[0], args[1:]
    if command == "serve":
        return _serve(rest)
    if command == "bench":
        return _bench(rest)

    print(f"hipengine: unknown command {command!r}\n", file=sys.stderr)
    _print_help(file=sys.stderr)
    return 2


def _serve(argv: Sequence[str]) -> int:
    from hipengine.server.__main__ import main as server_main

    old_argv = sys.argv
    sys.argv = ["hipengine serve", *argv]
    try:
        return int(server_main(argv) or 0)
    finally:
        sys.argv = old_argv


def _bench(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_bench_help()
        return 0
    if args[0] in {"list", "ls"}:
        _print_benchmarks()
        return 0

    name, rest = args[0], args[1:]
    if name not in _BENCHMARKS:
        print(f"hipengine bench: unknown benchmark {name!r}\n", file=sys.stderr)
        _print_bench_help(file=sys.stderr)
        return 2
    module_name, _description = _BENCHMARKS[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == "scripts":
            print(
                "hipengine bench could not find its packaged benchmark helpers. "
                "Reinstall hipengine or run from the repository checkout.",
                file=sys.stderr,
            )
            return 1
        raise
    entry = getattr(module, "main", None)
    if not callable(entry):
        print(f"hipengine bench: {module_name} does not expose main()", file=sys.stderr)
        return 1

    old_argv = sys.argv
    sys.argv = [f"hipengine bench {name}", *rest]
    try:
        result = entry()
    finally:
        sys.argv = old_argv
    return int(result or 0)


def _print_help(*, file=None) -> None:
    file = sys.stdout if file is None else file
    print(
        "usage: hipengine <command> [args]\n\n"
        "Commands:\n"
        "  serve        Run the OpenAI-compatible server\n"
        "  bench        Run or list benchmark helpers\n"
        "  version      Print the installed hipEngine version\n\n"
        "Examples:\n"
        "  hipengine serve --model shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed --quant w4_paro\n"
        "  hipengine bench list\n\n"
        "Use `hipengine <command> --help` for command-specific help.",
        file=file,
    )


def _print_bench_help(*, file=None) -> None:
    file = sys.stdout if file is None else file
    print("usage: hipengine bench <benchmark> [args]\n", file=file)
    _print_benchmarks(file=file)
    print("\nUse `hipengine bench <benchmark> --help` for benchmark-specific options.", file=file)


def _print_benchmarks(*, file=None) -> None:
    file = sys.stdout if file is None else file
    print("Benchmarks:", file=file)
    width = max(len(name) for name in _BENCHMARKS)
    for name, (_module, description) in sorted(_BENCHMARKS.items()):
        print(f"  {name:<{width}}  {description}", file=file)


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("hipengine")
    except Exception:  # pragma: no cover - editable fallback without metadata
        return "0+unknown"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
