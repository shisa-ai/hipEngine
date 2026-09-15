"""Unit tests for the native/Python staged-exchange A/B driver."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import stat
import sys

import pytest


def _load():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tp_staged_exchange_native_ab.py"
    spec = importlib.util.spec_from_file_location("tp_staged_exchange_native_ab", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def _fake_runner(tmp_path: pathlib.Path, *, payload: dict, exit_code: int = 0) -> pathlib.Path:
    script = tmp_path / "fake_runner.sh"
    script.write_text(
        "#!/bin/sh\n" + f"cat <<'EOF'\n{json.dumps(payload)}\nEOF\n" + f"exit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_marginal_is_a_ladder_slope_not_a_single_point(mod) -> None:
    """The marginal must divide by the depth difference, like the chain report."""

    report = mod._marginal([(1, 100.0), (128, 100.0 + 20.8 * 127)])
    assert report["from_depth"] == 1
    assert report["to_depth"] == 128
    assert report["overall_us_per_step"] == pytest.approx(20.8)


def test_marginal_requires_two_depths(mod) -> None:
    assert mod._marginal([(4, 100.0)])["overall_us_per_step"] is None
    assert mod._marginal([])["overall_us_per_step"] is None


def test_run_native_parses_the_runner_report(tmp_path: pathlib.Path, mod) -> None:
    payload = {
        "depth": 16,
        "total_median_us": 390.5,
        "per_step_us": 24.47,
        "verification": {"exact": True},
    }
    exe = _fake_runner(tmp_path, payload=payload)
    result = mod._run_native(exe, depth=16, count=5120, iterations=2, warmup=1)
    assert result["total_median_us"] == pytest.approx(390.5)
    assert result["verification"]["exact"] is True


def test_run_native_reports_a_failed_verification_as_an_error(tmp_path: pathlib.Path, mod) -> None:
    """The runner exits non-zero when the chain's value check fails."""

    exe = _fake_runner(tmp_path, payload={"verification": {"exact": False}}, exit_code=1)
    result = mod._run_native(exe, depth=16, count=5120, iterations=1, warmup=0)
    assert "error" in result
    assert "exit 1" in result["error"]


def test_build_refuses_an_uncached_executable_when_required(tmp_path: pathlib.Path, mod) -> None:
    """A profiled run must not compile inside the profiler."""

    source = tmp_path / "source.hip"
    source.write_text("// placeholder\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="require_cached"):
        mod._build(source, tmp_path / "build", "gfx1100", require_cached=True)


def _native_source() -> str:
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "micro"
        / "runners"
        / "hip_staged_exchange.hip"
    ).read_text(encoding="utf-8")


def test_the_native_source_preserves_the_protocol() -> None:
    """The native arm must keep the batched protocol's completion boundaries.

    Both device-to-host copies are submitted before either stream is waited on,
    the return copies are not awaited per step, and there is exactly one drain
    per chain. These are the properties the A/B compares, so a source edit that
    dropped one would make the comparison meaningless rather than merely slower.
    """

    source = _native_source()
    body = source[source.index("  void step(int index, Recurrence") : source.index("  void drain()")]
    submits = body.index("hipMemcpyAsync")
    waits = body.index("hipStreamSynchronize")
    assert submits < waits, "the D2H copies must be submitted before the first wait"
    assert body.count("hipStreamSynchronize") == 2, "one wait per rank, no return wait"


def test_verification_and_timing_share_one_step_implementation() -> None:
    """A second copy of the step would verify code the timing run never uses."""

    source = _native_source()
    assert source.count("void step(") == 1, "exactly one step implementation"
    assert "step_bounded" not in source, "the separate verification step must be gone"
    # Every chain, timed or verified, goes through run_chain -> step.
    run_chain = source[source.index("  void run_chain(") : source.index("  std::vector<float> read_rank(")]
    assert "step(index, recurrence, phase)" in run_chain
    verify = source[source.index("  Check verify(") : source.index("};\n\nArgs parse_args")]
    assert "run_chain(depth, recurrence, nullptr)" in verify, "verification must run the chain"


def test_verification_checks_both_ranks_and_rejects_nonfinite() -> None:
    """Element zero of rank zero is not a correctness gate."""

    source = _native_source()
    verify = source[source.index("  Check verify(") : source.index("};\n\nArgs parse_args")]
    assert "for (uint32_t rank = 0; rank < world; ++rank)" in verify
    assert "read_rank(rank, depth)" in verify
    assert "std::isfinite(value)" in verify, "NaN must be rejected explicitly"
    assert "nonfinite" in verify
    assert "elements_per_rank" in source


def _driver_source() -> str:
    return (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "tp_staged_exchange_native_ab.py"
    ).read_text(encoding="utf-8")


def test_the_timed_recurrence_is_checked_where_it_is_representable() -> None:
    """The timed path's own arithmetic must be verified, not only the bounded one."""

    source = _native_source()
    assert "Recurrence::kSum" in source
    # Saturation is allowed only when the closed form itself overflows fp32.
    assert "saturation_expected" in source
    assert "closed form overflows fp32" in source


def test_the_driver_keeps_every_repetition_verdict() -> None:
    """Timing from a failed repetition must not ride on a later passing verdict."""

    source = _driver_source()
    assert "python_ladders: list[dict[str, Any]] = []" in source
    assert "python_ladders.append(result)" in source
    # The verdict is an aggregate over repetitions, not the last one alone.
    assert "python_verdicts" in source
    assert "python_repetitions_passed" in source
    assert '"python_repetitions_passed"' in source
    assert '"native_repetitions_passed"' in source
    assert "len(python_verdicts) == args.reps" in source


def test_a_missing_vector_check_is_a_failure_not_an_unknown() -> None:
    """Every repetition must verify the whole vector on both ranks."""

    source = _driver_source()
    assert "full_vector_matches" in source
    assert "ranks_agree" in source
    assert "rank_seeds_differ" in source
    assert "A missing vector check is a failure, not an unknown" in source


def test_the_python_arm_closes_its_transport() -> None:
    """A live RCCL communicator holds device handles across repetitions."""

    source = _driver_source()
    assert "transport.close()" in source
    # Buffers first, then the transport.
    assert source.index("free(buffer)") < source.index("transport.close()")


def test_the_provisional_flag_is_derived_from_every_check() -> None:
    source = _driver_source()
    assert "matched = all(match.values())" in source
    assert '"provisional": not matched' in source
