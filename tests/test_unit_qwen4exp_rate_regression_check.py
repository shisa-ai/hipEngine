from scripts.qwen4exp_rate_regression_check import compare_runs


def _run(rates, *, machine="m1", fixture="f", sha="a", deterministic=True):
    return {
        "status": "completed",
        "fixture_sha256": fixture,
        "host": {"machine_id": machine},
        "summary": {
            "shapes": {
                shape: {"prefill_tok_s_weighted": pp, "decode_tok_s_weighted": tg}
                for shape, (pp, tg) in rates.items()
            },
            "cases": {
                f"code-p{shape}": {
                    "deterministic": deterministic,
                    "output_token_ids_sha256": [sha],
                }
                for shape in rates
            },
        },
    }


BASE = {"1024": (900.0, 20.0), "16384": (1100.0, 20.0)}


def test_equal_and_faster_runs_pass():
    assert compare_runs(_run(BASE), _run(BASE))["passed"]
    faster = {"1024": (950.0, 20.5), "16384": (1200.0, 20.0)}
    result = compare_runs(_run(BASE), _run(faster))
    assert result["passed"]
    assert round(result["shapes"]["16384"]["prefill_tok_s_weighted"]["delta"], 4) == 0.0909


def test_a_drop_beyond_tolerance_at_one_length_fails():
    slow_long = {"1024": (900.0, 20.0), "16384": (1050.0, 20.0)}
    result = compare_runs(_run(BASE), _run(slow_long))
    assert not result["passed"]
    assert any("p16384 prefill" in failure for failure in result["failures"])
    assert compare_runs(_run(BASE), _run(slow_long), prefill_tolerance=0.05)["passed"]
    slow_decode = {"1024": (900.0, 19.0), "16384": (1100.0, 20.0)}
    assert not compare_runs(_run(BASE), _run(slow_decode))["passed"]


def test_output_change_is_reported_and_binds_only_on_request():
    changed = _run(BASE, sha="b")
    result = compare_runs(_run(BASE), changed)
    assert result["passed"]
    assert result["changed_output_cases"] == ["code-p1024", "code-p16384"]
    assert not compare_runs(_run(BASE), changed, require_identical_output=True)["passed"]


def test_different_host_fixture_shapes_or_nondeterminism_fail():
    assert not compare_runs(_run(BASE), _run(BASE, machine="m2"))["passed"]
    assert not compare_runs(_run(BASE), _run(BASE, fixture="g"))["passed"]
    assert not compare_runs(_run(BASE), _run({"1024": (900.0, 20.0)}))["passed"]
    assert not compare_runs(_run(BASE), _run(BASE, deterministic=False))["passed"]
