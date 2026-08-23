from __future__ import annotations

from types import SimpleNamespace

from scripts.gguf_mtp_verifier_rocprof import _bucket, _native_spec_args_error


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "native_spec_target_cycle": True,
        "mode": "block-verify",
        "block_rows": 3,
        "block_verify_mode": "bulk",
        "block_wmma_prefill": False,
        "return_logits": False,
        "sync_stage_timings": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_native_spec_profiler_accepts_all_registered_b1_b7_row_buckets() -> None:
    for rows in range(2, 9):
        assert _native_spec_args_error(_args(block_rows=rows)) is None

    assert "2..8" in str(_native_spec_args_error(_args(block_rows=1)))
    assert "2..8" in str(_native_spec_args_error(_args(block_rows=9)))


def test_native_spec_profiler_preserves_fail_closed_mode_contract() -> None:
    assert "bulk non-WMMA" in str(
        _native_spec_args_error(_args(block_verify_mode="native"))
    )
    assert "bulk non-WMMA" in str(
        _native_spec_args_error(_args(block_wmma_prefill=True))
    )
    assert "does not support" in str(
        _native_spec_args_error(_args(return_logits=True))
    )


def test_qmicro_dense_gate_up_has_a_distinct_profile_bucket() -> None:
    assert (
        _bucket(
            "q4_k_qmicro_t16_dense_dual_q8_1x2_rowtile8_dp4a_silu_gemv"
        )
        == "dense_q4_gate_up"
    )
    assert _bucket("q4_k_t16_dense_rowtile_gemv") == "dense_q4_projection"
