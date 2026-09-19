from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.backends import detect_hip_target_arches
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFResidentSession,
    _chunk_ranges,
    _gguf_aotriton_prefill_mode,
)

MODEL = Path("/models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf")
_MODEL_REQUIRED = pytest.mark.skipif(
    not MODEL.exists(),
    reason=f"local GGUF fixture not found: {MODEL}",
)


def test_gguf_chunk_ranges_merge_tiny_tail() -> None:
    assert _chunk_ranges(4097, 4096, min_chunk_size=4) == ((0, 4097),)
    assert _chunk_ranges(8193, 4096, min_chunk_size=4) == ((0, 4096), (4096, 8193))


def test_gguf_aotriton_prefill_mode_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_AOTRITON_PREFILL", raising=False)
    assert _gguf_aotriton_prefill_mode(0, 4096, 4096) == "v3"
    assert _gguf_aotriton_prefill_mode(4096, 4096, 8192) == "v3"

    monkeypatch.setenv("HIPENGINE_GGUF_AOTRITON_PREFILL", "auto")
    assert _gguf_aotriton_prefill_mode(0, 4096, 4096) == "v2"
    assert _gguf_aotriton_prefill_mode(4096, 4096, 8192) == "v3"

    monkeypatch.setenv("HIPENGINE_GGUF_AOTRITON_PREFILL", "v2")
    assert _gguf_aotriton_prefill_mode(0, 4096, 4096) == "v2"
    with pytest.raises(ValueError, match="only valid for full-context prefill"):
        _gguf_aotriton_prefill_mode(4096, 4096, 8192)


@_MODEL_REQUIRED
def test_qwen35_gguf_chunked_prefill_matches_unchunked() -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    # 8 tokens prompt to test chunking into 2 chunks of size 4
    prompt_ids = [760, 4087, 369, 220, 760, 4087, 369, 220]

    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=16, prefill_chunk_size=999999) as unchunked:
        unchunked_res = unchunked.prefill(prompt_ids, use_bulk=True)

    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=16, prefill_chunk_size=4) as chunked:
        chunked_res = chunked.prefill(prompt_ids, use_bulk=True)

    assert chunked_res.token_id == unchunked_res.token_id
    assert chunked_res.logits.shape == unchunked_res.logits.shape == (1, 248320)
    assert np.all(np.isfinite(chunked_res.logits))
    assert _kl_divergence(unchunked_res.logits.reshape(-1), chunked_res.logits.reshape(-1)) <= 0.1


@_MODEL_REQUIRED
def test_qwen35_gguf_packed_prefill_returns_per_slot_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", "1")
    prompts = ([760, 4087, 369, 220], [760, 4087, 369, 221])

    with Qwen35GGUFResidentSession(MODEL, max_sequence_length=16) as owner:
        assert owner.runner is not None
        with Qwen35GGUFResidentSession(
            MODEL,
            shared_runner=owner.runner,
            max_sequence_length=16,
        ) as peer:
            packed = owner.prefill_batch_native(
                prompts,
                sessions=(owner, peer),
                return_logits=True,
            )
            packed_logits = [result.logits.copy() for result in packed if result is not None]
            packed_tokens = [int(result.token_id) for result in packed if result is not None]
            plan = dict(owner.last_packed_prefill_plan)

            owner.reset()
            peer.reset()
            scalar = [
                owner.prefill(prompts[0], return_logits=True),
                peer.prefill(prompts[1], return_logits=True),
            ]

    assert len(packed_logits) == len(packed_tokens) == len(scalar) == 2
    assert all(logits.shape == (1, 248320) for logits in packed_logits)
    assert all(np.all(np.isfinite(logits)) for logits in packed_logits)
    assert packed_tokens == [int(result.token_id) for result in scalar]
    assert all(
        _kl_divergence(result.logits.reshape(-1), logits.reshape(-1)) <= 0.05
        for result, logits in zip(scalar, packed_logits, strict=True)
    )
    assert plan["host_logits_d2h"] is True
    assert plan["host_logits_d2h_bytes"] == 2 * 248320 * np.dtype(np.float32).itemsize


@_MODEL_REQUIRED
@pytest.mark.parametrize(
    ("boundary", "suffix_tokens"),
    ((512, 200), (1024, 200)),
    ids=("paged-sub1024", "paged-crosses-1024-gate"),
)
def test_qwen35_gguf_packed_suffix_extend_matches_serial_oracle(
    monkeypatch: pytest.MonkeyPatch,
    boundary: int,
    suffix_tokens: int,
) -> None:
    """Batched suffix (\"extend\") prefill from a mid-sequence state is exact.

    A prefix-cache hit restores a session to a 256-aligned boundary and then
    needs the unmatched suffix prefilled *batched*: attention over the imported
    prefix KV, GDN slot state seeded from the restored state, positions offset
    by the boundary. The serial ``session.step()`` loop - today's fallback
    route at ~34 ms/token - is the oracle. The 1024-token case crosses
    ``PACKED_AR_PREFILL_CONTEXT_LIMIT``, which currently refuses the paged
    route with NotImplementedError.
    """
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", raising=False)
    total = boundary + suffix_tokens
    rng = np.random.RandomState(20260918)
    prompt = rng.randint(16, 248000, size=total).tolist()

    with Qwen35GGUFResidentSession(
        MODEL,
        max_sequence_length=total + 64,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as oracle:
        assert oracle.runner is not None
        with Qwen35GGUFResidentSession(
            MODEL,
            shared_runner=oracle.runner,
            max_sequence_length=total + 64,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as subject:
            oracle.prefill(prompt[:boundary], return_logits=False)
            subject.prefill(prompt[:boundary], return_logits=False)
            assert int(oracle.position) == int(subject.position) == boundary

            serial_result = None
            for index, token_id in enumerate(prompt[boundary:]):
                serial_result = oracle.step(
                    int(token_id), return_logits=index == suffix_tokens - 1
                )
            assert serial_result is not None
            assert int(oracle.position) == total

            extended = subject.prefill_batch_native(
                [prompt[boundary:]],
                sessions=[subject],
                full_prompt_lengths=[total],
                return_logits=True,
            )
            assert len(extended) == 1 and extended[0] is not None
            packed_result = extended[0]
            assert int(subject.position) == total

            assert int(packed_result.token_id) == int(serial_result.token_id)
            assert (
                _kl_divergence(
                    serial_result.logits.reshape(-1),
                    packed_result.logits.reshape(-1),
                )
                <= 0.05
            )
            # Decode continuation: the state left behind by the suffix prefill
            # (GDN recurrent state and appended KV) must drive the same token
            # stream, not just the same last-row logits.
            oracle_tokens = _greedy_continuation(oracle, int(serial_result.token_id), 8)
            packed_tokens = _greedy_continuation(subject, int(packed_result.token_id), 8)
            assert packed_tokens == oracle_tokens


def _greedy_continuation(session, first_token_id: int, steps: int) -> list[int]:
    tokens: list[int] = []
    token = int(first_token_id)
    for _ in range(steps):
        result = session.step(token, return_logits=False)
        token = int(result.token_id)
        tokens.append(token)
    return tokens


@_MODEL_REQUIRED
@pytest.mark.parametrize("verify_capture", ("0", "1"), ids=("no-gdn-capture", "gdn-capture"))
@pytest.mark.parametrize(
    ("boundary", "suffix_tokens"),
    ((512, 200), (1024, 200), (2048, 200), (4096, 200)),
    ids=("paged-712", "paged-1224", "paged-2248", "paged-4296"),
)
def test_qwen35_gguf_packed_suffix_extend_shared_prefix_matches_serial_oracle(
    monkeypatch: pytest.MonkeyPatch,
    verify_capture: str,
    boundary: int,
    suffix_tokens: int,
) -> None:
    """Batched suffix extend over a *shared* prefix matches the serial oracle.

    This is the true prefix-cache hit shape: the subject session binds a pool
    allocation whose prefix pages are shared with the source request and whose
    suffix pages sit behind a spacer, so the block table is non-contiguous and
    the slot-local (contiguous-slab AOTriton) route cannot represent it. The
    packed paged route must import the prefix KV, seed GDN state from
    ``clone_prefix_state_from``, and prefill the suffix in one batched call.
    The ``gdn-capture`` arm pins the generation layer's shipped configuration
    (``HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN=1`` wraps every packed prefill
    call there).
    """
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", verify_capture)
    total = boundary + suffix_tokens
    rng = np.random.RandomState(20260918)
    prompt = rng.randint(16, 248000, size=total).tolist()
    prefix_pages = boundary // 256
    request_pages = (total + 8 + 255) // 256
    pool_pages = 2 * request_pages + 2

    with Qwen35GGUFResidentSession(
        MODEL,
        max_sequence_length=total + 64,
        defer_kv_allocation=True,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as oracle:
        assert oracle.runner is not None
        with Qwen35GGUFResidentSession(
            MODEL,
            shared_runner=oracle.runner,
            max_sequence_length=total + 64,
            defer_kv_allocation=True,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as subject:
            pool = oracle.create_device_kv_pool(
                initial_pages=pool_pages,
                low_water_pages=pool_pages,
                high_water_pages=pool_pages,
                chunk_pages=pool_pages,
                idle_grace_seconds=0.0,
            )
            try:
                oracle_allocation = pool.allocate(
                    9001, request_pages, now_seconds=1.0
                )
                oracle.bind_device_kv_allocation(pool, oracle_allocation)
                # A spacer allocation forces the subject's suffix pages away
                # from the shared prefix, so the subject's block table has a
                # gap like any pool-placed shared admission.
                spacer_allocation = pool.allocate(9002, 1, now_seconds=1.0)
                subject_allocation = pool.admit_with_shared_prefix(
                    9003,
                    oracle_allocation.block_ids[:prefix_pages],
                    suffix_pages=request_pages - prefix_pages,
                    now_seconds=1.0,
                )
                assert tuple(subject_allocation.block_ids) != tuple(
                    range(
                        subject_allocation.block_ids[0],
                        subject_allocation.block_ids[0]
                        + len(subject_allocation.block_ids),
                    )
                ), "test requires a non-contiguous shared-prefix allocation"
                subject.bind_device_kv_allocation(pool, subject_allocation)

                oracle.prefill(prompt[:boundary], return_logits=False)
                assert int(oracle.position) == boundary
                cloned_bytes = subject.clone_prefix_state_from(
                    oracle, position=boundary
                )
                assert cloned_bytes > 0
                assert int(subject.position) == boundary

                serial_result = None
                for index, token_id in enumerate(prompt[boundary:]):
                    serial_result = oracle.step(
                        int(token_id), return_logits=index == suffix_tokens - 1
                    )
                assert serial_result is not None
                assert int(oracle.position) == total

                extended = subject.prefill_batch_native(
                    [prompt[boundary:]],
                    sessions=[subject],
                    full_prompt_lengths=[total],
                    return_logits=True,
                )
                assert len(extended) == 1 and extended[0] is not None
                packed_result = extended[0]
                assert int(subject.position) == total
                plan = dict(getattr(subject, "last_packed_prefill_plan", {}))
                assert plan.get("device_kv_nonidentity_scatter") is True, (
                    "the shared-prefix suffix must run on the paged route"
                )

                assert int(packed_result.token_id) == int(serial_result.token_id)
                assert (
                    _kl_divergence(
                        serial_result.logits.reshape(-1),
                        packed_result.logits.reshape(-1),
                    )
                    <= 0.05
                )
                # Decode continuation: the state left behind by the suffix
                # prefill (GDN recurrent state and appended KV) must drive the
                # same token stream, not just the same last-row logits.
                oracle_tokens = _greedy_continuation(
                    oracle, int(serial_result.token_id), 8
                )
                packed_tokens = _greedy_continuation(
                    subject, int(packed_result.token_id), 8
                )
                assert packed_tokens == oracle_tokens
            finally:
                subject.unbind_device_kv_allocation()
                oracle.unbind_device_kv_allocation()
                for request_id in (9003, 9002, 9001):
                    try:
                        pool.release(request_id, now_seconds=2.0)
                    except KeyError:
                        pass
                pool.close()


def _kl_divergence(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> float:
    ref = reference_logits.astype(np.float64, copy=False)
    cand = candidate_logits.astype(np.float64, copy=False)
    ref_exp = np.exp(ref - float(np.max(ref)))
    cand_exp = np.exp(cand - float(np.max(cand)))
    ref_prob = ref_exp / float(np.sum(ref_exp))
    cand_prob = cand_exp / float(np.sum(cand_exp))
    return float(np.sum(ref_prob * (np.log(ref_prob + 1.0e-30) - np.log(cand_prob + 1.0e-30))))


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


_PRODUCTION_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_S.gguf")


@pytest.mark.skipif(
    not _PRODUCTION_MODEL.exists(),
    reason=f"local GGUF fixture not found: {_PRODUCTION_MODEL}",
)
@pytest.mark.skipif(
    "gfx1151" not in detect_hip_target_arches(),
    reason="Qwen3.8 FP16-state packed AR gate requires physical gfx1151",
)
@pytest.mark.parametrize(
    ("state_env", "expected_fp16_state"),
    ((None, True), ("0", False), ("1", True)),
)
def test_qwen35_gguf_packed_ar_prefill_decode_runs_without_verify_capture(
    monkeypatch: pytest.MonkeyPatch,
    state_env: str | None,
    expected_fp16_state: bool,
) -> None:
    """Packed AR prefill+decode works in the production route (no verify-capture).

    Regression for the removed fail-closed guards that raised
    ``NotImplementedError`` when ``HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN``
    was unset.  The packed AR path is self-contained (segmented compact-peer
    per-slot state; c1-exact per-slot decode), so the production route must be
    supported.  Asserts the packed prefill+decode token streams match scalar
    prefill+decode on the same session pair.
    """
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    monkeypatch.delenv("HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN", raising=False)
    monkeypatch.delenv("HIPENGINE_GGUF_GDN_PREFILL_MODE", raising=False)
    if state_env is None:
        monkeypatch.delenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_GGUF_FP16_RECURRENT_STATE", state_env)
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFResidentSession,
        _gguf_verify_capture_prefill_gdn_enabled,
    )
    assert not _gguf_verify_capture_prefill_gdn_enabled()

    prompt_a = [760, 4087, 369, 220, 760, 4087, 369, 220]
    prompt_b = [760, 4087, 369, 221, 760, 4087, 369, 221]

    with Qwen35GGUFResidentSession(
        _PRODUCTION_MODEL,
        backend="hip_gfx1151",
        max_sequence_length=64,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as owner:
        assert owner.runner is not None
        assert owner.runner.fp16_recurrent_state is expected_fp16_state
        with Qwen35GGUFResidentSession(
            _PRODUCTION_MODEL,
            shared_runner=owner.runner,
            backend="hip_gfx1151",
            max_sequence_length=64,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as peer:
            packed = owner.prefill_batch_native(
                (prompt_a, prompt_b),
                sessions=(owner, peer),
                return_logits=True,
            )
            packed_tokens = [int(res.token_id) for res in packed if res is not None]
            packed_logits = [
                np.ascontiguousarray(res.logits, dtype=np.float32)
                for res in packed
                if res is not None
            ]
            dec = owner.step_batch_native(
                tuple(packed_tokens),
                sessions=(owner, peer),
                return_logits=True,
            )
            dec_tokens = [int(res.token_id) for res in dec]

            owner.reset()
            peer.reset()
            scalar_a = owner.prefill(prompt_a, return_logits=True)
            scalar_b = peer.prefill(prompt_b, return_logits=True)
            scalar_tokens = [int(scalar_a.token_id), int(scalar_b.token_id)]
            scalar_logits = [
                np.ascontiguousarray(scalar_a.logits, dtype=np.float32),
                np.ascontiguousarray(scalar_b.logits, dtype=np.float32),
            ]
            dec_a = owner.step(scalar_tokens[0], return_logits=True)
            dec_b = peer.step(scalar_tokens[1], return_logits=True)
            scalar_dec = [int(dec_a.token_id), int(dec_b.token_id)]

    assert packed_tokens == scalar_tokens
    assert all(
        _kl_divergence(p.reshape(-1), s.reshape(-1)) <= 0.05
        for p, s in zip(packed_logits, scalar_logits, strict=True)
    )
    assert dec_tokens == scalar_dec


@_MODEL_REQUIRED
@pytest.mark.parametrize(
    ("boundary", "suffix"),
    (
        (256, 2),
        (256, 200),
        (1024, 50),
        (1024, 200),
        (1024, 1024),
        (1536, 300),
    ),
)
def test_batched_suffix_extend_is_bit_exact_against_recompute(
    boundary: int, suffix: int
) -> None:
    """A prefix-cache HIT must return exactly what a MISS would have returned.

    This is the contract prefix caching actually owes its users: enabling the
    cache must not change the answer. It is deliberately NOT a comparison
    against a serial ``session.step`` reference - the engine's own default
    prefill is batched, and batched-versus-serial GDN arithmetic differs by
    ~5e-3 relative L2 whether or not any prefix is reused (measured at
    ``start_position == 0`` too). Holding a reused suffix to serial
    bit-equality would therefore impose a standard the engine does not meet
    for any prompt it prefills.

    Two shapes are excluded because they diverge on clean HEAD as well - they
    are pre-existing split-prefill properties, not reuse defects, and they are
    pinned separately below:
      * a single-token suffix (``rows == 1`` selects decode-shaped kernels),
      * a total context above 2048.
    """
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    total = boundary + suffix
    assert total <= 2048, "shapes above 2048 total are covered by the boundary pin"
    rng = np.random.RandomState(20260919)
    prompt = rng.randint(16, 140000, size=total).tolist()

    with Qwen35GGUFResidentSession(
        MODEL,
        max_sequence_length=total + 64,
        use_wmma_prefill=True,
        use_gemv_decode=True,
    ) as owner:
        assert owner.runner is not None

        def arm(split: bool) -> tuple[int, list[np.ndarray]]:
            with Qwen35GGUFResidentSession(
                MODEL,
                shared_runner=owner.runner,
                max_sequence_length=total + 64,
                use_wmma_prefill=True,
                use_gemv_decode=True,
            ) as session:
                if split:
                    session.prefill_batch_native(
                        [prompt[:boundary]],
                        sessions=[session],
                        full_prompt_lengths=[boundary],
                        return_logits=False,
                    )
                    result = session.prefill_batch_native(
                        [prompt[boundary:]],
                        sessions=[session],
                        full_prompt_lengths=[total],
                        return_logits=True,
                    )
                else:
                    result = session.prefill_batch_native(
                        [prompt],
                        sessions=[session],
                        full_prompt_lengths=[total],
                        return_logits=True,
                    )
                assert len(result) == 1 and result[0] is not None
                first = int(result[0].token_id)
                logits = []
                token = first
                for _ in range(6):
                    step = session.step(token, return_logits=True)
                    logits.append(np.asarray(step.logits).reshape(-1).copy())
                    token = int(step.token_id)
                return first, logits

        miss_first, miss_logits = arm(split=False)
        hit_first, hit_logits = arm(split=True)

    assert hit_first == miss_first
    for index, (miss_row, hit_row) in enumerate(
        zip(miss_logits, hit_logits, strict=True)
    ):
        assert np.array_equal(miss_row, hit_row), (
            f"reuse diverged from recompute at continuation row {index}"
        )


@_MODEL_REQUIRED
def test_split_prefill_divergence_boundaries_are_unchanged() -> None:
    """Pin the two shapes where split prefill differs from a single call.

    Both reproduce identically on clean HEAD (d01ec1b51), so they are
    pre-existing properties of chunked prefill rather than prefix-cache
    defects. They are pinned so that a future change to either boundary is a
    deliberate, visible decision instead of a silent drift:

      * ``suffix == 1``: a one-row prefill selects ``rows == 1`` decode-shaped
        kernels, whose arithmetic differs from the bulk prefill route. The
        prefix cache no longer routes into this shape - a one-token reused
        suffix takes the serial step instead - so this pin covers the raw
        ``prefill_batch_native`` API, which still exposes it.
      * ``total > 2048``: splitting anywhere diverges from one call; measured
        against a serial reference the single call is the closer of the two.

    If a fix lands for either, this test should fail and be replaced by
    coverage in ``test_batched_suffix_extend_is_bit_exact_against_recompute``.
    """
    if not _hip_available():
        pytest.skip("HIP runtime is not available")
    rng = np.random.RandomState(20260919)

    def diverges(boundary: int, suffix: int) -> bool:
        total = boundary + suffix
        prompt = rng.randint(16, 140000, size=total).tolist()
        with Qwen35GGUFResidentSession(
            MODEL,
            max_sequence_length=total + 64,
            use_wmma_prefill=True,
            use_gemv_decode=True,
        ) as owner:
            assert owner.runner is not None

            def arm(split: bool) -> np.ndarray:
                with Qwen35GGUFResidentSession(
                    MODEL,
                    shared_runner=owner.runner,
                    max_sequence_length=total + 64,
                    use_wmma_prefill=True,
                    use_gemv_decode=True,
                ) as session:
                    if split:
                        session.prefill_batch_native(
                            [prompt[:boundary]],
                            sessions=[session],
                            full_prompt_lengths=[boundary],
                            return_logits=False,
                        )
                        result = session.prefill_batch_native(
                            [prompt[boundary:]],
                            sessions=[session],
                            full_prompt_lengths=[total],
                            return_logits=True,
                        )
                    else:
                        result = session.prefill_batch_native(
                            [prompt],
                            sessions=[session],
                            full_prompt_lengths=[total],
                            return_logits=True,
                        )
                    return np.asarray(result[0].logits).reshape(-1).copy()

            return not np.array_equal(arm(split=False), arm(split=True))

    assert diverges(256, 1), "single-token-suffix divergence disappeared"
    assert diverges(1024, 1028), "above-2048 split divergence disappeared"
    assert not diverges(1024, 1024), "at-2048 split must still be bit-exact"
