from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation import (
    EngineLoopConfig,
    GeneratedToken,
    ResidentEngineLoop,
)
from hipengine.kvcache import (
    CompactExtentPool,
    DMSAdmissionManager,
    DMSCodecQualification,
    DMSCompactResidentRunnerAdapter,
    DMSRetrofitConfig,
    build_dms_live_mask,
    compact_attention_reference,
    create_dms_bf16_backend,
    create_dms_int8_backend,
    decode_dms_payload,
    encode_dms_payload,
    extract_dms_eviction_decisions,
    load_dms_retrofit_config,
)


def _retrofit(*, artifact: str = "artifact:dms-fixture") -> DMSRetrofitConfig:
    return DMSRetrofitConfig(
        artifact_fingerprint=artifact,
        model_family="fixture-qwen",
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=4,
        window_size=2,
        target_compression_ratio=4,
        alpha_scale=100.0,
        alpha_offset=5.0,
        borrowed_query_channel=-1,
        corrected_mask=True,
        trained_checkpoint=True,
        evidence_source="fixture://dms",
        source_path="/fixtures/dms_metadata.json",
    )


def _request(request_id: int, prompt_tokens: int = 8, max_new_tokens: int = 4):
    return SimpleNamespace(
        request_id=int(request_id),
        prompt_tokens=tuple(range(int(prompt_tokens))),
        max_new_tokens=int(max_new_tokens),
    )


def _backend(*, codec: str = "bf16", slots_per_layer: int = 64):
    kwargs = dict(
        retrofit=_retrofit(),
        slots_per_layer=slots_per_layer,
        max_request_rows=32,
        max_pack_rows=64,
    )
    if codec == "bf16":
        return create_dms_bf16_backend(**kwargs)
    qualification = DMSCodecQualification(
        codec="int8_per_token_head",
        artifact_fingerprint="artifact:dms-fixture",
        kl_divergence=0.01,
        top1_agreement=0.95,
        no_dense_shadow=True,
        evidence_source="fixture://int8-quality",
    )
    return create_dms_int8_backend(
        codec_qualification=qualification,
        **kwargs,
    )


def _prompt_arrays(tokens: int = 8):
    rng = np.random.default_rng(20260817)
    k = rng.normal(size=(tokens, 2, 2, 4)).astype(np.float32)
    v = rng.normal(size=(tokens, 2, 2, 4)).astype(np.float32)
    evict = np.zeros((tokens, 2, 2), dtype=np.bool_)
    return k, v, evict


def _write_metadata(path, *, artifact: str = "artifact:dms-fixture") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dms": {
                    "artifact_fingerprint": artifact,
                    "model_family": "fixture-qwen",
                    "num_layers": 2,
                    "num_q_heads": 4,
                    "num_kv_heads": 2,
                    "head_dim": 4,
                    "window_size": 2,
                    "target_compression_ratio": 4,
                    "alpha_scale": 100.0,
                    "alpha_offset": 5.0,
                    "borrowed_query_channel": -1,
                    "corrected_mask": True,
                    "trained_checkpoint": True,
                    "evidence_source": "fixture://dms",
                },
            }
        ),
        encoding="utf-8",
    )


def test_dms_metadata_loader_requires_packaged_qualified_checkpoint(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    _write_metadata(model / "dms_metadata.json", artifact="sha256:model")

    config = load_dms_retrofit_config(
        model,
        expected_artifact_fingerprint="sha256:model",
    )

    assert config.borrowed_query_channel == 3
    assert config.group_size == 2
    assert config.source_kind == "packaged_metadata"
    assert config.fingerprint == "029d407abc19685d55317bed54f9ab4bcaa0dc3222f8123c79571ad47a4e8649"
    with pytest.raises(ValueError, match="does not match"):
        load_dms_retrofit_config(
            model,
            expected_artifact_fingerprint="sha256:other",
        )
    (model / "dms_metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="no packaged"):
        load_dms_retrofit_config(model)


def test_dms_backend_gate_blocks_missing_metadata_and_accepts_fixture(tmp_path) -> None:
    from scripts import dms_backend_gate

    model = tmp_path / "model"
    model.mkdir()
    blocked_args = dms_backend_gate.build_parser().parse_args(
        ["--model", str(model)]
    )
    blocked = dms_backend_gate.run(blocked_args)
    assert blocked["status"] == "blocked_metadata"
    assert blocked["passed"] is False

    metadata = model / "dms_metadata.json"
    _write_metadata(metadata)
    accepted_args = dms_backend_gate.build_parser().parse_args(
        [
            "--model",
            str(model),
            "--expected-artifact",
            "artifact:dms-fixture",
            "--slots-per-layer",
            "1024",
            "--prompt-tokens",
            "8",
            "--decode-tokens",
            "2",
        ]
    )
    accepted = dms_backend_gate.run(accepted_args)

    assert accepted["status"] == "accepted_host_backend"
    assert accepted["passed"] is True
    assert [row["width"] for row in accepted["widths"]] == [1, 2, 4, 8, 16, 32]
    assert accepted["pressure"]["retryable_rejection_observed"] is True
    assert accepted["no_dense_shadow"] is True


def test_dms_metadata_rejects_untrained_or_uncorrected_checkpoint() -> None:
    common = dict(
        artifact_fingerprint="artifact",
        model_family="qwen",
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=4,
        window_size=2,
        target_compression_ratio=4,
        alpha_scale=1.0,
        alpha_offset=0.0,
        borrowed_query_channel=-1,
        evidence_source="fixture",
        source_path="fixture.json",
    )
    with pytest.raises(ValueError, match="corrected-mask"):
        DMSRetrofitConfig(corrected_mask=False, trained_checkpoint=True, **common)
    with pytest.raises(ValueError, match="trained"):
        DMSRetrofitConfig(corrected_mask=True, trained_checkpoint=False, **common)


def test_dms_extracts_group_decisions_and_zeros_borrowed_channel() -> None:
    config = _retrofit()
    q = np.zeros((3, 4, 4), dtype=np.float32)
    q[:, 0, -1] = [0.04, 0.06, 0.01]
    q[:, 2, -1] = [0.10, 0.00, 0.051]
    original = q.copy()

    cleaned, evict = extract_dms_eviction_decisions(q, config)

    np.testing.assert_array_equal(
        evict,
        np.asarray(
            [[False, True], [True, False], [False, True]],
            dtype=np.bool_,
        ),
    )
    assert np.all(cleaned[:, (0, 2), -1] == 0)
    np.testing.assert_array_equal(q, original)


def test_dms_live_mask_keeps_recent_window_and_non_evicted_rows() -> None:
    evict = np.asarray([[True, True, False, True, True]], dtype=np.bool_)

    live = build_dms_live_mask(
        evict,
        current_position=4,
        window_size=2,
    )

    np.testing.assert_array_equal(
        live,
        np.asarray([[False, False, True, True, True]]),
    )


def test_dms_int8_codec_is_independently_qualified_and_roundtrips() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(0.0, 0.2, size=(12, 8)).astype(np.float32)

    payload, scales = encode_dms_payload(values, codec="int8_per_token_head")
    restored = decode_dms_payload(payload, scales, codec="int8_per_token_head")

    assert payload.dtype == np.int8
    assert scales is not None and scales.shape == (12,)
    assert float(np.max(np.abs(restored - values))) < 0.005
    with pytest.raises(ValueError, match="KL"):
        DMSCodecQualification(
            codec="int8_per_token_head",
            artifact_fingerprint="artifact",
            kl_divergence=0.051,
            top1_agreement=1.0,
            no_dense_shadow=True,
            evidence_source="fixture",
        )


def test_dms_cpu_reference_kernels_register_and_stream_without_shadow() -> None:
    from hipengine.kernels.cpu_reference.dms import (
        dms_streaming_pack_reference,
        register_dms_cpu_reference_kernels,
    )
    from hipengine.kernels.registry import resolve

    register_dms_cpu_reference_kernels(replace=True)
    packed_kernel = resolve(
        backend="cpu_reference",
        layer="dms_streaming_pack",
        quant="bf16",
        variant="count_rank_scatter",
    )
    assert packed_kernel is dms_streaming_pack_reference
    k, v, _ = _prompt_arrays(tokens=6)
    evict = np.ones((6, 2, 2), dtype=np.bool_)
    evict[0] = False

    packed_k, packed_v, positions = packed_kernel(
        k,
        v,
        evict,
        current_position=5,
        window_size=2,
    )

    assert packed_k[0][0].shape == packed_v[0][0].shape == (4, 4)
    np.testing.assert_array_equal(positions[0][0], [0, 3, 4, 5])


def test_compact_attention_matches_dense_prefix_for_variable_gqa_counts() -> None:
    rng = np.random.default_rng(9)
    q = rng.normal(size=(2, 4, 4)).astype(np.float32)
    k = rng.normal(size=(2, 2, 6, 4)).astype(np.float32)
    v = rng.normal(size=(2, 2, 6, 4)).astype(np.float32)
    counts = np.asarray([[6, 3], [2, 5]], dtype=np.int32)

    output = compact_attention_reference(q, k, v, counts)

    expected = np.zeros_like(output)
    for row in range(2):
        for kv_head in range(2):
            for local_head in range(2):
                q_head = kv_head * 2 + local_head
                logits = (
                    q[row, q_head]
                    @ k[row, kv_head, : counts[row, kv_head]].T
                    * 0.5
                )
                probabilities = np.exp(logits - np.max(logits))
                probabilities /= probabilities.sum()
                expected[row, q_head] = (
                    probabilities
                    @ v[row, kv_head, : counts[row, kv_head]]
                )
    np.testing.assert_allclose(output, expected, atol=1e-6, rtol=1e-6)


def test_dms_int8_codec_passes_fixture_kl_and_top1_gate() -> None:
    rng = np.random.default_rng(77)
    q = rng.normal(size=(16, 4, 8)).astype(np.float32)
    k = rng.normal(0.0, 0.2, size=(16, 2, 12, 8)).astype(np.float32)
    v = rng.normal(0.0, 0.2, size=(16, 2, 12, 8)).astype(np.float32)
    counts = np.full((16, 2), 12, dtype=np.int32)
    teacher = compact_attention_reference(q, k, v, counts)
    k_payload, k_scales = encode_dms_payload(k, codec="int8_per_token_head")
    v_payload, v_scales = encode_dms_payload(v, codec="int8_per_token_head")
    candidate = compact_attention_reference(
        q,
        decode_dms_payload(k_payload, k_scales, codec="int8_per_token_head"),
        decode_dms_payload(v_payload, v_scales, codec="int8_per_token_head"),
        counts,
    )
    teacher_logits = teacher.reshape(16, -1)
    candidate_logits = candidate.reshape(16, -1)
    teacher_prob = np.exp(teacher_logits - teacher_logits.max(axis=1, keepdims=True))
    teacher_prob /= teacher_prob.sum(axis=1, keepdims=True)
    candidate_prob = np.exp(candidate_logits - candidate_logits.max(axis=1, keepdims=True))
    candidate_prob /= candidate_prob.sum(axis=1, keepdims=True)
    kl = float(
        np.mean(
            np.sum(
                teacher_prob
                * np.log(np.maximum(teacher_prob, 1e-30) / np.maximum(candidate_prob, 1e-30)),
                axis=1,
            )
        )
    )
    top1 = float(
        np.mean(
            np.argmax(teacher_logits, axis=1)
            == np.argmax(candidate_logits, axis=1)
        )
    )

    assert kl <= 0.05
    assert top1 >= 0.90
    qualification = DMSCodecQualification(
        codec="int8_per_token_head",
        artifact_fingerprint="artifact:dms-fixture",
        kl_divergence=kl,
        top1_agreement=top1,
        no_dense_shadow=True,
        evidence_source="fixture://measured-int8",
    )
    assert qualification.kl_divergence == pytest.approx(kl)


def test_compact_extent_pool_rolls_back_fragmented_failure_and_coalesces() -> None:
    pool = CompactExtentPool(num_layers=2, slots_per_layer=12)
    first = pool.allocate("first", per_head_slots=3, num_heads=2)
    second = pool.allocate("second", per_head_slots=2, num_heads=2)
    before = pool.snapshot()

    with pytest.raises(MemoryError, match="exhausted"):
        pool.allocate("too-large", per_head_slots=4, num_heads=2)

    assert pool.snapshot()["free_ranges_by_layer"] == before["free_ranges_by_layer"]
    pool.release("first")
    pool.release("second")
    pool.assert_conserved()
    assert pool.free_slots == 24
    assert pool.largest_free_extent == 12
    assert len(first) == len(second) == 4


def test_dms_work_item_claims_pack_workspace_not_dense_pages() -> None:
    backend = _backend()
    claims = backend.estimate(
        _request(1),
        None,
        {"kind": "work_item", "rows": 16},
    )

    assert claims.units_by_pool() == {"dms.pack_workspace_rows": 16}
    assert claims.claims[0].lifetime.value == "work_item"
    with pytest.raises(ValueError, match="exceed"):
        backend.estimate(
            _request(1),
            None,
            {"kind": "work_item", "rows": 65},
        )


def test_dms_bf16_streaming_pack_has_no_dense_shadow_and_exact_no_evict() -> None:
    backend = _backend()
    request = _request(1)
    claims = backend.estimate(request, None, {"kind": "admission"})
    lease = backend.reserve(claims)
    k, v, evict = _prompt_arrays()

    backend.streaming_pack(1, k, v, evict)
    state = backend.state_for_request(1)
    view = backend.prepare(
        SimpleNamespace(request_ids=(1,), span_role="decode")
    )

    assert lease.request_id == 1
    assert state.logical_tokens == 8
    assert np.all(state.live_counts == 8)
    assert set(state.k_payload) == {(layer, head) for layer in range(2) for head in range(2)}
    assert not hasattr(state, "dense_k") and not hasattr(state, "dense_v")
    assert view.live_spans.spans_mode == "per_head_variable"
    assert view.live_spans.live_counts.shape == (1, 2, 2)
    assert view.storage_view.layout_key == "dms-compact:bf16:g1"
    assert view.kernel_bundle_key == "dms_compact_bf16_streaming_v1"
    snapshot = backend.observability_snapshot()
    assert snapshot["backend"]["no_dense_shadow"] is True
    assert snapshot["operations"]["streaming_pack_calls"] == 1

    backend.reclaim(lease)
    backend.assert_conserved()
    assert backend.extents.free_slots == 128


def test_dms_streaming_pack_reduces_allocator_visible_live_rows() -> None:
    backend = _backend(slots_per_layer=48)
    request = _request(2, prompt_tokens=12, max_new_tokens=2)
    lease = backend.reserve(backend.estimate(request, None, {}))
    k, v, _ = _prompt_arrays(tokens=12)
    evict = np.ones((12, 2, 2), dtype=np.bool_)
    evict[::4] = False

    backend.streaming_pack(2, k, v, evict)
    snapshot = backend.observability_snapshot()

    assert np.all(backend.state_for_request(2).live_counts == 6)
    assert snapshot["capacity"]["logical_token_rows"] == 48
    assert snapshot["capacity"]["live_token_rows"] == 24
    assert snapshot["capacity"]["actual_compression_ratio"] == pytest.approx(2.0)
    backend.reclaim(lease)


def test_dms_decode_append_and_transaction_rollback_restore_canonical_payload() -> None:
    backend = _backend()
    request = _request(3)
    lease = backend.reserve(backend.estimate(request, None, {}))
    k, v, _ = _prompt_arrays()
    evict = np.ones((8, 2, 2), dtype=np.bool_)
    backend.streaming_pack(3, k, v, evict)
    state = backend.state_for_request(3)
    operation = backend.begin_transaction((lease,), None)
    before_payload = state.k_payload[(0, 0)].copy()

    backend.append_decode(
        3,
        np.ones((2, 2, 4), dtype=np.float32),
        np.full((2, 2, 4), 2.0, dtype=np.float32),
        np.ones((2, 2), dtype=np.bool_),
        position=8,
    )
    assert state.logical_tokens == 9
    backend.rollback(operation)

    assert state.logical_tokens == 8
    np.testing.assert_array_equal(state.k_payload[(0, 0)], before_payload)
    assert np.all(state.live_counts == 3)
    backend.reclaim(lease)


def test_dms_int8_composition_changes_only_codec_planes_and_uses_scales() -> None:
    backend = _backend(codec="int8_per_token_head")
    request = _request(4)
    lease = backend.reserve(backend.estimate(request, None, {}))
    k, v, evict = _prompt_arrays()

    backend.streaming_pack(4, k, v, evict)
    state = backend.state_for_request(4)
    view = backend.prepare(SimpleNamespace(request_ids=(4,)))

    assert state.k_payload[(0, 0)].dtype == np.int8
    assert state.k_scales[(0, 0)].shape == (8,)
    assert view.live_spans.scale_metadata is not None
    assert {plane.role for plane in view.storage_view.planes} >= {
        "k_payload",
        "v_payload",
        "k_scale",
        "v_scale",
        "base_offsets",
        "live_counts",
    }
    assert backend.spec.topology_key == "dms_compact"
    backend.reclaim(lease)


def test_dms_prefix_lookup_is_fail_closed() -> None:
    backend = _backend()
    miss = backend.prefix_lookup((1, 2, 3))

    assert miss.hit is False
    assert miss.reason == "dms_prefix_overlay_unqualified"
    with pytest.raises(ValueError, match="prefix reuse is disabled"):
        backend.estimate(_request(1), object(), {})


class _FakeCompactRunner:
    kv_kernel_bundle_key = "dms_compact_bf16_streaming_v1"

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.prefill_views = []
        self.decode_views = []

    def prefill_batch_with_kv(self, work, *, kv_batch_view, commit):
        self.prefill_views.append((work, kv_batch_view, bool(commit)))

    def decode_batch_with_kv(self, work, *, kv_batch_view, commit):
        self.decode_views.append((work, kv_batch_view, bool(commit)))
        return tuple(
            GeneratedToken(request_id=request_id, token_id=1000 + request_id)
            for request_id in work.request_ids
        )

    def reclaim(self, completed) -> None:
        del completed


def test_dms_common_scheduler_supports_c1_through_c32_and_final_drain() -> None:
    backend = create_dms_bf16_backend(
        retrofit=_retrofit(),
        slots_per_layer=512,
        max_request_rows=32,
        max_pack_rows=128,
        physical_widths=(1, 2, 4, 8),
    )
    admission = DMSAdmissionManager(backend)
    runner = _FakeCompactRunner(capacity=32)
    adapter = DMSCompactResidentRunnerAdapter(runner, admission)
    loop = ResidentEngineLoop(
        adapter,
        config=EngineLoopConfig(
            max_active_requests=32,
            max_pending_requests=64,
            prefill_decode_policy="token_budget",
            max_prefill_chunk_tokens=8,
        ),
    )

    request_ids = tuple(
        loop.submit([request_id], max_new_tokens=1)
        for request_id in range(32)
    )
    for _ in range(128):
        loop.tick()
        if all(request_id in loop.completed for request_id in request_ids):
            break

    assert all(request_id in loop.completed for request_id in request_ids)
    for request_id in request_ids:
        loop.release_completed(request_id)
    backend.assert_conserved()
    assert backend.extents.free_slots == 2 * 512
    assert backend.observability_snapshot()["extent_pool"]["owner_count"] == 0
    assert runner.prefill_views
    assert runner.decode_views
    assert all(
        call[1].live_spans.spans_mode == "per_head_variable"
        for call in (*runner.prefill_views, *runner.decode_views)
    )
