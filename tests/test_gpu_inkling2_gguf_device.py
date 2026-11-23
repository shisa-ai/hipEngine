import ctypes
from contextlib import ExitStack
from unittest.mock import Mock

import numpy as np
import pytest

from hipengine.core.hip import get_hip_runtime
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.backends import resolve_backend
from hipengine.kernels.cpu_reference.inkling2_mini import _sigmoid
from hipengine.loading.inkling2_gguf import Inkling2GGUFWeights
from hipengine.loading import inkling2_gguf_device as device
from hipengine.loading.inkling2_gguf_device import Inkling2GGUFDeviceProjections
from hipengine.quant.gguf import GGMLQuantizationType as Q
from tests.test_unit_inkling2_gguf import _file


@pytest.mark.parametrize("legacy", [True, False])
def test_inkling2_gguf_packed_projections_preserve_scale_and_route_order(legacy, monkeypatch):
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("librocblas.so")
    except OSError as exc:
        pytest.skip(f"HIP or rocBLAS unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")

    rng = np.random.default_rng(1983)
    reader = Mock()
    reader.info = _file(legacy)
    weights = Inkling2GGUFWeights(reader)
    mapping = weights.mapping
    matrices = [mapping.root["output.weight"], mapping.layers[2].gate_up.weight,
                mapping.layers[2].down.weight, mapping.layers[4].gate_up.weight, mapping.layers[4].down.weight]
    dense = mapping.layers[2].tensors["attn_q.weight"]
    raw = {}
    for t in (*matrices, dense):
        blocks = rng.integers(0, 256, t.byte_shape, dtype=np.uint8)
        if t.ggml_type == Q.NVFP4:
            blocks.reshape(-1, 36)[:, :4] = rng.integers(16, 63, (blocks.size // 36, 4), dtype=np.uint8)
        elif t.ggml_type == Q.Q8_0:
            blocks.reshape(-1, 34)[:, :2] = np.array([0.003], dtype="<f2").view(np.uint8)
        else:
            blocks.reshape(-1, 176)[:, :4] = np.array([0.002, 0.001], dtype="<f2").view(np.uint8)
        raw[t.name] = blocks
    raw[mapping.layers[2].gate_up.scale.name] = np.array([0.63, 0.33], np.float32)
    raw[mapping.layers[2].down.scale.name] = np.array([0.41, 0.72], np.float32)
    reader.tensor_data.side_effect = raw.__getitem__
    x = rng.normal(0, 0.1, (3, 256)).astype(np.float32)
    expert_ids = np.array([1, 0, 1], np.int32)
    blas = Rocblas.load()
    try:
        with Inkling2GGUFDeviceProjections(weights, runtime=get_hip_runtime(), blas=blas, backend=backend) as owner:
            head = matrices[0]
            np.testing.assert_allclose(owner.project(head.name, x, limit=mapping.vocab_size),
                                       x @ weights.decode(head, slice(mapping.vocab_size)).T, rtol=1e-5, atol=2e-6)
            launch_prefill = owner._q8_prefill
            assert launch_prefill is not None
            spy = Mock(wraps=launch_prefill)
            owner._q8_prefill = spy
            for rows in (3, 4, 9):
                inp = rng.normal(0, 0.1, (rows, 256)).astype(np.float32)
                actual = owner.project(dense.name, inp)
                try:
                    owner._q8_prefill = None
                    expected = owner.project(dense.name, inp)
                finally:
                    owner._q8_prefill = spy
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            assert spy.call_count == 3
            for width in (2, 3, 4):
                inp = rng.normal(0, 0.1, (width, 256)).astype(np.float32)
                scalar = np.stack([owner.project(head.name, row, limit=mapping.vocab_size) for row in inp])
                before = len(owner._matrices)
                actual = owner.project_decode_head(inp)
                np.testing.assert_array_equal(actual.view(np.uint32), scalar.view(np.uint32))
                assert len(owner._matrices) == before
                assert spy.call_args.args[-3:] == (width, 256, head.shape[0])
            assert spy.call_count == 6
            # One row and an output tail not divisible by eight use the parent.
            owner.project(dense.name, x[0, :])
            owner.project(head.name, x, limit=mapping.vocab_size)
            assert spy.call_count == 6
            # Force the minimum down in this small fixture to cover an aligned
            # head prefix plus a scalar tail without enlarging the model.
            with monkeypatch.context() as scope:
                scope.setattr(device, "_Q8_DECODE_MIN_OUTPUTS", 128)
                launch_decode = owner._q8_decode
                decode_spy = Mock(wraps=launch_decode)
                owner._q8_decode = decode_spy
                for width in (123, 240, 251):
                    actual = owner.project(dense.name, x[0], limit=width)
                    try:
                        owner._q8_decode = None
                        expected = owner.project(dense.name, x[0], limit=width)
                    finally:
                        owner._q8_decode = decode_spy
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                assert [call.args[-1] for call in decode_spy.call_args_list] == [240, 248]
                owner._q8_decode = launch_decode
            controls = [owner.project_groups((((dense.name, head.name), row),), pinned=True) for row in x]
            rt = get_hip_runtime()

            def zero_query(query, head):
                rt.memset_async(query[0], 0, 4, 0)

            def zero_head(query, head):
                rt.memset_async(head[0], 0, 4, 0)

            transforms = ((zero_query,), (zero_head,), None)
            batched = owner.project_shared_attention_rows([
                ((((dense.name, head.name), row),), consume) for row, consume in zip(x, transforms, strict=True)])
            controls[0][0][0][0] = 0
            controls[1][0][1][0] = 0
            for ((q, h),), ((expected_q, expected_h),) in zip(batched, controls, strict=True):
                np.testing.assert_array_equal(q.view(np.uint32), expected_q.view(np.uint32))
                np.testing.assert_array_equal(h.view(np.uint32), expected_h.view(np.uint32))
            for visit in (mapping.plan.visits[2], mapping.plan.visits[4], mapping.plan.visits[6]):
                for projection in ("gate_up", "down"):
                    observed = owner.project_selected(visit, projection, expert_ids, x)
                    expected = np.stack([weights.project_expert(visit, projection, expert, h)
                                         for expert, h in zip(expert_ids, x, strict=True)])
                    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=2e-5)
            assert owner.packed_bank_bytes == sum(t.nbytes for t in matrices[1:])
            for visit in (mapping.plan.visits[2], mapping.plan.visits[6]):
                for ids in (expert_ids[:2], np.array([[1, 0], [0, 1], [1, 1]], np.int32)):
                    activated = rng.normal(0, 0.1, (ids.size, 256)).astype(np.float32)
                    mix = rng.uniform(0.0, 1.0, ids.shape).astype(np.float32)
                    down = owner.project_selected(visit, "down", ids.ravel(), activated).reshape(*ids.shape, 256)
                    expected = np.zeros((*ids.shape[:-1], 256), dtype=np.float32)
                    for j in range(ids.shape[-1]):
                        expected += mix[..., j, None] * down[..., j, :]
                    transform = Mock()
                    with monkeypatch.context() as limit:
                        # Also cover advancing all route and output pointers across the launch limit.
                        limit.setattr(device, "_MAX_SELECTED_ROUTES", 4)
                        actual = owner.project_down_mixed(visit, ids, mix, activated, transform_output=transform)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    transform.assert_called_once()
                    assert transform.call_args.args[1] == expected.shape
            with Inkling2GGUFDeviceProjections(weights, runtime=get_hip_runtime(), blas=blas, backend=backend,
                                               fast_experts=True) as fast:
                assert fast.supports_device_activation
                exchange = fast._transfer.exchange = Mock(wraps=fast._transfer.exchange)
                for visit in (mapping.plan.visits[2], mapping.plan.visits[6]):
                    for shape in ((2,), (9, 2)):
                        ids = rng.integers(0, 2, shape, dtype=np.int32)
                        inp = rng.normal(0, 0.1, (*shape[:-1], 256)).astype(np.float32)
                        mix = rng.uniform(0, 1, shape).astype(np.float32)
                        expanded = np.repeat(np.atleast_2d(inp), shape[-1], axis=0)
                        packed = owner.project_selected(visit, "gate_up", ids.ravel(), expanded)
                        shared = owner.project_selected(visit, "gate_up", ids.ravel(), np.atleast_2d(inp),
                                                        routes_per_input=shape[-1])
                        np.testing.assert_array_equal(shared.view(np.uint32), packed.view(np.uint32))
                        gate, up = np.split(packed, 2, axis=-1)
                        a, b = mapping.gate_cap, mapping.up_cap
                        activated = (a * np.tanh(gate / a) * _sigmoid(gate)) * (b * np.tanh(up / b))
                        want = owner.project_down_mixed(visit, ids, mix, activated)
                        consumer = Mock()
                        exchange.reset_mock()
                        actual = fast.project_experts_mixed(visit, ids, mix, inp, transform_output=consumer)
                        # The activation is approximate; bound error through the
                        # down projection even when terms cancel in one output.
                        np.testing.assert_allclose(actual, want, rtol=2e-5, atol=1e-5)
                        np.testing.assert_array_equal(actual.argmax(axis=-1), want.argmax(axis=-1))
                        def log_probs(row):
                            logits = row.astype(np.float64)
                            return logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)

                        ref_log, got_log = (log_probs(row) for row in (want, actual))
                        assert np.max(np.sum(np.exp(ref_log) * (ref_log - got_log), axis=-1)) <= 0.05
                        exchange.assert_called_once()
                        consumer.assert_called_once()
                        assert consumer.call_args.args[1] == want.shape
                    requests = []
                    controls = []
                    for ids in (np.array([0, 1]), np.array([1, 0]), np.array([1, 1])):
                        row = rng.normal(0, 0.1, (256,)).astype(np.float32)
                        mix = rng.uniform(0, 1, ids.shape).astype(np.float32)
                        controls.append(fast.project_experts_mixed(visit, ids, mix, row))
                        requests.append((ids, mix, row, Mock()))
                    exchange.reset_mock()
                    grouped = fast.project_experts_mixed_rows(visit, requests)
                    exchange.assert_called_once()
                    ptrs = []
                    for output, control, (_, _, _, consumer) in zip(grouped, controls, requests, strict=True):
                        np.testing.assert_array_equal(output.view(np.uint32), control.view(np.uint32))
                        consumer.assert_called_once()
                        ptr, shape = consumer.call_args.args
                        assert shape == control.shape
                        ptrs.append(ptr)
                    assert len(set(ptrs)) == len(requests)
            # The final physical banks have different formats in the two layouts.
            visit = mapping.plan.visits[6]
            layer = mapping.layers[visit.weight_layer]
            for stem in ("gate_up", "down"):
                bank = getattr(layer, stem)
                quant = Q(bank.weight.ggml_type)
                launch = owner._selected_launches[quant]
                spy = Mock(wraps=launch)
                owner._selected_launches[quant] = spy
                try:
                    grouped = owner.project_selected(visit, stem, expert_ids, x)
                    assert spy.call_count == 1
                    owner._selected_launches.pop(quant)
                    scalar = owner.project_selected(visit, stem, expert_ids, x)
                finally:
                    owner._selected_launches[quant] = launch
                np.testing.assert_array_equal(grouped.view(np.uint32), scalar.view(np.uint32))
                assert quant in owner._tiled_launches
                # Two experts with multiple full tiles and nonuniform tails.
                many_ids = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1], np.int32)
                many_x = rng.normal(0, 0.1, (many_ids.size, 256)).astype(np.float32)
                tiled_launch = owner._tiled_launches[quant]
                try:
                    owner._tiled_launches[quant] = Mock(wraps=tiled_launch)
                    candidate = owner.project_selected(visit, stem, many_ids, many_x)
                    owner._tiled_launches[quant].assert_called_once()
                    owner._tiled_launches.pop(quant)
                    control = owner.project_selected(visit, stem, many_ids, many_x)
                finally:
                    owner._tiled_launches[quant] = tiled_launch
                np.testing.assert_array_equal(candidate.view(np.uint32), control.view(np.uint32))
    finally:
        blas.close()


@pytest.mark.parametrize("quant", ["gguf_q5_k", "gguf_q8_0"])
@pytest.mark.parametrize("outputs", [19, 2047, 2048, 2049, 2304])
def test_selected_tiles_match_individual_routes_at_pair_boundary(quant, outputs):
    from hipengine.core.hip import HipMemcpyKind as Copy
    from hipengine.kernels.backends import hip_target_arch_for_backend
    from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
        _make_selected_raw_f32, _make_selected_tile4_raw_f32, build_gguf_k_gemv,
    )
    from hipengine.loading.inkling2_device_experts import build_expert_route_tiles

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    runtime = get_hip_runtime()
    library = build_gguf_k_gemv(target_arch=hip_target_arch_for_backend(backend))
    rng = np.random.default_rng(1793)
    ids = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], np.int32)
    q5 = quant == "gguf_q5_k"
    inputs, block_size, block_bytes = (2304, 256, 176) if q5 else (1152, 32, 34)
    weights = rng.integers(0, 256, (2, outputs, inputs // block_size, block_bytes), dtype=np.uint8)
    params = np.array([0.002, 0.001] if q5 else [0.002], dtype="<f2").view(np.uint8)
    weights[..., :params.size] = params
    tokens = rng.standard_normal((len(ids) // 2, inputs), dtype=np.float32)
    x = np.repeat(tokens, 2, axis=0)
    scales = rng.uniform(-1, 1, len(ids)).astype(np.float32)
    tiles = build_expert_route_tiles(ids, 4)
    scores = np.empty((len(ids), outputs), dtype=np.float32)
    one = _make_selected_raw_f32(quant)
    tiled = _make_selected_tile4_raw_f32(quant)
    with ExitStack() as owned:
        def allocate(nbytes):
            ptr = runtime.malloc(nbytes)
            owned.callback(runtime.free, ptr)
            return ptr

        def upload(data):
            ptr = allocate(data.nbytes)
            runtime.memcpy(ptr, data.ctypes.data, data.nbytes, Copy.HOST_TO_DEVICE)
            return ptr

        xp, wp, ip, sp, tp = map(upload, (x, weights, ids, scales, tiles))
        out = allocate(scores.nbytes)
        one(xp, wp, out, ip, sp, len(ids), 2, x.shape[1], outputs, runtime=runtime, library=library)
        runtime.memcpy(scores.ctypes.data, out, scores.nbytes, Copy.DEVICE_TO_HOST)
        expected = scores.view(np.uint32).copy()
        tiled(xp, wp, out, tp, sp, len(tiles), len(ids), 2, x.shape[1], outputs, runtime=runtime, library=library)
        runtime.memcpy(scores.ctypes.data, out, scores.nbytes, Copy.DEVICE_TO_HOST)
        np.testing.assert_array_equal(scores.view(np.uint32), expected)
        if q5:
            shared = _make_selected_tile4_raw_f32(quant, shared_input=True)
            compact = upload(tokens)
            with pytest.raises(ValueError, match="routes per input"):
                shared(compact, wp, out, tp, sp, len(tiles), len(ids), 2, x.shape[1], outputs,
                       routes_per_input=3, runtime=runtime, library=library)
            shared(compact, wp, out, tp, sp, len(tiles), len(ids), 2, x.shape[1], outputs,
                   routes_per_input=2, runtime=runtime, library=library)
            runtime.memcpy(scores.ctypes.data, out, scores.nbytes, Copy.DEVICE_TO_HOST)
            np.testing.assert_array_equal(scores.view(np.uint32), expected)
