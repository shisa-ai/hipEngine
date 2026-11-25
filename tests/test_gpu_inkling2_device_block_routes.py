"""A prompt block of the resident Inkling GGUF traversal selects its routes on the device.

With the fast profile, the resident traversal and device routes, a block of 192 or more rows has
its router logits projected on the device, its routes selected there and its grouped schedule built
there, so its expert stage downloads nothing and uploads nothing. Its mix scales are the host
router's, a row's routes in expert order, and the rows it writes are bitwise those of the
host-routed block given the same routes and scales. Only visits whose experts are raw NVFP4 take
it; a mixed-quantization file keeps other layouts in some layers. The model is a multi-gigabyte
local file, so this test runs only when ``HIPENGINE_INKLING2_MINI_GGUF`` points at it and a HIP
runtime loads.
"""

from __future__ import annotations

import ctypes
import os
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from hipengine import LLM, SamplingParams
from hipengine.core.hip import HipMemcpyKind as Copy, get_hip_runtime
from hipengine.quant.gguf import GGMLQuantizationType as Q

PROMPT = (
    "Tide pools are shallow pools of seawater that form in the rocky intertidal zone, the part of "
    "the shore that is covered at high tide and exposed at low tide. Each pool is a small world of "
    "its own, cut off from the sea for hours at a time. During that time the water warms or cools, "
    "salt concentrates as it evaporates, and oxygen falls as animals and algae breathe. Organisms "
    "that live there must survive wide swings in temperature and salinity, strong waves, and long "
    "stretches of bright sun. Sea stars, anemones, barnacles, mussels, limpets, crabs, sculpins and "
    "many kinds of seaweed are common residents. Anemones fold their tentacles inward when the tide "
    "is out, storing water to keep from drying out. Barnacles close their plates. Crabs hide under "
    "rocks and weeds. When the tide returns, the pool is flooded with cold, oxygen-rich water and "
    "the animals open and start feeding again. Barnacles kick their legs to sweep food from the "
    "water. Anemones spread their tentacles to sting and capture small animals. Sea stars slowly "
    "pry open mussels. Biologists study tide pools because they are easy to reach and show how "
    "species compete for space. On many rocky shores, barnacles and mussels occupy most of the "
    "rock unless predators such as sea stars and whelks keep their numbers down. Removing the "
    "predators lets mussels crowd out nearly everything else, a classic demonstration of how a "
    "few species can shape a whole community. Tide pools also respond quickly to changes in water "
    "quality, so they serve as natural indicators of the health of the coast. Visitors are asked "
    "to walk carefully, to replace any overturned rocks, and to leave animals where they found "
    "them, because a creature moved to the wrong pool may not survive the next low tide. "
    "Repeated from the top of the page, the same account follows once more so that the prompt is "
    "long enough to fill several blocks of tokens. "
) * 2
PARAMS = dict(max_tokens=32, temperature=0.0, ignore_eos=True)


def _skip_unless_runnable() -> str:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    model = os.environ.get("HIPENGINE_INKLING2_MINI_GGUF", "")
    if not model or not Path(model).is_file():
        pytest.skip("HIPENGINE_INKLING2_MINI_GGUF must point at the Inkling 2 GGUF model")
    return model


def _router_name(text, visit):
    return text.weights.names[f"layers.{visit.weight_layer}.mlp.gate.weight"].name


def test_prompt_block_routes_on_the_device_like_the_host(monkeypatch):
    model = _skip_unless_runnable()
    monkeypatch.setenv("HIPENGINE_INKLING2_RESIDENT_DECODE", "1")
    monkeypatch.setenv("HIPENGINE_INKLING2_DEVICE_ROUTES", "1")
    with closing(LLM(model=model, backend="hip_gfx1100", execution_profile="fast", max_sequence_length=2048,
                     prefix_cache="off", speculative_mtp_serving="off")) as llm:
        text = llm._get_text_generator().model
        projection = text.projection
        visit = text.plan.visits[0]
        assert projection._wmma_banks(visit)
        assert projection.block_device_routes_supported(visit, 192)
        assert not projection.block_device_routes_supported(visit, 191)
        # Every layer of this file is read by the grouped kernel: NVFP4, and in the last four layers
        # Q5_K gate/up with Q8_0 down.
        assert not [item for item in text.plan.visits if not projection._wmma_banks(item)]
        mixed = [item for item in text.plan.visits
                 if Q(text.mapping.layers[item.weight_layer].gate_up.weight.ggml_type) is Q.Q5_K]
        assert mixed and Q(text.mapping.layers[mixed[0].weight_layer].down.weight.ggml_type) is Q.Q8_0
        assert projection.block_device_routes_supported(mixed[0], 192)

        width = text.mapping.layers[0].gate_up.weight.shape[2]
        rows = 256
        hidden = np.random.default_rng(23).normal(0, 0.1, (rows, width)).astype(np.float32)
        want_ids, want_mix = text.route(visit, hidden)
        top = want_ids.shape[1]
        rt = get_hip_runtime()
        hidden_buffer, result_buffer = rt.malloc(hidden.nbytes), rt.malloc(hidden.nbytes)
        try:
            rt.memcpy(hidden_buffer, hidden.ctypes.data, hidden.nbytes, Copy.HOST_TO_DEVICE)
            real_block = projection.project_experts_block_device_routes
            real_block(mixed[0], rows=rows, router=_router_name(text, mixed[0]),
                       hidden_ptr=hidden_buffer, out_ptr=result_buffer)
            rt.stream_synchronize(0)

            def block():
                real_block(visit, rows=rows, router=_router_name(text, visit), hidden_ptr=hidden_buffer,
                           out_ptr=result_buffer)

            # The first call uploads the router bias, which is kept.
            block()
            rt.stream_synchronize(0)
            download, upload = projection._transfer.download, projection._transfer.enqueue_upload
            downloads = projection._transfer.download = Mock(wraps=download)
            uploads = projection._transfer.enqueue_upload = Mock(wraps=upload)
            try:
                block()
            finally:
                projection._transfer.download, projection._transfer.enqueue_upload = download, upload
            downloads.assert_not_called()
            uploads.assert_not_called()
            rt.stream_synchronize(0)
            device_rows = np.full((rows, width), np.nan, np.float32)
            rt.memcpy(device_rows.ctypes.data, result_buffer, device_rows.nbytes, Copy.DEVICE_TO_HOST)
            # The device scales follow the host's: a row's routes in expert order.
            mix = np.empty(rows * top, np.float32)
            rt.memcpy(mix.ctypes.data, projection._schedule_state["mix"], mix.nbytes, Copy.DEVICE_TO_HOST)
            np.testing.assert_allclose(mix.reshape(rows, top), want_mix, rtol=1e-5, atol=1e-7)
            assert projection._schedule_state["routes"] >= rows * top
            # The same routes and scales through the host-routed block give the same rows, bitwise.
            projection.project_experts_mixed_rows(visit, ((want_ids, mix.reshape(rows, top), hidden, None),),
                                                  device=(hidden_buffer, result_buffer))
            host_rows = np.full((rows, width), np.nan, np.float32)
            rt.memcpy(host_rows.ctypes.data, result_buffer, host_rows.nbytes, Copy.DEVICE_TO_HOST)
            assert np.isfinite(device_rows).all()
            np.testing.assert_array_equal(device_rows, host_rows)
        finally:
            rt.free(hidden_buffer)
            rt.free(result_buffer)

        # Through the public route a long prompt reaches the device-routed block once per visit of
        # every full block whose experts are raw NVFP4, and never for the other visits or the tail.
        calls = []

        def counted(item, **kwargs):
            calls.append((item.weight_layer, kwargs["rows"]))
            return real_block(item, **kwargs)

        real_supported = projection.block_device_routes_supported
        projection.project_experts_block_device_routes = counted
        try:
            out = llm.generate_detailed([PROMPT], SamplingParams(**PARAMS))[0]
        finally:
            projection.project_experts_block_device_routes = real_block
        wmma_visits = [item.weight_layer for item in text.plan.visits if projection._wmma_banks(item)]
        # Every block of at least 192 rows: the full ones of the profile's width, and the tail when
        # it is wide enough.
        from hipengine.generation.inkling2_gguf import Inkling2GGUFGenerator

        tokens, width = len(llm.tokenize(PROMPT)), Inkling2GGUFGenerator.fast_prefill_block_width
        assert width > 192
        tail = tokens % width
        sizes = ({width} if tokens >= width else set()) | ({tail} if tail >= 192 else set())
        assert calls and {count for _, count in calls} == sizes
        assert len(calls) == len(wmma_visits) * (tokens // width + (tail >= 192))
        assert sorted(layer for layer, _ in calls) == sorted(wmma_visits * (len(calls) // len(wmma_visits)))
        assert out.telemetry.diagnostics["execution_path"] == "inkling2_gguf_packed_hip"
        # The same prompt with host-routed blocks gives the same greedy tokens.
        projection.block_device_routes_supported = lambda item, count: False
        try:
            host = llm.generate_detailed([PROMPT], SamplingParams(**PARAMS))[0]
        finally:
            projection.block_device_routes_supported = real_supported
        assert out.generated_token_ids == host.generated_token_ids
