"""C8 phase-1 static-batch kernels/runtime bit-exactness for CUDA ``sm_120a``.

The C8 batch milestones add static-B variants of every decoder primitive and a
``MoonshineCudaBatchRuntime`` whose rows must be **bit-exact** to B independent
c=1 sessions at the token, per-layer-boundary hidden, and self-cache levels.
Two retained gates live here:

1. Kernel-level: batched embedding, partial-RoPE+cache-append, advance-position,
   and fused LM-head are bit-exact vs B sequential single-row calls.
2. Runtime-level: at B=2/4/8, the batch runtime equals B independent
   ``MoonshineCudaResidentRuntime`` sessions on shared fixtures for tokens,
   every layer-boundary hidden state, sampled self caches, EOS transcripts,
   reset, and teardown.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from hipengine.core.cuda import get_cuda_runtime
from hipengine.core.device import Device
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.loading.moonshine import load_moonshine_model

_FIXTURE_DIR = os.environ.get(
    "HIPENGINE_MOONSHINE_SIX_FIXTURE_DIR",
    "/home/lhl/moonshine-prod-inference/results/raw/moonshine-fixtures-six",
)
_SNAPSHOT = os.environ.get(
    "HIPENGINE_MOONSHINE_SNAPSHOT",
    "/home/lhl/.cache/huggingface/hub/models--shisa-ai--shisa-realtime-asr-0.92b/snapshots/"
    "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
)
_SIX_FIXTURES = (
    "audio-hai-fp16",
    "audio-konichiwa-fp16",
    "audio-konichiwa.ogenkidesuka-fp16",
    "audio-kumbawa-fp16",
    "audio-sosososo-fp16",
    "audio-sumimasen-fp16",
)
_EOS = 2
_CACHE_SAMPLE_LAYERS = (0, 4, 7)
# Layer-boundary names in the order the resident decoder invokes them.
_BOUNDARY_NAMES = tuple(
    f"layer_{layer}.{kind}"
    for layer in range(8)
    for kind in ("after_self_attention", "after_cross_attention", "after_mlp")
) + ("final_hidden",)


def _cuda_sm120a_enabled() -> bool:
    import ctypes

    if os.environ.get("HIPENGINE_RUN_CUDA_SM120A") != "1":
        return False
    if os.environ.get("HIPENGINE_CUDA_ARCH") != "sm_120a":
        return False
    try:
        ctypes.CDLL("libcudart.so.13")
    except OSError:
        return False
    return True


def _fixtures_available() -> bool:
    return (
        all(
            os.path.isfile(os.path.join(_FIXTURE_DIR, f"{name}.npz"))
            and os.path.isfile(os.path.join(_FIXTURE_DIR, f"{name}.json"))
            for name in _SIX_FIXTURES
        )
        and os.path.isdir(_SNAPSHOT)
    )


def _upload(array: np.ndarray, runtime, allocations):
    host = np.ascontiguousarray(array)
    device = malloc(host.nbytes, runtime=runtime)
    allocations.append(device)
    copy_host_to_device(device, host_array_ptr(host), runtime=runtime)
    return device


def _alloc(shape: tuple[int, ...], dtype, runtime, allocations):
    device = malloc(int(np.prod(shape)) * np.dtype(dtype).itemsize, runtime=runtime)
    allocations.append(device)
    return device


def _zero(buf, runtime):
    runtime.memset_async(buf.ptr, 0, buf.nbytes, 0)
    runtime.device_synchronize()


def _download(device, shape: tuple[int, ...], dtype, runtime) -> np.ndarray:
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), device, runtime=runtime)
    return host


def _tensor_to_host(runtime, tensor) -> np.ndarray:
    host = np.empty(tensor.shape, dtype=np.float16)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(tensor.ptr, tensor.numel * tensor.dtype.itemsize),
        runtime=runtime,
    )
    return host


# ---------------------------------------------------------------------------
# Kernel-level gate: batched glue + LM-head vs B sequential single-row calls
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _cuda_sm120a_enabled(),
    reason="CUDA sm_120a gate is not enabled",
)
def test_moonshine_cuda_batch_glue_and_lm_head_bit_exact_vs_single_row() -> None:
    from hipengine.kernels.cuda_sm120a.fused.moonshine_glue import (
        build_moonshine_glue,
        moonshine_embedding_lookup_batch_fp16,
        moonshine_embedding_lookup_fp16,
        moonshine_partial_rope_cache_append_batch_fp16,
        moonshine_partial_rope_cache_append_fp16,
    )
    from hipengine.kernels.cuda_sm120a.linear.lm_head import (
        build_moonshine_lm_head,
        lm_head_argmax_scratch_elements,
        moonshine_lm_head_argmax_batch_fp16,
        moonshine_lm_head_argmax_fp16,
    )

    rng = np.random.default_rng(0xC8B47C8)
    runtime = get_cuda_runtime()
    runtime.set_device(0)
    glue = build_moonshine_glue(load=True)
    head = build_moonshine_lm_head(load=True)
    batch = 4
    heads, head_dim, capacity, rotary, max_pos = 8, 52, 194, 32, 100_000
    hidden, vocab, rpb = 416, 36_864, 8
    allocations: list[object] = []
    try:
        # ---- batched embedding lookup ------------------------------------
        embedding = rng.normal(0.0, 0.05, size=(vocab, hidden)).astype(np.float16)
        tokens = np.array([7, 0, vocab - 1, 12345], dtype=np.int64)
        d_emb = _upload(embedding, runtime, allocations)
        d_tok = _upload(tokens, runtime, allocations)
        d_out = _alloc((batch, hidden), np.float16, runtime, allocations)
        moonshine_embedding_lookup_batch_fp16(
            d_emb.ptr, d_tok.ptr, d_out.ptr, hidden, vocab, batch,
            library=glue, runtime=runtime,
        )
        runtime.device_synchronize()
        batched = _download(d_out, (batch, hidden), np.float16, runtime)
        single = _alloc((1, hidden), np.float16, runtime, allocations)
        for row in range(batch):
            d_tok1 = _upload(tokens[row : row + 1], runtime, allocations)
            moonshine_embedding_lookup_fp16(
                d_emb.ptr, d_tok1.ptr, single.ptr, hidden, vocab,
                library=glue, runtime=runtime,
            )
            runtime.device_synchronize()
            expected = _download(single, (1, hidden), np.float16, runtime)[0]
            assert np.array_equal(batched[row], expected), f"embedding row {row}"

        # ---- batched partial-RoPE + cache append --------------------------
        positions = np.array([0, 5, 9, 193], dtype=np.int64)
        cos = rng.normal(0.0, 1.0, size=(max_pos, rotary // 2)).astype(np.float16)
        sin = rng.normal(0.0, 1.0, size=(max_pos, rotary // 2)).astype(np.float16)
        d_cos = _upload(cos, runtime, allocations)
        d_sin = _upload(sin, runtime, allocations)
        d_pos = _upload(positions, runtime, allocations)
        host_q = rng.normal(0, 0.4, (batch, heads * head_dim)).astype(np.float16)
        host_k = rng.normal(0, 0.4, (batch, heads * head_dim)).astype(np.float16)
        host_v = rng.normal(0, 0.4, (batch, heads * head_dim)).astype(np.float16)
        d_query = _upload(host_q, runtime, allocations)
        d_key = _upload(host_k, runtime, allocations)
        d_value = _upload(host_v, runtime, allocations)
        d_qout = _alloc((batch, heads * head_dim), np.float16, runtime, allocations)
        d_kout = _alloc((batch, heads * head_dim), np.float16, runtime, allocations)
        d_kcache = _alloc((batch, heads * capacity * head_dim), np.float16, runtime, allocations)
        d_vcache = _alloc((batch, heads * capacity * head_dim), np.float16, runtime, allocations)
        _zero(d_kcache, runtime)
        _zero(d_vcache, runtime)
        moonshine_partial_rope_cache_append_batch_fp16(
            d_query.ptr, d_key.ptr, d_value.ptr, d_cos.ptr, d_sin.ptr, d_pos.ptr,
            d_qout.ptr, d_kout.ptr, d_kcache.ptr, d_vcache.ptr,
            heads, head_dim, rotary, capacity, max_pos, batch,
            library=glue, runtime=runtime,
        )
        runtime.device_synchronize()
        b_qout = _download(d_qout, (batch, heads * head_dim), np.float16, runtime)
        b_kout = _download(d_kout, (batch, heads * head_dim), np.float16, runtime)
        b_kcache = _download(d_kcache, (batch, heads * capacity * head_dim), np.float16, runtime)
        b_vcache = _download(d_vcache, (batch, heads * capacity * head_dim), np.float16, runtime)
        s_qout = _alloc((1, heads * head_dim), np.float16, runtime, allocations)
        s_kout = _alloc((1, heads * head_dim), np.float16, runtime, allocations)
        s_kcache = _alloc((1, heads * capacity * head_dim), np.float16, runtime, allocations)
        s_vcache = _alloc((1, heads * capacity * head_dim), np.float16, runtime, allocations)
        for row in range(batch):
            d_q1 = _upload(host_q[row : row + 1], runtime, allocations)
            d_k1 = _upload(host_k[row : row + 1], runtime, allocations)
            d_v1 = _upload(host_v[row : row + 1], runtime, allocations)
            d_p1 = _upload(positions[row : row + 1], runtime, allocations)
            _zero(s_kcache, runtime)
            _zero(s_vcache, runtime)
            moonshine_partial_rope_cache_append_fp16(
                d_q1.ptr, d_k1.ptr, d_v1.ptr, d_cos.ptr, d_sin.ptr, d_p1.ptr,
                s_qout.ptr, s_kout.ptr, s_kcache.ptr, s_vcache.ptr,
                heads, head_dim, rotary, capacity, max_pos,
                library=glue, runtime=runtime,
            )
            runtime.device_synchronize()
            assert np.array_equal(b_qout[row], _download(s_qout, (1, heads * head_dim), np.float16, runtime)[0]), f"rope qout row {row}"
            assert np.array_equal(b_kout[row], _download(s_kout, (1, heads * head_dim), np.float16, runtime)[0]), f"rope kout row {row}"
            assert np.array_equal(b_kcache[row], _download(s_kcache, (1, heads * capacity * head_dim), np.float16, runtime)[0]), f"rope kcache row {row}"
            assert np.array_equal(b_vcache[row], _download(s_vcache, (1, heads * capacity * head_dim), np.float16, runtime)[0]), f"rope vcache row {row}"

        # ---- batched fused LM-head ----------------------------------------
        input_rows = rng.normal(0.0, 1.0, size=(batch, hidden)).astype(np.float16)
        weight = rng.normal(0.0, 0.02, size=(vocab, hidden)).astype(np.float16)
        num_blocks = lm_head_argmax_scratch_elements(vocab, rpb)
        d_in = _upload(input_rows, runtime, allocations)
        d_w = _upload(weight, runtime, allocations)
        d_bv = _alloc((batch, num_blocks), np.float32, runtime, allocations)
        d_bi = _alloc((batch, num_blocks), np.int64, runtime, allocations)
        d_idx = _alloc((batch,), np.int64, runtime, allocations)
        d_val = _alloc((batch,), np.float32, runtime, allocations)
        moonshine_lm_head_argmax_batch_fp16(
            d_in.ptr, d_w.ptr, d_bv.ptr, d_bi.ptr, d_idx.ptr, d_val.ptr,
            hidden, vocab, batch, rows_per_block=rpb,
            library=head, runtime=runtime,
        )
        runtime.device_synchronize()
        b_idx = _download(d_idx, (batch,), np.int64, runtime)
        b_val = _download(d_val, (batch,), np.float32, runtime)
        s_idx = _alloc((1,), np.int64, runtime, allocations)
        s_val = _alloc((1,), np.float32, runtime, allocations)
        s_bv = _alloc((1, num_blocks), np.float32, runtime, allocations)
        s_bi = _alloc((1, num_blocks), np.int64, runtime, allocations)
        for row in range(batch):
            d_in1 = _upload(input_rows[row : row + 1], runtime, allocations)
            moonshine_lm_head_argmax_fp16(
                d_in1.ptr, d_w.ptr, s_bv.ptr, s_bi.ptr, s_idx.ptr, s_val.ptr,
                hidden, vocab, rows_per_block=rpb,
                library=head, runtime=runtime,
            )
            runtime.device_synchronize()
            assert int(b_idx[row]) == int(_download(s_idx, (1,), np.int64, runtime)[0]), f"lm-head token row {row}"
            assert float(b_val[row]) == float(_download(s_val, (1,), np.float32, runtime)[0]), f"lm-head value row {row}"
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)


# ---------------------------------------------------------------------------
# Runtime-level gate: batch runtime vs B independent c=1 sessions
# ---------------------------------------------------------------------------


def _pad_row_cache(array: np.ndarray, shared_frames: int) -> np.ndarray:
    arr = np.ascontiguousarray(array, dtype=np.float16)
    if arr.shape[2] == shared_frames:
        return arr
    out = np.zeros(
        (arr.shape[0], arr.shape[1], shared_frames, arr.shape[3]), dtype=np.float16
    )
    out[:, :, : arr.shape[2], :] = arr
    return out


def _load_fixture(name: str, shared_frames: int) -> dict[str, object]:
    with open(os.path.join(_FIXTURE_DIR, f"{name}.json")) as handle:
        manifest = json.load(handle)
    with np.load(os.path.join(_FIXTURE_DIR, f"{name}.npz")) as fixture:
        frames = int(manifest["input"]["encoder_frames"])
        reference = [int(token) for token in manifest["decoder"]["token_ids"]]
        keys = [fixture[f"cross.layer_{layer}.key"] for layer in range(8)]
        values = [fixture[f"cross.layer_{layer}.value"] for layer in range(8)]
    mask = np.zeros((1, shared_frames), dtype=np.int32)
    mask[0, :frames] = 1
    return {
        "name": name,
        "frames": frames,
        "reference": reference,
        "keys": [_pad_row_cache(k, shared_frames) for k in keys],
        "values": [_pad_row_cache(v, shared_frames) for v in values],
        "mask": mask,
    }


def _run_c1_window(
    runtime,
    loaded,
    encoder_frames: int,
    keys,
    values,
    mask,
    seed: int,
    steps: int,
) -> dict[str, object]:
    """Run one c=1 session for ``steps`` lockstep positions; capture everything."""
    from hipengine.runtime.moonshine_cuda import MoonshineCudaResidentRuntime

    spec = loaded.spec
    decoder = MoonshineCudaResidentRuntime(
        encoder_frames=encoder_frames, loaded_model=loaded, owns_weights=False
    )
    decoder.prepare_decoder_kernels()
    hidden: dict[tuple[int, str], np.ndarray] = {}
    caches: dict[tuple[int, int, str], np.ndarray] = {}
    try:
        decoder.load_cross_cache(keys, values, mask=mask)
        tokens: list[int] = []
        token_id = seed
        for position in range(steps):
            def callback(name, tensor, position=position):
                # A synchronous D2H copy races the runtime's nonblocking stream,
                # so sync the device before copying each boundary.
                runtime.device_synchronize()
                hidden[(position, name)] = _tensor_to_host(runtime, tensor).reshape(-1)

            decoder.set_decode_state(token_id=token_id, position=position)
            decoder.token_step(boundary_callback=callback)
            token_id = int(decoder.read_token())
            tokens.append(token_id)
            for layer in _CACHE_SAMPLE_LAYERS:
                view = decoder.self_cache(layer)
                caches[(position, layer, "key")] = _tensor_to_host(runtime, view.key).reshape(-1)
                caches[(position, layer, "value")] = _tensor_to_host(runtime, view.value).reshape(-1)
        return {"tokens": tokens, "hidden": hidden, "caches": caches}
    finally:
        decoder.close()


def _run_batch_window(
    runtime,
    loaded,
    decoder,
    seeds: np.ndarray,
    steps: int,
) -> dict[str, object]:
    """Run the batch runtime for ``steps`` lockstep positions; capture everything."""
    spec = loaded.spec
    batch = decoder.max_batch
    hidden: dict[tuple[int, str], np.ndarray] = {}
    caches: dict[tuple[int, int, str], np.ndarray] = {}
    tokens = seeds.astype(np.int64)
    token_rows: list[list[int]] = [[] for _ in range(batch)]
    for position in range(steps):
        def callback(name, tensor, position=position):
            runtime.device_synchronize()
            hidden[(position, name)] = _tensor_to_host(runtime, tensor)

        decoder.set_batch_decode_state(tokens=tokens.tolist(), position=position)
        decoder.batch_token_step(boundary_callback=callback)
        tokens = decoder.read_tokens()
        for layer in _CACHE_SAMPLE_LAYERS:
            view = decoder.self_cache(layer)
            caches[(position, layer, "key")] = _tensor_to_host(runtime, view.key)
            caches[(position, layer, "value")] = _tensor_to_host(runtime, view.value)
        for row in range(batch):
            token_rows[row].append(int(tokens[row]))
    return {"tokens": token_rows, "hidden": hidden, "caches": caches}


def _run_c1_eos(runtime, loaded, encoder_frames, keys, values, mask, seed) -> list[int]:
    from hipengine.runtime.moonshine_cuda import MoonshineCudaResidentRuntime

    spec = loaded.spec
    decoder = MoonshineCudaResidentRuntime(
        encoder_frames=encoder_frames, loaded_model=loaded, owns_weights=False
    )
    decoder.prepare_decoder_kernels()
    try:
        decoder.load_cross_cache(keys, values, mask=mask)
        transcript: list[int] = []
        token_id = seed
        for position in range(spec.self_cache_capacity):
            decoder.set_decode_state(token_id=token_id, position=position)
            decoder.token_step()
            token_id = int(decoder.read_token())
            if token_id == _EOS:
                break
            transcript.append(token_id)
        return transcript
    finally:
        decoder.close()


def _run_batch_eos(runtime, loaded, decoder, seeds) -> tuple[list[list[int]], list[int | None]]:
    spec = loaded.spec
    batch = decoder.max_batch
    tokens = seeds.astype(np.int64)
    done = np.zeros(batch, dtype=bool)
    transcripts: list[list[int]] = [[] for _ in range(batch)]
    eos_positions: list[int | None] = [None] * batch
    for position in range(spec.self_cache_capacity):
        decoder.set_batch_decode_state(tokens=tokens.tolist(), position=position)
        decoder.batch_token_step()
        tokens = decoder.read_tokens()
        for row in range(batch):
            if done[row]:
                continue
            if int(tokens[row]) == _EOS:
                done[row] = True
                eos_positions[row] = position
            else:
                transcripts[row].append(int(tokens[row]))
        if bool(done.all()):
            break
    return transcripts, eos_positions


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _fixtures_available(),
    reason="CUDA sm_120a gate or six audio fixtures are not available",
)
def test_moonshine_cuda_batch_token_graphs_bit_exact_vs_eager() -> None:
    """C8 phase-1: captured batch token DAGs replay bit-exact to eager decode.

    Graphs are captured in device-owned mode (the DAG tail advances the device
    position scalars), then replayed to all-EOS.  The transcripts must equal the
    eager lockstep transcripts and the graph contract must report both position
    buckets.
    """
    from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    loaded = load_moonshine_model(_SNAPSHOT, device=Device("cuda", 0), runtime=runtime)
    fixture_names = ["audio-hai-fp16", "audio-konichiwa-fp16"]
    shared_frames = 0
    for name in fixture_names:
        with open(os.path.join(_FIXTURE_DIR, f"{name}.json")) as handle:
            manifest = json.load(handle)
        shared_frames = max(shared_frames, int(manifest["input"]["encoder_frames"]))
    fixtures = [_load_fixture(name, shared_frames) for name in fixture_names]
    seeds = np.array([f["reference"][0] for f in fixtures], dtype=np.int64)
    keys_batch = [
        np.concatenate([f["keys"][layer] for f in fixtures], axis=0)
        for layer in range(8)
    ]
    values_batch = [
        np.concatenate([f["values"][layer] for f in fixtures], axis=0)
        for layer in range(8)
    ]
    masks_batch = np.concatenate([f["mask"] for f in fixtures], axis=0)

    decoder = MoonshineCudaBatchRuntime(
        max_batch=2,
        encoder_frames=shared_frames,
        loaded_model=loaded,
        owns_weights=False,
    )
    decoder.prepare_decoder_kernels()
    try:
        decoder.load_cross_cache_batch(keys_batch, values_batch, masks=masks_batch)

        def decode_eager() -> list[list[int]]:
            decoder.reset_generation(clear_cross_cache=False)
            tokens = seeds.astype(np.int64)
            done = np.zeros(2, dtype=bool)
            transcripts: list[list[int]] = [[], []]
            while not bool(done.all()):
                decoder.set_batch_decode_state(
                    tokens=tokens.tolist(), position=decoder.self_cache_length
                )
                decoder.batch_token_step()
                tokens = decoder.read_tokens()
                for row in range(2):
                    if not done[row]:
                        if int(tokens[row]) == _EOS:
                            done[row] = True
                        else:
                            transcripts[row].append(int(tokens[row]))
            return transcripts

        def decode_graph() -> list[list[int]]:
            decoder.reset_generation(clear_cross_cache=False)
            decoder.set_batch_device_owned_decode(True)
            decoder.set_batch_decode_seed(tokens=seeds.tolist())
            decoder.capture_batch_token_graphs()
            contract = decoder.batch_token_graph_contract()
            assert contract["captured"] is True
            assert set(contract["buckets"]) == {"positions_0_6", "positions_7_193"}
            done = np.zeros(2, dtype=bool)
            transcripts: list[list[int]] = [[], []]
            while not bool(done.all()):
                decoder.graph_batch_token_step()
                tokens = decoder.read_tokens()
                for row in range(2):
                    if not done[row]:
                        if int(tokens[row]) == _EOS:
                            done[row] = True
                        else:
                            transcripts[row].append(int(tokens[row]))
            return transcripts

        eager = decode_eager()
        graph = decode_graph()
        assert eager == graph, f"graph vs eager transcripts diverged: {eager} != {graph}"
        assert decoder.batch_token_graph_contract()["replay_count"] > 0
    finally:
        decoder.close()
        assert decoder.teardown_returned_to_baseline is True
        loaded.weights.free(runtime=runtime)


@pytest.mark.skipif(
    not _cuda_sm120a_enabled() or not _fixtures_available(),
    reason="CUDA sm_120a gate or six audio fixtures are not available",
)
@pytest.mark.parametrize("batch_size", [2, 4, 8])
def test_moonshine_cuda_batch_runtime_bit_exact_vs_c1_sessions(batch_size: int) -> None:
    """C8 phase-1: batch runtime equals B independent c=1 sessions exactly.

    For B=2/4/8 on shared-frame fixtures, compare the batch runtime against B
    independent ``MoonshineCudaResidentRuntime`` sessions at (a) a fixed 6-step
    lockstep window covering every layer-boundary hidden state and sampled self
    caches, and (b) EOS-terminated token transcripts.  All comparisons are
    bit-exact; the reset/teardown parity checks close the allocation contract.
    """
    from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime

    runtime = get_cuda_runtime()
    runtime.set_device(0)
    loaded = load_moonshine_model(_SNAPSHOT, device=Device("cuda", 0), runtime=runtime)
    fixture_names = [_SIX_FIXTURES[i % len(_SIX_FIXTURES)] for i in range(batch_size)]
    # Resolve the shared bucket from each fixture's native frame count, then
    # reload every row padded identically to that bucket.
    shared_frames = 0
    for name in fixture_names:
        with open(os.path.join(_FIXTURE_DIR, f"{name}.json")) as handle:
            manifest = json.load(handle)
        shared_frames = max(shared_frames, int(manifest["input"]["encoder_frames"]))
    loaded_fixtures = [_load_fixture(name, shared_frames) for name in fixture_names]
    seeds = np.array([f["reference"][0] for f in loaded_fixtures], dtype=np.int64)

    try:
        # ---- (a) fixed 6-step lockstep window ------------------------------
        steps = 6
        c1_results = []
        for fixture in loaded_fixtures:
            c1_results.append(
                _run_c1_window(
                    runtime, loaded, shared_frames,
                    fixture["keys"], fixture["values"], fixture["mask"],
                    int(fixture["reference"][0]), steps,
                )
            )
        decoder = MoonshineCudaBatchRuntime(
            max_batch=batch_size,
            encoder_frames=shared_frames,
            loaded_model=loaded,
            owns_weights=False,
        )
        decoder.prepare_decoder_kernels()
        try:
            keys_batch = [
                np.concatenate([f["keys"][layer] for f in loaded_fixtures], axis=0)
                for layer in range(8)
            ]
            values_batch = [
                np.concatenate([f["values"][layer] for f in loaded_fixtures], axis=0)
                for layer in range(8)
            ]
            masks_batch = np.concatenate(
                [f["mask"] for f in loaded_fixtures], axis=0
            )
            decoder.load_cross_cache_batch(
                keys_batch, values_batch, masks=masks_batch
            )
            batch_result = _run_batch_window(runtime, loaded, decoder, seeds, steps)

            for row in range(batch_size):
                c1 = c1_results[row]
                assert batch_result["tokens"][row] == c1["tokens"], (
                    f"B={batch_size} row {row} token window diverged: "
                    f"batch={batch_result['tokens'][row]} c1={c1['tokens']}"
                )
                for position in range(steps):
                    for name in _BOUNDARY_NAMES:
                        key = (position, name)
                        batched = batch_result["hidden"][key][row].reshape(-1)
                        single = c1["hidden"][key].reshape(-1)
                        assert np.array_equal(batched, single), (
                            f"B={batch_size} row {row} pos {position} {name} diverged"
                        )
                    for layer in _CACHE_SAMPLE_LAYERS:
                        for side in ("key", "value"):
                            batched = batch_result["caches"][(position, layer, side)][row].reshape(-1)
                            single = c1["caches"][(position, layer, side)].reshape(-1)
                            assert np.array_equal(batched, single), (
                                f"B={batch_size} row {row} pos {position} "
                                f"layer {layer} {side} cache diverged"
                            )

            # reset: a fresh batch decode without reloading cross cache must
            # reproduce the same window (fixed addresses, state zeroed).
            decoder.reset_generation(clear_cross_cache=False)
            again = _run_batch_window(runtime, loaded, decoder, seeds, steps)
            for row in range(batch_size):
                assert again["tokens"][row] == batch_result["tokens"][row]

            # ---- (b) EOS-terminated transcripts ----------------------------
            c1_eos = [
                _run_c1_eos(
                    runtime, loaded, shared_frames,
                    f["keys"], f["values"], f["mask"], int(f["reference"][0]),
                )
                for f in loaded_fixtures
            ]
            decoder.reset_generation(clear_cross_cache=False)
            batch_eos, eos_positions = _run_batch_eos(runtime, loaded, decoder, seeds)
            for row in range(batch_size):
                assert batch_eos[row] == c1_eos[row], (
                    f"B={batch_size} row {row} EOS transcript diverged: "
                    f"batch={batch_eos[row]} c1={c1_eos[row]}"
                )
                assert eos_positions[row] is not None
            contract = decoder.allocation_contract()
            assert contract["max_batch"] == batch_size
        finally:
            decoder.close()
            assert decoder.teardown_returned_to_baseline is True
    finally:
        loaded.weights.free(runtime=runtime)
