"""One-shot C8 phase-1 runtime verification: batch runtime vs B independent c=1 sessions.

GPU-gated; run under the same env as the hipEngine CUDA tests:
  HIPENGINE_RUN_CUDA_SM120A=1 HIPENGINE_CUDA_ARCH=sm_120a CUDA_VISIBLE_DEVICES=0
  PYTHONPATH=$PWD uv run --no-project python scripts/verify_c8_batch_runtime.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from hipengine.core.cuda import get_cuda_runtime
from hipengine.core.device import Device
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    host_array_ptr,
)
from hipengine.loading.moonshine import load_moonshine_model
from hipengine.runtime.moonshine_cuda import MoonshineCudaResidentRuntime
from hipengine.runtime.moonshine_cuda_batch import MoonshineCudaBatchRuntime

_FIXTURE_DIR = os.environ.get(
    "HIPENGINE_MOONSHINE_SIX_FIXTURE_DIR",
    "/home/lhl/moonshine-prod-inference/results/raw/moonshine-fixtures-six",
)
_SNAPSHOT = os.environ.get(
    "HIPENGINE_MOONSHINE_SNAPSHOT",
    "/home/lhl/.cache/huggingface/hub/models--shisa-ai--shisa-realtime-asr-0.92b/snapshots/"
    "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
)

_FIXTURES = (
    "audio-hai-fp16",
    "audio-konichiwa-fp16",
    "audio-sosososo-fp16",
    "audio-sumimasen-fp16",
)
_SHARED_FRAMES = 42
_EOS = 2
_BOUNDARIES = 25  # 8 layers x 3 + final_hidden


def _pad_row_cache(array, shared_frames):
    arr = np.ascontiguousarray(array, dtype=np.float16)
    if arr.shape[2] == shared_frames:
        return arr
    out = np.zeros(
        (arr.shape[0], arr.shape[1], shared_frames, arr.shape[3]), dtype=np.float16
    )
    out[:, :, : arr.shape[2], :] = arr
    return out


def _load_fixture(name):
    with open(os.path.join(_FIXTURE_DIR, f"{name}.json")) as handle:
        manifest = json.load(handle)
    with np.load(os.path.join(_FIXTURE_DIR, f"{name}.npz")) as fixture:
        frames = int(manifest["input"]["encoder_frames"])
        reference = [int(token) for token in manifest["decoder"]["token_ids"]]
        keys = [fixture[f"cross.layer_{layer}.key"] for layer in range(8)]
        values = [fixture[f"cross.layer_{layer}.value"] for layer in range(8)]
    padded_keys = [_pad_row_cache(k, _SHARED_FRAMES) for k in keys]
    padded_values = [_pad_row_cache(v, _SHARED_FRAMES) for v in values]
    mask = np.zeros((1, _SHARED_FRAMES), dtype=np.int32)
    mask[0, :frames] = 1
    return {
        "name": name,
        "frames": frames,
        "reference": reference,
        "keys": padded_keys,
        "values": padded_values,
        "mask": mask,
    }


def _tensor_to_host(runtime, tensor):
    host = np.empty(tensor.shape, dtype=np.float16)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(tensor.ptr, tensor.numel * tensor.dtype.itemsize),
        runtime=runtime,
    )
    return host


def _decode_c1(runtime, loaded, encoder_frames, keys, values, mask, seed):
    """Run one c=1 resident session to EOS; return transcript + hidden map."""
    spec = loaded.spec
    decoder = MoonshineCudaResidentRuntime(
        encoder_frames=encoder_frames,
        loaded_model=loaded,
        owns_weights=False,
    )
    decoder.prepare_decoder_kernels()
    hidden: dict[tuple[int, str], np.ndarray] = {}
    try:
        decoder.load_cross_cache(keys, values, mask=mask)
        transcript: list[int] = []
        token_id = seed
        for position in range(spec.self_cache_capacity):
            def callback(name, tensor, position=position):
                # A synchronous D2H copy races the nonblocking stream, so sync
                # the device before copying each boundary.
                runtime.device_synchronize()
                hidden[(position, name)] = _tensor_to_host(runtime, tensor)
            decoder.set_decode_state(token_id=token_id, position=position)
            decoder.token_step(boundary_callback=callback)
            token_id = int(decoder.read_token())
            if token_id == _EOS:
                break
            transcript.append(token_id)
        return transcript, hidden
    finally:
        decoder.close()


def main() -> None:
    cuda_runtime = get_cuda_runtime()
    cuda_runtime.set_device(0)
    loaded = load_moonshine_model(_SNAPSHOT, device=Device("cuda", 0), runtime=cuda_runtime)
    spec = loaded.spec
    try:
        fixtures = [_load_fixture(name) for name in _FIXTURES]
        batch = len(fixtures)

        c1_transcripts: list[list[int]] = []
        c1_hidden: list[dict[tuple[int, str], np.ndarray]] = []
        for fixture in fixtures:
            seed = fixture["reference"][0]
            transcript, hidden = _decode_c1(
                cuda_runtime, loaded, _SHARED_FRAMES,
                fixture["keys"], fixture["values"], fixture["mask"], seed,
            )
            c1_transcripts.append(transcript)
            c1_hidden.append(hidden)
            print(
                f"c=1 {fixture['name']:32s} frames={fixture['frames']:3d} "
                f"tokens={len(transcript)}"
            )

        keys_batch = [
            np.concatenate([fixture["keys"][layer] for fixture in fixtures], axis=0)
            for layer in range(8)
        ]
        values_batch = [
            np.concatenate([fixture["values"][layer] for fixture in fixtures], axis=0)
            for layer in range(8)
        ]
        masks_batch = np.concatenate([fixture["mask"] for fixture in fixtures], axis=0)
        seeds = np.array([fixture["reference"][0] for fixture in fixtures], dtype=np.int64)

        decoder = MoonshineCudaBatchRuntime(
            max_batch=batch,
            encoder_frames=_SHARED_FRAMES,
            loaded_model=loaded,
            owns_weights=False,
        )
        decoder.prepare_decoder_kernels()
        batch_hidden: dict[tuple[int, str], np.ndarray] = {}
        try:
            decoder.load_cross_cache_batch(keys_batch, values_batch, masks=masks_batch)
            tokens = seeds.astype(np.int64)
            done = np.zeros(batch, dtype=bool)
            transcripts: list[list[int]] = [[] for _ in range(batch)]
            eos_positions: list[int | None] = [None] * batch

            def callback(name, tensor, ):
                cuda_runtime.device_synchronize()
                batch_hidden[(tensor_position, name)] = _tensor_to_host(cuda_runtime, tensor)

            tensor_position = 0
            for position in range(spec.self_cache_capacity):
                tensor_position = position
                decoder.set_batch_decode_state(tokens=tokens.tolist(), position=position)
                decoder.batch_token_step(boundary_callback=callback)
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

            # ---- bit-exact comparisons -------------------------------------
            for row, fixture in enumerate(fixtures):
                assert transcripts[row] == c1_transcripts[row], (
                    f"{fixture['name']} row {row} diverged: "
                    f"batch={transcripts[row][:20]} c1={c1_transcripts[row][:20]}"
                )
            # hidden-state comparison across every position each row decoded,
            # at every layer boundary (25 boundaries/position).
            checked = 0
            diverged = 0
            for row, fixture in enumerate(fixtures):
                c1map = c1_hidden[row]
                for position in range(len(c1_transcripts[row]) + 1):
                    for boundary in range(_BOUNDARIES):
                        name = _boundary_name(boundary)
                        key = (position, name)
                        if key not in c1map:
                            continue
                        batched = batch_hidden[key]
                        row_slice = batched[row].reshape(-1)
                        single = c1map[key].reshape(-1)
                        checked += 1
                        if not np.array_equal(row_slice, single):
                            diverged += 1
                            if diverged <= 5:
                                print(
                                    f"  DIVERGED row {row} {fixture['name']} "
                                    f"pos {position} {name}: "
                                    f"max_abs={np.abs(row_slice.astype(np.float32) - single.astype(np.float32)).max():.3e}"
                                )
            print(f"hidden-state boundaries checked: {checked}, diverged: {diverged}")
            for row, fixture in enumerate(fixtures):
                print(
                    f"  row {row} ({fixture['name']}): {len(transcripts[row])} tokens BIT-EXACT, "
                    f"hidden BIT-EXACT, eos@pos={eos_positions[row]}"
                )
            assert diverged == 0, f"{diverged} hidden-state divergences"
            print("ALL BATCH ROWS BIT-EXACT vs independent c=1 sessions (tokens + hidden)")
        finally:
            decoder.close()
    finally:
        loaded.weights.free(runtime=cuda_runtime)


def _boundary_name(index: int) -> str:
    if index == 24:
        return "final_hidden"
    layer = index // 3
    kind = ("after_self_attention", "after_cross_attention", "after_mlp")[index % 3]
    return f"layer_{layer}.{kind}"


if __name__ == "__main__":
    main()
