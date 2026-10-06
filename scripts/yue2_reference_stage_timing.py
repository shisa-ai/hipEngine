#!/usr/bin/env python3
"""Matched NAR/VAE timing for the pinned upstream reference (oracle only).

The counterpart of ``scripts/yue2_stage_matched_timing.py``. Same case, same inputs:

* ``nar``: the case's prefix, codes and seed through the reference's own
  ``song_chunks`` and ``CachedNAR`` for the same number of steps, with the chunk noise
  digest recorded so the comparison can prove both sides started from the same draw.
* ``vae``: the case's recorded latents through the reference's own decoder, full and
  tiled at the product's 1024/16 tiling.

Oracle-only: imports torch and the pinned upstream package, and nothing here is
reachable from ``hipengine.LLM.generate()``.

    PYTHONPATH=~/yue2-shootout/shared/upstream \\
    ~/venvs/vibevoice-tts-oracle/bin/python scripts/yue2_reference_stage_timing.py \\
        --stage nar --case mandarin-off-s1234 --json /tmp/yue2_nar_ref.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.yue2_stage_matched_timing import (  # noqa: E402
    PROTOCOL,
    SAMPLE_RATE,
    array_digest,
    load_case,
)


def run_nar(case_data: dict, *, repeats: int) -> dict:
    import numpy as np
    import torch

    from scripts.yue2_oracle import _install_path, load_model

    _install_path()
    from yue2.nar import CachedNAR, song_chunks

    prefix = case_data["prefix"]
    codec = case_data["semantic"]
    seed = case_data["seed"]
    steps = case_data["steps"]
    chunks = song_chunks(prefix, codec, seed, context=24576)
    song_noise = np.asarray(chunks[0].noise.cpu().numpy(), dtype=np.float32)
    if len(chunks) > 1:
        song_noise = np.concatenate([np.asarray(c.noise.cpu().numpy(), dtype=np.float32)
                                     for c in chunks], axis=0)
    noise_digest = array_digest(song_noise)
    model = load_model()[1]

    def solve_once() -> tuple[float, np.ndarray]:
        latents = []
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for chunk in chunks:
                engine = CachedNAR(model, chunk)
                latents.append(engine.solve(steps).numpy())
                engine.close()
        torch.cuda.synchronize()
        return time.perf_counter() - started, np.concatenate(latents, axis=0)

    seconds, latents = solve_once()
    repeat_seconds = [solve_once()[0] for _ in range(max(0, repeats - 1))]
    return {
        "solve_seconds": seconds,
        "repeat_solve_seconds": repeat_seconds,
        "warm_solve_seconds": min([seconds, *repeat_seconds]),
        "ms_per_step": seconds / steps * 1000.0,
        "latent_digest": array_digest(np.asarray(latents, dtype=np.float32)),
        "noise_digest": noise_digest,
        "latent_norm": float(np.linalg.norm(latents.astype(np.float64))),
        "latents": latents,
        "song_noise": song_noise,
    }


def run_vae(case_data: dict, *, repeats: int) -> dict:
    import numpy as np
    import torch

    from scripts.yue2_oracle import VAE_DIR, _install_path

    _install_path()
    from yue2.modeling_vae import YuE2VAE

    model = YuE2VAE.from_pretrained(VAE_DIR, decoder_only=True, device="cuda")
    latent = np.asarray(case_data["latent"], dtype=np.float32)
    z = torch.from_numpy(latent.T[None, ...].copy()).to("cuda")

    def decode_tiled() -> np.ndarray:
        with torch.inference_mode():
            return model.decode_tiled(z, core_frames=1024, halo_frames=16,
                                      output_device="cpu").numpy()

    torch.cuda.synchronize()
    started = time.perf_counter()
    tiled = np.asarray(decode_tiled(), dtype=np.float32)
    torch.cuda.synchronize()
    tiled_seconds = time.perf_counter() - started
    tiled_repeats = []
    for _ in range(max(0, repeats - 1)):
        torch.cuda.synchronize()
        started = time.perf_counter()
        decode_tiled()
        torch.cuda.synchronize()
        tiled_repeats.append(time.perf_counter() - started)

    full = None
    full_seconds = None
    full_note = ""
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            full = model.decode(z).cpu().numpy()
        torch.cuda.synchronize()
        full_seconds = time.perf_counter() - started
    except Exception as error:  # device residency, not correctness
        full_note = f"full decode unavailable: {type(error).__name__}: {error}"
    frames = int(latent.shape[0])
    return {
        "frames": frames,
        "tiled_seconds": tiled_seconds,
        "repeat_tiled_seconds": tiled_repeats,
        "tiled_ms_per_frame": tiled_seconds / frames * 1000.0,
        "full_seconds": full_seconds,
        "full_ms_per_frame": (full_seconds / frames * 1000.0) if full_seconds else None,
        "full_note": full_note,
        "tiled_digest": array_digest(tiled),
        "full_digest": array_digest(full) if full is not None else "",
        "tiled_vs_full_max_abs": (float(np.abs(tiled - full).max()) if full is not None else None),
        "samples": int(tiled.shape[-1]),
        "audio_seconds": int(tiled.shape[-1]) / SAMPLE_RATE,
        "peak": float(np.abs(tiled).max()),
        "tiled": tiled,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["nar", "vae"], default="nar")
    parser.add_argument("--case", default="mandarin-off-s1234")
    parser.add_argument("--oracle", default=str(REPO / "artifacts" / "yue2" / "oracle" / "cases"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    import torch

    case_data = load_case(Path(args.oracle), args.case)
    result = run_nar(case_data, repeats=args.repeats) if args.stage == "nar" else \
        run_vae(case_data, repeats=args.repeats)
    song_noise = result.pop("song_noise", None)
    if song_noise is not None and args.json:
        # Our own seeded draw is not the reference's, so the matched comparison has to
        # hand this to the torch-free side rather than let it re-derive one.
        Path(args.json + ".noise.npy").parent.mkdir(parents=True, exist_ok=True)
        np.save(args.json + ".noise.npy", song_noise)
        print(f"wrote {args.json}.noise.npy")
    result.pop("latents", None)
    result.pop("tiled", None)
    payload = {
        "protocol": PROTOCOL,
        "side": "reference",
        "stage": args.stage,
        "case": case_data["case"],
        "steps": case_data["steps"],
        "prefix": case_data["prefix"],
        "semantic": case_data["semantic"],
        "prefix_digest": case_data["prefix_digest"],
        "semantic_digest": case_data["semantic_digest"],
        "latent_digest": case_data["latent_digest"],
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        **result,
    }
    if args.stage == "nar":
        print(f"nar {payload['case']}: {payload['steps']} steps, "
              f"solve {payload['solve_seconds']:.3f} s "
              f"({payload['ms_per_step']:.2f} ms per step), "
              f"latent norm {payload['latent_norm']:.3f}, noise {payload['noise_digest']}")
        if payload["repeat_solve_seconds"]:
            print(f"repeat solve: {['%.3f' % v for v in payload['repeat_solve_seconds']]}")
    else:
        print(f"vae {payload['case']}: {payload['frames']} frames, "
              f"tiled {payload['tiled_seconds']:.3f} s "
              f"({payload['tiled_ms_per_frame']:.2f} ms per frame), "
              f"full {payload['full_seconds'] and '%.3f s' % payload['full_seconds'] or payload['full_note']}")
        if payload["repeat_tiled_seconds"]:
            print(f"repeat tiled: {['%.3f' % v for v in payload['repeat_tiled_seconds']]}")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
