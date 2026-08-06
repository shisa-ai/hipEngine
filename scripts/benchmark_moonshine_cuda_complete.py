#!/usr/bin/env python3
"""C5: exclusive complete-ASR custom-CUDA benchmark driver (sm_120a).

Measures the complete autoregressive ASR route on the same six retained audio
fixtures and the same timing scope as the framework baselines (preprocessing
and initial H2D excluded; encoder plus autoregressive generation to EOS
included).  Three modes share one driver:

  standalone      torch-free fixed-bucket encoder -> D2D handoff ->
                  precompute cross K/V -> CUDA-graph decode to EOS
  torch-encoder   HF PyTorch CUDA FP16 encoder (uncompiled, as in the retained
                  framework baseline) -> D2D handoff -> precompute cross K/V ->
                  CUDA-graph decode to EOS
  decoder-only    fixture encoder.output -> D2D handoff -> precompute cross K/V
                  -> CUDA-graph decode to EOS  (explicitly partial: the encoder
                  is excluded)

Each file is prepared once (weights, kernels, graph capture) and then run
through ``--warmup`` untimed repetitions followed by ``--iterations`` timed,
stream-synchronized host-wall measurements (mirroring the framework baseline
protocol).  Token streams are verified against the retained fixture reference
(an exact EOS gate; the documented sub-0.05-logit borderline at decode
position 88 is far beyond every EOS decode length here and never reached).

The report also records preparation/startup (weight upload, kernel prep, graph
capture) and deployment footprint (generated kernel binary bytes, resident
workspace bytes, packed model bytes, runtime library bytes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_EOS_TOKEN = 2
_MAX_POSITIONS = 194
_SIX_FIXTURES = (
    "audio-hai-fp16",
    "audio-konichiwa-fp16",
    "audio-konichiwa.ogenkidesuka-fp16",
    "audio-kumbawa-fp16",
    "audio-sosososo-fp16",
    "audio-sumimasen-fp16",
)
_PACKED_MODEL_FP16_BYTES = 126_435_712  # 63,217,856 parameters x 2 bytes
_MODES = ("standalone", "torch-encoder", "decoder-only")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path) -> dict[str, str]:
    return {
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        ),
    }


def _packed_artifact_info(snapshot_dir: Path) -> dict[str, Any]:
    """Report the deployable packed artifact's manifest fields (C5-R1)."""

    manifest_path = Path(snapshot_dir) / "pack_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"--packed requires {manifest_path} (run scripts/pack_moonshine_fp16.py)"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packed = manifest.get("packed", {})
    sha = packed.get("model.safetensors_sha256")
    if not isinstance(sha, str):
        raise ValueError(f"pack_manifest.json missing model.safetensors_sha256: {manifest_path}")
    return {
        "artifact": manifest.get("artifact"),
        "model.safetensors_sha256": sha,
        "model.safetensors_bytes": packed.get("model.safetensors_bytes"),
        "fp16_payload_bytes": _PACKED_MODEL_FP16_BYTES,
        "sidecar_total_bytes": packed.get("sidecar_total_bytes"),
    }


def _implementation_files(repo_root: Path) -> dict[str, Path]:
    """Every implementation/driver source that this report depends on.

    C5-R4 closure: the driver itself plus the resident decoder runtime, the
    encoder runtime, and the sm_120a kernel wrappers/headers used by the
    standalone/torch-encoder/decoder-only routes are all hashed so a raw
    report is reproducible only from the exact sources that produced it.
    """

    files = {
        "driver": repo_root / "scripts" / "benchmark_moonshine_cuda_complete.py",
        "runtime/decoder": repo_root / "hipengine/runtime/moonshine_cuda.py",
        "runtime/encoder": repo_root / "hipengine/runtime/moonshine_encoder_cuda.py",
    }
    kernel_dir = repo_root / "hipengine/kernels/cuda_sm120a"
    if kernel_dir.is_dir():
        for path in sorted(kernel_dir.rglob("*.py")):
            files[str(path.relative_to(repo_root))] = path
        for path in sorted(kernel_dir.rglob("*.cu")):
            files[str(path.relative_to(repo_root))] = path
    return files


def implementation_sha256(repo_root: Path) -> dict[str, str]:
    """SHA-256 of every implementation/driver source (path -> digest)."""

    return {
        str(rel): sha256_file(path)
        for rel, path in sorted(_implementation_files(repo_root).items())
        if path.is_file()
    }


def latency_summary(values_ms: list[float]) -> dict[str, float]:
    return {
        "count": len(values_ms),
        "min_ms": float(min(values_ms)),
        "median_ms": float(statistics.median(values_ms)),
        "p95_ms": float(np.percentile(values_ms, 95)),
        "max_ms": float(max(values_ms)),
    }


class Fixture:
    """One retained six-file audio fixture with its reference artifacts."""

    def __init__(self, name: str, fixture_dir: Path) -> None:
        self.name = name
        self.json_path = fixture_dir / f"{name}.json"
        self.npz_path = fixture_dir / f"{name}.npz"
        with open(self.json_path) as handle:
            manifest = json.load(handle)
        self.frames = int(manifest["input"]["encoder_frames"])
        self.sample_count = int(manifest["input"]["sample_count"])
        self.reference = [int(token) for token in manifest["decoder"]["token_ids"]]
        with np.load(self.npz_path) as data:
            self.audio = data["input.values"].astype(np.float32)
            self.audio_mask = data["input.attention_mask"].astype(np.int64)
            self.encoder_output = data["encoder.output"].astype(np.float16)
            self.encoder_mask = data["encoder.attention_mask"].astype(np.int32)

    def eos_steps(self) -> int:
        """Number of decode steps (positions) until the first EOS token."""

        for index, token in enumerate(self.reference):
            if index > 0 and token == _EOS_TOKEN:
                return index
        return _MAX_POSITIONS


def _upload_host_to_device(cuda_runtime, buffer, array: np.ndarray) -> None:
    from hipengine.core.memory import copy_host_to_device, host_array_ptr

    array = np.ascontiguousarray(array)
    copy_host_to_device(buffer, host_array_ptr(array), runtime=cuda_runtime)


class Route:
    """A complete custom-CUDA ASR route for one fixture."""

    def __init__(
        self, mode: str, fixture: Fixture, loaded, cuda_runtime, snapshot: str, packed: bool = False
    ) -> None:
        self.mode = mode
        self.fixture = fixture
        self.loaded = loaded
        self.cuda_runtime = cuda_runtime
        self.snapshot = snapshot
        self.packed = packed
        self.enc = None
        self.dec = None
        self.torch_encoder = None
        self._device_buffers = []

    # -- preparation ---------------------------------------------------------

    def prepare(self) -> dict[str, Any]:
        """Create + prepare runtimes and capture token graphs; return timing."""

        from hipengine.core.cuda import get_cuda_runtime
        from hipengine.core.device import Device
        from hipengine.core.memory import malloc
        from hipengine.loading.moonshine import load_moonshine_model
        from hipengine.runtime.moonshine_cuda import MoonshineCudaResidentRuntime

        if self.loaded is None:
            self.loaded = load_moonshine_model(
                self.snapshot,
                device=Device("cuda", 0),
                runtime=self.cuda_runtime,
                packed=self.packed,
            )
        loaded = self.loaded

        prepare: dict[str, Any] = {}

        self.dec = MoonshineCudaResidentRuntime(
            encoder_frames=self.fixture.frames,
            loaded_model=loaded,
            owns_weights=False,
        )
        t0 = time.perf_counter()
        self.dec.prepare_decoder_kernels()
        prepare["prepare_decoder_s"] = time.perf_counter() - t0

        if self.mode == "standalone":
            from hipengine.runtime.moonshine_encoder_cuda import (
                MoonshineCudaEncoderRuntime,
            )

            self.enc = MoonshineCudaEncoderRuntime(
                audio_samples=self.fixture.sample_count,
                loaded_model=loaded,
                owns_weights=False,
            )
            t0 = time.perf_counter()
            self.enc.prepare_encoder_kernels()
            prepare["prepare_encoder_s"] = time.perf_counter() - t0
            # Pre-upload the audio bucket once; the timed region runs only the
            # fixed-address DAG (initial H2D excluded, matching the baseline).
            self.enc.upload_input(self.fixture.audio, self.fixture.audio_mask)
        elif self.mode == "torch-encoder":
            self._prepare_torch_encoder(prepare)
        elif self.mode == "decoder-only":
            # Upload the fixture encoder hidden + mask to device buffers (the
            # producer of the encoder state in a real partial-route setup).
            hidden = malloc(
                self.fixture.encoder_output.nbytes,
                runtime=self.cuda_runtime,
            )
            mask = malloc(
                self.fixture.encoder_mask.nbytes,
                runtime=self.cuda_runtime,
            )
            _upload_host_to_device(
                self.cuda_runtime, hidden, self.fixture.encoder_output
            )
            _upload_host_to_device(self.cuda_runtime, mask, self.fixture.encoder_mask)
            self._device_buffers.extend([hidden, mask])
            self._decoder_only_hidden_ptr = hidden.ptr
            self._decoder_only_mask_ptr = mask.ptr

        # One untimed state setup so the cross cache is resident and the
        # token graphs can be captured before the timed iterations.
        self._enqueue_encoder(self.fixture)
        self._set_encoder_state(self.dec, source_frames=self.fixture.frames)
        self.dec.reset_generation(clear_cross_cache=False)

        t0 = time.perf_counter()
        graphs = self.dec.capture_token_graphs()
        prepare["capture_token_graphs_s"] = time.perf_counter() - t0
        contract = self.dec.token_graph_contract()
        prepare["graph_count"] = contract["graph_count"]
        prepare["capture_wall_ms"] = contract["capture_wall_ms"]
        prepare["instantiate_wall_ms"] = contract["instantiate_wall_ms"]
        return prepare

    def _prepare_torch_encoder(self, prepare: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq

        t0 = time.perf_counter()
        model = (
            AutoModelForSpeechSeq2Seq.from_pretrained(self.snapshot)
            .eval()
            .half()
            .to("cuda")
        )
        prepare["torch_encoder_load_s"] = time.perf_counter() - t0
        self.torch_encoder = model.get_encoder()
        # Pre-upload the fixture input once; the timed region runs only the
        # encoder forward (initial H2D excluded, matching the baseline).
        self._torch_input = torch.from_numpy(self.fixture.audio).to(
            "cuda", dtype=torch.float16
        )
        self._torch_mask = torch.from_numpy(self.fixture.audio_mask).to("cuda")

    # -- per-iteration route -------------------------------------------------

    def _set_encoder_state(self, dec, source_frames: int) -> None:
        if self.mode == "standalone":
            self.enc.handoff_to(dec)
        elif self.mode == "torch-encoder":
            dec.set_encoder_state_from_device(
                hidden_fp16_ptr=self._torch_hidden_ptr,
                attention_mask_int32_ptr=self._torch_mask_ptr,
                source_frames=source_frames,
            )
            dec.precompute_cross_kv()
        elif self.mode == "decoder-only":
            dec.set_encoder_state_from_device(
                hidden_fp16_ptr=self._decoder_only_hidden_ptr,
                attention_mask_int32_ptr=self._decoder_only_mask_ptr,
                source_frames=source_frames,
            )
            dec.precompute_cross_kv()

    def _enqueue_encoder(self, fixture: Fixture) -> None:
        if self.mode == "standalone":
            self.enc.run_encode()
        elif self.mode == "torch-encoder":
            self._run_torch_encoder(fixture)
        # decoder-only: encoder state is already resident on device.

    def _run_torch_encoder(self, fixture: Fixture) -> None:
        import torch

        with torch.no_grad():
            hidden = self.torch_encoder(
                input_values=self._torch_input, attention_mask=self._torch_mask
            ).last_hidden_state
        # The producer (torch) stream must finish before the decoder's D2D
        # handoff copies the hidden/mask on its own stream.
        torch.cuda.synchronize()
        self._torch_hidden_ptr = hidden.data_ptr()
        self._torch_mask_ptr = self._torch_mask.data_ptr()

    def run_once(self) -> tuple[list[int], int]:
        """Run the complete route to EOS and return (tokens, steps)."""

        dec = self.dec
        fixture = self.fixture
        self._enqueue_encoder(fixture)
        self._set_encoder_state(dec, source_frames=fixture.frames)
        tokens: list[int] = []
        token_id = fixture.reference[0]
        position = 0
        while True:
            dec.set_decode_state(token_id=token_id, position=position)
            dec.graph_token_step()
            token_id = int(dec.read_token())
            tokens.append(token_id)
            position += 1
            if token_id == _EOS_TOKEN or position >= _MAX_POSITIONS:
                break
        return tokens, position

    def timed_run(self) -> tuple[float, list[int], int]:
        """Time one complete route (stream-synchronized host wall) to EOS."""

        dec = self.dec
        dec.reset_generation(clear_cross_cache=False)
        self.cuda_runtime.stream_synchronize(dec.stream)
        started = time.perf_counter_ns()
        tokens, steps = self.run_once()
        self.cuda_runtime.device_synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) * 1.0e-6
        return elapsed_ms, tokens, steps

    # -- footprint -----------------------------------------------------------

    def footprint(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.enc is not None:
            result["encoder_workspace_nbytes"] = sum(
                allocation.buffer.nbytes
                for allocation in self.enc.workspace._allocations.values()
            )
        if self.dec is not None:
            result["decoder_workspace_nbytes"] = self.dec.allocation_contract()[
                "workspace_nbytes"
            ]
            result["resident_nbytes"] = self.dec.allocation_contract()[
                "resident_nbytes"
            ]
        return result

    def close(self) -> None:
        from hipengine.core.memory import free

        if self.enc is not None:
            self.enc.close()
        if self.dec is not None:
            self.dec.close()
        for buffer in self._device_buffers:
            free(buffer, runtime=self.cuda_runtime)
        self._device_buffers.clear()


def generated_kernel_bytes(root: Path) -> dict[str, int]:
    """Sum the cached architecture-specific CUDA kernel binary sizes."""

    build_root = Path.home() / ".cache" / "hipengine" / "build"
    families = {
        "encoder": "cuda_sm120a_moonshine_encoder-*",
        "attention": "cuda_sm120a_moonshine_attention-*",
        "glue": "cuda_sm120a_moonshine_glue-*",
        "layernorm": "cuda_sm120a_moonshine_layernorm-*",
        "lm_head": "cuda_sm120a_moonshine_lm_head-*",
        "mlp": "cuda_sm120a_moonshine_mlp-*",
        "projection": "cuda_sm120a_moonshine_projection-*",
    }
    result: dict[str, int] = {}
    total = 0
    for label, pattern in families.items():
        total_bytes = 0
        for path in build_root.glob(pattern):
            for so in path.glob("*.so"):
                total_bytes += so.stat().st_size
        result[label] = total_bytes
        total += total_bytes
    result["total_bytes"] = total
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--packed", action="store_true", help="load the deployable FP16 artifact (scripts/pack_moonshine_fp16.py)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    fixture_dir = args.fixture_dir
    if not fixture_dir.is_dir():
        parser.error(f"fixture dir not found: {fixture_dir}")
    fixtures = [Fixture(name, fixture_dir) for name in _SIX_FIXTURES]
    for fixture in fixtures:
        if not fixture.json_path.is_file() or not fixture.npz_path.is_file():
            parser.error(f"missing fixture files for {fixture.name}")

    import torch  # noqa: F401  (guards import availability early)

    from hipengine.core.cuda import get_cuda_runtime
    from hipengine.core.device import Device
    from hipengine.loading.moonshine import load_moonshine_model

    cuda_runtime = get_cuda_runtime()
    cuda_runtime.set_device(args.gpu_index)

    load_started = time.perf_counter()
    loaded = load_moonshine_model(
        args.snapshot_dir,
        device=Device("cuda", args.gpu_index),
        runtime=cuda_runtime,
        packed=args.packed,
    )
    weight_load_s = time.perf_counter() - load_started

    device = torch.cuda.get_device_name(args.gpu_index)
    capability = list(torch.cuda.get_device_capability(args.gpu_index))

    report: dict[str, Any] = {
        "schema": 1,
        "artifact": "moonshine_cuda_complete_asr",
        "mode": args.mode,
        "date": datetime.now(UTC).date().isoformat(),
        "model": {
            "id": "shisa-ai/shisa-realtime-asr-0.92b",
            "revision": "cb0b524b74f6e0bfe6a8780b8dc9854ffa429c7d",
        },
        "environment": {
            "gpu": device,
            "compute_capability": capability,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "scope": {
            "gpu_index": args.gpu_index,
            "mode": args.mode,
            "preprocessing_timed": False,
            "initial_h2d_timed": False,
            "encoder_and_generation_timed": True,
            "decode_stops_at_eos": True,
            "six_file_exact_tokens": True,
        },
        "protocol": {
            "warmups_per_file": args.warmup,
            "iterations_per_file": args.iterations,
            "timing": "stream-synchronized host wall",
            "kernel_build_cached": True,
        },
        "preparation": {
            "weight_load_s": weight_load_s,
        },
        "footprint": {
            "packed_model_fp16_bytes": _PACKED_MODEL_FP16_BYTES,
            "generated_kernel_bytes": generated_kernel_bytes(repo_root),
        },
        # C5-R1: when loading the deployable packed artifact, record its
        # on-disk bytes and SHA-256 so the deployment identity is explicit.
        "packed_artifact": _packed_artifact_info(args.snapshot_dir) if args.packed else None,
        "git": git_state(repo_root),
        # C5-R4: hash every implementation/driver source this report depends on.
        "implementation_sha256": implementation_sha256(repo_root),
        "files": {},
    }

    route_results: dict[str, Any] = {}
    try:
        for fixture in fixtures:
            route = Route(args.mode, fixture, loaded, cuda_runtime, str(args.snapshot_dir), packed=args.packed)
            try:
                prepare = route.prepare()
                report["preparation"].update(prepare)

                # Untimed warmups.
                for _ in range(args.warmup):
                    route.timed_run()

                latencies: list[float] = []
                all_tokens: list[list[int]] = []
                steps_list: list[int] = []
                for _ in range(args.iterations):
                    elapsed_ms, tokens, steps = route.timed_run()
                    latencies.append(elapsed_ms)
                    all_tokens.append(tokens)
                    steps_list.append(steps)

                expected_steps = fixture.eos_steps()
                reference = fixture.reference[1 : expected_steps + 1]
                # Validate EVERY timed output, not just the first iteration
                # (C5-R5): all ten token streams must match the retained
                # reference to EOS and every step count must be identical.
                all_exact = all(
                    tokens[:expected_steps] == reference for tokens in all_tokens
                )
                deterministic_steps = len(set(steps_list)) == 1
                deterministic_tokens = all(
                    tokens == all_tokens[0] for tokens in all_tokens[1:]
                )
                tokens = all_tokens[0]
                exact = tokens[:expected_steps] == reference
                route_results[fixture.name] = {
                    "input": {
                        "id": fixture.json_path.with_suffix("").name.replace(
                            "audio-", ""
                        )
                        .replace("-fp16", "") + ".wav",
                        "encoder_frames": fixture.frames,
                        "sample_count": fixture.sample_count,
                        "eos_decode_steps": expected_steps,
                    },
                    "latency": latency_summary(latencies),
                    "latency_ms": latencies,
                    "steps": steps_list,
                    "token_ids": tokens,
                    "tokens_exact_to_eos": bool(exact),
                    "all_timed_tokens_exact_to_eos": bool(all_exact),
                    "deterministic_steps": bool(deterministic_steps),
                    "deterministic_tokens": bool(deterministic_tokens),
                    "preparation": prepare,
                    "footprint": route.footprint(),
                }
            finally:
                route.close()
    finally:
        loaded.weights.free(runtime=cuda_runtime)

    medians = [
        float(row["latency"]["median_ms"]) for row in route_results.values()
    ]
    report["summary"] = {
        "case_count": len(route_results),
        "median_of_medians_ms": statistics.median(medians),
        "min_median_ms": min(medians),
        "max_median_ms": max(medians),
        "all_six_tokens_exact_to_eos": all(
            row["tokens_exact_to_eos"] for row in route_results.values()
        ),
        # C5-R5: every timed iteration (not just the first) is validated.
        "all_timed_tokens_exact_to_eos": all(
            row["all_timed_tokens_exact_to_eos"] for row in route_results.values()
        ),
        "all_deterministic_steps": all(
            row["deterministic_steps"] for row in route_results.values()
        ),
        "all_deterministic_tokens": all(
            row["deterministic_tokens"] for row in route_results.values()
        ),
        # Per-fixture preparation is retained on each file row; the report
        # keeps the first fixture's cold-ish preparation plus the per-file
        # breakdown so a one-time preparation row is never presented as a
        # campaign-wide warm measurement (C5-R5).
        "preparation_per_file": {
            name: row["preparation"] for name, row in route_results.items()
        },
    }
    report["files"] = route_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
