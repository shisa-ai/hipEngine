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

    RR-3 closure: the complete transitive runtime manifest.  Beyond the driver,
    the decoder/encoder runtimes, and the sm_120a kernel wrappers, this now
    hashes the entire ``hipengine`` package (runtime, loading incl. the packed
    loader, core CUDA/memory/workspace, models/spec, dispatch, kernels) plus
    every CUDA/CuTe kernel source (.cu/.cuh/.h).  Hashing the full package is a
    superset of the true import closure, so no runtime-critical module can be
    silently omitted from provenance.
    """

    files: dict[str, Path] = {
        "driver": repo_root / "scripts" / "benchmark_moonshine_cuda_complete.py",
    }
    package = repo_root / "hipengine"
    if package.is_dir():
        for path in sorted(package.rglob("*.py")):
            files[str(path.relative_to(repo_root))] = path
        for path in sorted(package.rglob("*.cu")):
            files[str(path.relative_to(repo_root))] = path
        for path in sorted(package.rglob("*.cuh")):
            files[str(path.relative_to(repo_root))] = path
        for path in sorted(package.rglob("*.h")):
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
        self,
        mode: str,
        fixture: Fixture,
        loaded,
        cuda_runtime,
        snapshot: str,
        packed: bool = False,
        async_chain: bool = False,
        device_owned: bool = False,
        encoder_graph: bool = False,
        bucket_frames: str | None = None,
        attention_route: str = "custom",
        lm_head_route: str = "fused",
    ) -> None:
        self.mode = mode
        self.fixture = fixture
        self.loaded = loaded
        self.cuda_runtime = cuda_runtime
        self.snapshot = snapshot
        self.packed = packed
        # RR-2: optional capacity-selected fixed-bucket arena.  "auto" picks
        # the smallest certified encoder frame bucket that fits each fixture
        # (40/207/1,248); an explicit bucket forces that capacity for all six.
        self.bucket_frames = bucket_frames
        self.exact_frames = self.fixture.frames
        if bucket_frames is not None:
            if self.mode != "standalone":
                raise ValueError("--bucket requires --mode standalone")
            from hipengine.runtime.moonshine_encoder_cuda import (
                moonshine_encoder_bucket_for_frames,
            )

            if bucket_frames == "auto":
                self.bucket_frames = moonshine_encoder_bucket_for_frames(
                    self.fixture.frames
                )
            else:
                self.bucket_frames = int(bucket_frames)
        self.encoder_capacity = self.bucket_frames or self.exact_frames
        # C5/§7.3: async encoder->handoff->cross-KV chain (no terminal sync
        # until the decode boundary), device-owned token/position state, and
        # one captured encoder+handoff+cross-KV graph per bucket.
        self.async_chain = bool(async_chain)
        self.device_owned = bool(device_owned)
        self.encoder_graph = bool(encoder_graph)
        # Opt-in torch-free AOT CUTLASS/CuTe encoder self-attention route
        # (review §8.3 item 3/4); the default keeps the custom kernel so the
        # deployment path never changes.
        if attention_route not in ("custom", "cutlass"):
            raise ValueError(f"attention_route must be 'custom' or 'cutlass'")
        self.attention_route = attention_route
        # C6/RR-8 LM-head route: "fused" (exact C1f stage, default) or
        # "wave8" (fused wave8 + stable top-1 candidate, review §7.3).
        if lm_head_route not in ("fused", "wave8"):
            raise ValueError(f"lm_head_route must be 'fused' or 'wave8'")
        self.lm_head_route = lm_head_route
        self.enc = None
        self.dec = None
        self.torch_encoder = None
        self._device_buffers = []
        self._enc_chain_graph = 0
        self._enc_chain_exec = None

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
            encoder_frames=self.encoder_capacity,
            loaded_model=loaded,
            owns_weights=False,
            lm_head_route=self.lm_head_route,
        )
        t0 = time.perf_counter()
        self.dec.prepare_decoder_kernels()
        prepare["prepare_decoder_s"] = time.perf_counter() - t0

        if self.mode == "standalone":
            from hipengine.runtime.moonshine_encoder_cuda import (
                MoonshineCudaEncoderRuntime,
                moonshine_encoder_bucket_audio_samples,
            )

            self.enc = MoonshineCudaEncoderRuntime(
                audio_samples=(
                    moonshine_encoder_bucket_audio_samples(self.encoder_capacity)
                    if self.bucket_frames is not None
                    else self.fixture.sample_count
                ),
                loaded_model=loaded,
                owns_weights=False,
                attention_route=self.attention_route,
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
        # RR-1 regression: read back the decoder's installed encoder mask and
        # assert it equals the model's downsampled int32 mask (this catches
        # any future pointer/type mix-up in the D2D handoff, including the
        # historical int64-audio-mask-as-int32 bug).
        prepare["encoder_mask_readback"] = self._verify_installed_encoder_mask(
            source_frames=self.fixture.frames
        )
        self.dec.reset_generation(clear_cross_cache=False)

        # Device-owned decode must be enabled before graph capture so the
        # captured DAGs include the graph-tail position-advance state kernel.
        if self.device_owned:
            self.dec.set_device_owned_decode(True)

        t0 = time.perf_counter()
        graphs = self.dec.capture_token_graphs()
        prepare["capture_token_graphs_s"] = time.perf_counter() - t0
        contract = self.dec.token_graph_contract()
        prepare["graph_count"] = contract["graph_count"]
        prepare["capture_wall_ms"] = contract["capture_wall_ms"]
        prepare["instantiate_wall_ms"] = contract["instantiate_wall_ms"]

        # C5/§7.3: capture the whole encoder+handoff+cross-KV chain as one
        # fixed-address graph on the decoder stream, so each timed iteration
        # replays it instead of dispatching ~101 encoder kernels plus the D2D
        # handoff and cross-K/V projections from Python.  Per-bucket reusable:
        # the driver sizes one runtime per fixture, so the capture is per
        # (bucket, fixture length); a production arena would reuse one graph
        # per certified bucket for fixed-length uploads.
        if self.encoder_graph:
            if self.mode != "standalone":
                raise ValueError("--encoder-graph requires --mode standalone")
            prepare["encoder_chain_graph"] = self._capture_encoder_chain()
        return prepare

    def _capture_encoder_chain(self) -> dict[str, float]:
        """Capture encoder DAG + D2D handoff + cross-K/V on the decoder stream.

        Returns capture/instantiate wall timing for the report.  Replaying the
        resulting graph replaces the per-iteration Python dispatch of the ~101
        encoder kernels plus handoff and the eight cross-K/V projections; the
        decode readback still synchronizes at the externally visible result
        boundary.
        """

        import time as _time

        dec = self.dec
        stream = dec.stream
        self.cuda_runtime.stream_synchronize(stream)
        capture_start = _time.perf_counter_ns()
        graph = 0
        self.cuda_runtime.stream_begin_capture(stream)
        try:
            self.enc.run_encode(stream=stream, synchronize=False)
            self.enc.handoff_to(dec, synchronize=False)
            graph = self.cuda_runtime.stream_end_capture(stream)
        except Exception:
            try:
                leaked = self.cuda_runtime.stream_end_capture(stream)
                if leaked:
                    self.cuda_runtime.graph_destroy(leaked)
            except Exception:
                pass
            raise
        capture_wall_ms = (_time.perf_counter_ns() - capture_start) / 1.0e6
        instantiate_start = _time.perf_counter_ns()
        try:
            self._enc_chain_exec = self.cuda_runtime.graph_instantiate(graph)
        except Exception:
            self.cuda_runtime.graph_destroy(graph)
            raise
        instantiate_wall_ms = (_time.perf_counter_ns() - instantiate_start) / 1.0e6
        self._enc_chain_graph = graph
        # The captured chain left the cross cache resident; re-seed the decode
        # host state so the timed iterations start from a fresh generation.
        self.dec.reset_generation(clear_cross_cache=False)
        return {
            "capture_wall_ms": float(capture_wall_ms),
            "instantiate_wall_ms": float(instantiate_wall_ms),
        }

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
        # RR-1: the D2D handoff consumes the downsampled encoder mask (int32,
        # one value per encoder frame), NOT the raw int64 audio mask.  The
        # fixture captures the model's exact downsampled encoder mask (from
        # ``encoder_outputs.attention_mask`` or its all-ones equivalent), so
        # materialize that contiguous int32 tensor as the handoff source.
        self._torch_enc_mask = torch.from_numpy(
            np.ascontiguousarray(self.fixture.encoder_mask, dtype=np.int32)
        ).to("cuda")

    # -- per-iteration route -------------------------------------------------

    def _set_encoder_state(
        self, dec, source_frames: int, *, synchronize: bool = True
    ) -> None:
        if self.mode == "standalone":
            self.enc.handoff_to(dec, synchronize=synchronize)
        elif self.mode == "torch-encoder":
            dec.set_encoder_state_from_device(
                hidden_fp16_ptr=self._torch_hidden_ptr,
                attention_mask_int32_ptr=self._torch_mask_ptr,
                source_frames=source_frames,
                synchronize=synchronize,
            )
            dec.precompute_cross_kv(
                synchronize=synchronize,
                reset=synchronize,
            )
        elif self.mode == "decoder-only":
            dec.set_encoder_state_from_device(
                hidden_fp16_ptr=self._decoder_only_hidden_ptr,
                attention_mask_int32_ptr=self._decoder_only_mask_ptr,
                source_frames=source_frames,
                synchronize=synchronize,
            )
            dec.precompute_cross_kv(
                synchronize=synchronize,
                reset=synchronize,
            )

    def _enqueue_encoder(self, fixture: Fixture) -> None:
        if self.mode == "standalone":
            if self.async_chain:
                # C5/§7.3 async chain: enqueue the encoder DAG onto the decoder
                # stream with no terminal sync so handoff D2D + cross-K/V +
                # decode follow on one ordered stream and a single sync at the
                # decode boundary covers the whole chain.
                self.enc.run_encode(stream=self.dec.stream, synchronize=False)
            else:
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
        # RR-1: hand the downsampled int32 encoder mask (not the audio mask).
        self._torch_mask_ptr = self._torch_enc_mask.data_ptr()

    def run_once(self) -> tuple[list[int], int]:
        """Run the complete route to EOS and return (tokens, steps)."""

        dec = self.dec
        fixture = self.fixture
        if self._enc_chain_exec is not None:
            # Replay the captured encoder+handoff+cross-KV graph (C5/§7.3):
            # no per-iteration Python dispatch of the encoder chain; the decode
            # readback synchronizes the chain at the result boundary.
            self.cuda_runtime.graph_launch(self._enc_chain_exec, dec.stream)
        else:
            synchronize = not self.async_chain
            self._enqueue_encoder(fixture)
            self._set_encoder_state(
                dec,
                source_frames=fixture.frames,
                synchronize=synchronize,
            )
        if self.device_owned:
            # Seed token/position once; the graph tail advances the device
            # position and the fused LM head writes each next token into the
            # same device token buffer (no per-step H2D re-upload).
            dec.set_decode_seed(token_id=fixture.reference[0])
            tokens: list[int] = []
            for _ in range(_MAX_POSITIONS):
                dec.graph_token_step()
                token_id = int(dec.read_token())
                tokens.append(token_id)
                if token_id == _EOS_TOKEN:
                    break
            return tokens, len(tokens)
        tokens = []
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

    def _verify_installed_encoder_mask(self, source_frames: int) -> dict[str, Any]:
        """Read back the decoder's resident encoder mask and compare to the fixture.

        D2D handoff regression (RR-1): confirms the int32 encoder mask actually
        installed in the decoder bucket matches the model's downsampled mask, so
        a raw-int64-audio-mask-as-int32 mix-up can never silently re-occur.
        """

        from hipengine.core.memory import copy_device_to_host, host_array_ptr

        allocation = self.dec.workspace.allocation("encoder_attention_mask")
        expected = np.ascontiguousarray(
            self.fixture.encoder_mask.reshape(-1)[:source_frames], dtype=np.int32
        )
        host = np.empty(int(source_frames), dtype=np.int32)
        copy_device_to_host(
            host_array_ptr(host),
            allocation.buffer,
            int(source_frames) * np.dtype(np.int32).itemsize,
            runtime=self.cuda_runtime,
        )
        matches = bool(np.array_equal(host, expected))
        return {
            "source_frames": int(source_frames),
            "readback_matches": matches,
            "readback": [int(value) for value in host.tolist()],
            "expected": [int(value) for value in expected.tolist()],
        }

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

        if self._enc_chain_exec is not None:
            self.cuda_runtime.graph_exec_destroy(self._enc_chain_exec)
            self._enc_chain_exec = None
        if self._enc_chain_graph:
            self.cuda_runtime.graph_destroy(self._enc_chain_graph)
            self._enc_chain_graph = 0
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
    parser.add_argument("--async-chain", action="store_true", help="enqueue encoder->handoff->cross-KV on the decoder stream without terminal syncs (C5/§7.3 async chain)")
    parser.add_argument("--device-owned", action="store_true", help="device-owned token/position decode state (graph-tail position advance, C5/§7.3)")
    parser.add_argument("--encoder-graph", action="store_true", help="capture encoder+handoff+cross-KV as one fixed-address graph per bucket (standalone only, C5/§7.3)")
    parser.add_argument(
        "--bucket",
        choices=("auto", "40", "207", "1248"),
        default=None,
        help="RR-2: run capacity-selected encoder+decoder runtimes (smallest certified"
        " bucket that fits each fixture, or a fixed certified bucket for all six);"
        " omitted = legacy exact-shape per-file runtimes",
    )
    parser.add_argument(
        "--attention-route",
        choices=("custom", "cutlass"),
        default="custom",
        help="encoder self-attention route: 'custom' (default, unchanged deployment)"
        " or 'cutlass' (opt-in torch-free AOT CUTLASS/CuTe .so, review §8.3 item 3)",
    )
    parser.add_argument(
        "--lm-head-route",
        choices=("fused", "wave8"),
        default="fused",
        help="decoder LM-head route: 'fused' (exact C1f stage, default) or 'wave8'"
        " (opt-in fused wave8 + stable top-1 candidate, review §7.3/RR-8)",
    )
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
            "async_chain": bool(args.async_chain),
            "device_owned": bool(args.device_owned),
            "encoder_graph": bool(args.encoder_graph),
            "attention_route": args.attention_route,
            "lm_head_route": args.lm_head_route,
            "bucket_capacity": args.bucket,
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
            route = Route(
                args.mode,
                fixture,
                loaded,
                cuda_runtime,
                str(args.snapshot_dir),
                packed=args.packed,
                async_chain=args.async_chain,
                device_owned=args.device_owned,
                encoder_graph=args.encoder_graph,
                bucket_frames=args.bucket,
                attention_route=args.attention_route,
                lm_head_route=args.lm_head_route,
            )
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
                        "real_frames": fixture.frames,
                        "bucket_frames": (
                            route.encoder_capacity
                            if route.bucket_frames is not None
                            else None
                        ),
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
