"""One model-owning loop driving MLP-only TP2 Qwen3.8-27B to generated tokens.

This is handoff step 2 of the TP2 plan
(``docs/QWEN38-27B-GFX1100-TP2.md`` Packet 3): a *diagnostic checkpoint* that
runs the full hybrid model - replicated attention/GDN, MLP-only TP2 - to
actual generated tokens on the W7900 + RX 7900 XTX host.

The session owns the model. One loop drives every rank: for each layer, each
rank's runner enqueues its own replicated attention/GDN segment and its own
post-attention residual+norm, the shard group executes that layer's MLP
across both ranks and reduces the f32 partials, and each rank adds the
residual once on its own device-resident hidden state. The control rank (the
first device) owns the final norm, vocabulary head and sampling; the token it
selects is the only token either rank advances. Nothing reads a hidden state
back to the host except the pinned partial copies the staged exchange stages
and the logits the sampler consumes.

Sharding scope (labeled, not implied): attention/GDN weights and compute are
replicated on every rank; only the MLP gate/up/down projections are sharded,
so one staged reduction runs per layer per token and the full replicated
attention/GDN cost is inside the measured walls. Prefill runs token-by-token
through the same per-layer schedule (the shard GEMVs are single-row decode
kernels), so rank-local KV/GDN state is produced by the identical TP2
schedule that decode consumes.

The same composition with ``devices=(d,)`` and no shard group is the matched
TP1 control: the identical layer recipe with the local full-width unfused MLP
chain, on one device. Fresh controls per physical GPU are run by constructing
one session per device.

Arithmetic: attention/GDN take the runner's own eager route (the resolved
add+RMSNorm leaf, plain bf16 residual add, the same GEMV kernels the shard
chain uses); the MLP shard chain resolves its gate/up route through the
shape-qualified fused pair+SiLU policy at the shard shape when the policy row
admits it (bit-identical to the unfused chain per the slice probe), else the
unfused gate GEMV / up GEMV / SiLU-multiply chain. It is a production-profile
candidate, not a strict-parity route: the reduction order changes, so the
binding contract is the full-logit teacher gate against the TP1 model, not
bitwise equality.

Failure semantics: any failure inside a step poisons the session and raises
:class:`TP2GroupError`; the session then refuses every further step. A
partially advanced session is never reused.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.core.device import scoped_current_device
from hipengine.core.hip import get_hip_runtime
from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, malloc
from hipengine.distributed.device_exchange_compiled import CompiledDeviceExchange
from hipengine.distributed.head_shard import materialize_head_shards
from hipengine.distributed.shard_exec import upload_shard_weight
from hipengine.distributed.shard_group import MlpShardGroup
from hipengine.distributed.tp2_prefill import run_sharded_mlp_with_residual
from hipengine.distributed.shard_weights import (
    MLP_TENSORS,
    materialize_mlp_shards,
    resolve_mlp_shard_context,
    upload_mlp_shard_weights,
)
from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION
from hipengine.loading.qwen35_gguf_admission import build_qwen35_gguf_role_manifest
from hipengine.loading.qwen35_gguf_materialize import build_qwen35_gguf_tensor_map
from hipengine.loading.gguf import scan_gguf
from hipengine.kernels.hip_gfx1100.fused.gguf_ops import (
    gguf_bf16_add,
    gguf_rmsnorm_bf16_f32_weight,
)
from hipengine.kernels.hip_gfx1100.convert.cast import f32_to_bf16
from hipengine.kernels.hip_gfx1100.fused.paro_silu import silu_mul_separate_out_bf16
from hipengine.runtime.gguf_linear import (
    GGUF_OUTPUT_F32,
    launch_gguf_linear,
    resident_session_wmma_prefill_default,
)
from hipengine.runtime.gguf_embedding import launch_gguf_embedding
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFFullStackRunner,
    _FullStackScratch,
    _GGUFFullAttentionPrefillScratch,
    _gguf_dense_pair_silu_decode_variant,
    _gguf_norm_residual_decode_kernel,
    resident_prefill_dispatch_session,
)


class TP2GroupError(RuntimeError):
    """The rank group failed; the session is poisoned and not reusable."""


@dataclass
class StepTrace:
    """Host-side walls for one token step.

    ``total_s`` is real: the step ends with a synchronized logits readback,
    so it includes every rank's execution. The per-stage walls are host
    submission times except ``exchange_s``, which blocks until both ranks'
    chains for that layer have completed - the staged schedule's real
    per-layer synchronization cost. They are attribution aids, not device
    times; correctness instrumentation (logit capture, boundary taps) is
    recorded separately and never inside a timed run.
    """

    kind: str
    position: int
    total_s: float
    stages: dict[str, float] = field(default_factory=dict)
    exchange_layers: int = 0


@dataclass
class GenerationResult:
    """One generation's tokens, logits (optional) and per-step traces."""

    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    finished_on_eos: bool
    step_traces: tuple[StepTrace, ...]
    logits: np.ndarray | None = None


def rank_slot_allowlist_from_records(
    records: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    """The rank's resident slot allowlist for a file's role manifest records.

    A ``tp2`` rank's MLP is entirely served by its own shard of
    ``ffn_gate``/``ffn_up``/``ffn_down`` (``_shard_group.forward``), so the
    full-width MLP copy the generic runner would materialize is dead residency:
    it is never read on this path. The only reader of a runner's resident MLP is
    ``_local_mlp``, which runs in ``mode != 'tp2'``. Measured on the supplied
    Q4_K_M GGUF that copy is 9.650 GiB of source weights, so every rank was
    holding half of it (4.825 GiB) for nothing - the shard's own half is
    materialized and uploaded separately.

    The leaves come from the shard module's own tensor list, so the allowlist
    and the shard materializer cannot disagree about what the shard replaces.
    """

    slots = tuple(str(record[0]) for record in records)
    mlp_leaves = {name.split(".")[0] for name in MLP_TENSORS}
    excluded = {slot for slot in slots if slot.rsplit(".", 1)[-1] in mlp_leaves}
    if not excluded:
        raise TP2GroupError(
            "the file's role manifest has no MLP shard leaves to exclude"
        )
    allowed = tuple(slot for slot in slots if slot not in excluded)
    if not allowed:
        raise TP2GroupError("the rank slot allowlist resolved empty")
    return allowed


class MlpTP2GenerationSession:
    """The model-owning TP2 (or matched TP1) generation session."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        devices: Sequence[int] = (0, 1),
        mode: str = "tp2",
        max_sequence_length: int = 2048,
        stage_trace: bool = True,
        driver: str = "compiled",
        schedule: str | None = None,
        reduce_mode: str | None = None,
        head_shard: bool | None = None,
        bulk_prefill: bool = False,
        bulk_prefill_rows: int | None = None,
        use_wmma_prefill: bool | None = None,
        use_gemv_decode: bool | None = None,
        decode_partial_dtype: str = "bf16",
        uneven_split: Any = None,
    ) -> None:
        self.model_path = str(model_path)
        self.mode = str(mode)
        # An :class:`~hipengine.loading.qwen35_gguf_shards.UnevenSplitPolicy`
        # (or a plain ``(fractions, ...)`` spec) gives the MLP projections an
        # explicit per-rank boundary instead of an even one. ``None`` keeps the
        # even split, which is the default until the uneven path has passed the
        # production numerical gate.
        if uneven_split is not None and not hasattr(uneven_split, "ranges"):
            from hipengine.loading.qwen35_gguf_shards import UnevenSplitPolicy

            fractions = (
                tuple(float(v) for v in uneven_split)
                if isinstance(uneven_split, (tuple, list))
                else None
            )
            if fractions is None:
                raise ValueError(
                    f"uneven_split must be an UnevenSplitPolicy or a fraction sequence, "
                    f"got {type(uneven_split).__name__}"
                )
            uneven_split = UnevenSplitPolicy(fractions=fractions)
        self.uneven_split = uneven_split
        if self.mode not in {"tp1", "tp2"}:
            raise ValueError(f"unknown mode {self.mode!r}; expected 'tp1' or 'tp2'")
        if driver not in {"python", "compiled"}:
            raise ValueError(
                f"unknown exchange driver {driver!r}; expected 'python' or 'compiled'"
            )
        if schedule is None:
            # Production default: tp2 sessions run the captured per-layer
            # graph schedule (validated, -46% decode p50 vs eager); the eager
            # per-launch schedule stays the explicit opt-out and the TP1
            # control's schedule.
            schedule = "graphed" if mode == "tp2" else "eager"
        if schedule not in {"eager", "graphed"}:
            raise ValueError(
                f"unknown schedule {schedule!r}; expected 'eager' or 'graphed'"
            )
        if schedule == "graphed" and mode != "tp2":
            raise ValueError("the graphed schedule is tp2-only; the matched TP1 "
                             "control always runs eager")
        if schedule == "graphed" and driver != "compiled":
            raise ValueError(
                "the graphed schedule needs the compiled staged-exchange driver "
                "for fixed per-layer payload slots"
            )
        if reduce_mode is None:
            # The device-side exchange is the graphed schedule's production
            # default (validated: bit-identical teacher gates at fixture and
            # model scale, bounded spin-timeout failure, -9.4% decode p50 vs
            # the host-summed reduction). The host-summed transport reduction
            # stays the eager schedule's path and the explicit opt-out.
            reduce_mode = "device" if schedule == "graphed" else "host"
        if reduce_mode not in {"host", "device"}:
            raise ValueError(
                f"unknown reduce_mode {reduce_mode!r}; expected 'host' or 'device'"
            )
        if reduce_mode == "device" and schedule != "graphed":
            raise ValueError(
                "the device-side reduction needs the graphed schedule; the eager "
                "schedule reduces through the host transport"
            )
        self.schedule = str(schedule)
        self.reduce_mode = str(reduce_mode)
        if head_shard is None:
            # The sharded head is the tp2 production default: bit-identical
            # teacher gates and determinism at fixture and model scale,
            # -3.2% decode p50 vs the replicated head (the single largest
            # per-token weight read now runs concurrently as two shards).
            # head_shard=False is the explicit opt-out and bisection control.
            head_shard = self.mode == "tp2"
        self.head_shard = bool(head_shard)
        if self.head_shard and self.mode != "tp2":
            raise ValueError("the sharded head is tp2-only")
        self._head_plan: Any | None = None
        self._head_weights: dict[int, Any] = {}
        self._head_logits_bufs: dict[int, Any] = {}
        self.driver = str(driver)
        self.devices = tuple(int(d) for d in devices)
        if not self.devices:
            raise ValueError("at least one device is required")
        if self.mode == "tp2" and len(self.devices) < 2:
            raise ValueError("tp2 mode needs two rank devices")
        self.control_device = self.devices[0]
        self.max_sequence_length = int(max_sequence_length)
        self.stage_trace = bool(stage_trace)
        # Opt-in experimental rank-local bulk prefill. Off by default: the
        # token-serial prefill stays the committed route until this candidate
        # passes the production numerical envelope end to end.
        self.bulk_prefill_enabled = bool(bulk_prefill)
        self.bulk_prefill_rows = (
            int(bulk_prefill_rows) if bulk_prefill_rows is not None else None
        )
        if self.bulk_prefill_enabled:
            if self.mode != "tp2":
                raise ValueError("bulk_prefill is tp2-only (it needs the sharded MLP)")
            # ``bulk_prefill_rows`` is an explicit capacity: it is allocated
            # here and the prompt must fit it. Left unset, the workspace is
            # sized to the prompt on first use, which is both leaner and
            # faster (a capacity-sized workspace costs ~7 GiB more per rank at
            # a 512-token prompt and measured 386 against 440 tok/s, because
            # every layer then moves more memory than the prompt has rows).
            if self.bulk_prefill_rows is not None:
                if self.bulk_prefill_rows < 1:
                    raise ValueError("bulk_prefill_rows must be positive")
                if self.bulk_prefill_rows > int(self.max_sequence_length):
                    raise ValueError(
                        f"bulk_prefill_rows {self.bulk_prefill_rows} exceeds the "
                        f"session capacity {self.max_sequence_length}"
                    )
        # The rank-local bulk prefill replaces the resident session's bulk
        # prefill, so it has to resolve the same GGUF linear kernels for the
        # same weight and rows. ``None`` means "the shipped resident policy"
        # (the same resolution ``hipengine.generation.qwen35_gguf`` uses for
        # its resident sessions); an explicit value is a route A/B override.
        # These are read only inside :meth:`bulk_prefill`; the single-row
        # decode route keeps its own explicit per-launch arguments.
        self.use_wmma_prefill = (
            resident_session_wmma_prefill_default()
            if use_wmma_prefill is None
            else bool(use_wmma_prefill)
        )
        self.use_gemv_decode = True if use_gemv_decode is None else bool(use_gemv_decode)

        self.runtime = get_hip_runtime()
        if self.runtime.device_count() < len(self.devices):
            raise TP2GroupError(
                f"need {len(self.devices)} devices, found "
                f"{self.runtime.device_count()}"
            )

        self._poisoned = False
        self._closed = False
        self._runners: dict[int, Qwen35GGUFFullStackRunner] = {}
        self._scratches: dict[int, Any] = {}
        self._extra_buffers: dict[int, list[Any]] = {d: [] for d in self.devices}
        self._shard_group: MlpShardGroup | None = None
        self._bulk_shard_group: MlpShardGroup | None = None
        self._bulk_scratch: dict[int, Any] = {}
        self._bulk_rows = 0
        self._bulk_buffers: dict[int, list[Any]] = {d: [] for d in self.devices}
        self._bulk_chunk_scratch: dict[int, Any] = {}
        self._bulk_hidden: dict[int, tuple[int, int]] = {}
        self._bulk_token_buf: dict[int, Any] = {}
        self._bulk_logits_buf: dict[int, Any] = {}
        self._bulk_host_tokens: np.ndarray | None = None
        self._bulk_final_hidden: dict[int, int] = {}
        self._uploaded_shard_weights: Any | None = None
        self._per_rank_ffn: dict[int, int] | None = None
        self._fused_shard_variant: dict[int, str | None] | None = None
        # Partial dtype of the single-row decode/serial-prefill exchange. The
        # row-parallel down projection rounds its partial once before the
        # staged f32 reduction, so the dtype is an arithmetic choice: ``bf16``
        # is the original schedule, ``f32`` removes that rounding wherever the
        # layer's registered down consumer admits an f32 output. The bulk
        # prefill group keeps ``bf16`` (its rows>1 f32-out leaves are not
        # registered); see docs/REFACTOR.md.
        if decode_partial_dtype not in {"bf16", "f32"}:
            raise ValueError(
                f"unknown decode_partial_dtype {decode_partial_dtype!r}; "
                "expected 'bf16' or 'f32'"
            )
        if decode_partial_dtype == "f32" and self.reduce_mode == "device":
            # The compiled device-exchange driver stages and spin-sums bf16
            # rows (``row_bytes = hidden * 2``), so an f32 partial would be
            # read at half width and summed as bf16 - silent numerical
            # corruption, not a slowdown. The host transport already reduces
            # f32 rows, so require it explicitly; a device-side f32 driver is
            # the tracked follow-up (docs/REFACTOR.md).
            raise ValueError(
                "decode_partial_dtype='f32' needs reduce_mode='host': the "
                "device exchange stages bf16 rows"
            )
        self.decode_partial_dtype = str(decode_partial_dtype)
        self._tp1_mlp_ptrs: dict[int, tuple[int, int, int]] = {}
        # A tp2 rank never reads its runner's full-width MLP (see
        # ``_rank_slot_allowlist``), so the runner is built without those slots
        # instead of materializing them and holding them for the session's
        # lifetime. The tp1 control mode still needs them for ``_local_mlp``.
        self._slot_allowlist: tuple[str, ...] | None = (
            self._rank_slot_allowlist() if self.mode == "tp2" else None
        )
        self._step_buffers: dict[int, Any] = {}
        self._add_norm_cache: dict[int, Any] = {}
        self._layer_graphs: dict[tuple[int, int], int] = {}
        self._layer_execs: dict[tuple[int, int], int] = {}
        self._layer_partials: dict[tuple[int, int], int] = {}
        self._graph_schedule_ready = False
        self._capture_position = max(int(max_sequence_length) - 1, 1)
        self._created_streams: dict[int, int] = {}
        self._device_exchange: CompiledDeviceExchange | None = None
        # Graph capture is not permitted on the legacy default stream, so the
        # graphed schedule creates one non-blocking stream per rank and runs
        # ALL of that rank's work on it; the eager schedule keeps stream 0.
        if schedule == "graphed":
            try:
                for device in self.devices:
                    with scoped_current_device(self.runtime, device):
                        self._created_streams[device] = self.runtime.stream_create()
            except Exception:
                self.close()
                raise

        try:
            for device in self.devices:
                self._build_rank(device)
            self._config = self._runners[self.control_device].weights.config
            if self.mode == "tp2":
                self._build_shard_group()
            if self.head_shard:
                self._build_head_shards()
            for device in self.devices:
                self._alloc_step_buffers(device)
            if self.bulk_prefill_enabled and self.bulk_prefill_rows is not None:
                # An explicit capacity is pre-allocated here, which is how this
                # route was validated. Deferring the *large* workspace instead
                # (a 2048-row one costs ~10 GiB) measured erratic and much
                # slower - 25 to 137 tok/s against 386 when built here. The
                # prompt-sized default is the deferred one.
                self._build_bulk_prefill_workspace(int(self.bulk_prefill_rows))
        except Exception:
            self.close()
            raise

        self._hidden: dict[int, tuple[int, int]] = {
            device: (
                self._step_buffers[device]["hidden_a"].ptr,
                self._step_buffers[device]["hidden_b"].ptr,
            )
            for device in self.devices
        }
        self._token_ids_host = np.empty(1, dtype=np.int64)

    # -- construction ------------------------------------------------------

    def _rank_stream(self, device: int) -> int:
        """The one stream this rank's work runs on (created for graphed)."""

        return self._created_streams.get(int(device), 0)

    def _rank_slot_allowlist(self) -> tuple[str, ...]:
        """Every resident slot except the MLP leaves the shard group supplies.

        A ``tp2`` rank's MLP is entirely served by its own shard of
        ``ffn_gate``/``ffn_up``/``ffn_down`` (``_shard_group.forward``), so the
        full-width MLP copy the generic runner would materialize is dead
        residency: it is never read on this path. The only reader of a runner's
        resident MLP is ``_local_mlp``, which runs in ``mode != 'tp2'``. Measured
        on the supplied Q4_K_M GGUF that copy is 9.650 GiB of source weights, so
        every rank was holding half of it (4.825 GiB) for nothing - the shard's
        own half is still materialized and uploaded separately.

        The allowlist is derived from the file's own role manifest, never from a
        hardcoded geometry, so a different model or layer count stays correct.
        """

        info = scan_gguf(self.model_path)
        manifest = build_qwen35_gguf_role_manifest(
            build_qwen35_gguf_tensor_map(info)
        )
        return rank_slot_allowlist_from_records(manifest.records)

    def _build_rank(self, device: int) -> None:
        """One rank's runner and scratch, entirely inside its device scope."""

        with scoped_current_device(self.runtime, device):
            runner = Qwen35GGUFFullStackRunner(
                self.model_path,
                backend="hip_gfx1100",
                execution_routes=("eager",),
                runtime=self.runtime,
                deferred_device_slots=("root.lm_head",) if self.head_shard else (),
                selected_slots=self._slot_allowlist,
            )
            scratch = _FullStackScratch.allocate(
                runner,
                runtime=self.runtime,
                max_sequence_length=self.max_sequence_length,
            )
            scratch.zero_states(self.runtime)
            self._runners[device] = runner
            self._scratches[device] = scratch

    def _build_shard_group(self) -> None:
        """Materialize and upload every layer's rank shards, then group them."""

        materialization, _context, tensor_plans, config = resolve_mlp_shard_context(
            self.model_path,
            world_size=len(self.devices),
            uneven_split=self.uneven_split,
        )
        shards = materialize_mlp_shards(
            self.model_path,
            world_size=len(self.devices),
            uneven_split=self.uneven_split,
        )
        # Each rank's shard width. An even split is an even division and needs
        # no manifest; an uneven split takes its widths from the manifest,
        # which is the shape authority - never from a re-derived division.
        if self.uneven_split is None:
            width = int(config.feed_forward_length) // len(self.devices)
            per_rank_ffn = {device: width for device in self.devices}
        else:
            if not tensor_plans:
                raise TP2GroupError(
                    "an uneven split needs the shard manifest to place its boundaries"
                )
            gate_plan = tensor_plans[min(tensor_plans)]["ffn_gate"]
            per_rank_ffn = {
                device: int(gate_plan.slice_for(rank).local_shape[0])
                for rank, device in enumerate(self.devices)
            }
        # The MLP gate/up route resolves through the same shape-qualified
        # policy lookup the TP1 runner uses, at *this rank's* shard shape -
        # never a hardcoded variant. An uneven split gives the ranks different
        # shapes, so the lookup is per rank; each rank that the policy admits
        # runs the fused pair+SiLU chain (bit-identical to the unfused chain
        # and faster, per the slice probe), and each rank it does not admits
        # takes the unfused chain.
        fused_variants = {
            device: _gguf_dense_pair_silu_decode_variant(
                self._runners[device],
                rows=1,
                in_features=int(config.hidden_size),
                out_features=per_rank_ffn[device],
            )
            for device in self.devices
        }
        uploaded = upload_mlp_shard_weights(
            self.runtime, shards, devices=self.devices
        )
        self._uploaded_shard_weights = uploaded
        self._per_rank_ffn = per_rank_ffn
        self._fused_shard_variant = fused_variants
        self._shard_group = MlpShardGroup(
            self.runtime,
            devices=self.devices,
            streams={device: self._rank_stream(device) for device in self.devices},
            hidden=int(config.hidden_size),
            per_rank_ffn=per_rank_ffn,
            weights=uploaded,
            # One uniform partial schedule for the decode/serial-prefill group:
            # the layer's registered down consumer decides the dtype, and the
            # exchange reduces in f32 either way. ``f32`` stages the partial
            # unrounded; ``bf16`` rounds each rank's partial first (the shipped
            # schedule - measured against the resident TP1 teacher, the f32
            # partial does not improve agreement; see docs/REFACTOR.md).
            staging_dtype=self.decode_partial_dtype,
            driver=self.driver,
            mlp_decode_variant=fused_variants,
            # Graphed schedules give every layer its own fixed mapped payload
            # slot so a captured graph's deferred consumer (the bf16 cast and
            # residual add folded into the next layer's graph) reads a stable
            # pointer. The eager schedule keeps the two-slot alternation.
            slot_sets=len(self._config.layer_types) if self.schedule == "graphed" else 2,
        )

    def _alloc_step_buffers(self, device: int) -> None:
        """Persistent per-token buffers for one rank, under its device scope."""

        runner = self._runners[device]
        hidden_size = runner.hidden_size
        buffers: dict[str, Any] = {}
        with scoped_current_device(self.runtime, device):
            buffers["token_buf"] = malloc(np.int64().nbytes, runtime=self.runtime)
            buffers["hidden_a"] = malloc(hidden_size * 2, runtime=self.runtime)
            buffers["hidden_b"] = malloc(hidden_size * 2, runtime=self.runtime)
            if device == self.control_device and not self.head_shard:
                buffers["logits_buf"] = malloc(
                    runner.vocab_size * 4, runtime=self.runtime
                )
            if self.mode == "tp1":
                buffers["mlp_gate"] = malloc(runner.ffn_size * 2, runtime=self.runtime)
                buffers["mlp_up"] = malloc(runner.ffn_size * 2, runtime=self.runtime)
                buffers["mlp_act"] = malloc(runner.ffn_size * 2, runtime=self.runtime)
                buffers["mlp_out"] = malloc(runner.hidden_size * 2, runtime=self.runtime)
        self._step_buffers[device] = buffers
        self._extra_buffers[device].extend(buffers.values())

    # -- opt-in rank-local bulk prefill ------------------------------------

    def _build_bulk_prefill_workspace(self, rows: int) -> None:
        """Allocate the per-rank bulk-prefill scratch and a weight-sharing group.

        The bulk MLP group reuses the decode group's uploaded shard weights
        (``owns_weights=False``) so closing the two groups cannot double-free
        them. Every bulk buffer is device-scoped to its own rank and tracked in
        ``_bulk_buffers`` so growing or closing the session frees each exactly
        once - the workspace is sized to the prompt, so it is not a build-time
        allocation.
        """

        rows = int(rows)
        assert rows > 0
        for device in self.devices:
            runner = self._runners[device]
            with scoped_current_device(self.runtime, device):
                scratch = _GGUFFullAttentionPrefillScratch.allocate(
                    runner,
                    rows=rows,
                    capacity=rows,
                    allocate_kv_cache=False,
                    runtime=self.runtime,
                )
                hidden_a = malloc(rows * runner.hidden_size * 2, runtime=self.runtime)
                hidden_b = malloc(rows * runner.hidden_size * 2, runtime=self.runtime)
                token_buf = malloc(rows * np.int64().nbytes, runtime=self.runtime)
                if self.head_shard:
                    logits_buf = malloc(
                        rows * int(self._head_plan.rows_per_rank) * 4,
                        runtime=self.runtime,
                    )
                elif device == self.control_device:
                    logits_buf = malloc(
                        rows * runner.vocab_size * 4, runtime=self.runtime
                    )
                else:
                    logits_buf = None
            self._bulk_scratch[device] = scratch
            self._bulk_hidden[device] = (hidden_a.ptr, hidden_b.ptr)
            self._bulk_token_buf[device] = token_buf
            if logits_buf is not None:
                self._bulk_logits_buf[device] = logits_buf
            self._bulk_buffers[device].extend([hidden_a, hidden_b, token_buf])
            if logits_buf is not None:
                self._bulk_buffers[device].append(logits_buf)
            self._bulk_buffers[device].extend(scratch.buffers)
        self._bulk_rows = rows
        self._bulk_host_tokens = np.empty(rows, dtype=np.int64)
        self._bulk_shard_group = MlpShardGroup(
            self.runtime,
            devices=self.devices,
            streams={device: self._rank_stream(device) for device in self.devices},
            hidden=self.hidden_size,
            per_rank_ffn=self._per_rank_ffn,
            weights=self._uploaded_shard_weights,
            staging_dtype="bf16",
            driver=self.driver,
            mlp_decode_variant=self._fused_shard_variant,
            slot_sets=2,
            rows=rows,
            owns_weights=False,
            reduce_mode=self.reduce_mode,
        )

    def _release_bulk_prefill_workspace(self) -> None:
        """Free the bulk prefill scratch, group and buffers, exactly once.

        The bulk group owns no weights (it shares the decode group's), so
        closing it is safe at any point; the buffers are freed here rather than
        by ``close``'s ``_extra_buffers`` sweep because growing the workspace
        replaces them mid-session.
        """

        if self._bulk_shard_group is not None:
            try:
                self._bulk_shard_group.close()
            except Exception:  # noqa: BLE001 - teardown continues
                pass
            self._bulk_shard_group = None
        for scratch in self._bulk_scratch.values():
            for buffer in getattr(scratch, "full_attn_split_growth_buffers", ()) or ():
                try:
                    free(buffer, runtime=self.runtime)
                except Exception:  # noqa: BLE001 - teardown continues
                    pass
        self._bulk_scratch.clear()
        self._bulk_chunk_scratch.clear()
        self._bulk_hidden.clear()
        self._bulk_token_buf.clear()
        self._bulk_logits_buf.clear()
        for device, buffers in self._bulk_buffers.items():
            for buffer in buffers:
                try:
                    with scoped_current_device(self.runtime, device):
                        free(buffer, runtime=self.runtime)
                except Exception:  # noqa: BLE001 - teardown continues
                    pass
            buffers.clear()
        self._bulk_host_tokens = None
        self._bulk_rows = 0

    def _ensure_bulk_prefill_workspace(self, rows: int) -> None:
        """Make the bulk workspace big enough for ``rows`` active rows.

        Called before every bulk prefill. The first call allocates, a later
        longer prompt grows it (the old workspace is released first), and
        repeat prompts at or below the current size reuse it. With an explicit
        ``bulk_prefill_rows`` the caller has asked for a fixed capacity and
        that is what gets allocated.
        """

        rows = int(rows)
        if rows < 1:
            raise ValueError("rows must be positive")
        target = int(self.bulk_prefill_rows) if self.bulk_prefill_rows is not None else rows
        target = max(target, rows)
        if self._bulk_shard_group is not None and self._bulk_rows >= target:
            return
        self._release_bulk_prefill_workspace()
        self._build_bulk_prefill_workspace(target)

    def bulk_prefill(
        self, token_ids: Sequence[int], *, logits_rows: int | None = None
    ) -> np.ndarray:
        """Whole-prompt rank-local bulk prefill; returns ``(rows, vocab)`` logits.

        Experimental opt-in candidate. Each layer runs the attention/GDN
        helper, the post-attention norm+residual helper, the batched sharded
        MLP exchange and one residual add on every rank, then the final norm
        and head. The prompt must fit ``bulk_prefill_rows``; chunked bulk
        prefill is not implemented, so an over-capacity prompt is rejected
        before any allocation or launch. Every call zeroes the KV/GDN state
        first, so it is repeatable and never inherits a previous sequence.

        ``logits_rows`` projects the head for only the trailing N prompt rows
        and returns that many rows. The product path needs the last row alone,
        which is what the single-card and token-serial routes compute; the
        teacher-forced diagnostic needs every row and leaves it unset.
        """

        self._require_live()
        if not self.bulk_prefill_enabled:
            raise TP2GroupError("bulk prefill is not enabled on this session")
        tokens = tuple(int(token) for token in token_ids)
        if not tokens:
            raise ValueError("token_ids must not be empty")
        rows = len(tokens)
        capacity = (
            int(self.bulk_prefill_rows)
            if self.bulk_prefill_rows is not None
            else int(self.max_sequence_length)
        )
        if rows > capacity:
            raise ValueError(
                f"prompt of {rows} tokens exceeds the bulk prefill capacity "
                f"{capacity}; chunked bulk prefill is not implemented"
            )
        for token in tokens:
            if token < 0 or token >= self.vocab_size:
                raise ValueError(
                    f"token_id {token} outside [0, {self.vocab_size})"
                )
        # Validated here rather than in the launcher so a bad argument is a
        # plain ValueError before any allocation, and does not poison the
        # session through the launch-failure handler.
        if logits_rows is not None and not 1 <= int(logits_rows) <= rows:
            raise ValueError(
                f"logits_rows {int(logits_rows)} is outside [1, {rows}]"
            )
        # Sized to the prompt unless the caller pinned a capacity. This is the
        # first thing the route does, so a session that never bulk-prefills
        # never pays for the workspace.
        self._ensure_bulk_prefill_workspace(rows)
        # Capture/warmup zeroes the state and rewinds positions, so build the
        # decode schedule BEFORE writing the prefill state - otherwise the
        # capture would overwrite the newly computed KV/GDN state.
        self._ensure_graph_schedule()
        try:
            for device in self.devices:
                with scoped_current_device(self.runtime, device):
                    self._scratches[device].zero_states(
                        self.runtime, stream=self._rank_stream(device)
                    )
            assert self._bulk_host_tokens is not None
            self._bulk_host_tokens[:rows] = tokens
            for device in self.devices:
                runner = self._runners[device]
                with scoped_current_device(self.runtime, device):
                    copy_host_to_device(
                        self._bulk_token_buf[device],
                        self._bulk_host_tokens.ctypes.data,
                        rows * np.int64().nbytes,
                        runtime=self.runtime,
                    )
                    launch_gguf_embedding(
                        runner.weights.root("token_embedding"),
                        self._bulk_token_buf[device].ptr,
                        self._bulk_hidden[device][0],
                        rows=rows,
                        hidden_size=runner.hidden_size,
                        vocab_size=runner.vocab_size,
                        stream=self._rank_stream(device),
                        runtime=self.runtime,
                    )
            # The resident bulk caller narrows the capacity-sized scratch to
            # the active chunk with ``for_chunk`` before every layer. The
            # full-attention helper reads its row count from ``scratch.rows``
            # (it has no explicit ``rows`` argument), so without this the
            # full-attention layers would run at the whole bulk capacity
            # instead of the prompt length.
            for device in self.devices:
                with scoped_current_device(self.runtime, device):
                    self._bulk_chunk_scratch[device] = self._bulk_scratch[
                        device
                    ].for_chunk(
                        0,
                        rows,
                        rows,
                        runtime=self.runtime,
                        stream=self._rank_stream(device),
                    )
            self._run_bulk_prefill_layers(rows)
            logits = self._finish_bulk_prefill(rows, project_rows=logits_rows)
        except Exception as error:
            self._poisoned = True
            raise TP2GroupError(
                f"bulk prefill failed at {rows} rows: "
                f"{type(error).__name__}: {error}"
            ) from error
        return logits

    def _run_bulk_prefill_layers(self, rows: int) -> None:
        src = {device: self._bulk_hidden[device][0] for device in self.devices}
        dst = {device: self._bulk_hidden[device][1] for device in self.devices}
        group = self._bulk_shard_group
        # A device-reduced group enqueues every layer without a host wait, so it
        # needs its spin-timeout flags cleared once here and one wait at the end
        # (which also surfaces a timeout as a failure rather than leaving a
        # stale boundary row in place). The staged route is a no-op on both.
        if group is not None:
            group.begin_device_group()
        for layer_id, layer_type in enumerate(self._config.layer_types):
            self._bulk_attention_layer(layer_id, layer_type, src, rows)
            self._bulk_norm_residual_layer(layer_id, src, rows)
            self._bulk_sharded_mlp_layer(layer_id, src, dst, rows)
            src, dst = dst, src
        if group is not None:
            group.finish_device_group()
        self._bulk_final_hidden = dict(src)

    def _bulk_attention_layer(
        self,
        layer_id: int,
        layer_type: str,
        src: Mapping[int, int],
        rows: int,
    ) -> None:
        for device in self.devices:
            runner = self._runners[device]
            scratch = self._bulk_chunk_scratch.get(
                device, self._bulk_scratch[device]
            )
            decode_scratch = self._scratches[device]
            stream = self._rank_stream(device)
            with (
                scoped_current_device(self.runtime, device),
                resident_prefill_dispatch_session(
                    runner,
                    prompt_tokens=rows,
                    use_wmma_prefill=self.use_wmma_prefill,
                    use_gemv_decode=self.use_gemv_decode,
                ),
            ):
                if layer_type == LINEAR_ATTENTION:
                    runner._run_linear_attention_prefill_attn_rows(
                        layer_id,
                        int(src[device]),
                        scratch,
                        rows=rows,
                        decode_scratch=decode_scratch,
                        stream=stream,
                    )
                elif layer_type == FULL_ATTENTION:
                    key_cache, value_cache = decode_scratch.full_cache(layer_id)
                    layer_scratch = replace(
                        scratch,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        retained_key_cache=None,
                        retained_value_cache=None,
                        retained_append_spans=None,
                        int8_kv_value_bf16=False,
                    )
                    runner._run_full_attention_prefill_attn_rows(
                        layer_id,
                        int(src[device]),
                        layer_scratch,
                        cos_table_ptr=decode_scratch.cos_table_buf.ptr,
                        sin_table_ptr=decode_scratch.sin_table_buf.ptr,
                        max_positions=int(decode_scratch.max_positions),
                        stream=stream,
                    )
                else:
                    raise TP2GroupError(
                        f"unsupported GGUF layer type {layer_type!r}"
                    )

    def _bulk_norm_residual_layer(
        self,
        layer_id: int,
        src: Mapping[int, int],
        rows: int,
    ) -> None:
        for device in self.devices:
            runner = self._runners[device]
            scratch = self._bulk_chunk_scratch.get(
                device, self._bulk_scratch[device]
            )
            with (
                scoped_current_device(self.runtime, device),
                resident_prefill_dispatch_session(
                    runner,
                    prompt_tokens=rows,
                    use_wmma_prefill=self.use_wmma_prefill,
                    use_gemv_decode=self.use_gemv_decode,
                ),
            ):
                runner._run_post_attention_norm_residual_rows(
                    layer_id,
                    int(src[device]),
                    scratch.attn_out.ptr,
                    scratch,
                    rows=rows,
                    stream=self._rank_stream(device),
                )

    def _bulk_sharded_mlp_layer(
        self,
        layer_id: int,
        src: Mapping[int, int],
        dst: Mapping[int, int],
        rows: int,
    ) -> None:
        group = self._bulk_shard_group
        assert group is not None
        hidden = self.hidden_size

        def add_residual(
            device: int,
            residual_ptr: int,
            mlp_out_ptr: int,
            out_ptr: int,
            layer_rows: int,
        ) -> None:
            with scoped_current_device(self.runtime, device):
                gguf_bf16_add(
                    residual_ptr,
                    mlp_out_ptr,
                    out_ptr,
                    layer_rows * hidden,
                    stream=self._rank_stream(device),
                    runtime=self.runtime,
                )

        # The shard chain launches its own GGUF linears through
        # ``MlpShardRank.forward_partial``, so it needs the same session-scoped
        # owners. Those six owners are device-free and resolve from
        # ``(backend, geometry, rows)``, which are identical on every rank of
        # this group, so one entry around the group forward sets exactly the
        # same context the per-rank loops above set. They hold no device
        # pointers, so there is nothing to leak into the other rank's device
        # scope; the group's own ``scoped_current_device`` stays per rank.
        with resident_prefill_dispatch_session(
            self._runners[self.control_device],
            prompt_tokens=rows,
            use_wmma_prefill=self.use_wmma_prefill,
            use_gemv_decode=self.use_gemv_decode,
        ):
            run_sharded_mlp_with_residual(
                group,
                layer_id=layer_id,
                rows=rows,
                post_norm_ptrs={
                    device: self._bulk_chunk_scratch.get(
                        device, self._bulk_scratch[device]
                    ).post_norm.ptr
                    for device in self.devices
                },
                residual_ptrs={
                    device: self._bulk_chunk_scratch.get(
                        device, self._bulk_scratch[device]
                    ).residual.ptr
                    for device in self.devices
                },
                out_ptrs={device: int(dst[device]) for device in self.devices},
                add_residual=add_residual,
            )

    def _finish_bulk_prefill(
        self, rows: int, *, project_rows: int | None = None
    ) -> np.ndarray:
        """Project the head for the trailing ``project_rows`` rows (default: all).

        Picking the next token needs the last row's logits and nothing else,
        which is what both the single-card route (``_sample_from_hidden`` at
        rows=1) and this session's own token-serial route compute. The head is
        a ``hidden x vocab`` matrix, so projecting every prompt row costs 512x
        the arithmetic and 512x the device-to-host traffic for logits no caller
        reads. ``None`` keeps the all-rows behaviour the teacher-forced
        diagnostic needs.
        """

        rows = int(rows)
        want = rows if project_rows is None else int(project_rows)
        if want < 1 or want > rows:
            raise ValueError(f"project_rows {want} is outside [1, {rows}]")
        # The projected rows are the last ``want`` of the pre-norm hidden, so
        # the norm and the projection both start at this row offset.
        offset_rows = rows - want
        src = self._bulk_final_hidden
        if self.head_shard:
            plan = self._head_plan
            shard_rows = int(plan.rows_per_rank)
            for device in self.devices:
                runner = self._runners[device]
                scratch = self._bulk_scratch[device]
                stream = self._rank_stream(device)
                # Deliberately NOT inside ``resident_prefill_dispatch_session``:
                # the head projection is not a resident layer helper. The
                # resident session samples through its dedicated ``lm_head``
                # kernel (``_sample_from_hidden``), and the Q6_K planar
                # ``t16_wmma_prefill_bf16_f32_out`` variant this route would
                # select is not registered - it silently falls back to the CPU
                # reference. See docs/REFACTOR.md.
                with scoped_current_device(self.runtime, device):
                    gguf_rmsnorm_bf16_f32_weight(
                        int(src[device]) + offset_rows * runner.hidden_size * 2,
                        runner.weights.root("output_norm").allocation().tensor.ptr,
                        scratch.norm.ptr,
                        want,
                        runner.hidden_size,
                        runner.weights.config.rms_norm_eps,
                        stream=stream,
                        runtime=self.runtime,
                    )
                    launch_gguf_linear(
                        self._head_weights[device],
                        scratch.norm.ptr,
                        self._bulk_logits_buf[device].ptr,
                        want,
                        runner.hidden_size,
                        shard_rows,
                        output_dtype=GGUF_OUTPUT_F32,
                        stream=stream,
                        runtime=self.runtime,
                    )
            logits = np.empty((want, int(plan.vocab_rows)), dtype="<f4")
            for device in self.devices:
                stream = self._rank_stream(device)
                with scoped_current_device(self.runtime, device):
                    if stream:
                        self.runtime.stream_synchronize(stream)
                    # A column slice of the 2-D logits array is not contiguous,
                    # so read each rank's shard into a contiguous buffer first.
                    chunk = np.empty((want, shard_rows), dtype="<f4")
                    copy_device_to_host(
                        chunk.ctypes.data,
                        self._bulk_logits_buf[device],
                        want * shard_rows * 4,
                        runtime=self.runtime,
                    )
                    start = int(plan.rank_row_start(device))
                    logits[:, start:start + shard_rows] = chunk
            return logits
        device = self.control_device
        runner = self._runners[device]
        scratch = self._bulk_scratch[device]
        stream = self._rank_stream(device)
        # See the head_shard branch above: the head is not a resident layer
        # helper and must keep its registered f32-out route.
        with scoped_current_device(self.runtime, device):
            gguf_rmsnorm_bf16_f32_weight(
                int(src[device]) + offset_rows * runner.hidden_size * 2,
                runner.weights.root("output_norm").allocation().tensor.ptr,
                scratch.norm.ptr,
                want,
                runner.hidden_size,
                runner.weights.config.rms_norm_eps,
                stream=stream,
                runtime=self.runtime,
            )
            launch_gguf_linear(
                runner.weights.root("lm_head"),
                scratch.norm.ptr,
                self._bulk_logits_buf[device].ptr,
                want,
                runner.hidden_size,
                runner.vocab_size,
                output_dtype=GGUF_OUTPUT_F32,
                stream=stream,
                runtime=self.runtime,
            )
            if stream:
                self.runtime.stream_synchronize(stream)
        logits = np.empty((want, runner.vocab_size), dtype="<f4")
        copy_device_to_host(
            logits.ctypes.data,
            self._bulk_logits_buf[device],
            want * runner.vocab_size * 4,
            runtime=self.runtime,
        )
        return logits

    # -- state -------------------------------------------------------------

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def hidden_size(self) -> int:
        return int(self._runners[self.control_device].hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self._runners[self.control_device].vocab_size)

    @property
    def prefill_schedule(self) -> str:
        """Prefill arithmetic schedule; TP2 drives token-by-token prefill only.

        This is an ownership/provenance fact, not a performance claim. It exists
        so a teacher-forced comparison cannot silently mix a bulk prefill teacher
        with a token-serial candidate and misattribute the schedule difference to
        a distributed numerical failure.
        """

        return 'bulk-tp2' if self.bulk_prefill_enabled else 'token-serial'

    def reset(self) -> None:
        """Zero every rank's KV/GDN state and rewind positions to zero."""

        self._require_live()
        for device in self.devices:
            with scoped_current_device(self.runtime, device):
                self._scratches[device].zero_states(self.runtime)

    # -- generation --------------------------------------------------------

    def generate(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int = 32,
        eos_token_id: int | None = None,
        capture_logits: bool = False,
    ) -> GenerationResult:
        """Greedy-generate tokens after a TP2/TP1 prefill.

        Each call is a fresh sequence: every rank's scratch state (KV spans,
        GDN conv/recurrent state) is zeroed first, so back-to-back calls on
        one session never inherit the previous call's state.

        When the session was built with ``bulk_prefill=True`` the prompt is
        consumed by the rank-local bulk prefill candidate in one shot and the
        loop below starts at the first decode position; otherwise the prompt is
        walked token by token, which is the committed schedule.
        """

        self._require_live()
        for device, scratch in self._scratches.items():
            with scoped_current_device(self.runtime, device):
                scratch.zero_states(self.runtime)
        prompt = tuple(int(t) for t in prompt_token_ids)
        if not prompt:
            raise ValueError("prompt_token_ids must not be empty")
        if len(prompt) + max(0, int(max_new_tokens)) > self.max_sequence_length:
            raise ValueError(
                f"prompt ({len(prompt)}) plus max_new_tokens ({max_new_tokens}) "
                f"exceeds the session's {self.max_sequence_length}-token capacity"
            )
        traces: list[StepTrace] = []
        logits_rows: list[np.ndarray] = []
        generated: list[int] = []
        finished_on_eos = False
        next_token: int | None = None
        total_positions = len(prompt) + int(max_new_tokens)
        self._ensure_graph_schedule()
        first_decode_position = 0
        if self.bulk_prefill_enabled:
            # One shot for the whole prompt. ``bulk_prefill`` zeroes state,
            # captures/warms the decode schedule and writes the prompt's KV and
            # GDN state, so the decode loop below continues from it exactly as
            # it would from a token-serial prefill. The whole prompt is one
            # prefill step in the trace, which is what a rate measured from
            # ``step_traces`` must see.
            started = time.perf_counter()
            bulk_logits = self.bulk_prefill(prompt, logits_rows=1)
            traces.append(
                StepTrace(
                    kind="prefill",
                    position=len(prompt) - 1,
                    total_s=time.perf_counter() - started,
                )
            )
            next_token = int(np.argmax(np.asarray(bulk_logits[-1]).reshape(-1)))
            first_decode_position = len(prompt)
        for position in range(first_decode_position, total_positions):
            if position < len(prompt):
                token_id = prompt[position]
                kind = "prefill"
            else:
                assert next_token is not None
                token_id = next_token
                kind = "decode"
                generated.append(token_id)
                if eos_token_id is not None and token_id == int(eos_token_id):
                    finished_on_eos = True
                    break
            step_logits, trace = self._forward_token(token_id, position, kind=kind)
            next_token = int(np.argmax(step_logits))
            if capture_logits and kind == "decode":
                logits_rows.append(step_logits)
            traces.append(trace)
        return GenerationResult(
            prompt_token_ids=prompt,
            token_ids=tuple(generated),
            finished_on_eos=finished_on_eos,
            step_traces=tuple(traces),
            logits=(
                np.stack(logits_rows, axis=0) if logits_rows else None
            ),
        )

    def teacher_forced_logits(
        self,
        token_ids: Sequence[int],
    ) -> np.ndarray:
        """Full-vocabulary logits for every teacher-forced position.

        Diagnostics only: this reads a full logit row to the host per position
        and must never wrap a timed run. Each call is a fresh sequence: every
        rank's scratch state is zeroed first, so the logits never inherit a
        previous generation's GDN/KV state.
        """

        self._require_live()
        if self.bulk_prefill_enabled:
            return self.bulk_prefill(token_ids)
        for device, scratch in self._scratches.items():
            with scoped_current_device(self.runtime, device):
                scratch.zero_states(self.runtime, stream=self._rank_stream(device))
        tokens = tuple(int(t) for t in token_ids)
        if not tokens:
            raise ValueError("token_ids must not be empty")
        if len(tokens) > self.max_sequence_length:
            raise ValueError("token_ids exceed the session capacity")
        self._ensure_graph_schedule()
        rows: list[np.ndarray] = []
        for position, token_id in enumerate(tokens):
            logits, _trace = self._forward_token(token_id, position, kind="prefill")
            rows.append(logits)
        return np.stack(rows, axis=0)

    # -- one token step ----------------------------------------------------

    def _forward_token(
        self,
        token_id: int,
        position: int,
        *,
        kind: str,
    ) -> tuple[np.ndarray, StepTrace]:
        if self._poisoned:
            raise TP2GroupError("the session is poisoned by an earlier failure")
        if position >= self.max_sequence_length:
            raise TP2GroupError(
                f"position {position} exceeds the {self.max_sequence_length}-token capacity"
            )
        if self.schedule == "graphed" and self._graph_schedule_ready:
            return self._forward_token_graphed(token_id, position, kind=kind)
        return self._forward_token_eager(token_id, position, kind=kind)

    def _forward_token_eager(
        self,
        token_id: int,
        position: int,
        *,
        kind: str,
    ) -> tuple[np.ndarray, StepTrace]:
        if self._poisoned:
            raise TP2GroupError("the session is poisoned by an earlier failure")
        started = time.perf_counter()
        stages: dict[str, float] = {}
        exchanges_before = (
            len(self._shard_group.exchange_walls_s)
            if self._shard_group is not None
            else 0
        )
        try:
            hidden_ptrs = self._enqueue_embedding(token_id, position, stages)
            for layer_id, layer_type in enumerate(self._config.layer_types):
                self._enqueue_layer(layer_id, layer_type, hidden_ptrs, position, stages)
            logits = self._finish_step(stages)
        except Exception as error:
            self._poisoned = True
            raise TP2GroupError(
                f"{self.mode} rank group failed at position {position}: "
                f"{type(error).__name__}: {error}"
            ) from error
        total = time.perf_counter() - started
        trace = StepTrace(
            kind=kind,
            position=position,
            total_s=total,
            stages=stages if self.stage_trace else {},
            exchange_layers=(
                len(self._shard_group.exchange_walls_s) - exchanges_before
                if self._shard_group is not None
                else 0
            ),
        )
        return logits, trace

    def _enqueue_embedding(
        self,
        token_id: int,
        position: int,
        stages: dict[str, float],
    ) -> dict[int, tuple[int, int]]:
        mark = time.perf_counter()
        self._token_ids_host[0] = int(token_id)
        for device in self.devices:
            with scoped_current_device(self.runtime, device):
                scratch = self._scratches[device]
                runner = self._runners[device]
                scratch.set_full_attention_position(position, self.runtime)
                buffers = self._step_buffers[device]
                copy_host_to_device(
                    buffers["token_buf"],
                    self._token_ids_host.ctypes.data,
                    runtime=self.runtime,
                )
                launch_gguf_embedding(
                    runner.weights.root("token_embedding"),
                    buffers["token_buf"].ptr,
                    buffers["hidden_a"].ptr,
                    rows=1,
                    hidden_size=runner.hidden_size,
                    vocab_size=runner.vocab_size,
                    stream=self._rank_stream(device),
                    runtime=self.runtime,
                )
        stages["embedding"] = time.perf_counter() - mark
        return dict(self._hidden)

    def _enqueue_layer(
        self,
        layer_id: int,
        layer_type: str,
        hidden_ptrs: Mapping[int, tuple[int, int]],
        position: int,
        stages: dict[str, float],
    ) -> None:
        mark = time.perf_counter()
        for device in self.devices:
            src, _dst = hidden_ptrs[device]
            stream = self._rank_stream(device)
            with scoped_current_device(self.runtime, device):
                runner = self._runners[device]
                scratch = self._scratches[device]
                attn_out = scratch.attn_out.ptr
                if layer_type == LINEAR_ATTENTION:
                    runner._run_linear_attention_attn_only(
                        layer_id, src, attn_out, scratch, stream=stream
                    )
                elif layer_type == FULL_ATTENTION:
                    runner._run_full_attention_attn_only(
                        layer_id, src, attn_out, scratch, position=position, stream=stream
                    )
                else:
                    raise TP2GroupError(
                        f"unsupported GGUF layer type {layer_type!r}"
                    )
        stages["attention"] = stages.get("attention", 0.0) + (
            time.perf_counter() - mark
        )

        mark = time.perf_counter()
        for device in self.devices:
            src, _dst = hidden_ptrs[device]
            stream = self._rank_stream(device)
            with scoped_current_device(self.runtime, device):
                runner = self._runners[device]
                scratch = self._scratches[device]
                self._add_norm_kernel(runner)(
                    src,
                    scratch.attn_out.ptr,
                    runner.weights.layer(layer_id)
                    .weight("post_attention_norm")
                    .allocation()
                    .tensor.ptr,
                    scratch.post_norm.ptr,
                    scratch.residual.ptr,
                    1,
                    runner.hidden_size,
                    runner.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=self.runtime,
                )
        stages["add_norm"] = stages.get("add_norm", 0.0) + (
            time.perf_counter() - mark
        )

        mark = time.perf_counter()
        if self.mode == "tp2":
            assert self._shard_group is not None
            outputs = self._shard_group.forward(
                layer_id,
                {
                    device: self._scratches[device].post_norm.ptr
                    for device in self.devices
                },
            )
        else:
            outputs = {
                device: self._local_mlp(device, layer_id) for device in self.devices
            }
        stages["mlp"] = stages.get("mlp", 0.0) + (time.perf_counter() - mark)

        mark = time.perf_counter()
        for device in self.devices:
            src, dst = hidden_ptrs[device]
            with scoped_current_device(self.runtime, device):
                scratch = self._scratches[device]
                gguf_bf16_add(
                    scratch.residual.ptr,
                    outputs[device],
                    dst,
                    self.hidden_size,
                    stream=self._rank_stream(device),
                    runtime=self.runtime,
                )
        stages["residual_add"] = stages.get("residual_add", 0.0) + (
            time.perf_counter() - mark
        )
        # Swap both ranks' hidden buffers: the summed output becomes the next
        # layer's residual-chain input on every rank.
        for device in self.devices:
            src, dst = hidden_ptrs[device]
            hidden_ptrs[device] = (dst, src)

    # -- graphed schedule --------------------------------------------------

    def _ensure_graph_schedule(self) -> None:
        """Build the per-layer captured graphs once, on first use.

        The schedule is captured eagerly first (one warmup token to force
        every lazy allocation and JIT build), then each (layer, rank) pair's
        capturable segment - the prior layer's deferred bf16 cast and residual
        add, attention, add+norm, the D2D input copy and the fused shard
        chain - is recorded into one instantiated graph. The transport
        reduction stays host-driven between graph segments.
        """

        if self.schedule != "graphed" or self._graph_schedule_ready:
            return
        if self.mode != "tp2" or self._shard_group is None:
            raise TP2GroupError("the graphed schedule is tp2-only")
        try:
            self._build_graph_schedule()
        except Exception as error:
            self._poisoned = True
            raise TP2GroupError(
                f"graph schedule capture failed: {type(error).__name__}: {error}"
            ) from error

    def _build_graph_schedule(self) -> None:
        capture_position = self._capture_position
        # The device-side exchange serves one slot per layer; the config is
        # only known after the ranks are built.
        if self.reduce_mode == "device" and self._device_exchange is None:
            self._device_exchange = CompiledDeviceExchange(
                self.runtime,
                devices=self.devices,
                streams={
                    device: self._created_streams[device] for device in self.devices
                },
                num_layers=len(self._config.layer_types),
                hidden=self.hidden_size,
            )
        # Warmup: one eager token forces every lazy allocation (split-decode
        # rows, launcher workspaces) and JIT build, so nothing allocates or
        # compiles inside a capture.
        self._forward_token_eager(0, 0, kind="prefill")
        group = self._shard_group
        assert group is not None
        group.reset_exchange_walls()
        for device in self.devices:
            with scoped_current_device(self.runtime, device):
                self.runtime.device_synchronize()
                self._scratches[device].zero_states(
                    self.runtime, stream=self._rank_stream(device)
                )
                self._scratches[device].set_full_attention_position(
                    capture_position, self.runtime
                )
        try:
            for layer_id, layer_type in enumerate(self._config.layer_types):
                for device in self.devices:
                    self._capture_layer_graph(
                        layer_id, layer_type, device, capture_position
                    )
        except Exception:
            self._destroy_graphs()
            raise
        self._graph_schedule_ready = True

    def _capture_layer_graph(
        self,
        layer_id: int,
        layer_type: str,
        device: int,
        capture_position: int,
    ) -> None:
        group = self._shard_group
        assert group is not None
        runner = self._runners[device]
        scratch = self._scratches[device]
        src, dst = self._hidden[device]
        if layer_id % 2 == 1:
            src, dst = dst, src
        out_buf = group.output_ptr(device)
        runtime = self.runtime
        stream = self._rank_stream(device)
        with scoped_current_device(self.runtime, device):
            runtime.stream_begin_capture(stream)
            try:
                # The prior layer's reduced row. Device mode: this rank's
                # exchange for the prior slot (stage own partial, publish the
                # flag, spin-sum the other rank's staged row) writes the bf16
                # output directly - no host wait. Host mode: cast the fixed
                # mapped payload slot the host transport published. Pointers
                # are stable either way: one slot per layer.
                if layer_id > 0:
                    prior_partial = self._layer_partials[(layer_id - 1, device)]
                    if self.reduce_mode == "device":
                        rank_index = self.devices.index(device)
                        self._device_exchange.enqueue_rank(
                            rank_index,
                            prior_partial,
                            layer_id - 1,
                            out_buf,
                        )
                    else:
                        prior_reduced = group.reduced_payload_ptr(layer_id - 1)
                        f32_to_bf16(
                            prior_reduced,
                            out_buf,
                            self.hidden_size,
                            stream=stream,
                            runtime=runtime,
                        )
                    prior_residual = scratch.residual.ptr
                    gguf_bf16_add(
                        prior_residual,
                        out_buf,
                        src,
                        self.hidden_size,
                        stream=stream,
                        runtime=runtime,
                    )
                attn_out = scratch.attn_out.ptr
                if layer_type == LINEAR_ATTENTION:
                    runner._run_linear_attention_attn_only(
                        layer_id, src, attn_out, scratch, stream=stream
                    )
                elif layer_type == FULL_ATTENTION:
                    runner._run_full_attention_attn_only(
                        layer_id,
                        src,
                        attn_out,
                        scratch,
                        position=capture_position,
                        stream=stream,
                    )
                else:
                    raise TP2GroupError(
                        f"unsupported GGUF layer type {layer_type!r}"
                    )
                self._add_norm_kernel(runner)(
                    src,
                    scratch.attn_out.ptr,
                    runner.weights.layer(layer_id)
                    .weight("post_attention_norm")
                    .allocation()
                    .tensor.ptr,
                    scratch.post_norm.ptr,
                    scratch.residual.ptr,
                    1,
                    runner.hidden_size,
                    runner.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=runtime,
                )
                partial = group.enqueue_rank_chain(
                    layer_id, device, scratch.post_norm.ptr
                )
            except Exception:
                leaked = runtime.stream_end_capture(stream)
                if leaked:
                    runtime.graph_destroy(leaked)
                raise
            graph = runtime.stream_end_capture(stream)
            try:
                graph_exec = runtime.graph_instantiate(graph)
            except Exception:
                runtime.graph_destroy(graph)
                raise
        self._layer_graphs[(layer_id, device)] = graph
        self._layer_execs[(layer_id, device)] = graph_exec
        self._layer_partials[(layer_id, device)] = partial

    def _forward_token_graphed(
        self,
        token_id: int,
        position: int,
        *,
        kind: str,
    ) -> tuple[np.ndarray, StepTrace]:
        started = time.perf_counter()
        stages: dict[str, float] = {}
        exchanges_before = (
            len(self._shard_group.exchange_walls_s) if self._shard_group else 0
        )
        group = self._shard_group
        assert group is not None
        try:
            # Eager per-token metadata: token id and the pinned position/
            # context refresh the captured kernels read through device
            # tensors, then the token embedding. The device exchange's step
            # counters bump once per rank, before any graph reads them.
            if self.reduce_mode == "device":
                self._device_exchange.step_begin()
            self._enqueue_embedding(token_id, position, stages)
            mark = time.perf_counter()
            for layer_id, _layer_type in enumerate(self._config.layer_types):
                for device in self.devices:
                    with scoped_current_device(self.runtime, device):
                        self.runtime.graph_launch(
                            self._layer_execs[(layer_id, device)],
                            self._rank_stream(device),
                        )
                if self.reduce_mode == "host":
                    partials = {
                        d: self._layer_partials[(layer_id, d)] for d in self.devices
                    }
                    group.reduce_partials(partials, slot=layer_id)
            stages["layers"] = time.perf_counter() - mark
            mark = time.perf_counter()
            # The tail writes the LAST layer's destination buffer - the one
            # the head reads - not the unswapped embedding pair: layer L-1
            # swaps the pair iff L-1 is odd.
            layer_count = len(self._config.layer_types)
            last_slot = layer_count - 1
            if self.reduce_mode == "device":
                # The last layer's exchange runs eagerly (nothing follows it
                # in the step to carry it); one wait per rank covers both
                # spin-sums, then the residual adds.
                for device in self.devices:
                    rank_index = self.devices.index(device)
                    self._device_exchange.enqueue_rank(
                        rank_index,
                        self._layer_partials[(last_slot, device)],
                        last_slot,
                        group.output_ptr(device),
                    )
                self._device_exchange.wait()
            for device in self.devices:
                stream = self._rank_stream(device)
                with scoped_current_device(self.runtime, device):
                    scratch = self._scratches[device]
                    dst = self._hidden[device][1 - ((layer_count - 1) % 2)]
                    out_buf = group.output_ptr(device)
                    if self.reduce_mode == "host":
                        f32_to_bf16(
                            group.reduced_payload_ptr(last_slot),
                            out_buf,
                            self.hidden_size,
                            stream=stream,
                            runtime=self.runtime,
                        )
                    gguf_bf16_add(
                        scratch.residual.ptr,
                        out_buf,
                        dst,
                        self.hidden_size,
                        stream=stream,
                        runtime=self.runtime,
                    )
            stages["tail"] = time.perf_counter() - mark
            logits = self._finish_step(stages)
        except Exception as error:
            self._poisoned = True
            raise TP2GroupError(
                f"graphed {self.mode} rank group failed at position {position}: "
                f"{type(error).__name__}: {error}"
            ) from error
        total = time.perf_counter() - started
        trace = StepTrace(
            kind=kind,
            position=position,
            total_s=total,
            stages=stages if self.stage_trace else {},
            exchange_layers=(
                len(self._shard_group.exchange_walls_s) - exchanges_before
                if self._shard_group
                else 0
            ),
        )
        return logits, trace

    def _destroy_graphs(self) -> None:
        for (layer_id, device), graph_exec in self._layer_execs.items():
            try:
                with scoped_current_device(self.runtime, device):
                    self.runtime.graph_exec_destroy(graph_exec)
                    self.runtime.graph_destroy(self._layer_graphs[(layer_id, device)])
            except Exception:  # noqa: BLE001 - teardown continues
                pass
        self._layer_execs.clear()
        self._layer_graphs.clear()
        self._layer_partials.clear()
        for device, stream in self._created_streams.items():
            try:
                with scoped_current_device(self.runtime, device):
                    self.runtime.stream_destroy(stream)
            except Exception:  # noqa: BLE001 - teardown continues
                pass
        self._created_streams.clear()

    def _local_mlp(self, device: int, layer_id: int) -> int:
        """The matched TP1 control's full-width unfused MLP chain (bf16 out).

        Owner boundary: the caller (``_enqueue_layer``) invokes this outside any
        device scope, and the preceding scoped attention/add-norm blocks restore
        the ambient device, so this must select ``device`` for every launch and
        use that rank's own stream. Without it a rank-1 session launched the
        full-width gate/up/down GEMV against device 1 pointers from device 0's
        context, which silently wrote nothing on device 1.
        """

        runner = self._runners[device]
        scratch = self._scratches[device]
        buffers = self._step_buffers[device]
        layer = runner.weights.layer(layer_id)
        stream = self._rank_stream(device)
        with scoped_current_device(self.runtime, device):
            launch_gguf_linear(
                layer.weight("ffn_gate"),
                scratch.post_norm.ptr,
                buffers["mlp_gate"].ptr,
                1,
                runner.hidden_size,
                runner.ffn_size,
                use_gemv_decode=True,
                stream=stream,
                runtime=self.runtime,
            )
            launch_gguf_linear(
                layer.weight("ffn_up"),
                scratch.post_norm.ptr,
                buffers["mlp_up"].ptr,
                1,
                runner.hidden_size,
                runner.ffn_size,
                use_gemv_decode=True,
                stream=stream,
                runtime=self.runtime,
            )
            silu_mul_separate_out_bf16(
                buffers["mlp_gate"].ptr,
                buffers["mlp_up"].ptr,
                buffers["mlp_act"].ptr,
                1,
                runner.ffn_size,
                stream=stream,
                runtime=self.runtime,
            )
            launch_gguf_linear(
                layer.weight("ffn_down"),
                buffers["mlp_act"].ptr,
                buffers["mlp_out"].ptr,
                1,
                runner.ffn_size,
                runner.hidden_size,
                use_gemv_decode=True,
                stream=stream,
                runtime=self.runtime,
            )
        return buffers["mlp_out"].ptr

    # -- sharded output head -------------------------------------------------

    def _build_head_shards(self) -> None:
        """One block-aligned head row shard per rank, plus its logits buffer.

        Each rank's runner defers the replicated head allocation; the shard is
        a contiguous Q6_K block range of the source tensor repacked to the
        runtime-resolved layout and uploaded device-scoped to its rank.
        """

        if self.mode != "tp2":
            raise TP2GroupError("the sharded head is tp2-only")
        plan, rank_payloads = materialize_head_shards(
            self.model_path, world_size=len(self.devices)
        )
        self._head_plan = plan
        for device in self.devices:
            payload = rank_payloads[device]["tiles"]
            with scoped_current_device(self.runtime, device):
                weight = upload_shard_weight(
                    self.runtime,
                    device=device,
                    name="tiles",
                    layout=plan.layout,
                    quant_key=plan.quant_key,
                    payload=payload,
                )
                self._head_weights[device] = weight
                buf = malloc(plan.rows_per_rank * 4, runtime=self.runtime)
                self._head_logits_bufs[device] = buf
                self._extra_buffers[device].append(buf)

    def _finish_step(self, stages: dict[str, float]) -> np.ndarray:
        if self.head_shard:
            return self._finish_step_sharded_head(stages)
        return self._finish_step_replicated_head(stages)

    def _finish_step_replicated_head(
        self,
        stages: dict[str, float],
    ) -> np.ndarray:
        """The committed default: the full head GEMV on the control rank."""

        mark = time.perf_counter()
        device = self.control_device
        stream = self._rank_stream(device)
        with scoped_current_device(self.runtime, device):
            runner = self._runners[device]
            scratch = self._scratches[device]
            src, _dst = self._hidden[device]
            gguf_rmsnorm_bf16_f32_weight(
                src,
                runner.weights.root("output_norm").allocation().tensor.ptr,
                scratch.norm.ptr,
                1,
                runner.hidden_size,
                runner.weights.config.rms_norm_eps,
                stream=stream,
                runtime=self.runtime,
            )
            buffers = self._step_buffers[device]
            launch_gguf_linear(
                runner.weights.root("lm_head"),
                scratch.norm.ptr,
                buffers["logits_buf"].ptr,
                1,
                runner.hidden_size,
                runner.vocab_size,
                output_dtype=GGUF_OUTPUT_F32,
                stream=stream,
                runtime=self.runtime,
            )
            # The blocking logits readback must not race the non-blocking
            # rank stream's pending head work.
            if stream:
                self.runtime.stream_synchronize(stream)
            logits = np.empty(runner.vocab_size, dtype="<f4")
            copy_device_to_host(
                logits.ctypes.data,
                buffers["logits_buf"],
                runner.vocab_size * 4,
                runtime=self.runtime,
            )
        stages["head_sample"] = time.perf_counter() - mark
        return logits

    def _finish_step_sharded_head(
        self,
        stages: dict[str, float],
    ) -> np.ndarray:
        """Both ranks run their contiguous head row shard concurrently.

        By this point the last layer's exchange has synchronized both rank
        streams, so the two shard GEMVs overlap; the host then reads both f32
        rows back and takes the concatenated argmax - the exact global greedy
        token with the replicated head's first-maximum tie-break, because the
        GEMV is row-independent and the shards are contiguous vocab ranges.
        """

        mark = time.perf_counter()
        plan = self._head_plan
        rows = int(plan.rows_per_rank)
        for device in self.devices:
            stream = self._rank_stream(device)
            with scoped_current_device(self.runtime, device):
                runner = self._runners[device]
                scratch = self._scratches[device]
                src, _dst = self._hidden[device]
                gguf_rmsnorm_bf16_f32_weight(
                    src,
                    runner.weights.root("output_norm").allocation().tensor.ptr,
                    scratch.norm.ptr,
                    1,
                    runner.hidden_size,
                    runner.weights.config.rms_norm_eps,
                    stream=stream,
                    runtime=self.runtime,
                )
                launch_gguf_linear(
                    self._head_weights[device],
                    scratch.norm.ptr,
                    self._head_logits_bufs[device].ptr,
                    1,
                    runner.hidden_size,
                    rows,
                    output_dtype=GGUF_OUTPUT_F32,
                    stream=stream,
                    runtime=self.runtime,
                )
        logits = np.empty(plan.vocab_rows, dtype="<f4")
        for device in self.devices:
            stream = self._rank_stream(device)
            with scoped_current_device(self.runtime, device):
                if stream:
                    self.runtime.stream_synchronize(stream)
                copy_device_to_host(
                    logits[plan.rank_row_start(device):].ctypes.data,
                    self._head_logits_bufs[device],
                    rows * 4,
                    runtime=self.runtime,
                )
        stages["head_sample"] = time.perf_counter() - mark
        return logits

    def _add_norm_kernel(self, runner: Qwen35GGUFFullStackRunner) -> Any:
        """The runner's resolved add+RMSNorm leaf for rows=1 at hidden size."""

        cache = getattr(self, "_add_norm_cache", None)
        if cache is None:
            cache = {}
            self._add_norm_cache = cache
        key = id(runner)
        if key not in cache:
            cache[key] = _gguf_norm_residual_decode_kernel(
                runner,
                layer="add_rmsnorm",
                rows=1,
                hidden_size=runner.hidden_size,
            )
        return cache[key]

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        """Free every rank's buffers, weights and shards, exactly once."""

        if self._closed:
            return
        self._closed = True
        self._destroy_graphs()
        if self._device_exchange is not None:
            try:
                self._device_exchange.close()
            except Exception:  # noqa: BLE001 - teardown continues
                pass
            self._device_exchange = None
        if self._bulk_shard_group is not None or self._bulk_rows:
            self._release_bulk_prefill_workspace()
        if self._shard_group is not None:
            self._shard_group.close()
            self._shard_group = None
        for device, weight in self._head_weights.items():
            try:
                with scoped_current_device(self.runtime, device):
                    weight.allocation().free()
            except Exception:  # noqa: BLE001 - teardown continues
                pass
        self._head_weights.clear()
        self._head_logits_bufs.clear()
        self._head_plan = None
        for device, buffers in self._extra_buffers.items():
            for buffer in buffers:
                try:
                    with scoped_current_device(self.runtime, device):
                        free(buffer, runtime=self.runtime)
                except Exception:  # noqa: BLE001 - teardown continues
                    pass
        self._extra_buffers.clear()
        self._step_buffers.clear()
        for device, runner in self._runners.items():
            weights = runner.weights
            runner.weights = None
            if weights is not None:
                try:
                    with scoped_current_device(self.runtime, device):
                        weights.free(runtime=self.runtime)
                except Exception:  # noqa: BLE001 - teardown continues
                    pass
        self._runners.clear()
        self._scratches.clear()

    def __enter__(self) -> "MlpTP2GenerationSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _require_live(self) -> None:
        if self._closed:
            raise TP2GroupError("the session is closed")
        if self._poisoned:
            raise TP2GroupError(
                "the session is poisoned by an earlier failure; "
                "a partially advanced rank group is not reusable"
            )
