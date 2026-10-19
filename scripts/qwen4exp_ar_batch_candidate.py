"""Numerical candidate adapter: Qwen4Exp c1 prefill into the AR batch runner.

This module drives the real :class:`Qwen4ExpGGUFBatchRunner` (through
:class:`Qwen4ExpARBatchOwner`) as a production-profile candidate for the
normative evaluator in ``scripts/qwen4exp_layer2_profile_gate.py``.

What is batched and what is not, stated once so no result can be read as
something else:

* **Prefill is the committed c1 path.** Each request is prefilled on the
  request-owned resident runner, exactly as serving does, and its state is
  imported into one physical batch slot once. No prefill row is batched here.
* **Decode is a genuine N-row step.** At each teacher-forced position every
  row of the composition group is advanced by one ``owner.step`` call with that
  request's own forced token. The adapter never runs N single-row decode loops
  and labels them width N.
* **Composition is explicit.** Requests are grouped in fixture order in
  consecutive chunks of ``width``; the final chunk keeps its natural width
  (1..width). The plan, the observed step-width histogram, and the
  request/slot/owner map are all recorded.
* **Ownership is checked, not assumed.** Every request gets a distinct slot and
  owner token, its import position must equal its prompt length, and every step
  must leave the slot position at ``prompt_len + step``. The owner returns
  results keyed by request id, so a row cannot be attributed to its neighbour.

The state fingerprint covers the request-local batch slot (live KV blocks,
index keys/counts, GDN conv/recurrent rows, PLE history, PLE n-gram history)
and is compared across repeats. It is deliberately *not* compared byte-for-byte
to the c1 state: the batch runner owns a different physical layout, and the
profile contract requires exact ownership and determinism, not a shared layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

import numpy as np

#: ``candidate_mode`` recorded by the normative evaluator for this adapter.
BATCH_CANDIDATE_MODE = "qwen4_exp_ar_batch"
#: Widths the live qualification measured; anything else is refused.
BATCH_WIDTHS = (1, 2, 4, 8)
#: Label that states prefill is not part of the batch claim.
BATCH_PREFILL_SOURCE = "c1_qwen4_exp_gguf_resident_model_runner"

#: Direct-call identities the batch decode path uses. The generator's c1
#: profile manifest is not evidence for these kernels: each row must be named
#: by a manifest selection before the candidate is admission-eligible.
BATCH_PATH_IDENTITIES: tuple[Mapping[str, Any], ...] = (
    {
        "role": "gdn_conv_indexed",
        "identity": "qwen4_exp_linear_attn_conv_decode_indexed_f32",
        "source": "hipengine/runtime/qwen4_exp_batch_gdn.py",
        "manifest_match": "conv_decode_indexed",
    },
    {
        "role": "gdn_recurrence_norm_gate",
        "identity": "qwen4_exp_gdn_decode_rowsplit_indexed_f32",
        "source": "hipengine/runtime/qwen4_exp_batch_gdn.py",
        "manifest_match": "rowsplit_indexed",
    },
    {
        "role": "qsa_mixed_mixer",
        "identity": "run_qwen4_exp_batch_qsa_mixed_token_mixer",
        "source": "hipengine/runtime/qwen4_exp_batch_qsa.py",
        "manifest_match": ("mixed_token_mixer", "qsa_mixed"),
    },
    {
        "role": "qsa_paged_kv_write",
        "identity": "qwen35_write_paged_kv_f32_batch_spans",
        "source": "hipengine/runtime/qwen4_exp_batch_qsa.py",
        "manifest_match": ("batch_spans", "batch_decode"),
    },
    {
        "role": "qsa_paged_decode",
        "identity": "qwen35_paged_full_attn_decode_context_bf16_batch_spans",
        "source": "hipengine/runtime/qwen4_exp_batch_qsa.py",
        "manifest_match": ("batch_spans", "batch_decode"),
    },
    {
        "role": "batch_ple",
        "identity": "run_qwen4_exp_batch_ple_stage",
        "source": "hipengine/runtime/qwen4_exp_batch_ple.py",
        "manifest_match": ("batch_ple",),
    },
)


def batch_path_identities() -> tuple[dict[str, Any], ...]:
    """Return the batch path's direct-call identities as fresh dicts."""

    return tuple(dict(identity) for identity in BATCH_PATH_IDENTITIES)


def batch_manifest_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Check whether a profile manifest names the batch decode path.

    A match is a manifest selection whose layer, scope, selected variant, or
    strict fallback contains the identity's distinctive token. The c1 profile
    names prefill GDN and c1 sparse-decode scopes, so it does not bind the batch
    rowsplit-indexed recurrence, the mixed QSA mixer, the batch-span paged
    attention, or the batch PLE stage. Missing bindings are reported
    individually and make the candidate ineligible for admission.
    """

    selections = manifest.get("selections", ())
    if not isinstance(selections, Sequence) or isinstance(selections, (str, bytes)):
        selections = ()
    fields: list[str] = []
    for selection in selections:
        if not isinstance(selection, Mapping):
            continue
        fields.extend(
            str(selection.get(key, ""))
            for key in ("layer", "scope", "selected_variant", "strict_fallback_variant")
        )
    haystack = " ".join(fields).lower()
    identities = []
    for identity in BATCH_PATH_IDENTITIES:
        row = dict(identity)
        tokens = row["manifest_match"]
        if isinstance(tokens, str):
            tokens = (tokens,)
        row["manifest_match"] = list(tokens)
        row["bound"] = any(str(token).lower() in haystack for token in tokens)
        identities.append(row)
    missing = [dict(row) for row in identities if not row["bound"]]
    return {
        "bound": not missing,
        "identities": identities,
        "missing": missing,
        "manifest_selection_count": len(selections),
    }


@dataclass(frozen=True)
class BatchCandidateRequest:
    """One fixture row's prompt, teacher-forced schedule, and request id."""

    prompt_id: str
    request_id: int
    prompt_ids: tuple[int, ...]
    forced_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.prompt_id:
            raise ValueError("batch candidate request needs a prompt id")
        if isinstance(self.request_id, bool) or int(self.request_id) <= 0:
            raise ValueError("batch candidate request id must be a positive integer")
        if not self.prompt_ids:
            raise ValueError(f"batch candidate request {self.prompt_id!r} has no prompt")
        if not self.forced_ids:
            raise ValueError(
                f"batch candidate request {self.prompt_id!r} has no teacher-forced tokens"
            )
        if any(int(token) < 0 for token in (*self.prompt_ids, *self.forced_ids)):
            raise ValueError("batch candidate token ids must be non-negative")


@dataclass(frozen=True)
class BatchCompositionGroup:
    """Consecutive fixture rows that share one physical N-row decode step."""

    index: int
    requests: tuple[BatchCandidateRequest, ...]

    @property
    def width(self) -> int:
        return len(self.requests)

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        return tuple(request.prompt_id for request in self.requests)


def plan_batch_composition(
    requests: Sequence[BatchCandidateRequest],
    width: int,
) -> tuple[BatchCompositionGroup, ...]:
    """Group fixture rows in order into chunks of at most ``width``.

    The final chunk keeps its natural width, so a 5-row selection at width 4 is
    one width-4 group and one width-1 group. Every group is recorded; a result
    from a width-1 chunk is never reported as width 4.
    """

    parsed_width = int(width)
    if parsed_width not in BATCH_WIDTHS:
        raise ValueError(
            f"batch width must be one of {BATCH_WIDTHS}, got {width!r}"
        )
    rows = tuple(requests)
    if not rows:
        raise ValueError("batch composition needs at least one request")
    prompt_ids = [row.prompt_id for row in rows]
    request_ids = [int(row.request_id) for row in rows]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("batch composition has duplicate prompt ids")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("batch composition has duplicate request ids")
    forced_lengths = {len(row.forced_ids) for row in rows}
    if len(forced_lengths) != 1:
        raise ValueError(
            "batch composition requires one teacher-forced length per group; "
            f"got {sorted(forced_lengths)}"
        )
    return tuple(
        BatchCompositionGroup(index=index, requests=rows[start : start + parsed_width])
        for index, start in enumerate(range(0, len(rows), parsed_width))
    )


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _read_device(runtime: Any, ptr: int, nbytes: int) -> np.ndarray:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    size = int(nbytes)
    host = np.empty(size, dtype=np.uint8)
    copy_device_to_host(
        host_array_ptr(host),
        DeviceBuffer(int(ptr), size),
        size,
        runtime=runtime,
    )
    return host


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def batch_slot_state(
    runner: Any,
    slot: int,
    *,
    prompt_id: str,
) -> dict[str, Any]:
    """Fingerprint one admitted slot's request-local batch state.

    Reads the slot's live KV blocks, index keys and counts, every GDN
    conv/recurrent row, the PLE conv history, and the PLE n-gram history. The
    owner token is deliberately excluded: it is a monotonic admission handle,
    not request state, so including it would make a correct deterministic
    rerun look different.
    """

    parsed_slot = int(slot)
    position = int(runner.position(parsed_slot))
    runtime = runner.runtime
    digest = hashlib.sha256()
    layout: dict[str, int] = {}
    finite = True
    attention_positions: list[list[int]] = []
    index_counts: list[list[int]] = []

    for layer, state in enumerate(runner.attention_states):
        blocks = np.asarray(state.block_host[parsed_slot], dtype=np.int64)
        needed = (position + int(state.block_size) - 1) // int(state.block_size)
        row_bytes = int(state.kv_heads) * int(state.head_dim) * 2
        block_bytes = int(state.block_size) * row_bytes
        for name, cache in (("k", state.key_cache), ("v", state.value_cache)):
            for block in blocks[:needed].tolist():
                raw = _read_device(
                    runtime, cache.ptr + int(block) * block_bytes, block_bytes
                )
                digest.update(f"kv:{layer}:{name}:{block}".encode())
                digest.update(raw)
                finite = finite and bool(
                    np.isfinite(_bf16_to_f32(raw.view(np.uint16))).all()
                )
        layout[f"kv_layer_{layer}_blocks"] = int(needed) * block_bytes * 2
        attention_positions.append([position, position])

    for layer, index_state in enumerate(runner.index_states):
        count = int(index_state.cursor_host[parsed_slot])
        pooled = int(index_state.pooled_count_host[parsed_slot])
        blocks = np.asarray(
            runner.attention_states[layer].block_host[parsed_slot], dtype=np.int64
        )
        needed = (position + int(index_state.block_size) - 1) // int(
            index_state.block_size
        )
        index_row_bytes = int(index_state.index_dim) * 4
        index_block_bytes = int(index_state.block_size) * index_row_bytes
        for block in blocks[:needed].tolist():
            raw = _read_device(
                runtime,
                index_state.raw_keys.ptr + int(block) * index_block_bytes,
                index_block_bytes,
            )
            digest.update(f"index:{layer}:{block}".encode())
            digest.update(raw)
            finite = finite and bool(
                np.isfinite(raw.view(np.float32)).all()
            )
        layout[f"index_layer_{layer}_bytes"] = int(needed) * index_block_bytes
        index_counts.append([count, pooled])

    for layer, gdn in enumerate(runner.gdn_states):
        for name, ptr, nbytes in (
            ("conv", gdn.conv_slot_ptr(parsed_slot), gdn.conv_bytes_per_slot),
            (
                "recurrent",
                gdn.recurrent_slot_ptr(parsed_slot),
                gdn.recurrent_bytes_per_slot,
            ),
        ):
            raw = _read_device(runtime, ptr, nbytes)
            digest.update(f"gdn:{layer}:{name}".encode())
            digest.update(raw)
            finite = finite and bool(np.isfinite(raw.view(np.float32)).all())
            layout[f"gdn_layer_{layer}_{name}"] = int(nbytes)

    if runner.ple_state is not None:
        raw = _read_device(
            runtime,
            runner.ple_state.history_slot_ptr(parsed_slot),
            runner.ple_state.history_bytes_per_slot,
        )
        digest.update(b"ple:history")
        digest.update(raw)
        finite = finite and bool(np.isfinite(raw.view(np.float32)).all())
        layout["ple_history"] = int(runner.ple_state.history_bytes_per_slot)

    ple_hash = runner.ple_hash_states.get(parsed_slot)
    ple_metadata = (
        None
        if ple_hash is None
        else {
            "tokens": [int(token) for token in ple_hash.tokens],
            "next_position": int(ple_hash.next_position),
        }
    )
    metadata = {
        "position": position,
        "attention_positions": attention_positions,
        "index_counts": index_counts,
        "ple": ple_metadata,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    return {
        "prompt_id": str(prompt_id),
        "state_sha256": digest.hexdigest(),
        "layout_sha256": _hash_json(layout),
        "buffer_bytes": layout,
        "finite": bool(finite),
        **metadata,
    }


@dataclass(frozen=True)
class BatchGroupRun:
    """One composition group's trajectories, state fingerprints, and evidence."""

    group: BatchCompositionGroup
    trajectories: Mapping[str, tuple[Mapping[str, Any], ...]]
    states: Mapping[str, Mapping[str, Any]]
    evidence: Mapping[str, Any]


class Qwen4ExpARBatchCandidate:
    """Drive the real AR batch owner as a teacher-forced candidate.

    ``owner_factory`` exists for CPU tests; production callers leave it unset
    and the adapter builds a :class:`Qwen4ExpARBatchOwner` over the generator's
    model-owned residency.
    """

    def __init__(
        self,
        generator: Any,
        *,
        width: int,
        label: str = BATCH_CANDIDATE_MODE,
        owner_factory: Callable[[int], Any] | None = None,
    ) -> None:
        parsed_width = int(width)
        if parsed_width not in BATCH_WIDTHS:
            raise ValueError(
                f"batch width must be one of {BATCH_WIDTHS}, got {width!r}"
            )
        self.generator = generator
        self.width = parsed_width
        self.label = str(label)
        self._owner_factory = owner_factory
        self._owner: Any | None = None
        self._max_positions: int | None = None
        self._failure: str | None = None
        self._groups: list[dict[str, Any]] = []
        self._step_widths: dict[int, int] = {}
        self._dispatch_records: list[dict[str, Any]] = []
        self._variant_manifest: dict[str, Any] | None = None
        self._imports = 0
        self._steps = 0

    # -- lifecycle --------------------------------------------------------

    def open(self) -> Any:
        if self._failure is not None:
            raise RuntimeError(
                f"batch candidate is disabled after a failed call: {self._failure}"
            )
        if self._owner is not None:
            return self._owner
        if self._owner_factory is not None:
            owner = self._owner_factory(self.width)
        else:
            from hipengine.generation.qwen4_exp_ar_batch import (
                Qwen4ExpARBatchOwner,
                batch_slot_positions,
                make_qwen4_exp_ar_batch_runner,
            )

            # The serving slot cap: a longer request fails at reserve here,
            # where serving would decode it on a c1 staging runner instead.
            max_positions = batch_slot_positions(self.generator.runner.max_sequence_length)
            owner = Qwen4ExpARBatchOwner(
                batch_factory=lambda: make_qwen4_exp_ar_batch_runner(
                    self.generator, state_slots=self.width,
                    max_positions=max_positions,
                ),
                state_slots=self.width,
                max_positions=max_positions,
            )
        owner.ensure()
        self._owner = owner
        self._max_positions = getattr(owner, "max_positions", None)
        return owner

    def close(self) -> None:
        owner, self._owner = self._owner, None
        if owner is not None:
            owner.close()

    @property
    def step_widths(self) -> dict[int, int]:
        return dict(self._step_widths)

    def build_variant_manifest(self, *, source_revision: str) -> dict[str, Any]:
        """Build the structured batch manifest from the live runner and records.

        This must be called before :meth:`close`: the manifest names the live
        runner class and its config, and the projection dispatch records were
        collected around the measured steps.
        """

        if self._variant_manifest is not None:
            return dict(self._variant_manifest)
        owner = self._owner
        if owner is None:
            raise RuntimeError("batch candidate has no open owner")
        runner = getattr(owner, "runner", None)
        if runner is None:
            raise RuntimeError("batch owner exposes no runner")
        profile_manifest = getattr(self.generator, "execution_profile_manifest", None)
        if not isinstance(profile_manifest, Mapping):
            raise RuntimeError(
                "candidate generator has no execution profile manifest; the batch "
                "manifest cannot bind the execution profile"
            )
        from hipengine.runtime.qwen4_exp_batch_manifest import (
            build_qwen4_exp_batch_variant_manifest,
        )

        manifest = build_qwen4_exp_batch_variant_manifest(
            runner=runner,
            profile_manifest=profile_manifest,
            source_revision=source_revision,
            dispatch_records=self._dispatch_records,
            step_widths=self._step_widths,
        )
        self._variant_manifest = manifest
        return dict(manifest)

    @property
    def evidence(self) -> dict[str, Any]:
        manifest = self._variant_manifest
        return {
            "mode": self.label,
            "configured_width": self.width,
            "max_positions": self._max_positions,
            "prefill_source": BATCH_PREFILL_SOURCE,
            "path_identities": (
                [
                    {
                        "role": entry["role"],
                        "binding": entry["binding"],
                        "identity": entry["identity"],
                    }
                    for entry in manifest["entries"]
                ]
                if manifest is not None
                else list(batch_path_identities())
            ),
            "variant_manifest": dict(manifest) if manifest is not None else None,
            "dispatch_record_count": len(self._dispatch_records),
            "decode_step": "owner.step(rows, capture_logits=True)",
            "composition": "fixture_order_consecutive_groups",
            "imports": self._imports,
            "steps": self._steps,
            "step_widths": {str(k): v for k, v in sorted(self._step_widths.items())},
            "groups": list(self._groups),
            "failure": self._failure,
        }

    # -- runs -------------------------------------------------------------

    def run_group(self, group: BatchCompositionGroup) -> BatchGroupRun:
        """Prefill each request on c1, import, then step the group N rows wide."""

        owner = self.open()
        source = self.generator.runner
        trajectories: dict[str, list[dict[str, Any]]] = {
            request.prompt_id: [] for request in group.requests
        }
        slots: dict[int, int] = {}
        imported_positions: dict[int, int] = {}
        from hipengine.runtime.gguf_linear import record_gguf_linear_dispatch
        try:
            for request in group.requests:
                source.reset()
                prefill = source.prefill([int(token) for token in request.prompt_ids])
                slot = int(
                    owner.reserve(
                        request.request_id,
                        positions=len(request.prompt_ids)
                        + len(request.forced_ids),
                    )
                )
                position = int(
                    owner.import_prefilled(request.request_id, source=source)
                )
                if position != len(request.prompt_ids):
                    raise RuntimeError(
                        f"request {request.prompt_id!r} imported at position "
                        f"{position}, expected its prompt length "
                        f"{len(request.prompt_ids)}"
                    )
                slots[request.request_id] = slot
                imported_positions[request.request_id] = position
                trajectories[request.prompt_id].append(
                    {
                        "token_id": int(prefill.token_id),
                        "logits": np.ascontiguousarray(prefill.logits, dtype=np.float32),
                    }
                )
                self._imports += 1

            self._require_distinct_leases(owner, group)
            teacher_steps = len(group.requests[0].forced_ids)
            for step in range(teacher_steps):
                rows = [
                    (request.request_id, int(request.forced_ids[step]))
                    for request in group.requests
                ]
                with record_gguf_linear_dispatch(self._dispatch_records):
                    results = owner.step(rows, capture_logits=True)
                self._steps += 1
                self._step_widths[len(rows)] = self._step_widths.get(len(rows), 0) + 1
                for request in group.requests:
                    result = results[request.request_id]
                    logits = np.ascontiguousarray(result.logits, dtype=np.float32)
                    if logits.ndim != 1 or logits.size == 0:
                        raise RuntimeError(
                            f"request {request.prompt_id!r} returned logits with "
                            f"shape {logits.shape}"
                        )
                    trajectories[request.prompt_id].append(
                        {
                            "token_id": int(result.token_id),
                            "logits": logits,
                        }
                    )
                    expected = len(request.prompt_ids) + step + 1
                    actual = int(owner.runner.position(slots[request.request_id]))
                    if actual != expected:
                        raise RuntimeError(
                            f"request {request.prompt_id!r} slot advanced to "
                            f"position {actual}, expected {expected}"
                        )

            states = {
                request.prompt_id: batch_slot_state(
                    owner.runner,
                    slots[request.request_id],
                    prompt_id=request.prompt_id,
                )
                for request in group.requests
            }
            evidence = self._group_evidence(
                group,
                owner,
                slots,
                imported_positions,
                states,
                decode_steps=len(group.requests[0].forced_ids),
            )
            self._groups.append(evidence)
            return BatchGroupRun(
                group=group,
                trajectories={
                    prompt_id: tuple(run) for prompt_id, run in trajectories.items()
                },
                states=states,
                evidence=evidence,
            )
        except BaseException as error:
            self._failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            for request in group.requests:
                owner.release(request.request_id)

    def run_free_group(
        self,
        group: BatchCompositionGroup,
        *,
        steps: int,
    ) -> dict[str, dict[str, Any]]:
        """Run the group free-running, feeding each row its own greedy token."""

        parsed_steps = int(steps)
        if parsed_steps <= 0:
            raise ValueError("free-running batch candidate needs positive steps")
        owner = self.open()
        source = self.generator.runner
        slots: dict[int, int] = {}
        ids: dict[str, list[int]] = {request.prompt_id: [] for request in group.requests}
        current: dict[str, int] = {}
        from hipengine.runtime.gguf_linear import record_gguf_linear_dispatch
        try:
            for request in group.requests:
                source.reset()
                prefill = source.prefill([int(token) for token in request.prompt_ids])
                slot = int(
                    owner.reserve(
                        request.request_id,
                        positions=len(request.prompt_ids) + parsed_steps - 1,
                    )
                )
                owner.import_prefilled(request.request_id, source=source)
                slots[request.request_id] = slot
                current[request.prompt_id] = int(prefill.token_id)
                ids[request.prompt_id].append(int(prefill.token_id))
                self._imports += 1

            self._require_distinct_leases(owner, group)
            for _ in range(parsed_steps - 1):
                rows = [
                    (request.request_id, current[request.prompt_id])
                    for request in group.requests
                ]
                with record_gguf_linear_dispatch(self._dispatch_records):
                    results = owner.step(rows, capture_logits=False)
                self._steps += 1
                self._step_widths[len(rows)] = self._step_widths.get(len(rows), 0) + 1
                for request in group.requests:
                    token = int(results[request.request_id].token_id)
                    current[request.prompt_id] = token
                    ids[request.prompt_id].append(token)

            return {
                request.prompt_id: {
                    "ids": list(ids[request.prompt_id]),
                    "ids_sha256": hashlib.sha256(
                        np.asarray(ids[request.prompt_id], dtype="<i8").tobytes()
                    ).hexdigest(),
                    "text": str(
                        self.generator.tokenizer.decode(
                            ids[request.prompt_id], skip_special=False
                        )
                    ),
                    "state": batch_slot_state(
                        owner.runner,
                        slots[request.request_id],
                        prompt_id=request.prompt_id,
                    ),
                }
                for request in group.requests
            }
        except BaseException as error:
            self._failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            for request in group.requests:
                owner.release(request.request_id)

    # -- evidence ---------------------------------------------------------

    @staticmethod
    def _require_distinct_leases(
        owner: Any, group: BatchCompositionGroup
    ) -> None:
        leases = [owner.lease(request.request_id) for request in group.requests]
        if any(lease is None for lease in leases):
            raise RuntimeError("a batch candidate request has no pinned slot")
        slots = [int(lease[0]) for lease in leases]
        owners = [int(lease[1]) for lease in leases]
        if len(set(slots)) != len(slots):
            raise RuntimeError(f"batch candidate slots are not distinct: {slots}")
        if len(set(owners)) != len(owners):
            raise RuntimeError(f"batch candidate owner tokens are not distinct: {owners}")

    def _group_evidence(
        self,
        group: BatchCompositionGroup,
        owner: Any,
        slots: Mapping[int, int],
        imported_positions: Mapping[int, int],
        states: Mapping[str, Mapping[str, Any]],
        *,
        decode_steps: int,
    ) -> dict[str, Any]:
        rows = []
        for request in group.requests:
            lease = owner.lease(request.request_id)
            assert lease is not None
            state = states[request.prompt_id]
            rows.append(
                {
                    "prompt_id": request.prompt_id,
                    "request_id": int(request.request_id),
                    "slot": int(lease[0]),
                    "owner": int(lease[1]),
                    "import_position": int(imported_positions[request.request_id]),
                    "final_position": int(state["position"]),
                    "state_sha256": str(state["state_sha256"]),
                }
            )
        hashes = [row["state_sha256"] for row in rows]
        return {
            "group_index": int(group.index),
            "width": int(group.width),
            "prompt_ids": list(group.prompt_ids),
            "rows": rows,
            "distinct_state_sha256": len(set(hashes)),
            "state_distinct": len(set(hashes)) == len(hashes),
            "ownership_sha256": _hash_json(
                [
                    {
                        "prompt_id": row["prompt_id"],
                        "slot": row["slot"],
                        "import_position": row["import_position"],
                        "final_position": row["final_position"],
                    }
                    for row in rows
                ]
            ),
            "decode_steps": int(decode_steps),
        }


def batch_state_gate(
    strict_states: Sequence[Mapping[str, Any]],
    candidate_states: Sequence[Sequence[Mapping[str, Any]]],
    *,
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Repeat-determinism and ownership gate for the batch slot state.

    The strict teacher's c1 state and a batch slot are different physical
    layouts, so byte equality across them is not a requirement and is reported
    as not applicable. What is required: every slot is finite, the same request
    reproduces the same slot state on every repeat, the slot position equals the
    strict teacher's final position, and requests in one group never share a
    state fingerprint.
    """

    if len(strict_states) != len(candidate_states):
        raise ValueError("strict and batch candidate state prompt counts differ")
    prompts: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for strict, candidate_values in zip(strict_states, candidate_states, strict=True):
        runs = tuple(candidate_values)
        if len(runs) < 3:
            raise ValueError("batch state gate requires at least three candidate runs")
        first = runs[0]
        finite = bool(strict["finite"]) and all(bool(run["finite"]) for run in runs)
        layout_exact = all(
            run["layout_sha256"] == first["layout_sha256"] for run in runs[1:]
        )
        repeat_exact = all(
            run["state_sha256"] == first["state_sha256"] for run in runs[1:]
        )
        position_exact = all(
            int(run["position"]) == int(strict["position"]) for run in runs
        )
        metadata_exact = all(
            len(run["attention_positions"]) == len(strict["attention_positions"])
            and len(run["index_counts"]) == len(strict["index_counts"])
            for run in runs
        )
        result = {
            "prompt_id": str(strict["prompt_id"]),
            "finite": finite,
            "layout_exact": layout_exact,
            "repeat_exact": repeat_exact,
            "position_exact": position_exact,
            "metadata_exact": metadata_exact,
            "batch_state_sha256": [str(run["state_sha256"]) for run in runs],
            "strict_position": int(strict["position"]),
            "batch_position": int(first["position"]),
            "strict_candidate_state_exact": False,
        }
        prompts.append(result)
        if not (finite and layout_exact and repeat_exact and position_exact and metadata_exact):
            mismatches.append(result)

    group_mismatches = [
        dict(group)
        for group in groups
        if not bool(group.get("state_distinct", False))
    ]
    passed = not mismatches and not group_mismatches
    return {
        "kind": "batch_slot_state",
        "layout_equivalence": (
            "not_applicable: the batch runner owns a different physical layout; "
            "exactness is checked against repeats, positions, and ownership"
        ),
        "passed": passed,
        "mismatches": mismatches,
        "group_mismatches": group_mismatches,
        "prompts": prompts,
    }


__all__ = [
    "BATCH_CANDIDATE_MODE",
    "BATCH_PATH_IDENTITIES",
    "BATCH_PREFILL_SOURCE",
    "BATCH_WIDTHS",
    "batch_manifest_binding",
    "batch_path_identities",
    "BatchCandidateRequest",
    "BatchCompositionGroup",
    "BatchGroupRun",
    "Qwen4ExpARBatchCandidate",
    "batch_slot_state",
    "batch_state_gate",
    "plan_batch_composition",
]
