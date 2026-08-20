"""Explicit public Poolside Laguna DFlash text-generation provider.

The adapter binds the pinned B4 drafter to a source-identified Laguna target,
retains one resettable target/drafter/cycle owner, and leaves ordinary generation
on the target-only AR route.  D4 rejected automatic promotion, so this module
advertises no performance claim and admits only raw greedy target top-1.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.kernels.backends import backend_package_capability
from hipengine.generation.laguna_gguf import (
    LagunaGGUFGenerator,
    _IncrementalLagunaDecoder,
    _filter_output_specials,
    _laguna_finish_details,
    _suppressed_suffix_length,
    _visible_generated_ids,
    _with_total_timing,
)
from hipengine.generation.registry import (
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationTelemetry,
)
from hipengine.generation.sampling import speculative_mtp_sampling_blockers
from hipengine.speculative.laguna_dflash import (
    LagunaDFlashResidentCycle,
    LagunaDFlashResidentDrafter,
)
from hipengine.speculative.registry import (
    SpeculativeProviderConfig,
    SpeculativeProviderKey,
    register_speculative_provider,
)

LAGUNA_DFLASH_TARGET_SHA256 = (
    "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f"
)
LAGUNA_DFLASH_DRAFTER_SHA256 = (
    "f24f08781c697c19952c02fb2e7e9bdf2071b79a711c2a44b836a74b9b62a1f4"
)
LAGUNA_DFLASH_DRAFTER_REVISION = "b0486d1586daa0d56435c508108171fc1c8daff9"
LAGUNA_DFLASH_CANDIDATE_BUDGET = 4
LAGUNA_DFLASH_EXECUTION_PATH = "laguna_dflash_b4_c1"
LAGUNA_DFLASH_IQ3_SELECTED_DOWN_TILE_ENV = (
    "HIPENGINE_LAGUNA_DFLASH_IQ3_SELECTED_DOWN_TILE"
)
LAGUNA_DFLASH_ECONOMICS_EVIDENCE = (
    "benchmarks/results/2026-07-23-gfx1151-"
    "laguna-dflash-category-economics-post-prefill.json"
)
LAGUNA_DFLASH_FALLBACK_REASON = "d4_full_suite_speedup_0p9469x_below_1p10"


def _laguna_dflash_iq3_selected_down_tile(backend: str) -> int:
    raw = os.environ.get(LAGUNA_DFLASH_IQ3_SELECTED_DOWN_TILE_ENV)
    if raw is None or not raw.strip():
        value = backend_package_capability(
            backend,
            "LAGUNA_DFLASH_IQ3_SELECTED_DOWN_TILE",
            1,
        )
    else:
        value = raw
    try:
        tile = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{LAGUNA_DFLASH_IQ3_SELECTED_DOWN_TILE_ENV} must be 1 or 4"
        ) from exc
    if tile not in {1, 4}:
        raise ValueError(
            f"{LAGUNA_DFLASH_IQ3_SELECTED_DOWN_TILE_ENV} must be 1 or 4"
        )
    return tile


@dataclass(frozen=True)
class _DFlashCounters:
    cycles: int = 0
    accepted_draft_tokens: int = 0
    draft_tokens_proposed: int = 0
    target_verify_rows: int = 0
    proposal_seconds: float = 0.0
    target_verify_seconds: float = 0.0
    draft_commit_enqueue_seconds: float = 0.0
    decode_seconds: float = 0.0

    def add_cycle(self, result: Any, *, wall_seconds: float) -> "_DFlashCounters":
        return _DFlashCounters(
            cycles=self.cycles + 1,
            accepted_draft_tokens=(
                self.accepted_draft_tokens
                + int(result.target_result.accepted_draft_count)
            ),
            draft_tokens_proposed=(
                self.draft_tokens_proposed
                + len(tuple(result.proposal.candidate_token_ids))
            ),
            target_verify_rows=(
                self.target_verify_rows + len(tuple(result.target_batch.tokens))
            ),
            proposal_seconds=self.proposal_seconds + float(result.proposal_seconds),
            target_verify_seconds=(
                self.target_verify_seconds + float(result.target_verify_seconds)
            ),
            draft_commit_enqueue_seconds=(
                self.draft_commit_enqueue_seconds
                + float(result.draft_commit_enqueue_seconds)
            ),
            decode_seconds=self.decode_seconds + float(wall_seconds),
        )


@dataclass(frozen=True)
class _DFlashTokenStep:
    token_id: int
    generated_ids: tuple[int, ...]
    finish_details: FinishDetails | None
    telemetry: GenerationTelemetry


class LagunaDFlashTextProvider:
    """Pinned, explicit-only B4 DFlash owner for one Laguna target generator."""

    provider_name = "dflash"

    def __init__(
        self,
        target_generator: LagunaGGUFGenerator,
        config: SpeculativeProviderConfig,
    ) -> None:
        if str(config.provider) != self.provider_name:
            raise ValueError("Laguna DFlash provider config must select dflash")
        if int(config.candidate_budget) != LAGUNA_DFLASH_CANDIDATE_BUDGET:
            raise ValueError("Laguna public DFlash admits only candidate budget B4")
        self.target = target_generator
        self.config = config
        self.target_iq3_selected_down_tile = (
            _laguna_dflash_iq3_selected_down_tile(self.target.backend)
        )
        self.drafter_model = Path(config.draft_model).expanduser().resolve()
        self._target_session: Any | None = None
        self._drafter: LagunaDFlashResidentDrafter | None = None
        self._cycle: LagunaDFlashResidentCycle | None = None
        self._load_seconds: float | None = None
        self._closed = False
        self._validate_identities()
        self.target.bind_repacked_cache_source_sha256(
            LAGUNA_DFLASH_TARGET_SHA256
        )

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "policy": "explicit_only",
            "default_enabled": False,
            "streaming_compatible": True,
            "candidate_budget": LAGUNA_DFLASH_CANDIDATE_BUDGET,
            "exactness_mode": "target_corrected_greedy",
            "processed_target_verification": False,
            "target": {
                "model": "poolside/Laguna-S-2.1-GGUF",
                "sha256": LAGUNA_DFLASH_TARGET_SHA256,
                "quant": "Q4_K_M",
                "iq3_selected_down_tile": self.target_iq3_selected_down_tile,
            },
            "drafter": {
                "model": "poolside/Laguna-S-2.1-DFlash",
                "revision": LAGUNA_DFLASH_DRAFTER_REVISION,
                "sha256": LAGUNA_DFLASH_DRAFTER_SHA256,
                "dtype": "bf16",
            },
            "fallback_reason": LAGUNA_DFLASH_FALLBACK_REASON,
            "performance_claim": False,
            "economics_evidence": LAGUNA_DFLASH_ECONOMICS_EVIDENCE,
        }

    def generate_detailed(
        self,
        request: GenerationRequest,
    ) -> list[GenerationOutput]:
        prompt_ids = self._prepare_request(request)
        if request.max_tokens == 0:
            finish = FinishDetails(
                reason="length",
                length_limit=0,
                sampler_mode="greedy_fast",
            )
            output = GenerationOutput(
                text="",
                generated_token_ids=(),
                finish_details=finish,
                telemetry=self._telemetry(
                    prompt_ids,
                    (),
                    finish=finish,
                    prefill_seconds=0.0,
                    counters=_DFlashCounters(),
                ),
            )
            self._record_output(output, prompt_ids, request)
            return [output]

        started = time.perf_counter()
        generated_ids: list[int] = []
        finish: FinishDetails | None = None
        final_telemetry: GenerationTelemetry | None = None
        for step in self._token_steps(request, prompt_ids):
            generated_ids.append(step.token_id)
            finish = step.finish_details or finish
            final_telemetry = step.telemetry
        if finish is None or final_telemetry is None:
            raise RuntimeError("Laguna DFlash generation ended without terminal metadata")
        visible_ids = _visible_generated_ids(
            generated_ids,
            finish,
            tokenizer=self.target.tokenizer,
        )
        output = GenerationOutput(
            text=self.target.tokenizer.decode(visible_ids),
            generated_token_ids=tuple(generated_ids),
            finish_details=finish,
            telemetry=_with_total_timing(final_telemetry, started),
        )
        self._record_output(output, prompt_ids, request)
        return [output]

    def stream_detailed(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationStreamChunk]:
        prompt_ids = self._prepare_request(request)
        if request.max_tokens == 0:
            output = self.generate_detailed(request)[0]
            yield GenerationStreamChunk(
                text="",
                finish_details=output.finish_details,
                telemetry=output.telemetry,
                generated_token_ids=(),
            )
            return

        started = time.perf_counter()
        pending: list[int] = []
        visible_parts: list[str] = []
        decoder = _IncrementalLagunaDecoder(self.target.tokenizer)
        longest_sequence = max(
            (len(sequence) for sequence in request.stop_token_sequences),
            default=1,
        )
        hold_tokens = max(0, longest_sequence - 1)
        terminal_output: GenerationOutput | None = None
        for step in self._token_steps(request, prompt_ids):
            pending.append(step.token_id)
            finish = step.finish_details
            if finish is None:
                safe_count = max(0, len(pending) - hold_tokens)
                safe_ids = pending[:safe_count]
                del pending[:safe_count]
                text = decoder.feed(
                    _filter_output_specials(safe_ids, self.target.tokenizer)
                )
                if text:
                    visible_parts.append(text)
                    yield GenerationStreamChunk(text=text, telemetry=step.telemetry)
                continue

            suppressed = _suppressed_suffix_length(finish)
            safe_ids = pending[:-suppressed] if suppressed else pending
            text = decoder.feed(
                _filter_output_specials(safe_ids, self.target.tokenizer),
                final=True,
            )
            if text:
                visible_parts.append(text)
            terminal_telemetry = _with_total_timing(step.telemetry, started)
            terminal_output = GenerationOutput(
                text="".join(visible_parts),
                generated_token_ids=step.generated_ids,
                finish_details=finish,
                telemetry=terminal_telemetry,
            )
            yield GenerationStreamChunk(
                text=text,
                finish_details=finish,
                telemetry=terminal_telemetry,
                generated_token_ids=step.generated_ids,
            )
        if terminal_output is None:
            raise RuntimeError("Laguna DFlash streaming ended without a terminal chunk")
        self._record_output(terminal_output, prompt_ids, request)

    def close(self) -> None:
        with self.target._lock:
            self._close_locked(suppress_errors=False)

    def _record_output(
        self,
        output: GenerationOutput,
        prompt_ids: Sequence[int],
        request: GenerationRequest,
    ) -> None:
        self.target.last_generation_outputs = (output,)
        telemetry = output.telemetry
        diagnostics = (
            {}
            if telemetry is None or telemetry.diagnostics is None
            else dict(telemetry.diagnostics)
        )
        self.target.last_batch_generation = {
            "path": LAGUNA_DFLASH_EXECUTION_PATH,
            "backend": self.target.backend,
            "quant": "gguf_q4_k_m",
            "batch_size": 1,
            "prompt_lengths": [len(tuple(prompt_ids))],
            "decode_steps": len(output.generated_token_ids or ()),
            "max_tokens": int(request.max_tokens),
            "resident_weights": getattr(self.target, "resident_weights", None)
            is not None,
            "throughput_claim_eligible": False,
            "speculative": {
                "provider": self.provider_name,
                "candidate_budget": LAGUNA_DFLASH_CANDIDATE_BUDGET,
                "target_iq3_selected_down_tile": self.target_iq3_selected_down_tile,
                "cycles": int(diagnostics.get("cycles", 0)),
                "accepted_draft_tokens": int(
                    diagnostics.get("accepted_draft_tokens", 0)
                ),
                "draft_tokens_proposed": int(
                    diagnostics.get("draft_tokens_proposed", 0)
                ),
                "target_verify_rows": int(
                    diagnostics.get("target_verify_rows", 0)
                ),
                "exactness_mode": "target_corrected_greedy",
                "performance_claim": False,
            },
        }

    def _prepare_request(self, request: GenerationRequest) -> tuple[int, ...]:
        self._check_open()
        blockers = [
            blocker
            for blocker in speculative_mtp_sampling_blockers(request)
            if blocker not in {"stop_token_ids", "stop_token_sequences"}
        ]
        if request.top_p != 1.0:
            blockers.append("top_p")
        if request.top_k != 0:
            blockers.append("top_k")
        if request.min_p != 0.0:
            blockers.append("min_p")
        if request.kv_storage not in {"auto", "bf16"}:
            blockers.append("kv_storage")
        if request.kv_scale_dtype != "fp16":
            blockers.append("kv_scale_dtype")
        if request.kv_scale_granularity != "per_token_head":
            blockers.append("kv_scale_granularity")
        blockers = list(dict.fromkeys(blockers))
        if blockers:
            raise NotImplementedError(
                "Laguna DFlash supports raw greedy BF16 c=1 only; "
                f"unsupported request fields: {', '.join(blockers)}"
            )
        return self.target._prepare_request(request)

    def _token_steps(
        self,
        request: GenerationRequest,
        prompt_ids: tuple[int, ...],
    ) -> Iterator[_DFlashTokenStep]:
        with self.target._lock:
            self._prepare_locked()
            assert self._target_session is not None
            assert self._drafter is not None
            assert self._cycle is not None
            self._reset_locked()
            prefill_started = time.perf_counter()
            counters = _DFlashCounters()
            generated: list[int] = []
            try:
                raise_if_generation_deadline_expired(request)
                result = self._cycle.prefill(prompt_ids)
                self._target_session.runtime.device_synchronize()
                prefill_seconds = time.perf_counter() - prefill_started
                token_id = int(result.next_token_id)
                generated.append(token_id)
                finish = _laguna_finish_details(
                    generated,
                    tokenizer=self.target.tokenizer,
                    request=request,
                )
                yield _DFlashTokenStep(
                    token_id=token_id,
                    generated_ids=tuple(generated),
                    finish_details=finish,
                    telemetry=self._telemetry(
                        prompt_ids,
                        generated,
                        finish=finish,
                        prefill_seconds=prefill_seconds,
                        counters=counters,
                    ),
                )
                if finish is not None:
                    return
                root = token_id
                stop_ids = self._cycle_stop_token_ids(request)
                while len(generated) < int(request.max_tokens):
                    raise_if_generation_deadline_expired(request)
                    remaining = int(request.max_tokens) - len(generated)
                    cycle_started = time.perf_counter()
                    cycle_result = self._cycle.run_cycle(
                        root,
                        remaining_decode=remaining,
                        stop_token_ids=stop_ids,
                    )
                    self._target_session.runtime.device_synchronize()
                    cycle_wall = time.perf_counter() - cycle_started
                    counters = counters.add_cycle(
                        cycle_result,
                        wall_seconds=cycle_wall,
                    )
                    visible = tuple(
                        int(token) for token in cycle_result.visible_output_ids
                    )
                    if not visible or len(visible) > remaining:
                        raise RuntimeError(
                            "Laguna DFlash cycle emitted an invalid visible-output count"
                        )
                    for visible_token in visible:
                        generated.append(visible_token)
                        finish = _laguna_finish_details(
                            generated,
                            tokenizer=self.target.tokenizer,
                            request=request,
                        )
                        yield _DFlashTokenStep(
                            token_id=visible_token,
                            generated_ids=tuple(generated),
                            finish_details=finish,
                            telemetry=self._telemetry(
                                prompt_ids,
                                generated,
                                finish=finish,
                                prefill_seconds=prefill_seconds,
                                counters=counters,
                            ),
                        )
                        if finish is not None:
                            return
                    next_token = cycle_result.target_result.next_token_id
                    if next_token is None:
                        raise RuntimeError(
                            "Laguna DFlash cycle ended without a terminal finish"
                        )
                    root = int(next_token)
            finally:
                self._reset_after_request_locked()

    def _cycle_stop_token_ids(self, request: GenerationRequest) -> tuple[int, ...]:
        stops = {
            int(token)
            for token in (
                self.target.tokenizer.eos_token_id,
                self.target.tokenizer.eot_token_id,
                *request.stop_token_ids,
            )
            if token is not None
        }
        return tuple(sorted(stops))

    def _telemetry(
        self,
        prompt_ids: Sequence[int],
        generated_ids: Sequence[int],
        *,
        finish: FinishDetails | None,
        prefill_seconds: float,
        counters: _DFlashCounters,
    ) -> GenerationTelemetry:
        suppressed = 0 if finish is None else _suppressed_suffix_length(finish)
        diagnostics = {
            "backend": self.target.backend,
            "model": "laguna_gguf",
            "quant": "gguf_q4_k_m",
            "provider": self.provider_name,
            "candidate_budget": LAGUNA_DFLASH_CANDIDATE_BUDGET,
            "target_iq3_selected_down_tile": self.target_iq3_selected_down_tile,
            "exactness_mode": "target_corrected_greedy",
            "performance_claim": False,
            "fallback_reason": LAGUNA_DFLASH_FALLBACK_REASON,
            "cycles": counters.cycles,
            "accepted_draft_tokens": counters.accepted_draft_tokens,
            "draft_tokens_proposed": counters.draft_tokens_proposed,
            "target_verify_rows": counters.target_verify_rows,
        }
        return GenerationTelemetry.from_decode_counts(
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated_ids),
            phase="done" if finish is not None else "answer",
            sampler_mode="greedy_fast",
            answer_tokens=max(0, len(generated_ids) - suppressed),
            execution_path=LAGUNA_DFLASH_EXECUTION_PATH,
            native_compact_prefill=False,
            native_caware_decode=False,
            serial_decode_fallback=False,
            native_sampler_rows=False,
            event="completed" if finish is not None else "token",
            timing={
                "prefill_ms": float(prefill_seconds) * 1_000.0,
                "decode_ms": counters.decode_seconds * 1_000.0,
                "draft_ms": counters.proposal_seconds * 1_000.0,
                "target_verify_ms": counters.target_verify_seconds * 1_000.0,
                "draft_commit_enqueue_ms": (
                    counters.draft_commit_enqueue_seconds * 1_000.0
                ),
                "provider_load_ms": float(self._load_seconds or 0.0) * 1_000.0,
            },
            timing_scope="request",
            group_rows=1,
            timing_owner=True,
            usage={
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(prompt_ids) + len(generated_ids),
            },
            diagnostics=diagnostics,
        )

    def _prepare_locked(self) -> None:
        self._check_open()
        if self._cycle is not None:
            return
        started = time.perf_counter()
        self.target._prepare_locked()
        target_session = self.target._open_session_locked(
            iq3_selected_down_tile=self.target_iq3_selected_down_tile,
        )
        drafter: LagunaDFlashResidentDrafter | None = None
        cycle: LagunaDFlashResidentCycle | None = None
        try:
            drafter = LagunaDFlashResidentDrafter(
                target_session,
                self.drafter_model,
                candidate_budget=LAGUNA_DFLASH_CANDIDATE_BUDGET,
                top_k=1,
                max_append_rows=64,
            )
            cycle = LagunaDFlashResidentCycle(target_session, drafter)
        except BaseException:
            if cycle is not None:
                cycle.close()
            if drafter is not None:
                drafter.close()
            target_session.close()
            raise
        self._target_session = target_session
        self._drafter = drafter
        self._cycle = cycle
        self._load_seconds = time.perf_counter() - started

    def _reset_locked(self) -> None:
        assert self._target_session is not None
        assert self._drafter is not None
        self._target_session.reset_state()
        self._drafter.reset_state()

    def _reset_after_request_locked(self) -> None:
        if self._target_session is None or self._drafter is None:
            return
        if bool(getattr(self._target_session, "closed", False)) or bool(
            getattr(self._drafter, "_closed", False)
        ):
            self._close_locked(suppress_errors=True)
            return
        try:
            self._reset_locked()
        except BaseException:
            self._close_locked(suppress_errors=True)
            raise

    def _validate_identities(self) -> None:
        cache = self.target.repacked_cache_path
        if cache is None:
            raise ValueError(
                "Laguna DFlash requires the source-bound sibling repacked cache"
            )
        manifest_path = Path(cache) / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            target_sha = str(manifest["source"]["sha256"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Laguna DFlash target cache manifest lacks source SHA-256"
            ) from exc
        if target_sha != LAGUNA_DFLASH_TARGET_SHA256:
            raise ValueError(
                "Laguna DFlash target SHA-256 does not match the admitted Q4_K_M"
            )
        model_file = self.drafter_model / "model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(
                f"Laguna DFlash model.safetensors not found: {model_file}"
            )
        resolved = model_file.resolve()
        content_addressed = (
            resolved.name == LAGUNA_DFLASH_DRAFTER_SHA256
            and resolved.parent.name == "blobs"
        )
        if not content_addressed and _sha256_file(resolved) != LAGUNA_DFLASH_DRAFTER_SHA256:
            raise ValueError(
                "Laguna DFlash drafter SHA-256 does not match the pinned artifact"
            )
        snapshot_revision = (
            self.drafter_model.name
            if self.drafter_model.parent.name == "snapshots"
            else None
        )
        if (
            snapshot_revision is not None
            and snapshot_revision != LAGUNA_DFLASH_DRAFTER_REVISION
        ):
            raise ValueError(
                "Laguna DFlash drafter revision does not match the pinned artifact"
            )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Laguna DFlash public provider is closed")

    def _close_locked(self, *, suppress_errors: bool) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for owner_name in ("_cycle", "_drafter", "_target_session"):
            owner = getattr(self, owner_name)
            setattr(self, owner_name, None)
            if owner is None:
                continue
            try:
                owner.close()
            except BaseException as exc:  # pragma: no cover - defensive cleanup
                errors.append(exc)
        if errors and not suppress_errors:
            raise errors[0]


def _sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def make_laguna_dflash_text_provider(
    *,
    target_generator: LagunaGGUFGenerator,
    config: SpeculativeProviderConfig,
) -> LagunaDFlashTextProvider:
    return LagunaDFlashTextProvider(target_generator, config)


register_speculative_provider(
    SpeculativeProviderKey(
        provider="dflash",
        target_model="laguna_gguf",
        backend="hip_gfx1151",
        quant="gguf_q4_k_m",
    ),
    make_laguna_dflash_text_provider,
    replace=True,
)


__all__ = [
    "LAGUNA_DFLASH_CANDIDATE_BUDGET",
    "LAGUNA_DFLASH_DRAFTER_REVISION",
    "LAGUNA_DFLASH_DRAFTER_SHA256",
    "LAGUNA_DFLASH_ECONOMICS_EVIDENCE",
    "LAGUNA_DFLASH_EXECUTION_PATH",
    "LAGUNA_DFLASH_FALLBACK_REASON",
    "LAGUNA_DFLASH_IQ3_SELECTED_DOWN_TILE_ENV",
    "LAGUNA_DFLASH_TARGET_SHA256",
    "LagunaDFlashTextProvider",
    "make_laguna_dflash_text_provider",
]
