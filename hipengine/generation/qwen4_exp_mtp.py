"""Public correctness-first Qwen4Exp MTP provider at 512/1K context."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any

from hipengine.generation.deadline import raise_if_generation_deadline_expired
from hipengine.generation.registry import (
    DecodeState,
    FinishDetails,
    GenerationOutput,
    GenerationRequest,
    GenerationStreamChunk,
    GenerationTelemetry,
)
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.loading.qwen4_exp_mtp_gguf import build_qwen4_exp_mtp_gguf_map
from hipengine.loading.qwen4_exp_mtp_materialize import (
    Qwen4ExpMTPResidentWeights,
    materialize_qwen4_exp_mtp_weights,
    plan_qwen4_exp_mtp_residency,
)
from hipengine.runtime.qwen4_exp_mtp import Qwen4ExpGGUFMTPDraftRunner
from hipengine.speculative.registry import (
    SpeculativeProviderCapabilities,
    SpeculativeProviderConfig,
    SpeculativeProviderKey,
    register_speculative_provider,
)

_PROVIDER = "qwen4_exp_mtp"
_SIDECAR_SHA256 = "9db03a687670608286e99b563fcc86d0ee76c8dd863f64b2afc0b54eb0eb975d"


@dataclass(frozen=True)
class Qwen4ExpMTPCycle:
    start_position: int
    candidates: tuple[int, ...]
    accepted: int
    mismatch_token: int | None
    committed_position: int


class Qwen4ExpMTPTextProvider:
    """Greedy MTP with exact serial target verification and draft-cursor trim."""

    provider_name = _PROVIDER

    def __init__(
        self,
        *,
        target_generator: Any,
        config: SpeculativeProviderConfig,
        draft_runner: Any | None = None,
        draft_resident: Qwen4ExpMTPResidentWeights | None = None,
    ) -> None:
        self.target_generator = target_generator
        self.config = config
        self.candidate_budget = int(config.candidate_budget)
        if not 1 <= self.candidate_budget <= 4:
            raise ValueError("Qwen4Exp MTP candidate budget must be in 1..4")
        self._resident = draft_resident
        self._owns_resident = draft_resident is None
        self._owns_runner = draft_runner is None
        self.closed = False
        self.last_cycles: tuple[Qwen4ExpMTPCycle, ...] = ()
        if draft_runner is None:
            paths = discover_gguf_files(Path(config.draft_model))
            readers = tuple(GGUFReader(path) for path in paths)
            model_map = build_qwen4_exp_mtp_gguf_map(
                tuple(reader.info for reader in readers)
            )
            plan = plan_qwen4_exp_mtp_residency(model_map)
            self._resident = materialize_qwen4_exp_mtp_weights(
                readers,
                plan=plan,
                backend=target_generator.backend,
                runtime=target_generator.runner.runtime,
            )
            try:
                draft_runner = Qwen4ExpGGUFMTPDraftRunner(
                    self._resident,
                    target_config=target_generator.runner.config,
                    max_sequence_length=min(
                        1_024, int(target_generator.runner.max_sequence_length)
                    ),
                    backend=target_generator.backend,
                    runtime=target_generator.runner.runtime,
                )
            except Exception:
                self._resident.close()
                self._resident = None
                raise
        self.draft = draft_runner

    def capabilities(self) -> dict[str, Any]:
        row = SpeculativeProviderCapabilities(
            provider_name=self.provider_name,
            artifact_fingerprint=_SIDECAR_SHA256,
            attachment_mode="model_attached",
            supported_modes=("verify_chain",),
            max_verifier_rows=4,
            transaction_mode="serial_exact_target_verify",
            provider_state_key="qwen4exp_mtp_dense_state",
            provider_kv_key="qwen4exp_mtp_dense_kv",
            fixed_transaction_units=(("qwen4exp_mtp.request", 1),),
            per_candidate_units=(("qwen4exp_mtp.candidate", 1),),
            strict_fallback="target_ar",
        )
        return {
            "provider_name": row.provider_name,
            "artifact_fingerprint": row.artifact_fingerprint,
            "attachment_mode": row.attachment_mode,
            "supported_modes": list(row.supported_modes),
            "max_verifier_rows": row.max_verifier_rows,
            "transaction_mode": row.transaction_mode,
            "strict_fallback": row.strict_fallback,
            "candidate_budget": self.candidate_budget,
            "context_limit": min(1_024, int(self.draft.max_sequence_length)),
            "streaming_mode": "buffered_public",
        }

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        self._require_open()
        if request.temperature != 0.0 or request.top_k not in (0, 1):
            raise NotImplementedError("Qwen4Exp MTP bring-up supports greedy generation only")
        outputs = [self._generate_one(prompt, request) for prompt in request.prompts]
        return outputs

    def _generate_one(self, prompt: Any, request: GenerationRequest) -> GenerationOutput:
        raise_if_generation_deadline_expired(request)
        token_ids = (
            [int(token) for token in prompt]
            if not isinstance(prompt, str)
            else [int(token) for token in self.target_generator.tokenizer.encode(prompt)]
        )
        if not token_ids:
            raise ValueError("Qwen4Exp MTP prompt produced no token IDs")
        if len(token_ids) + request.max_tokens > min(
            1_024,
            int(self.target_generator.runner.max_sequence_length),
            int(self.draft.max_sequence_length),
        ):
            raise ValueError("Qwen4Exp MTP request exceeds the 512/1K bring-up scope")
        if request.max_tokens == 0:
            return GenerationOutput(
                text="",
                generated_token_ids=(),
                finish_details=FinishDetails(
                    reason="length", length_limit=0, sampler_mode="greedy_speculative_mtp"
                ),
            )

        target = self.target_generator.runner
        phase_ms: dict[str, float] = {}
        phase_calls: dict[str, int] = {}
        resident_handoff = (
            os.environ.get("HIPENGINE_QWEN4_EXP_MTP_COMPACT_OUTPUT", "0")
            not in {"", "0", "false", "False"}
            and self.draft.runtime is target.runtime
        )

        def record_phase(name: str, started: float) -> None:
            phase_ms[name] = phase_ms.get(name, 0.0) + (
                time.perf_counter() - started
            ) * 1_000.0
            phase_calls[name] = phase_calls.get(name, 0) + 1

        prefill_started = time.perf_counter()
        result = target.prefill(token_ids, capture_hidden_seeds=True)
        record_phase("target_prefill_hidden_export", prefill_started)
        if result.hidden_seeds is None or result.hidden_seed is None:
            raise RuntimeError("Qwen4Exp target did not expose MTP hidden rows")
        self.draft.prime_prompt(token_ids, result.hidden_seeds)
        if int(target.position) != int(self.draft.position):
            raise RuntimeError("Qwen4Exp target/draft prompt cursors diverged")

        generated: list[int] = [int(result.token_id)]
        root_token = generated[0]
        root_hidden = result.hidden_seed
        cycles: list[Qwen4ExpMTPCycle] = []
        reason = "length"
        eos = int(self.target_generator.tokenizer.eos_token_id)
        if not request.ignore_eos and root_token == eos:
            reason = "eos"

        while len(generated) < request.max_tokens and reason != "eos":
            raise_if_generation_deadline_expired(request)
            budget = min(self.candidate_budget, request.max_tokens - len(generated))
            start_position = int(target.position)
            if int(self.draft.position) != start_position:
                raise RuntimeError("Qwen4Exp target/draft cursors diverged before proposal")
            proposal_started = time.perf_counter()
            proposal = self.draft.propose_chain(
                start_token=root_token,
                target_hidden_seed=(None if resident_handoff else root_hidden),
                target_hidden_seed_ptr=(
                    target.last_target_hidden.ptr if resident_handoff else None
                ),
                draft_n_max=budget,
            )
            record_phase("proposal", proposal_started)
            for name, elapsed_ms in getattr(
                self.draft, "last_proposal_stage_timings_ms", {}
            ).items():
                phase_ms[name] = phase_ms.get(name, 0.0) + float(elapsed_ms)
            candidates = tuple(int(row.token_id) for row in proposal)
            accepted = 0
            mismatch: int | None = None
            for candidate in candidates:
                raise_if_generation_deadline_expired(request)
                verify_started = time.perf_counter()
                verified = target.step(
                    root_token, capture_hidden_seed=not resident_handoff
                )
                record_phase("target_verify", verify_started)
                acceptance_started = time.perf_counter()
                if not resident_handoff and verified.hidden_seed is None:
                    raise RuntimeError("Qwen4Exp target verify row has no hidden seed")
                truth = int(verified.token_id)
                root_hidden = verified.hidden_seed
                root_token = truth
                generated.append(truth)
                if truth == candidate:
                    accepted += 1
                else:
                    mismatch = truth
                if not request.ignore_eos and truth == eos:
                    reason = "eos"
                should_stop = (
                    mismatch is not None
                    or reason == "eos"
                    or len(generated) >= request.max_tokens
                )
                record_phase("acceptance_control", acceptance_started)
                if should_stop:
                    break
            trim_started = time.perf_counter()
            self.draft.trim(int(target.position))
            record_phase("draft_commit_or_rollback", trim_started)
            cycles.append(
                Qwen4ExpMTPCycle(
                    start_position=start_position,
                    candidates=candidates,
                    accepted=accepted,
                    mismatch_token=mismatch,
                    committed_position=int(target.position),
                )
            )
        self.last_cycles = tuple(cycles)
        proposed = sum(len(cycle.candidates) for cycle in cycles)
        accepted = sum(cycle.accepted for cycle in cycles)
        telemetry = GenerationTelemetry(
            decode_state=DecodeState(
                prompt_tokens=len(token_ids),
                generated_tokens=len(generated),
                step_index=len(generated),
                sampler_mode="greedy_speculative_mtp",
                execution_path="qwen4exp_mtp_serial_exact_verify",
            ),
            event="generation_complete",
            diagnostics={
                "speculative_provider": self.provider_name,
                "candidate_budget": self.candidate_budget,
                "proposed_draft_tokens": proposed,
                "accepted_draft_tokens": accepted,
                "draft_acceptance": (
                    float(accepted / proposed) if proposed else 0.0
                ),
                "cycles": [
                    {
                        "start_position": cycle.start_position,
                        "candidates": list(cycle.candidates),
                        "accepted": cycle.accepted,
                        "mismatch_token": cycle.mismatch_token,
                        "committed_position": cycle.committed_position,
                    }
                    for cycle in cycles
                ],
                "target_verify": "serial_exact",
                "draft_rollback": "cursor_trim",
                "target_hidden_handoff": (
                    "device_to_device" if resident_handoff else "host"
                ),
                "phase_census": {
                    "cycles": len(cycles),
                    "target_verify_rows": phase_calls.get("target_verify", 0),
                    "target_prefill_hidden_export": {
                        "calls": phase_calls.get("target_prefill_hidden_export", 0),
                        "ms": phase_ms.get("target_prefill_hidden_export", 0.0),
                    },
                    "proposal": {
                        "calls": phase_calls.get("proposal", 0),
                        "ms": phase_ms.get("proposal", 0.0),
                    },
                    "target_verify": {
                        "calls": phase_calls.get("target_verify", 0),
                        "ms": phase_ms.get("target_verify", 0.0),
                        "includes_hidden_export": True,
                    },
                    "acceptance_control": {
                        "calls": phase_calls.get("acceptance_control", 0),
                        "ms": phase_ms.get("acceptance_control", 0.0),
                    },
                    "draft_commit_or_rollback": {
                        "calls": phase_calls.get("draft_commit_or_rollback", 0),
                        "ms": phase_ms.get("draft_commit_or_rollback", 0.0),
                    },
                    "draft_stages_ms": {
                        name: elapsed_ms
                        for name, elapsed_ms in sorted(phase_ms.items())
                        if name.startswith("draft_")
                        and name != "draft_commit_or_rollback"
                    },
                },
            },
        )
        return GenerationOutput(
            text=self.target_generator.tokenizer.decode(
                generated, skip_special=False
            ),
            generated_token_ids=tuple(generated),
            telemetry=telemetry,
            finish_details=FinishDetails(
                reason=reason,
                eos_token_id=eos if reason == "eos" else None,
                length_limit=request.max_tokens if reason == "length" else None,
                sampler_mode="greedy_speculative_mtp",
            ),
        )

    def stream_detailed(self, request: GenerationRequest):
        self._require_open()
        if len(request.prompts) != 1:
            raise ValueError("Qwen4Exp MTP streaming requires exactly one prompt")
        output = self.generate_detailed(request)[0]
        # Correctness-first public SSE is explicitly buffered. Token-live
        # streaming follows after the basic product gate, without relabeling ITL.
        yield GenerationStreamChunk(
            text=output.text,
            finish_details=output.finish_details,
            telemetry=output.telemetry,
            generated_token_ids=output.generated_token_ids,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._owns_runner and self.draft is not None:
            self.draft.close()
        if self._owns_resident and self._resident is not None:
            self._resident.close()
            self._resident = None

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Qwen4Exp MTP provider is closed")


def make_qwen4_exp_mtp_provider(
    *,
    target_generator: Any,
    config: SpeculativeProviderConfig,
) -> Qwen4ExpMTPTextProvider:
    return Qwen4ExpMTPTextProvider(target_generator=target_generator, config=config)


def register_qwen4_exp_mtp_providers(*, replace: bool = True) -> None:
    for quant in ("gguf_q4_k_m", "gguf_ud_q4_k_xl"):
        register_speculative_provider(
            SpeculativeProviderKey(
                provider=_PROVIDER,
                target_model="qwen4_exp_gguf",
                backend="hip_gfx1151",
                quant=quant,
            ),
            make_qwen4_exp_mtp_provider,
            replace=replace,
        )


register_qwen4_exp_mtp_providers()


__all__ = [
    "Qwen4ExpMTPCycle",
    "Qwen4ExpMTPTextProvider",
    "make_qwen4_exp_mtp_provider",
    "register_qwen4_exp_mtp_providers",
]
