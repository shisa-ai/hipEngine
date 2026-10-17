"""Public correctness-first Qwen4Exp MTP provider at 512/1K context."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Generator

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


@dataclass(frozen=True)
class Qwen4ExpMTPProgress:
    """One committed step of a Qwen4Exp MTP generation.

    ``token_ids`` are the tokens this step newly committed, never the running
    total. The initial root-token step and scheduler-selected K0 steps carry
    ``cycle=None``; speculative steps carry one :class:`Qwen4ExpMTPCycle`.
    ``finished`` marks the last step the iterator emits.
    """

    token_ids: tuple[int, ...]
    cycle: Qwen4ExpMTPCycle | None
    finished: bool


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
                    max_sequence_length=int(target_generator.runner.max_sequence_length),
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
            "context_limit": int(self.draft.max_sequence_length),
            "streaming_mode": "buffered_public",
        }

    def generate_detailed(self, request: GenerationRequest) -> list[GenerationOutput]:
        self._require_open()
        if request.temperature != 0.0 or request.top_k not in (0, 1):
            raise NotImplementedError("Qwen4Exp MTP bring-up supports greedy generation only")
        outputs = [self._generate_one(prompt, request) for prompt in request.prompts]
        return outputs

    def iter_generate(
        self,
        prompt: Any,
        request: GenerationRequest,
        *,
        target: Any | None = None,
        draft: Any | None = None,
        prefill_result: Any | None = None,
        draft_primed: bool = False,
    ) -> Generator[Qwen4ExpMTPProgress, int | None, GenerationOutput]:
        """Yield one committed MTP step at a time and return the final output.

        Lazy: creating the iterator performs no model work. The first step is
        the initial root token with ``cycle=None``; by default each later step
        is one existing verification cycle plus its committed tokens.
        After the step carrying ``finished=True``, the next ``next()`` raises
        ``StopIteration`` whose value is the complete ``GenerationOutput`` with
        the usual telemetry.

        ``target`` and ``draft`` are request-owned overrides for the loaded
        owners; the provider never swaps ``target_generator.runner`` or
        ``draft``. ``prefill_result`` reuses an existing target prefill and
        ``draft_primed=True`` reuses an already primed draft cursor. After each
        yield, ``send(k)`` selects the next cycle's candidate budget (zero is a
        target-only step with draft catchup). ``next()`` retains the configured
        budget. The scheduler owns cancellation rollback around a step.
        """

        self._require_open()
        if request.temperature != 0.0 or request.top_k not in (0, 1):
            raise NotImplementedError("Qwen4Exp MTP supports greedy generation only")
        raise_if_generation_deadline_expired(request)
        token_ids = (
            [int(token) for token in prompt]
            if not isinstance(prompt, str)
            else [int(token) for token in self.target_generator.tokenizer.encode(prompt)]
        )
        if not token_ids:
            raise ValueError("Qwen4Exp MTP prompt produced no token IDs")
        target_runner = self.target_generator.runner if target is None else target
        draft_runner = self.draft if draft is None else draft
        capacity = min(
            int(target_runner.max_sequence_length),
            int(draft_runner.max_sequence_length),
        )
        if len(token_ids) + request.max_tokens > capacity:
            raise ValueError(
                "Qwen4Exp MTP request exceeds the admitted target/draft context "
                f"capacity ({capacity} tokens)"
            )
        if request.max_tokens == 0:
            return GenerationOutput(
                text="",
                generated_token_ids=(),
                finish_details=FinishDetails(
                    reason="length", length_limit=0, sampler_mode="greedy_speculative_mtp"
                ),
            )

        phase_ms: dict[str, float] = {}
        phase_calls: dict[str, int] = {}

        def record_phase(name: str, started: float) -> None:
            phase_ms[name] = phase_ms.get(name, 0.0) + (
                time.perf_counter() - started
            ) * 1_000.0
            phase_calls[name] = phase_calls.get(name, 0) + 1

        prefill_started = time.perf_counter()
        if prefill_result is None:
            result = target_runner.prefill(token_ids, capture_hidden_seeds=True)
            record_phase("target_prefill_hidden_export", prefill_started)
        else:
            # The caller already prefilled this prompt; reuse those hidden rows
            # and the cursor they left behind instead of prefilling twice.
            result = prefill_result
        if result.hidden_seeds is None or result.hidden_seed is None:
            raise RuntimeError("Qwen4Exp target did not expose MTP hidden rows")
        if not draft_primed:
            draft_runner.prime_prompt(token_ids, result.hidden_seeds)
        if int(target_runner.position) != int(draft_runner.position):
            raise RuntimeError("Qwen4Exp target/draft prompt cursors diverged")

        generated: list[int] = [int(result.token_id)]
        root_token = generated[0]
        root_hidden = result.hidden_seed
        cycles: list[Qwen4ExpMTPCycle] = []
        reason = "length"
        eos = int(self.target_generator.tokenizer.eos_token_id)
        if not request.ignore_eos and root_token == eos:
            reason = "eos"

        next_budget = yield Qwen4ExpMTPProgress(
            token_ids=(root_token,),
            cycle=None,
            finished=not (len(generated) < request.max_tokens and reason != "eos"),
        )

        # One prefill-shaped verify pass per cycle, committing the accepted
        # rows' states (no replay); otherwise serial target steps.
        batched_verify = os.environ.get(
            "HIPENGINE_QWEN4_EXP_MTP_BATCHED_VERIFY", "1"
        ) not in {"", "0", "false", "False"} and hasattr(
            target_runner, "verify_target_block_state_rows"
        )
        while len(generated) < request.max_tokens and reason != "eos":
            step_start = len(generated)
            raise_if_generation_deadline_expired(request)
            remaining = request.max_tokens - len(generated)
            selected_budget = self.candidate_budget if next_budget is None else int(next_budget)
            if not 0 <= selected_budget <= self.candidate_budget:
                raise ValueError("scheduler candidate budget exceeds provider bounds")
            if selected_budget == 0:
                if int(target_runner.position) != int(draft_runner.position):
                    raise RuntimeError("Qwen4Exp target/draft cursors diverged before K0")
                catchup_started = time.perf_counter()
                draft_runner.forward(
                    root_token, root_hidden, capture_logits=False,
                    capture_hidden_seed=False, capture_token_id=False, run_head=False,
                )
                record_phase("draft_k0_catchup", catchup_started)
                target_started = time.perf_counter()
                verified = target_runner.step(root_token, capture_hidden_seed=True)
                record_phase("target_k0", target_started)
                if verified.hidden_seed is None:
                    raise RuntimeError("Qwen4Exp target did not expose K0 hidden row")
                if int(target_runner.position) != int(draft_runner.position):
                    raise RuntimeError("Qwen4Exp target/draft cursors diverged after K0")
                root_token = int(verified.token_id)
                root_hidden = verified.hidden_seed
                generated.append(root_token)
                if not request.ignore_eos and root_token == eos:
                    reason = "eos"
                next_budget = yield Qwen4ExpMTPProgress(
                    token_ids=(root_token,), cycle=None,
                    finished=len(generated) >= request.max_tokens or reason == "eos",
                )
                continue
            # A batched cycle commits up to budget + 1 tokens (the row after the
            # last draft is a bonus), so leave that row of headroom.
            bonus_row = batched_verify and remaining >= 2
            budget = min(
                selected_budget, remaining - 1 if bonus_row else remaining,
                7 if bonus_row else remaining,
            )
            start_position = int(target_runner.position)
            if int(draft_runner.position) != start_position:
                raise RuntimeError("Qwen4Exp target/draft cursors diverged before proposal")
            proposal_started = time.perf_counter()
            proposal = draft_runner.propose_chain(
                start_token=root_token,
                target_hidden_seed=root_hidden,
                draft_n_max=budget,
                # Device argmax and device-resident chaining; only the ids
                # come back. Same draft tokens as the logits path.
                compact_output=os.environ.get(
                    "HIPENGINE_QWEN4_EXP_MTP_COMPACT_PROPOSAL", "1"
                ) not in {"", "0", "false", "False"},
            )
            record_phase("proposal", proposal_started)
            for name, elapsed_ms in getattr(
                draft_runner, "last_proposal_stage_timings_ms", {}
            ).items():
                phase_ms[name] = phase_ms.get(name, 0.0) + float(elapsed_ms)
            candidates = tuple(int(row.token_id) for row in proposal)
            accepted = 0
            mismatch: int | None = None
            serial_candidates = candidates
            if bonus_row and candidates:
                # Rows [root, c1 .. cn] score every candidate: row i's target
                # token checks candidate i, the first mismatch ends the cycle
                # with the target's own token for that row, and when every
                # candidate matches the last row is one more (bonus) token.
                serial_candidates = ()
                inputs = (root_token, *candidates)
                verify_started = time.perf_counter()
                block = target_runner.verify_target_block_state_rows(inputs)
                record_phase("target_verify", verify_started)
                acceptance_started = time.perf_counter()
                consumed = 0
                for truth, candidate in zip(
                        block.token_ids, (*candidates, None), strict=True):
                    consumed += 1
                    generated.append(int(truth))
                    if candidate is not None and int(truth) != candidate:
                        mismatch = int(truth)
                    elif candidate is not None:
                        accepted += 1
                    if not request.ignore_eos and int(truth) == eos:
                        reason = "eos"
                    if (mismatch is not None or reason == "eos"
                            or len(generated) >= request.max_tokens):
                        break
                target_runner.commit_target_block_rows(consumed)
                assert block.hidden_seeds is not None
                if consumed == len(inputs):
                    # Every draft matched and the bonus row was committed: the
                    # draft has consumed root .. c(n-1), so it also takes cn,
                    # with the target's hidden row that produced it.
                    draft_runner.forward(
                        candidates[-1], block.hidden_seeds[len(candidates) - 1],
                        capture_logits=False, capture_hidden_seed=False,
                        capture_token_id=False, run_head=False,
                    )
                root_token = int(block.token_ids[consumed - 1])
                root_hidden = block.hidden_seeds[consumed - 1]
                record_phase("acceptance_control", acceptance_started)
            for candidate in serial_candidates:
                raise_if_generation_deadline_expired(request)
                verify_started = time.perf_counter()
                verified = target_runner.step(root_token, capture_hidden_seed=True)
                record_phase("target_verify", verify_started)
                acceptance_started = time.perf_counter()
                if verified.hidden_seed is None:
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
            draft_runner.trim(int(target_runner.position))
            record_phase("draft_commit_or_rollback", trim_started)
            cycles.append(
                Qwen4ExpMTPCycle(
                    start_position=start_position,
                    candidates=candidates,
                    accepted=accepted,
                    mismatch_token=mismatch,
                    committed_position=int(target_runner.position),
                )
            )
            next_budget = yield Qwen4ExpMTPProgress(
                token_ids=tuple(generated[step_start:]),
                cycle=cycles[-1],
                finished=not (
                    len(generated) < request.max_tokens and reason != "eos"
                ),
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
                execution_path=(
                    "qwen4exp_mtp_batched_state_rows_verify"
                    if batched_verify else "qwen4exp_mtp_serial_exact_verify"
                ),
            ),
            event="generation_complete",
            timing={
                "mtp_generated_draft_tokens": proposed,
                "mtp_accepted_draft_tokens": accepted,
                "mtp_cycles_count": len(cycles),
            },
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
                "target_verify": (
                    "batched_state_rows" if batched_verify else "serial_exact"
                ),
                "draft_rollback": "cursor_trim",
                "target_hidden_handoff": "host",
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
                    "target_k0": {
                        "calls": phase_calls.get("target_k0", 0),
                        "ms": phase_ms.get("target_k0", 0.0),
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

    def _generate_one(self, prompt: Any, request: GenerationRequest) -> GenerationOutput:
        steps = self.iter_generate(prompt, request)
        while True:
            try:
                next(steps)
            except StopIteration as stop:
                return stop.value

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
    "Qwen4ExpMTPProgress",
    "Qwen4ExpMTPTextProvider",
    "make_qwen4_exp_mtp_provider",
    "register_qwen4_exp_mtp_providers",
]
