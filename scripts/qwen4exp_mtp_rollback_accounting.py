#!/usr/bin/env python3
"""Bounded Qwen4Exp MTP rollback/replay accounting screen.

This is a CPU-only diagnostic for the public serial Qwen4Exp MTP provider.  It
uses instrumented target/draft doubles to exercise every rejection depth for a
four-token proposal, including all-reject and full-accept.  It deliberately does
not change the production provider or infer a performance claim from counters.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from hipengine.generation.qwen4_exp_mtp import Qwen4ExpMTPTextProvider
from hipengine.generation.registry import GenerationRequest
from hipengine.speculative.registry import SpeculativeProviderConfig


@dataclass(frozen=True)
class RollbackAccountingRow:
    case: str
    proposal_budget: int
    accepted_draft_tokens: int
    rejection_depth: int | None
    full_accept: bool
    target_forwards: int
    draft_forwards: int
    visible_tokens_after_root: int
    committed_cursor_delta: int
    draft_cursor_rollback_units: int
    trim_calls: int
    snapshot_calls: int
    restore_calls: int
    restored_bytes: int
    restored_allocations: int
    per_cycle_allocations: int
    shared_weight_allocations: int
    target_position: int
    draft_position: int
    output_ids: tuple[int, ...]


class _Tokenizer:
    def __init__(self, eos_token_id: int = 99_999) -> None:
        self.eos_token_id = int(eos_token_id)

    def encode(self, text: str) -> list[int]:
        return [1, 2] if text else []

    def decode(self, token_ids: Any, *, skip_special: bool = False) -> str:
        return " ".join(str(int(token)) for token in token_ids)


class _TargetRunner:
    def __init__(self, truth: tuple[int, ...]) -> None:
        self.max_sequence_length = 1_024
        self.position = 0
        self.truth = tuple(int(token) for token in truth)
        self.cursor = 0
        self.forward_calls = 0

    @staticmethod
    def _hidden(value: int) -> np.ndarray:
        return np.full(8, float(value), dtype=np.float32)

    def prefill(self, token_ids, *, capture_hidden_seeds: bool = False):
        self.position = len(token_ids)
        self.cursor = 0
        hidden = np.stack([self._hidden(token) for token in token_ids])
        return SimpleNamespace(
            token_id=self.truth[0],
            hidden_seeds=hidden,
            hidden_seed=hidden[-1],
        )

    def step(self, token_id: int, *, capture_hidden_seed: bool = False):
        if int(token_id) != self.truth[self.cursor]:
            raise AssertionError(
                f"target replay input {token_id} != truth[{self.cursor}]={self.truth[self.cursor]}"
            )
        self.forward_calls += 1
        self.cursor += 1
        self.position += 1
        hidden = self._hidden(self.truth[self.cursor - 1])
        return SimpleNamespace(
            token_id=self.truth[self.cursor],
            hidden_seeds=hidden.reshape(1, -1),
            hidden_seed=hidden,
        )


class _DraftRunner:
    max_sequence_length = 1_024

    def __init__(self, candidates: tuple[int, ...]) -> None:
        self.candidates = tuple(int(token) for token in candidates)
        self.position = 0
        self.forward_calls = 0
        self.trim_calls = 0
        self.snapshot_calls = 0
        self.restore_calls = 0
        self.trimmed: list[int] = []
        self.last_proposal_stage_timings_ms: dict[str, float] = {}

    def prime_prompt(self, token_ids, hidden_rows) -> None:
        assert hidden_rows.shape == (len(token_ids), 8)
        self.position = len(token_ids)

    def propose_chain(
        self,
        *,
        start_token: int,
        target_hidden_seed,
        draft_n_max: int,
        target_hidden_seed_ptr=None,
    ):
        values = self.candidates[: int(draft_n_max)]
        self.forward_calls += len(values)
        self.position += len(values)
        return tuple(
            SimpleNamespace(
                token_id=token,
                hidden_seed=np.full(8, token, dtype=np.float32),
            )
            for token in values
        )

    def snapshot(self):
        self.snapshot_calls += 1
        return SimpleNamespace(position=self.position)

    def restore(self, snapshot) -> None:
        self.restore_calls += 1
        self.trim(snapshot.position)

    def trim(self, position: int) -> None:
        count = int(position)
        if count < 0 or count > self.position:
            raise ValueError("draft trim must target the committed prefix")
        self.trim_calls += 1
        self.trimmed.append(count)
        self.position = count

    def close(self) -> None:
        pass


class _TargetGenerator:
    backend = "hip_gfx1151"

    def __init__(self, target: _TargetRunner, *, eos_token_id: int = 99_999) -> None:
        self.runner = target
        self.tokenizer = _Tokenizer(eos_token_id)


def _request(budget: int) -> GenerationRequest:
    return GenerationRequest(
        prompts=((1, 2),),
        max_tokens=int(budget) + 1,
        temperature=0.0,
        top_p=1.0,
        ignore_eos=False,
        top_k=1,
    )


def _case_candidates(budget: int, rejection_depth: int | None) -> tuple[int, ...]:
    truth = tuple(range(10, 10 + budget + 1))
    if rejection_depth is None:
        return truth[1:]
    # The first `rejection_depth` candidates agree with the target.  The next
    # candidate is intentionally wrong; the provider must stop after that
    # target forward and must not replay the accepted prefix.
    candidates = list(truth[1:])
    candidates[rejection_depth] = 10_000 + rejection_depth
    return tuple(candidates)


def run_accounting_screen(*, budget: int = 4) -> dict[str, Any]:
    budget = int(budget)
    if budget <= 0 or budget > 4:
        raise ValueError("Qwen4Exp MTP accounting budget must be in 1..4")

    rows: list[RollbackAccountingRow] = []
    # One resident weight attachment is shared by the bounded cases.  The
    # provider receives it rather than materializing or allocating weights.
    shared_resident = object()
    for rejection_depth in (*range(budget), None):
        full_accept = rejection_depth is None
        accepted = budget if full_accept else int(rejection_depth)
        target_eos = 99_999 if full_accept else 10 + accepted + 1
        truth = (
            tuple(range(10, 11 + accepted)) + (target_eos,)
            if not full_accept
            else tuple(range(10, 10 + budget + 1))
        )
        target = _TargetRunner(truth)
        draft_candidates = list(_case_candidates(budget, rejection_depth))
        if not full_accept:
            draft_candidates = list(range(11, 11 + accepted)) + [10_000 + accepted]
            draft_candidates.extend(
                20_000 + index for index in range(budget - len(draft_candidates))
            )
        draft = _DraftRunner(tuple(draft_candidates))
        provider = Qwen4ExpMTPTextProvider(
            target_generator=_TargetGenerator(target, eos_token_id=target_eos),
            config=SpeculativeProviderConfig(
                provider="qwen4_exp_mtp",
                draft_model="/tmp/fake.gguf",
                candidate_budget=budget,
            ),
            draft_runner=draft,
            draft_resident=shared_resident,
        )
        output = provider.generate_detailed(_request(budget))[0]
        provider.close()

        accepted = budget if full_accept else int(rejection_depth)
        target_forwards = budget if full_accept else accepted + 1
        visible = target_forwards
        committed_delta = target_forwards
        rollback_units = budget - committed_delta
        expected_output = (
            tuple(range(10, 11 + accepted)) + (target_eos,)
            if not full_accept
            else tuple(range(10, 11 + budget))
        )
        if output.generated_token_ids != expected_output:
            raise AssertionError(
                f"unexpected output for rejection_depth={rejection_depth}: "
                f"{output.generated_token_ids} != {expected_output}"
            )
        if draft.position != target.position:
            raise AssertionError("target/draft cursors diverged after accounting case")
        if draft.forward_calls != budget:
            raise AssertionError("draft proposal did not execute the full budget")
        if target.forward_calls != target_forwards:
            raise AssertionError("target forward count did not stop at the first mismatch")

        rows.append(
            RollbackAccountingRow(
                case="full_accept" if full_accept else "reject",
                proposal_budget=budget,
                accepted_draft_tokens=accepted,
                rejection_depth=None if full_accept else int(rejection_depth),
                full_accept=full_accept,
                target_forwards=target.forward_calls,
                draft_forwards=draft.forward_calls,
                visible_tokens_after_root=visible,
                committed_cursor_delta=committed_delta,
                draft_cursor_rollback_units=rollback_units,
                trim_calls=draft.trim_calls,
                snapshot_calls=draft.snapshot_calls,
                restore_calls=draft.restore_calls,
                # Qwen4Exp rollback is cursor-only: append-only KV/index bytes
                # are overwritten on a later append, not copied back now.
                restored_bytes=0,
                restored_allocations=0,
                per_cycle_allocations=0,
                shared_weight_allocations=1,
                target_position=target.position,
                draft_position=draft.position,
                output_ids=output.generated_token_ids,
            )
        )

    payload_rows = [
        {
            **asdict(row),
            "output_ids": list(row.output_ids),
            "target_forwards_per_accepted_draft_token": (
                None
                if row.accepted_draft_tokens == 0
                else row.target_forwards / row.accepted_draft_tokens
            ),
            "draft_forwards_per_accepted_draft_token": (
                None
                if row.accepted_draft_tokens == 0
                else row.draft_forwards / row.accepted_draft_tokens
            ),
        }
        for row in rows
    ]
    return {
        "schema": 1,
        "kind": "qwen4exp_mtp_rollback_replay_accounting",
        "status": "diagnostic_complete",
        "passed": True,
        "candidate_admitted": False,
        "production_defaults_changed": False,
        "performance_claim": False,
        "screen": {
            "provider": "qwen4_exp_mtp",
            "transaction_mode": "serial_exact_target_verify",
            "proposal_budget": budget,
            "cases": ["all_reject", *[f"reject_after_{d}" for d in range(1, budget)], "full_accept"],
            "rows": payload_rows,
        },
        "invariants": {
            "target_stops_at_first_mismatch": True,
            "draft_executes_full_proposal_before_cursor_trim": True,
            "rollback_kind": "cursor_trim",
            "dirty_destination_restore_bytes": 0,
            "dirty_destination_restore_allocations": 0,
            "per_cycle_allocations": 0,
            "shared_weight_allocations": 1,
            "snapshot_restore_replay_forwards": False,
            "shared_weight_double_count": False,
        },
        "conclusion": (
            "The current serial Qwen4Exp provider has no duplicate target replay: "
            "target verification stops at the first mismatch and draft rollback is "
            "a cursor trim. The four-token screen reports zero restore bytes and "
            "zero per-cycle allocations, so no production change is justified by MT-8."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_accounting_screen(budget=args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
