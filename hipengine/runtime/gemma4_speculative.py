"""Speculative MTP decoding for the Gemma 4 runner.

:class:`Gemma4MtpDrafter` proposes tokens. This module is the other half: the
bounded draft/verify/accept cycle the engine's staged speculative protocol
expects, expressed over three primitives that already exist.

The ordering is not arbitrary and the drafter's docstring states why:

* **Prime and draft first.** ``draft`` seeds the head from the backbone's own
  last hidden row, so it must run straight after the forward that produced the
  token and before any later forward. It also captures the shared-KV views, and
  ``shared_kv`` snapshots the live position count when it is called -- so the
  views have to be taken before the verify extends the cache, or the head would
  read a moved count or be handed the query's own K/V.
* **Then verify.** One target forward over the draft, reading
  ``logits_rows=len(draft)`` so every drafted position gets the target's own
  distribution rather than only the last. That is what ``logits_rows`` exists
  for. A block produces exactly as many rows as it has tokens, so there is no
  row past the last draft token: when the whole draft is accepted the bonus is
  one more single-token forward rather than a wider block.
* **Then accept.** Greedy acceptance walks the draft while the target's argmax
  agrees, and ``rewind`` truncates the KV to the accepted length. The verify
  forward appended the whole draft, so without the rewind the cache would carry
  positions the target never chose.

Nothing here is new arithmetic. The draft, the multi-row logits read, and the
rewind are all existing paths; this is the sequencing that joins them, which is
the part that is easy to get wrong from outside.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hipengine.runtime.gemma4 import Gemma4Runner
from hipengine.runtime.gemma4_assistant import Gemma4MtpDrafter

__all__ = ["Gemma4SpeculativeCycle", "Gemma4SpeculativeAdapter"]


@dataclass
class Gemma4SpeculativeCycle:
    """One bounded draft/verify/accept cycle, and what it decided."""

    drafted: list[int]
    accepted: list[int]
    # The target's argmax at the position after the last accepted token, which is
    # the next token when nothing was accepted past the prompt.
    bonus: int | None
    proposed: int
    agreed: int

    @property
    def acceptance_rate(self) -> float:
        return 0.0 if self.proposed == 0 else self.agreed / self.proposed

    @property
    def tokens_per_cycle(self) -> int:
        """Tokens this cycle advances the sequence by, including the bonus."""

        return len(self.accepted) + (0 if self.bonus is None else 1)


@dataclass
class Gemma4SpeculativeAdapter:
    """Draft/verify/accept over a runner and its assistant head.

    ``max_drafts`` bounds one cycle's proposal width. The engine owns the
    decision to speculate at all; this adapter owns what one cycle does and
    reports what it cost.
    """

    runner: Gemma4Runner
    drafter: Gemma4MtpDrafter
    cycles: list[Gemma4SpeculativeCycle] = field(default_factory=list, repr=False)

    @property
    def max_drafts(self) -> int:
        return int(self.drafter.max_drafts)

    def capability(self):
        """What this adapter can do, for the engine's cold capability hook."""

        return {
            "max_drafts": self.max_drafts,
            "acceptance": "greedy",
            # The verify reads the target at max_drafts + 1 rows in one call, so
            # the runner it is given must have been built with at least that many
            # logits rows. Reported here because the engine's capability hook is
            # where a caller can find out before constructing a cycle.
            "required_logits_rows": self.max_drafts + 1,
            # The verify is an arithmetic change: it reads the target at several
            # rows at once, so it is bit-exact against the decode only within one
            # execution profile. See docs/EXECUTION-PROFILES.md.
            "requires_profile_pin": True,
        }

    def draft(self, token: int, *, hidden_row: int = -1) -> list[int]:
        return self.drafter.draft(token, hidden_row=hidden_row)

    def verify_and_accept(
        self,
        token: int,
        draft: list[int],
        *,
        start_position: int,
    ) -> Gemma4SpeculativeCycle:
        """Verify ``draft`` against the target and truncate to what it agrees with.

        ``token`` is the seed the draft was proposed from, and it is part of the
        verify block rather than a separate forward: forwarding
        ``[token] + draft`` makes row ``i`` the target's distribution after
        ``draft[:i]``, which is the row that decides ``draft[i]``. Forwarding the
        draft alone would shift every row by one and compare each drafted token
        against the target's answer at the wrong position.

        ``start_position`` is the runner position the draft was proposed from.
        The runner must still be at that position when this is called, which is
        the same "call it straight after the producing forward" constraint the
        drafter documents.
        """

        if self.runner.position != start_position:
            raise ValueError(
                f"runner is at position {self.runner.position}, expected "
                f"{start_position}: the draft must be verified before any other "
                "forward moves the sequence"
            )
        if not draft:
            raise ValueError("draft must not be empty")
        # The verify reads the target at every drafted position plus the bonus, so
        # the runner's projection buffers have to be sized for that many rows.
        # The default of one row is what keeps a wide prefill from sizing its
        # logits scratch for the whole block, so this is a real requirement the
        # caller has to meet rather than something to work around.

        # The verify reads the target at every drafted position. Row i is the
        # target's distribution after draft[:i], so the row that decides draft[i]
        # is row i -- and there is no row past the last draft token, because a
        # block produces exactly as many rows as it has tokens.
        block = [int(token)] + [int(t) for t in draft]
        needed = len(block)
        logits = self.runner.forward(block, logits_rows=needed)
        rows = np.asarray(logits)
        if rows.shape[0] != needed:
            raise ValueError(
                f"verify returned {rows.shape[0]} rows for a {len(block)}-token "
                "verify block; the accepted length cannot be decided"
            )

        agreed = 0
        for index, token in enumerate(draft):
            if int(np.argmax(rows[index])) != int(token):
                break
            agreed += 1

        accepted = draft[:agreed]
        # Row `agreed` is the target's distribution after the seed plus the
        # accepted prefix, so its argmax is the bonus token -- available whether
        # or not every draft token was accepted, because the block carries one
        # row per position through `draft[-1]` inclusive.
        bonus = int(np.argmax(rows[agreed]))

        # The verify appended the seed plus the whole draft; the seed was already
        # in the sequence, so keeping `agreed` past start_position is exactly the
        # accepted prefix.
        self.runner.rewind(start_position + agreed)

        cycle = Gemma4SpeculativeCycle(
            drafted=list(draft),
            accepted=list(accepted),
            bonus=bonus,
            proposed=len(draft),
            agreed=agreed,
        )
        self.cycles.append(cycle)
        return cycle

    def run_cycle(self, token: int, *, hidden_row: int = -1) -> Gemma4SpeculativeCycle:
        """One full cycle: draft from ``token``, then verify and accept."""

        start_position = self.runner.position
        draft = self.draft(token, hidden_row=hidden_row)
        return self.verify_and_accept(token, draft, start_position=start_position)
