"""Continuous-batching owner loop for Maple M6 batch decode (D5).

A thin scheduler that keeps a fixed batch of independent requests resident in a
``MapleBatchRunner``, steps them together each round, and reclaims slots as
requests finish so new requests can be admitted. This mirrors the GGUF/PARO
server admission/decode/reclaim pattern at the kernel batch granularity.
"""

from __future__ import annotations

from hipengine.runtime.maple import MapleBatchRunner


class _Request:
    __slots__ = ("done", "generated", "max_new", "seed")

    def __init__(self, seed: int, max_new: int) -> None:
        self.seed = int(seed)
        self.generated: list[int] = []
        self.max_new = int(max_new)
        self.done = False

    def next_input(self, step: int) -> int:
        # First step consumes the seed; subsequent steps feed the previous
        # generated token (autoregressive decode).
        return self.seed if step == 0 else self.generated[-1]


class MapleContinuousBatcher:
    """Fixed-batch continuous-batching owner loop over a MapleBatchRunner.

    The batch is kept full: every active slot contributes one input token per
    round to ``batch_step``. A slot is reclaimed the moment its request reaches
    ``max_new`` generated tokens, and the freed slot can immediately host a new
    request via :meth:`submit`.
    """

    def __init__(self, runner: MapleBatchRunner) -> None:
        self.runner = runner
        self.c = runner.batch_size
        self.slots: list[_Request | None] = [None] * self.c
        self.steps = [0] * self.c
        self.total_generated = 0
        self.completions: list[list[int]] = []

    def submit(self, seed: int, max_new: int) -> int:
        """Admit a new request into the first free slot; return its slot id."""
        if self.runner.closed:
            raise RuntimeError("Maple continuous batcher runner is closed")
        for r in range(self.c):
            if self.slots[r] is None:
                self.runner.reset_request(r)
                self.slots[r] = _Request(seed, max_new)
                self.steps[r] = 0
                return r
        raise RuntimeError("batch is full; call step() before submitting more")

    def step(self) -> int:
        """Run one decode round over all slots; return the number completed."""
        if self.runner.closed:
            raise RuntimeError("Maple continuous batcher runner is closed")
        ids = [0] * self.c
        for r, req in enumerate(self.slots):
            if req is None:
                raise RuntimeError(
                    "step() requires a full batch; submit() until all slots are active"
                )
            ids[r] = req.next_input(self.steps[r])
        outs = self.runner.batch_step(ids)
        completed = 0
        for r, req in enumerate(self.slots):
            req.generated.append(int(outs[r]))
            self.steps[r] += 1
            self.total_generated += 1
            if len(req.generated) >= req.max_new:
                req.done = True
                self.completions.append(list(req.generated))
                self.slots[r] = None
                self.runner.reset_request(r)
                completed += 1
        return completed

    def active(self) -> int:
        return sum(1 for s in self.slots if s is not None)

    def requests(self) -> list[list[int]]:
        return [s.generated for s in self.slots if s is not None]

    def close(self) -> None:
        self.runner.close()
