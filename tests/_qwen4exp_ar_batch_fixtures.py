"""Fake c1 / batch / MTP owners for the AR batch decode unit tier.

The fakes implement the contracts the generator relies on, not the device work:
the c1 runner records ``prefill_append``/``step`` calls and advances a position,
the batch runner enforces slot ownership and returns per-row tokens derived from
the row's own slot, and the MTP provider yields one root token so a speculative
row can be driven without a real draft model.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator

VOCAB = 64


class FakeTokenizer:
    eos_token_id = VOCAB - 1

    def encode(self, text):
        return [ord(char) % (VOCAB - 2) + 1 for char in str(text)]

    def decode(self, ids, *, skip_special=False):
        del skip_special
        return ":".join(str(int(value)) for value in ids)


class FakeC1Runner:
    """One request's c1 runner: chunked prefill plus a single-row step."""

    max_sequence_length = 64
    prefill_chunk_size = 8
    runtime = None
    config = None

    def __init__(self, bias=0):
        self.bias = bias
        self.position = 0
        self.closed = False
        self.calls = []
        self.prompt_ids = []
        self.transactions = []

    def reset(self):
        self.position = 0
        self.prompt_ids.clear()

    def prefill_append(self, tokens, **kwargs):
        self.prompt_ids.extend(int(token) for token in tokens)
        self.position += len(tokens)
        self.calls.append(("prefill_append", dict(kwargs)))
        logits = None
        if kwargs.get("capture_logits"):
            logits = np.zeros(VOCAB, dtype=np.float32)
        return SimpleNamespace(
            token_id=sum(self.prompt_ids) % 50 + self.bias,
            logits=logits,
            hidden_seeds=(),
            hidden_seed=None,
        )

    def prefill(self, tokens, **kwargs):
        self.position = len(tokens)
        self.calls.append(("prefill", dict(kwargs)))
        logits = None
        if kwargs.get("capture_logits"):
            logits = np.zeros(VOCAB, dtype=np.float32)
        return SimpleNamespace(token_id=sum(tokens) % 50 + self.bias, logits=logits)

    def step(self, token, **kwargs):
        self.position += 1
        self.calls.append(("step", dict(kwargs)))
        return SimpleNamespace(token_id=int(token) + 1, logits=None)

    def begin_device_transaction(self, *, reuse_snapshot=False):
        self.transactions.append(("begin", bool(reuse_snapshot)))
        return SimpleNamespace(position=self.position)

    def commit_device_transaction(self, checkpoint):
        self.transactions.append(("commit", int(checkpoint.position)))

    def rollback_device_transaction(self, checkpoint):
        self.transactions.append(("rollback", int(checkpoint.position)))
        self.position = int(checkpoint.position)

    def close(self):
        self.closed = True


class FakeBatchRunner:
    """Stand-in for ``Qwen4ExpGGUFBatchRunner`` with the same control contract."""

    token_bias = 100

    #: Small blocks so a test can fill the pool with short prompts.
    block_size = 4
    #: Blocks an ``"auto"`` pool gets; tests override it to force exhaustion.
    auto_pool_blocks = 256

    def __init__(
        self, resident=None, *, state_slots, max_positions, backend=None, runtime=None,
        pool_blocks=None, graph_segments=True,
    ):
        del graph_segments
        self.resident = resident
        self.state_slots = int(state_slots)
        self.max_positions = int(max_positions)
        self.pool_blocks = (
            type(self).auto_pool_blocks if pool_blocks == "auto"
            else self.state_slots * self.blocks_for_positions(self.max_positions)
            if pool_blocks is None else int(pool_blocks)
        )
        self._held = [0] * self.state_slots
        self.backend = backend
        self.runtime = runtime
        self.free_slot_order = "ascending"
        self.closed = False
        self.admit_calls = []
        self.import_calls = []
        self.release_calls = []
        self.step_calls = []
        self.fail_next_step = None
        self.fail_next_import = None
        self.fail_next_close = None
        # Deliberate post-step corruption: "short_tokens", "nan_logits",
        # "no_logits" or "wrong_logits_rows". The step still advances state, the
        # way a real runner would before its result is read back.
        self.malformed = None
        self.close_calls = 0
        self._owners = [0] * self.state_slots
        self._positions = [0] * self.state_slots
        self._next_owner = 0

    @property
    def free_slots(self):
        slots = tuple(
            slot for slot, owner in enumerate(self._owners) if not owner
        )
        return tuple(reversed(slots)) if self.free_slot_order == "descending" else slots

    def owners_of(self, slots):
        return tuple(self._owners[int(slot)] for slot in slots)

    def blocks_for_positions(self, positions):
        return (int(positions) + self.block_size - 1) // self.block_size

    @property
    def free_block_count(self):
        return self.pool_blocks - sum(self._held)

    def admit(self, slot, *, positions=None):
        parsed = int(slot)
        if self._owners[parsed]:
            raise RuntimeError(
                f"slot {parsed} is already owned by request {self._owners[parsed]}"
            )
        if positions is not None:
            need = self.blocks_for_positions(positions)
            if need > self.free_block_count:
                raise RuntimeError(
                    f"batch KV pool has {self.free_block_count} free blocks; "
                    f"{positions} positions need {need}"
                )
            self._held[parsed] = need
        self._next_owner += 1
        self._owners[parsed] = self._next_owner
        self._positions[parsed] = 0
        self.admit_calls.append(parsed)
        return self._next_owner

    def import_prefilled(self, slot, owner, *, source):
        parsed = int(slot)
        if self._owners[parsed] != int(owner):
            raise RuntimeError(
                f"slot {parsed} is owned by request {self._owners[parsed]}, not {owner}"
            )
        if self._positions[parsed]:
            raise ValueError(f"slot {parsed} already holds request history")
        if self.fail_next_import is not None:
            error, self.fail_next_import = self.fail_next_import, None
            raise error
        position = int(getattr(source, "position", -1))
        if position <= 0:
            raise ValueError("import source has no prefilled position")
        self._positions[parsed] = position
        self.import_calls.append((parsed, int(owner), position))
        return position

    def step(self, token_ids, *, slots=None, owners=None, capture_logits=True):
        tokens = tuple(int(token) for token in token_ids)
        if slots is None:
            chosen = tuple(
                slot for slot, owner in enumerate(self._owners) if owner
            )
            granted = self.owners_of(chosen)
        else:
            if owners is None:
                raise ValueError("step with explicit slots requires owner tokens")
            chosen = tuple(int(slot) for slot in slots)
            granted = tuple(int(owner) for owner in owners)
        if len(chosen) != len(tokens) or len(granted) != len(tokens):
            raise ValueError("slots, owners and token ids must align")
        if len(set(chosen)) != len(chosen):
            raise ValueError("a slot appears twice in one step's row set")
        for slot, owner in zip(chosen, granted):
            if self._owners[slot] != owner:
                raise RuntimeError(
                    f"slot {slot} is owned by request {self._owners[slot]}, not {owner}"
                )
        if self.fail_next_step is not None:
            error, self.fail_next_step = self.fail_next_step, None
            self.step_calls.append(("failed", chosen, granted, tokens, bool(capture_logits)))
            raise error
        decoded = tuple(
            token + self.token_bias * (slot + 1)
            for token, slot in zip(tokens, chosen)
        )
        for slot in chosen:
            self._positions[slot] += 1
        self.step_calls.append((chosen, granted, tokens, bool(capture_logits)))
        logits = None
        if capture_logits:
            logits = np.zeros((len(tokens), VOCAB), dtype=np.float32)
            for row, token in enumerate(decoded):
                logits[row, token % VOCAB] = 1.0
        decoded, logits = self._malform(decoded, logits)
        return SimpleNamespace(
            slot_ids=chosen,
            positions=np.asarray([self._positions[slot] - 1 for slot in chosen]),
            tokens=np.asarray(decoded, dtype=np.int64),
            logits=logits,
        )

    def _malform(self, decoded, logits):
        mode = self.malformed
        if mode == "short_tokens":
            return decoded[:-1], logits
        if mode == "wrong_logits_rows":
            return decoded, None if logits is None else logits[:-1]
        if mode == "no_logits":
            return decoded, None
        if mode == "nan_logits":
            if logits is not None:
                logits[0, 0] = np.nan
            return decoded, logits
        return decoded, logits

    def release(self, slot, owner):
        parsed = int(slot)
        if self._owners[parsed] != int(owner):
            raise RuntimeError(
                f"slot {parsed} is owned by request {self._owners[parsed]}, not {owner}"
            )
        self._owners[parsed] = 0
        self._positions[parsed] = 0
        self._held[parsed] = 0
        self.release_calls.append((parsed, int(owner)))

    def close(self):
        self.close_calls += 1
        if self.fail_next_close is not None:
            error, self.fail_next_close = self.fail_next_close, None
            # A partial free leaves the runner open, like the real reverse free.
            raise error
        self.closed = True


class FakeDraft:
    max_sequence_length = 64

    def __init__(self):
        self.position = 0
        self.closed = False
        self.primed = []

    def reset(self):
        self.position = 0

    def prime_prompt_chunk(self, chunk, hidden_seeds, *, previous_hidden=None, check_cancelled=None):
        del hidden_seeds, previous_hidden
        if check_cancelled is not None:
            check_cancelled()
        self.primed.append(tuple(int(token) for token in chunk))
        self.position += len(chunk)

    def snapshot(self):
        return SimpleNamespace(position=self.position)

    def restore(self, snapshot):
        self.position = int(snapshot.position)

    def close(self):
        self.closed = True


class FakeMTPProvider:
    """One root token per request, then a finished iterator."""

    candidate_budget = 2
    root_token = 21

    def __init__(self):
        self.draft = FakeDraft()
        self._resident = None
        self.iter_calls = []

    def close(self):
        self.draft.close()

    def _validate_sampling(self, request):
        del request

    def iter_generate(self, prompt_ids, request, **kwargs):
        self.iter_calls.append((tuple(prompt_ids), request, kwargs))
        draft = kwargs.get("draft")
        if draft is not None:
            draft.position = int(kwargs["target"].position)

        def steps():
            yield SimpleNamespace(
                token_ids=(self.root_token,), finished=False, token_logprobs=()
            )
            return SimpleNamespace(finish_details=None)

        return steps()


def make_generator():
    runner = FakeC1Runner()
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf",
        weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(),
        tokenizer=FakeTokenizer(),
        runner=runner,
    )
    generator._resident = SimpleNamespace(close=lambda: None)
    return generator


__all__ = [
    "FakeBatchRunner",
    "FakeC1Runner",
    "FakeDraft",
    "FakeMTPProvider",
    "FakeTokenizer",
    "VOCAB",
    "make_generator",
]
