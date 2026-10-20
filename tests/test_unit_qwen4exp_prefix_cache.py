"""Flash-Next prompt-prefix checkpoints over a host-memory device model.

The fake runtime backs every "device" allocation with one bytearray, so the
store's copies run for real. The fake runner writes position- and
token-dependent rows into its append-only planes and folds every token into its
private state, which makes a wrong offset, granularity or segment visible as a
byte mismatch against a cold prefill.
"""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.dispatch import RequestState
from hipengine.generation.engine_loop import EngineLoopConfig, SubmitPollTextGenerator
from hipengine.generation.qwen4_exp_ar_batch import AR_BATCH_ENV
from hipengine.generation.qwen4_exp_gguf import (
    Qwen4ExpGGUFTextGenerator,
    Qwen4ExpResidentServingRunner,
)
from hipengine.generation.registry import GenerationRequest
from hipengine.runtime.qwen4_exp_prefix_cache import (
    PrefixChain,
    Qwen4ExpPrefixCache,
    capture_targets,
)

ROW = 8  # K/V bytes per token per layer
INDEX = 4  # raw index bytes per token
RATIO = 4  # tokens per pooled index block
STATE = 16  # private recurrent bytes per owner


class _Arena:
    """Bump-allocated host bytes standing in for device memory."""

    def __init__(self, size=1 << 22):
        self.memory = bytearray(size)
        self.top = 64
        self.live = {}

    def malloc(self, nbytes):
        ptr = self.top
        self.top += (int(nbytes) + 63) // 64 * 64
        if self.top > len(self.memory):
            raise MemoryError("arena exhausted")
        self.live[ptr] = int(nbytes)
        return ptr

    def free(self, ptr):
        self.live.pop(int(ptr))

    def memcpy(self, dst, src, nbytes, kind):
        del kind
        self.memory[dst:dst + nbytes] = self.memory[src:src + nbytes]

    def memset(self, ptr, value, nbytes):
        self.memory[ptr:ptr + nbytes] = bytes([value]) * nbytes

    def mem_get_info(self):
        return len(self.memory) - self.top, len(self.memory)

    def buffer(self, nbytes):
        return DeviceBuffer(self.malloc(nbytes), nbytes)

    def read(self, buffer, start=0, nbytes=None):
        size = buffer.nbytes if nbytes is None else nbytes
        return bytes(self.memory[buffer.ptr + start:buffer.ptr + start + size])


class _Attention:
    def __init__(self, arena, positions):
        self.block_size = 4
        self.block_host = np.arange(positions // 4, dtype=np.int32)
        self.max_positions = positions
        self.key_cache = arena.buffer(positions * ROW)
        self.value_cache = arena.buffer(positions * ROW)
        self.cursor = 0

    def set_position(self, value):
        self.cursor = int(value)


class _Index:
    def __init__(self, arena, positions):
        self.capacity = positions
        self.compression_ratio = RATIO
        self.raw_keys = arena.buffer(positions * INDEX)
        self.pooled_keys = arena.buffer(positions // RATIO * INDEX)
        self.count = 0

    def restore_count(self, count):
        self.count = int(count)


def _fill(arena, buffer, offset, nbytes, value):
    arena.memory[buffer.ptr + offset:buffer.ptr + offset + nbytes] = bytes([value % 251]) * nbytes


class _ArenaRunner:
    """A c1 runner whose state is a deterministic function of its tokens."""

    prefill_chunk_size = 64

    def __init__(self, arena, *, layers=2, positions=2048, bias=0):
        self.runtime = arena
        self.max_sequence_length = positions
        self.attention_states = tuple(_Attention(arena, positions) for _ in range(layers))
        self.index_states = tuple(_Index(arena, positions) for _ in range(layers))
        self.state = SimpleNamespace(owned_buffers=MappingProxyType({
            "gdn_matrix": arena.buffer(STATE), "ple_conv": arena.buffer(STATE),
        }))
        self.tokens: list[int] = []
        self.position = 0
        self.bias = bias
        self._ple = {}
        self.prefill_calls: list[int] = []
        self.closed = False

    @property
    def ple_hash_states(self):
        return MappingProxyType(dict(self._ple))

    def reset(self):
        for buffer in self.state.owned_buffers.values():
            self.runtime.memset(buffer.ptr, 0, buffer.nbytes)
        self.tokens.clear()
        self.position = 0
        self._ple = {}

    def adopt_prefix_cursor(self, position, ple_hash_states):
        for attention in self.attention_states:
            attention.set_position(max(position - 1, 0))
        for index in self.index_states:
            index.restore_count(position)
        self._ple = dict(ple_hash_states)
        self.position = int(position)

    def _state_digest(self):
        raw = self.runtime.read(self.state.owned_buffers["gdn_matrix"])
        return sum(raw) % 251

    def prefill_append(self, tokens, **kwargs):
        del kwargs
        self.prefill_calls.append(len(tokens))
        for token in tokens:
            position = self.position
            for layer, (attention, index) in enumerate(
                zip(self.attention_states, self.index_states)
            ):
                _fill(self.runtime, attention.key_cache, position * ROW, ROW, token + position + layer)
                _fill(self.runtime, attention.value_cache, position * ROW, ROW, 3 * token + layer)
                _fill(self.runtime, index.raw_keys, position * INDEX, INDEX, token * 7 + layer)
                if (position + 1) % RATIO == 0:
                    block = position // RATIO
                    _fill(self.runtime, index.pooled_keys, block * INDEX, INDEX, token + block)
            for name, buffer in self.state.owned_buffers.items():
                old = self.runtime.read(buffer)
                mixed = bytes((value * 5 + token + len(name)) % 251 for value in old)
                self.runtime.memory[buffer.ptr:buffer.ptr + buffer.nbytes] = mixed
            self._ple = {0: (self._ple.get(0, 0) * 3 + token) % 1009}
            self.position += 1
        return SimpleNamespace(token_id=(self._state_digest() + self.bias) % 50 + 1)

    def step(self, token, **kwargs):
        del kwargs
        return self.prefill_append([int(token)])

    def dump(self):
        """Every byte that describes positions below the cursor."""

        parts = []
        for attention, index in zip(self.attention_states, self.index_states):
            parts.append(self.runtime.read(attention.key_cache, 0, self.position * ROW))
            parts.append(self.runtime.read(attention.value_cache, 0, self.position * ROW))
            parts.append(self.runtime.read(index.raw_keys, 0, self.position * INDEX))
            parts.append(self.runtime.read(index.pooled_keys, 0, self.position // RATIO * INDEX))
        parts.extend(self.runtime.read(buffer) for buffer in self.state.owned_buffers.values())
        return b"".join(parts), dict(self._ple), self.position

    def close(self):
        self.closed = True


def _tokens(count, seed):
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(1, 200, size=count)]


def _prefill(runner, tokens, chunk=64):
    for start in range(0, len(tokens), chunk):
        runner.prefill_append(tokens[start:start + chunk])


def test_restore_plus_suffix_matches_cold_prefill_through_shared_segments() -> None:
    arena = _Arena()
    cache = Qwen4ExpPrefixCache(arena, limit_bytes=1 << 20)
    first = _tokens(1003, seed=1)
    writer = _ArenaRunner(arena)
    chain = PrefixChain()
    # Two checkpoints from one request: the second shares the first's rows.
    _prefill(writer, first[:640])
    chain, early, _ = cache.capture(first[:640], writer, None, chain=chain,
                                    previous_hidden=None, kind="grid")
    _prefill(writer, first[640:])
    chain, late, _ = cache.capture(first, writer, None, chain=chain,
                                   previous_hidden=None, kind="prompt_end")
    assert early is not None and late is not None
    assert late.segments[0] is early.segments[0]
    assert len(late.segments) == 2
    cache.release_chain(chain)

    second = first + _tokens(301, seed=2)
    branch = first[:700] + _tokens(400, seed=3)
    for prompt, expected_depth in ((second, 1003), (branch, 640)):
        lookup = cache.lookup(prompt, need_draft=False)
        assert lookup.snapshot is not None
        assert lookup.snapshot.depth == expected_depth
        warm = _ArenaRunner(arena)
        warm.reset()
        restored_chain, copied = cache.restore(lookup.snapshot, warm, None)
        assert copied > 0 and warm.position == expected_depth
        _prefill(warm, prompt[expected_depth:])
        cold = _ArenaRunner(arena)
        _prefill(cold, prompt)
        assert warm.dump() == cold.dump()
        cache.release_chain(restored_chain)
    assert cache.lookup(branch, need_draft=False).matched_tokens == 700


def test_lookup_needs_an_unmatched_suffix_and_draft_state_for_mtp() -> None:
    arena = _Arena()
    cache = Qwen4ExpPrefixCache(arena, limit_bytes=1 << 20)
    prompt = _tokens(512, seed=4)
    runner = _ArenaRunner(arena)
    _prefill(runner, prompt)
    chain, snapshot, _ = cache.capture(prompt, runner, None, chain=PrefixChain(),
                                       previous_hidden=None, kind="prompt_end")
    assert snapshot is not None and not snapshot.has_draft
    identical = cache.lookup(prompt, need_draft=False)
    assert identical.snapshot is None and identical.matched_tokens == 512
    longer = prompt + [7]
    assert cache.lookup(longer, need_draft=False).snapshot is snapshot
    mtp = cache.lookup(longer, need_draft=True)
    assert mtp.snapshot is None and mtp.reason == "draft_state_not_cached"
    # A repeat capture of identical tokens refreshes the entry, never duplicates it.
    _, again, reason = cache.capture(prompt, runner, None, chain=chain,
                                     previous_hidden=None, kind="prompt_end")
    assert again is None and reason == "duplicate" and cache.entries == 1


def test_budget_evicts_least_recently_used_and_frees_unshared_rows() -> None:
    arena = _Arena()
    runner = _ArenaRunner(arena)
    prompt = _tokens(600, seed=5)
    _prefill(runner, prompt)
    per_entry = 600 * (2 * ROW + INDEX) * 2 + 600 // RATIO * INDEX * 2 + 2 * STATE
    cache = Qwen4ExpPrefixCache(arena, limit_bytes=int(per_entry * 1.5))
    other = _ArenaRunner(arena)
    other_prompt = _tokens(600, seed=6)
    _prefill(other, other_prompt)
    baseline = len(arena.live)
    live, first, _ = cache.capture(prompt, runner, None, chain=PrefixChain(),
                                   previous_hidden=None, kind="prompt_end")
    # While the first row is live its rows stay resident, so nothing fits.
    _, refused, reason = cache.capture(other_prompt, other, None, chain=PrefixChain(),
                                       previous_hidden=None, kind="prompt_end")
    assert first is not None and refused is None and reason == "budget"
    assert cache.entries == 1 and cache.counters["evictions"] == 0
    cache.release_chain(live)
    _, second, _ = cache.capture(other_prompt, other, None, chain=PrefixChain(),
                                 previous_hidden=None, kind="prompt_end")
    assert second is not None
    assert cache.entries == 1 and cache.counters["evictions"] == 1
    assert cache.lookup(prompt + [1], need_draft=False).snapshot is None
    assert cache.resident_bytes <= cache.limit_bytes
    cache.close()
    assert len(arena.live) == baseline


def test_capture_targets_cover_the_tail_and_long_prompt_grid() -> None:
    assert capture_targets(200, 1024) == ()
    assert capture_targets(3000, 1024) == (1976,)
    targets = capture_targets(40_000, 1024)
    assert 38_976 in targets
    assert {10_000, 20_000, 30_000} <= set(targets)


# -- resident serving integration --------------------------------------------


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text):
        return [ord(char) % 32 + 1 for char in str(text)]

    def decode(self, ids, *, skip_special=False):
        del skip_special
        return ":".join(str(value) for value in ids)


def _serving(monkeypatch, *, prefix_cache):
    monkeypatch.setenv(AR_BATCH_ENV, "0")
    arena = _Arena(1 << 24)
    primary = _ArenaRunner(arena)
    generator = Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf", weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(), tokenizer=_Tokenizer(), runner=primary,
    )
    generator._resident = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(
        "hipengine.generation.qwen4_exp_gguf.Qwen4ExpGGUFResidentModelRunner",
        lambda *args, **kwargs: _ArenaRunner(arena),
    )
    driver = SubmitPollTextGenerator(
        generator,
        config=EngineLoopConfig(max_active_requests=1, max_prefill_chunk_tokens=64,
                                prefix_cache=prefix_cache),
    )
    return driver, generator


def _turns():
    system = "".join(chr(65 + (index * 7) % 26) for index in range(900))
    return (system + " first question?", system + " first question? answer. second?")


def test_resident_runner_reuses_the_previous_turn_and_matches_cold_output(monkeypatch) -> None:
    results = {}
    for mode in ("off", "radix"):
        driver, generator = _serving(monkeypatch, prefix_cache=mode)
        native = generator._resident_model_runner
        assert isinstance(native, Qwen4ExpResidentServingRunner)
        outputs = []
        try:
            for prompt in _turns():
                request = GenerationRequest(prompts=(prompt,), max_tokens=4,
                                            temperature=0.0, top_p=1.0, ignore_eos=True)
                outputs.extend(driver.generate_detailed(request))
            observability = native.observability_snapshot()["prefix_cache"]
        finally:
            driver.close()
        results[mode] = (outputs, observability)
    cold, _ = results["off"]
    warm, observability = results["radix"]
    assert [o.generated_token_ids for o in warm] == [o.generated_token_ids for o in cold]
    blocks = [o.telemetry.diagnostics["prefix_cache"] for o in warm]
    assert blocks[0]["hit"] is False and blocks[0]["fallback_reason"] == "no_cached_prefix"
    assert blocks[1]["hit"] is True
    first_prompt = len(_Tokenizer().encode(_turns()[0]))
    assert blocks[1]["reused_tokens"] == first_prompt
    assert blocks[1]["executed_prefill_tokens"] == len(_Tokenizer().encode(_turns()[1])) - first_prompt
    assert observability["mode"] == "radix" and observability["snapshot_entries"] >= 2
    off_block = cold[1].telemetry.diagnostics["prefix_cache"]
    assert off_block["hit"] is False and off_block["fallback_reason"] == "cache_off"


def test_reserve_admission_moves_the_prompt_cursor_past_reused_tokens(monkeypatch) -> None:
    driver, generator = _serving(monkeypatch, prefix_cache="radix")
    native = generator._resident_model_runner
    try:
        first, second = _turns()
        request = GenerationRequest(prompts=(first,), max_tokens=1, temperature=0.0,
                                    top_p=1.0, ignore_eos=True)
        driver.generate_detailed(request)
        ids = tuple(_Tokenizer().encode(second))
        follow = GenerationRequest(prompts=(second,), max_tokens=1, temperature=0.0,
                                   top_p=1.0, ignore_eos=True)
        native.register_batch((901,), follow, prompt_rows=(ids,))
        moved = native.reserve_admission(RequestState.from_tokens(901, ids, max_new_tokens=1))
        assert moved is not None
        assert moved.next_prompt_index == len(_Tokenizer().encode(first))
        native.discard((901,))
    finally:
        driver.close()
