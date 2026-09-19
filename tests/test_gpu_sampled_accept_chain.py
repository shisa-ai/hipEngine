"""Device sampled accept: the kernel's decision and induced law must match the host oracle.

The host oracle in ``hipengine/generation/mtp_sampled_accept.py`` defines what a
sampled MTP accept means: accept the drafted token with ``min(1, p(x)/q(x))`` and,
on rejection, draw from ``normalize(max(0, p - q))``. The device kernel must make
the *same* decision from device logits, or a sampled row's output stops being
``p``-distributed and the sampled route cannot be measured against its
autoregressive control.

These tests pin three things:

* the device walk consumes the same uniforms in the same order as the oracle, so
  replaying the oracle with the device's own draws reproduces the device's
  decision exactly (accepted count, committed row, emitted token);
* the first emitted token is distributed as the target law ``p`` itself - the
  speculative-sampling identity - on both paths;
* the law comparison has teeth: a wrong accept rule (argmax comparison, the
  greedy rule) is detected by the same measurement.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest

from hipengine.generation.mtp_sampled_accept import sampled_accept_uniform
from hipengine.speculative.interfaces import TargetVerifyBatch
from hipengine.speculative.sampling import sampled_accept_from_distributions


def _has_hip() -> bool:
    try:
        from hipengine.core.hip import get_hip_runtime
    except Exception:
        return False
    try:
        get_hip_runtime()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_hip(),
    reason="device sampled accept requires a working HIP runtime",
)


def _chain_batch(
    tokens: tuple[int, ...],
    *,
    request_id: int = 7,
    root_position: int = 11,
) -> TargetVerifyBatch:
    """Build a single-request drafted chain of ``tokens[1:]`` after ``tokens[0]``."""

    rows = len(tokens)
    return TargetVerifyBatch(
        request_ids=(request_id,),
        tokens=tuple(int(token) for token in tokens),
        positions=tuple(root_position + index for index in range(rows)),
        row_to_request=tuple(request_id for _ in range(rows)),
        parent_rows=(-1, *range(rows - 1)),
        root_rows=(0,),
        candidate_rows=tuple(range(1, rows)),
        draft_depths=(0, *range(1, rows)),
        active_mask=(True,) * rows,
    )


class _DeviceChain:
    """Device buffers and one launch of the sampled accept chain."""

    def __init__(self, logits: np.ndarray, tokens: tuple[int, ...], *, temperature: float):
        from hipengine.core.hip import get_hip_runtime
        from hipengine.core.memory import (
            copy_device_to_host,
            copy_host_to_device,
            free,
            host_array_ptr,
            malloc,
        )
        from hipengine.kernels.backends import hip_target_arch_environment
        from hipengine.kernels.hip_gfx1100.speculative.sampled_accept import (
            build_sampled_accept,
            sampled_accept_chain_i32,
            sampled_accept_row_stats_f32,
        )

        compiler_file = os.environ.get("HIPENGINE_COMPILER_VERSION_FILE")
        compiler_version = (
            pathlib.Path(compiler_file).read_text(encoding="utf-8")
            if compiler_file
            else None
        )
        arch = os.environ.get("HIPENGINE_HIP_ARCH", "gfx1100")
        with hip_target_arch_environment(arch):
            library = build_sampled_accept(load=True, compiler_version=compiler_version)

        self._free = free
        self._copy_device_to_host = copy_device_to_host
        self._host_array_ptr = host_array_ptr
        self._runtime = get_hip_runtime()
        self._library = library
        self._chain = sampled_accept_chain_i32
        self._stats = sampled_accept_row_stats_f32
        self._buffers: list = []

        self.logits = np.ascontiguousarray(logits, dtype=np.float32)
        self.rows, self.vocab = self.logits.shape
        self.tokens = tuple(int(token) for token in tokens)
        self.temperature = float(temperature)
        self.batch = _chain_batch(self.tokens)
        self.output_stride = self.rows

        self._logits_d = self._upload(self.logits)
        self._temperatures_d = self._upload(
            np.full((self.rows,), self.temperature, dtype=np.float32)
        )
        self._row_max_d = self._alloc(self.rows * 4)
        self._row_inv_sum_d = self._alloc(self.rows * 4)
        self._token_ids_d = self._upload(np.asarray(self.tokens, dtype=np.int32))
        self._positions_d = self._upload(
            np.asarray(self.batch.positions, dtype=np.int32)
        )
        self._parent_rows_d = self._upload(
            np.asarray(self.batch.parent_rows, dtype=np.int32)
        )
        self._draft_depths_d = self._upload(
            np.asarray(self.batch.draft_depths, dtype=np.int32)
        )
        self._active_mask_d = self._upload(
            np.asarray(self.batch.active_mask, dtype=np.uint8)
        )
        self._accepted_d = self._alloc(4)
        self._commit_rows_d = self._alloc(4)
        self._commit_tokens_d = self._alloc(4)
        self._commit_positions_d = self._alloc(4)
        self._next_tokens_d = self._alloc(4)
        self._full_accept_d = self._alloc(1)
        self._committed_ids_d = self._alloc(self.output_stride * 4)
        self._committed_lengths_d = self._alloc(4)
        self._packed_d = self._alloc(7 * 4)
        self._visible_ids_d = self._alloc(self.output_stride * 4)
        self._visible_lengths_d = self._alloc(4)
        self._resident_positions_d = self._alloc(8)
        self._resident_contexts_d = self._alloc(8)

        self._stats(
            self._logits_d.ptr,
            self._temperatures_d.ptr,
            self._row_max_d.ptr,
            self._row_inv_sum_d.ptr,
            self.rows,
            self.vocab,
            library=library,
            runtime=self._runtime,
        )

    # -- buffer helpers -------------------------------------------------
    def _upload(self, array: np.ndarray):
        from hipengine.core.memory import copy_host_to_device, host_array_ptr, malloc

        arr = np.ascontiguousarray(array)
        buf = malloc(max(int(arr.nbytes), 4), runtime=self._runtime)
        self._buffers.append(buf)
        copy_host_to_device(
            buf, host_array_ptr(arr), int(arr.nbytes), runtime=self._runtime
        )
        return buf

    def _alloc(self, nbytes: int):
        from hipengine.core.memory import malloc

        buf = malloc(max(int(nbytes), 4), runtime=self._runtime)
        self._buffers.append(buf)
        return buf

    def _download(self, buf, dtype, count: int) -> np.ndarray:
        out = np.zeros(count, dtype=dtype)
        self._copy_device_to_host(
            self._host_array_ptr(out), buf, int(out.nbytes), runtime=self._runtime
        )
        return out

    def close(self) -> None:
        for buf in self._buffers:
            self._free(buf, runtime=self._runtime)
        self._buffers.clear()

    # -- one cycle ------------------------------------------------------
    def run(self, seed: int, step_index: int) -> dict:
        seeds_d = self._upload(np.asarray([seed], dtype=np.uint64))
        steps_d = self._upload(np.asarray([step_index], dtype=np.uint64))
        try:
            self._chain(
                self._logits_d.ptr,
                self._temperatures_d.ptr,
                self._row_max_d.ptr,
                self._row_inv_sum_d.ptr,
                seeds_d.ptr,
                steps_d.ptr,
                self._token_ids_d.ptr,
                self._positions_d.ptr,
                self._parent_rows_d.ptr,
                self._draft_depths_d.ptr,
                self._active_mask_d.ptr,
                None,
                self._accepted_d.ptr,
                self._commit_rows_d.ptr,
                self._commit_tokens_d.ptr,
                self._commit_positions_d.ptr,
                self._next_tokens_d.ptr,
                self._full_accept_d.ptr,
                self._committed_ids_d.ptr,
                self._committed_lengths_d.ptr,
                self._packed_d.ptr,
                self._visible_ids_d.ptr,
                self._visible_lengths_d.ptr,
                self._resident_positions_d.ptr,
                self._resident_contexts_d.ptr,
                1,
                self.rows,
                1,
                self.output_stride,
                self.vocab,
                library=self._library,
                runtime=self._runtime,
            )
        finally:
            self._free(seeds_d, runtime=self._runtime)
            self._free(steps_d, runtime=self._runtime)
            self._buffers = [b for b in self._buffers if b is not seeds_d and b is not steps_d]
        return {
            "accepted": int(self._download(self._accepted_d, np.int32, 1)[0]),
            "commit_row": int(self._download(self._commit_rows_d, np.int32, 1)[0]),
            "next_token": int(self._download(self._next_tokens_d, np.int32, 1)[0]),
            "full_accept": int(self._download(self._full_accept_d, np.uint8, 1)[0]),
            "visible_ids": self._download(
                self._visible_ids_d, np.int32, self.output_stride
            ),
            "visible_length": int(
                self._download(self._visible_lengths_d, np.int32, 1)[0]
            ),
        }


def _target_probabilities(logits: np.ndarray, row: int, temperature: float) -> np.ndarray:
    """Reference softmax over one row, in float64 like the host oracle."""

    scaled = np.asarray(logits[row], dtype=np.float64) / float(temperature)
    shifted = scaled - float(scaled.max())
    weights = np.exp(shifted)
    return weights / float(weights.sum())


def _host_decision(
    device: _DeviceChain,
    seed: int,
    step_index: int,
    walk: dict,
) -> tuple[int, int, int]:
    """Replay the host oracle with the draws the device walk consumed.

    The device walk consumes one uniform per acceptance test (slot 0 at each
    visited row) and then one for the residual or bonus sample (slot 1 at the
    row where it stopped). Reproducing that sequence on the host and feeding it
    to the oracle checks both the decision rule and the draw order. The returned
    triple is ``(accepted, next_token, first_visible_token)``, where the first
    visible token is the first accepted draft or, when none was accepted, the
    correction.
    """

    chain = list(range(device.rows))
    accepted = int(walk["accepted"])
    # A walk that stopped on a rejection consumed one acceptance test at every
    # visited row plus one residual draw; a walk that exhausted the chain
    # consumed one test per drafted token and then one bonus draw.
    tests = accepted if accepted == device.rows - 1 else accepted + 1
    stop_row = chain[accepted]
    draws: list[float] = [
        sampled_accept_uniform(seed, step_index, chain[index], 0)
        for index in range(tests)
    ]
    draws.append(sampled_accept_uniform(seed, step_index, stop_row, 1))
    stream = iter(draws)
    summary = sampled_accept_from_distributions(
        device.batch,
        tuple(_sparse_row(device, row) for row in chain),
        tuple(_point_mass(device.tokens[row + 1] if row + 1 < device.rows else device.tokens[row]) for row in chain),
        draws=lambda: next(stream),
    )
    return (
        int(summary.accepted_counts[0]),
        int(summary.next_tokens[0]),
        int(
            summary.accepted_tokens[0][0]
            if summary.accepted_tokens[0]
            else summary.next_tokens[0]
        ),
    )


def _sparse_row(device: _DeviceChain, row: int):
    from hipengine.speculative.sampling import SparseDistribution

    probabilities = _target_probabilities(device.logits, row, device.temperature)
    return SparseDistribution.from_pairs(range(device.vocab), probabilities)


def _point_mass(token_id: int):
    from hipengine.speculative.sampling import SparseDistribution

    return SparseDistribution.point_mass(int(token_id))


def test_device_walk_matches_the_host_oracle_on_its_own_draws() -> None:
    """Same logits, same uniforms: the oracle reproduces the device decision."""

    rng = np.random.default_rng(20260919)
    vocab = 64
    logits = rng.normal(size=(3, vocab)).astype(np.float32) * 1.5
    tokens = (5, 11, 23)
    device = _DeviceChain(logits, tokens, temperature=0.7)
    try:
        for seed in range(12):
            walk = device.run(seed, step_index=seed + 3)
            host_accepted, host_next, _first = _host_decision(device, seed, seed + 3, walk)
            assert walk["accepted"] == host_accepted, (seed, walk)
            assert walk["next_token"] == host_next, (seed, walk)
    finally:
        device.close()


def test_device_walk_uses_every_row_of_a_wider_chain() -> None:
    """A deeper chain exercises every visited row, not just the root."""

    rng = np.random.default_rng(4242)
    vocab = 512
    logits = rng.normal(size=(4, vocab)).astype(np.float32) * 2.0
    # Make the chain easy to accept so the walk reaches later rows.
    tokens = (7, 7, 7, 7)
    for row in range(1, 4):
        logits[row, 7] = 9.0
    device = _DeviceChain(logits, tokens, temperature=0.7)
    try:
        accepted_seen = set()
        for seed in range(40):
            walk = device.run(seed, step_index=1)
            host_accepted, host_next, _first = _host_decision(device, seed, 1, walk)
            assert walk["accepted"] == host_accepted, (seed, walk)
            assert walk["next_token"] == host_next, (seed, walk)
            accepted_seen.add(int(walk["accepted"]))
        assert accepted_seen & {1, 2, 3}, accepted_seen
    finally:
        device.close()


def _first_visible_token(walk: dict) -> int:
    """Return the cycle's first visible token: first accepted draft, else correction."""

    length = int(walk["visible_length"])
    assert length >= 1
    return int(walk["visible_ids"][0])


def _empirical_law(tokens: list[int], vocab: int) -> np.ndarray:
    counts = np.bincount(np.asarray(tokens, dtype=np.int64), minlength=vocab)
    return counts / float(counts.sum())


def test_device_accept_emits_the_target_law() -> None:
    """The speculative identity: the emitted token is distributed as ``p``.

    With a depth-1 chain whose child row carries the same law as the root, the
    accept branch contributes ``p(x) * p`` and the rejection branch contributes
    ``(1 - p(x)) * normalize(p without x)``, which sums to ``p`` exactly. The
    measurement therefore fails for any rule that is not the coupled one.
    """

    rng = np.random.default_rng(99)
    vocab = 6
    logits = rng.normal(size=(2, vocab)).astype(np.float32)
    logits[1] = logits[0]
    tokens = (3, 1)
    temperature = 0.8
    target = _target_probabilities(logits, 0, temperature)

    device = _DeviceChain(logits, tokens, temperature=temperature)
    try:
        device_tokens: list[int] = []
        host_tokens: list[int] = []
        for seed in range(4000):
            walk = device.run(seed, 5)
            device_tokens.append(_first_visible_token(walk))
            _accepted, _next, first_visible = _host_decision(device, seed, 5, walk)
            host_tokens.append(int(first_visible))
    finally:
        device.close()

    device_law = _empirical_law(device_tokens, vocab)
    host_law = _empirical_law(host_tokens, vocab)
    assert np.max(np.abs(device_law - target)) < 0.04, (device_law, target)
    assert np.max(np.abs(host_law - target)) < 0.04, (host_law, target)
    assert np.max(np.abs(device_law - host_law)) < 0.03, (device_law, host_law)


def test_argmax_accept_rule_is_detected_by_the_same_measurement() -> None:
    """RED control: the greedy rule does not emit the target law.

    This is the failure the device kernel would have if it reused the argmax
    accept: accepting only the top-1 token biases the emitted distribution
    toward the target's mode. The control proves the law measurement above has
    the power to reject that rule.
    """

    rng = np.random.default_rng(7)
    vocab = 6
    logits = rng.normal(size=(2, vocab)).astype(np.float32)
    logits[1] = logits[0]
    temperature = 0.8
    target = _target_probabilities(logits, 0, temperature)
    top1 = int(np.argmax(logits[0]))

    # The greedy rule emits the drafted token when it is the top-1 token and the
    # top-1 token otherwise: a point mass on the mode, not the target law.
    greedy_tokens = [top1] * 4000
    greedy_law = _empirical_law(greedy_tokens, vocab)
    assert np.max(np.abs(greedy_law - target)) > 0.2
