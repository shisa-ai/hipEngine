"""Prompt-prefix reuse for the Flash-Next resident serving runner.

Admission looks up the deepest cached checkpoint that is a strict prefix of the
new prompt, writes it into the request's freshly reset staging runner (and MTP
draft), and advances the scheduler's prompt cursor past the reused tokens. Only
the suffix is prefilled. Each request then checkpoints its own prompt end and a
few earlier prefill boundaries for the requests after it. See
``hipengine/runtime/qwen4_exp_prefix_cache.py`` for what a checkpoint holds.
"""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

import numpy as np

from hipengine.kvcache import resolve_prefix_cache_mode
from hipengine.runtime.qwen4_exp_prefix_cache import (
    MIN_CAPTURE_SPACING_TOKENS,
    MIN_PREFIX_TOKENS,
    PrefixCacheUnsupported,
    PrefixChain,
    Qwen4ExpPrefixCache,
    capture_targets,
)


class Qwen4ExpServingPrefix:
    """Mixin over the resident runner's row table and staging runners."""

    def _init_prefix(self) -> None:
        self._prefix_mode = "off"
        self._prefix: Qwen4ExpPrefixCache | None = None
        self._prefix_unsupported: str | None = None

    def configure_engine_loop(self, config: Any) -> None:
        mode = resolve_prefix_cache_mode(getattr(config, "prefix_cache", "off"))
        if mode == self._prefix_mode:
            return
        if self._rows:
            raise RuntimeError("cannot change the Flash-Next prefix cache while requests are active")
        self._close_prefix()
        self._prefix_mode = mode
        if mode == "radix":
            self._prefix = Qwen4ExpPrefixCache(self.generator.runner.runtime)

    def _close_prefix(self) -> None:
        cache, self._prefix = self._prefix, None
        if cache is not None:
            for row in self._rows.values():
                chain = getattr(row, "prefix_chain", None)
                if chain is not None:
                    cache.release_chain(chain)
            cache.close()

    # -- admission --------------------------------------------------------

    def _restore_prefix(self, row, request):
        """Restore the best checkpoint into ``row``; return the moved request."""

        cache = self._prefix
        row.prefix_chain = PrefixChain()
        if cache is None:
            row.prefix_reason = "cache_off"
            return None
        row.prefix_eligible = self._prefix_unsupported is None
        if not row.prefix_eligible:
            row.prefix_reason = self._prefix_unsupported
            return None
        length = len(row.prompt_ids)
        chunk = max(1, int(self.generator.runner.prefill_chunk_size))
        restored = 0
        if length <= MIN_PREFIX_TOKENS:
            row.prefix_reason = "prompt_too_short"
        else:
            started = time.perf_counter()
            lookup = cache.lookup(row.prompt_ids, need_draft=row.draft is not None)
            row.prefix_lookup = True
            row.prefix_matched_tokens = int(lookup.matched_tokens)
            snapshot = lookup.snapshot
            if snapshot is None:
                row.prefix_reason = lookup.reason
            elif snapshot.depth < MIN_PREFIX_TOKENS:
                row.prefix_reason = "cached_prefix_too_short"
            else:
                try:
                    chain, copied = cache.restore(snapshot, row.runner, row.draft)
                except PrefixCacheUnsupported as error:
                    self._prefix_unsupported = f"unsupported_layout: {error}"
                    row.prefix_reason = self._prefix_unsupported
                    self._reset_after_failed_restore(row, snapshot.depth)
                except Exception as error:  # a failed copy leaves partial state
                    row.prefix_reason = f"restore_failed: {type(error).__name__}"
                    self._reset_after_failed_restore(row, snapshot.depth)
                else:
                    row.prefix_chain = chain
                    restored = snapshot.depth
                    row.prefix_restore_bytes = int(copied)
                    if row.draft is not None:
                        row.previous_hidden = np.array(snapshot.previous_hidden, copy=True)
                    row.prefill_tokens_seen = restored
                    row.prefix_reason = None
            row.prefix_restore_ms = (time.perf_counter() - started) * 1e3
            if row.prefix_reason is not None:
                cache.note_fallback(row.prefix_reason.split(":", 1)[0])
        row.prefix_reused_tokens = restored
        row.prefix_last_capture = restored
        targets = [(target, length - 1, "grid") for target in capture_targets(length, chunk)
                   if target > restored + MIN_CAPTURE_SPACING_TOKENS // 2]
        matched = int(getattr(row, "prefix_matched_tokens", 0))
        if restored + MIN_CAPTURE_SPACING_TOKENS <= matched < length:
            # A new branch off cached history: checkpoint near the divergence so
            # the next branch from the same point restores there.
            targets.append((max(restored + 1, matched - chunk + 1), matched, "divergence"))
        row.prefix_targets = sorted(targets)
        if restored:
            return replace(request, next_prompt_index=restored)
        return None

    def _reset_after_failed_restore(self, row, depth: int) -> None:
        # The bounded K/V reset clears below the cursor only, so point it past
        # every row the partial restore may have written first.
        row.runner.adopt_prefix_cursor(depth, {})
        row.runner.reset()
        if row.draft is not None:
            row.draft.reset()
        row.previous_hidden = None
        row.prefill_tokens_seen = 0

    # -- prefill ----------------------------------------------------------

    def _capture_prefix(self, row) -> None:
        """Checkpoint a completed prefill boundary when it is worth keeping."""

        cache = self._prefix
        if cache is None or row.runner is None or not row.prefix_eligible:
            return
        depth = int(row.prefill_tokens_seen)
        final = depth == len(row.prompt_ids)
        kind = None
        if final:
            kind = "prompt_end"
        else:
            due = [target for target in row.prefix_targets if target[0] <= depth]
            row.prefix_targets = [target for target in row.prefix_targets if target[0] > depth]
            hits = [target for target in due if depth <= target[1]]
            if hits and depth - row.prefix_last_capture >= MIN_CAPTURE_SPACING_TOKENS:
                kind = hits[-1][2]
        if kind is None or depth < MIN_PREFIX_TOKENS or depth <= row.prefix_last_capture:
            return
        draft = row.draft if row.draft is not None and row.previous_hidden is not None else None
        try:
            chain, snapshot, _ = cache.capture(
                row.prompt_ids[:depth], row.runner, draft,
                chain=row.prefix_chain, previous_hidden=row.previous_hidden, kind=kind,
            )
        except PrefixCacheUnsupported as error:
            self._prefix_unsupported = f"unsupported_layout: {error}"
            row.prefix_eligible = False
            return
        row.prefix_chain = chain
        if snapshot is not None:
            row.prefix_captures += 1
            row.prefix_last_capture = depth

    def _release_prefix(self, row) -> None:
        chain = getattr(row, "prefix_chain", None)
        if chain is not None and self._prefix is not None:
            self._prefix.release_chain(chain)
        row.prefix_chain = None

    # -- telemetry --------------------------------------------------------

    def _prefix_request_telemetry(self, row) -> dict[str, Any]:
        cache = self._prefix
        reused = int(row.prefix_reused_tokens)
        reason = row.prefix_reason
        if reason is None and cache is None:
            reason = "cache_off"
        return {
            "mode": self._prefix_mode,
            "block_size_tokens": 256,
            "reuse_scope": "prompt_prefix",
            "eligible": bool(row.prefix_eligible),
            "lookup": bool(row.prefix_lookup),
            "hit": bool(reused),
            "source": "checkpoint" if reused else None,
            "matched_tokens": int(row.prefix_matched_tokens),
            "reused_tokens": reused,
            "avoided_prefill_tokens": reused,
            "executed_prefill_tokens": max(0, len(row.prompt_ids) - reused),
            "state_clone_bytes": int(row.prefix_restore_bytes),
            "restore_ms": round(float(row.prefix_restore_ms), 3),
            "captures": int(row.prefix_captures),
            "fallback_reason": reason,
            "cache_resident_entries": 0 if cache is None else cache.entries,
            "cache_resident_bytes": 0 if cache is None else cache.resident_bytes,
        }

    def prefix_cache_observability(self) -> dict[str, Any]:
        cache = self._prefix
        stats = {} if cache is None else cache.stats()
        resident = int(stats.get("resident_bytes", 0))
        return {
            "mode": self._prefix_mode,
            "block_size_tokens": 256,
            "snapshot_entries": int(stats.get("snapshot_entries", 0)),
            "snapshot_limit": int(stats.get("snapshot_limit", 0)),
            "retained_snapshot_entries": int(stats.get("snapshot_entries", 0)),
            "snapshot_bytes": resident,
            "retained_kv_pages": 0,
            "retained_kv_bytes": int(stats.get("segment_bytes", 0)),
            "resident_bytes": resident,
            "resident_limit_bytes": int(stats.get("resident_limit_bytes", 0)),
            "unsupported": self._prefix_unsupported,
            "stats": stats,
        }

    def observability_snapshot(self) -> dict[str, Any]:
        return {"prefix_cache": self.prefix_cache_observability()}
