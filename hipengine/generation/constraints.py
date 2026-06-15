"""Tokenizer-agnostic decode constraint primitives."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


_PHASE_THINK = "think"
_PHASE_CLOSING_THINK = "closing_think"
_PHASE_ANSWER = "answer"
_PHASE_DONE = "done"
_THINKING_PHASES = {_PHASE_THINK, _PHASE_CLOSING_THINK, _PHASE_ANSWER, _PHASE_DONE}
_SOFT_CLOSE_MAX_LOGIT_BIAS = 8.0


@dataclass(frozen=True, slots=True)
class TokenSequenceDFAState:
    """Incremental matcher for token-id sequences.

    The state keeps only the longest suffix that is also a prefix of at least
    one configured sequence. That is enough to detect future full matches while
    staying independent of tokenizer/model code.
    """

    sequences: tuple[tuple[int, ...], ...] = ()
    suffix: tuple[int, ...] = ()
    matched_sequence: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequences", normalize_token_sequences(self.sequences))
        object.__setattr__(self, "suffix", tuple(int(token) for token in self.suffix))
        object.__setattr__(self, "matched_sequence", tuple(int(token) for token in self.matched_sequence))

    @classmethod
    def from_sequences(cls, sequences: Iterable[Iterable[int]] | None) -> "TokenSequenceDFAState":
        return cls(sequences=normalize_token_sequences(sequences))

    @property
    def matched(self) -> bool:
        return bool(self.matched_sequence)

    def observe(self, token_id: int) -> "TokenSequenceDFAState":
        if not self.sequences or self.matched_sequence:
            return self
        history = (*self.suffix, int(token_id))
        matched = _matched_sequence(history, self.sequences)
        if matched:
            return TokenSequenceDFAState(
                sequences=self.sequences,
                suffix=matched,
                matched_sequence=matched,
            )
        return TokenSequenceDFAState(
            sequences=self.sequences,
            suffix=_longest_prefix_suffix(history, self.sequences),
        )

    def observe_many(self, token_ids: Iterable[int]) -> "TokenSequenceDFAState":
        state = self
        for token_id in token_ids:
            state = state.observe(int(token_id))
            if state.matched:
                break
        return state

    def to_json_dict(self) -> dict[str, Any]:
        if self.matched_sequence:
            return {"matched_sequence": list(self.matched_sequence)}
        if not self.suffix:
            return {}
        return {
            "partial_suffix": list(self.suffix),
            "candidate_sequences": [
                list(sequence)
                for sequence in self.sequences
                if sequence[: len(self.suffix)] == self.suffix
            ],
        }


@dataclass(slots=True)
class ForcedTokenQueue:
    """Mutable FIFO queue for tokens that must be emitted before sampling."""

    tokens: Iterable[int] = ()
    reason: str | None = None
    _pending: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pending = [int(token) for token in self.tokens]
        if any(token < 0 for token in pending):
            raise ValueError("forced tokens must be non-negative token ids")
        self._pending = pending
        self.reason = None if self.reason is None else str(self.reason)

    @property
    def pending_tokens(self) -> tuple[int, ...]:
        return tuple(self._pending)

    def __bool__(self) -> bool:
        return bool(self._pending)

    def __len__(self) -> int:
        return len(self._pending)

    def peek(self) -> int | None:
        return self._pending[0] if self._pending else None

    def pop(self) -> int | None:
        if not self._pending:
            return None
        return self._pending.pop(0)

    def extend(self, token_ids: Iterable[int], *, reason: str | None = None) -> None:
        tokens = tuple(int(token) for token in token_ids)
        if any(token < 0 for token in tokens):
            raise ValueError("forced tokens must be non-negative token ids")
        self._pending.extend(tokens)
        if reason is not None:
            self.reason = str(reason)

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"pending_tokens": list(self._pending)}
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(slots=True)
class JsonObjectConstraintState:
    """Incremental, tokenizer-agnostic balance state for JSON-object output.

    This is deliberately a structural primitive, not a full JSON parser. It
    tracks the root object, strings/escapes, object/array nesting, trailing
    content, and the deterministic suffix needed to close the current structure
    when closing is safe.
    """

    started: bool = False
    complete: bool = False
    invalid: bool = False
    error_reason: str | None = None
    stack: Iterable[str] = ()
    in_string: bool = False
    escaping: bool = False
    observed_text: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        self.started = bool(self.started)
        self.complete = bool(self.complete)
        self.invalid = bool(self.invalid)
        self.error_reason = None if self.error_reason is None else str(self.error_reason)
        self.stack = [str(item) for item in self.stack]
        if any(item not in {"}", "]"} for item in self.stack):
            raise ValueError("JsonObjectConstraintState.stack may contain only '}' and ']'")
        self.in_string = bool(self.in_string)
        self.escaping = bool(self.escaping)
        self.observed_text = str(self.observed_text)

    @property
    def forced_close_suffix(self) -> str:
        """Return the close suffix for an incomplete object when safe to force."""

        if self.invalid or not self.started or self.complete or self.escaping:
            return ""
        close_suffix = "".join(reversed(self.stack))
        if not close_suffix:
            return ""
        if self.in_string:
            close_suffix = f'"{close_suffix}'
        if _is_complete_json_root_object(f"{self.observed_text}{close_suffix}"):
            return close_suffix
        return ""

    @property
    def needs_close(self) -> bool:
        return bool(self.forced_close_suffix)

    def observe_text(self, text: str) -> "JsonObjectConstraintState":
        for char in str(text):
            self.observe_char(char)
            if self.invalid:
                break
        return self

    def observe_char(self, char: str) -> "JsonObjectConstraintState":
        if self.invalid:
            return self
        if len(char) != 1:
            raise ValueError("observe_char expects exactly one character")
        self.observed_text += char

        if not self.started:
            if char.isspace():
                return self
            if char != "{":
                return self._mark_invalid("root_must_be_object")
            self.started = True
            self.stack.append("}")
            return self

        if self.complete:
            if char.isspace():
                return self
            return self._mark_invalid("trailing_content")

        if self.in_string:
            if self.escaping:
                self.escaping = False
            elif char == "\\":
                self.escaping = True
            elif char == '"':
                self.in_string = False
            return self

        if char == '"':
            self.in_string = True
        elif char == "{":
            self.stack.append("}")
        elif char == "[":
            self.stack.append("]")
        elif char in {"}", "]"}:
            if not self.stack:
                return self._mark_invalid("unmatched_closing_delimiter")
            expected = self.stack[-1]
            if char != expected:
                return self._mark_invalid("mismatched_closing_delimiter")
            self.stack.pop()
            if not self.stack:
                self.complete = True
        return self

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "started": bool(self.started),
            "complete": bool(self.complete),
            "invalid": bool(self.invalid),
        }
        if self.error_reason is not None:
            payload["error_reason"] = self.error_reason
        if self.stack:
            payload["expected_close_stack"] = list(self.stack)
        if self.in_string:
            payload["in_string"] = True
        if self.escaping:
            payload["escaping"] = True
        suffix = self.forced_close_suffix
        if suffix:
            payload["forced_close_suffix"] = suffix
        return payload

    def _mark_invalid(self, reason: str) -> "JsonObjectConstraintState":
        self.invalid = True
        self.error_reason = reason
        return self


@dataclass(slots=True)
class ThinkingBudgetState:
    """Tokenizer-agnostic state for decode-time thinking budget control.

    The state does not choose tokens by itself. It tracks reasoning/answer token
    counts, detects soft/hard budget pressure, and enqueues a tokenizer-lowered
    close sequence through ``ForcedTokenQueue`` when a controller asks it to.
    """

    close_sequence: Iterable[int] = ()
    hard_token_cap: int | None = None
    soft_close_window: int = 0
    phase: str = _PHASE_THINK
    reasoning_tokens: int = 0
    answer_tokens: int = 0
    forced_tokens: ForcedTokenQueue | Iterable[int] = field(default_factory=ForcedTokenQueue)
    close_state: TokenSequenceDFAState = field(init=False)

    def __post_init__(self) -> None:
        close_sequence = tuple(int(token) for token in self.close_sequence)
        if any(token < 0 for token in close_sequence):
            raise ValueError("close_sequence must contain non-negative token ids")
        if self.hard_token_cap is not None and int(self.hard_token_cap) < 0:
            raise ValueError("hard_token_cap must be non-negative")
        if int(self.soft_close_window) < 0:
            raise ValueError("soft_close_window must be non-negative")
        phase = str(self.phase)
        if phase not in _THINKING_PHASES:
            raise ValueError(f"phase must be one of {sorted(_THINKING_PHASES)}")
        forced_tokens = self.forced_tokens
        if not isinstance(forced_tokens, ForcedTokenQueue):
            forced_tokens = ForcedTokenQueue(forced_tokens)
        self.close_sequence = close_sequence
        self.hard_token_cap = None if self.hard_token_cap is None else int(self.hard_token_cap)
        self.soft_close_window = int(self.soft_close_window)
        self.phase = phase
        self.reasoning_tokens = _nonnegative_int(self.reasoning_tokens, name="reasoning_tokens")
        self.answer_tokens = _nonnegative_int(self.answer_tokens, name="answer_tokens")
        self.forced_tokens = forced_tokens
        self.close_state = TokenSequenceDFAState.from_sequences((close_sequence,) if close_sequence else ())

    @property
    def remaining_think_tokens(self) -> int | None:
        if self.hard_token_cap is None:
            return None
        return max(0, int(self.hard_token_cap) - int(self.reasoning_tokens))

    @property
    def hard_close_due(self) -> bool:
        return (
            self.phase == _PHASE_THINK
            and self.hard_token_cap is not None
            and int(self.reasoning_tokens) >= int(self.hard_token_cap)
        )

    @property
    def soft_close_active(self) -> bool:
        if self.phase != _PHASE_THINK or self.hard_token_cap is None or self.soft_close_window <= 0:
            return False
        threshold = max(0, int(self.hard_token_cap) - int(self.soft_close_window))
        return int(self.reasoning_tokens) >= threshold

    @property
    def soft_close_progress(self) -> float | None:
        """Return soft-close window progress in ``(0, 1]`` when active."""

        if not self.soft_close_active or self.hard_close_due:
            return None
        threshold = max(0, int(self.hard_token_cap or 0) - int(self.soft_close_window))
        consumed = max(1, int(self.reasoning_tokens) - threshold + 1)
        return min(1.0, float(consumed) / float(max(1, int(self.soft_close_window))))

    @property
    def soft_close_bias(self) -> float | None:
        """Positive sparse logit bias for the first close token during soft close."""

        progress = self.soft_close_progress
        if progress is None:
            return None
        return float(_SOFT_CLOSE_MAX_LOGIT_BIAS * progress)

    @property
    def budget_pressure(self) -> str | None:
        if self.hard_close_due:
            return "hard_close"
        if self.soft_close_active:
            return "soft_close"
        return None

    @property
    def eos_suppression_active(self) -> bool:
        """Return whether EOS should be suppressed until visible answer phase."""

        return self.phase in {_PHASE_THINK, _PHASE_CLOSING_THINK}

    def force_close(self, *, reason: str = "manual_close") -> bool:
        """Queue the full close sequence if a close is possible and not pending."""

        if not self.close_sequence or self.phase in {_PHASE_ANSWER, _PHASE_DONE} or self.forced_tokens:
            return False
        self.forced_tokens.extend(self.close_sequence, reason=reason)
        self.phase = _PHASE_CLOSING_THINK
        return True

    def ensure_hard_close(self, *, reason: str = "thinking_hard_close") -> bool:
        """Queue the close sequence when the hard cap has been reached."""

        if not self.hard_close_due:
            return False
        return self.force_close(reason=reason)

    def observe(self, token_id: int) -> "ThinkingBudgetState":
        token = int(token_id)
        if token < 0:
            raise ValueError("observed token must be non-negative")
        if self.phase in {_PHASE_THINK, _PHASE_CLOSING_THINK}:
            self.reasoning_tokens += 1
            self.close_state = self.close_state.observe(token)
            if self.close_state.matched:
                self.phase = _PHASE_ANSWER
            elif self.phase == _PHASE_THINK and self.soft_close_active and self.close_state.suffix:
                remaining = self.close_sequence[len(self.close_state.suffix) :]
                if remaining:
                    self.forced_tokens.extend(remaining, reason="thinking_soft_close")
                    self.phase = _PHASE_CLOSING_THINK
        elif self.phase == _PHASE_ANSWER:
            self.answer_tokens += 1
        return self

    def to_json_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "phase": self.phase,
            "reasoning_tokens": int(self.reasoning_tokens),
            "answer_tokens": int(self.answer_tokens),
        }
        if self.hard_token_cap is not None:
            payload["hard_token_cap"] = int(self.hard_token_cap)
            payload["remaining_think_tokens"] = self.remaining_think_tokens
        if self.soft_close_window:
            payload["soft_close_window"] = int(self.soft_close_window)
        if self.budget_pressure is not None:
            payload["budget_pressure"] = self.budget_pressure
        if self.soft_close_bias is not None:
            payload["soft_close_bias"] = self.soft_close_bias
        if self.close_sequence:
            payload["close_sequence"] = list(self.close_sequence)
        close_state = self.close_state.to_json_dict()
        if close_state:
            payload["close_state"] = close_state
        if self.forced_tokens:
            payload["forced_tokens"] = self.forced_tokens.to_json_dict()
        return payload


def normalize_token_sequences(sequences: Iterable[Iterable[int]] | None) -> tuple[tuple[int, ...], ...]:
    if sequences is None:
        return ()
    normalized: list[tuple[int, ...]] = []
    for raw_sequence in sequences:
        sequence = tuple(int(token) for token in raw_sequence)
        if not sequence:
            continue
        if any(token < 0 for token in sequence):
            raise ValueError("token sequences must contain non-negative token ids")
        if sequence not in normalized:
            normalized.append(sequence)
    return tuple(normalized)


def token_sequence_state_for_tokens(
    token_ids: Iterable[int],
    sequences: Iterable[Iterable[int]] | None,
) -> TokenSequenceDFAState:
    return TokenSequenceDFAState.from_sequences(sequences).observe_many(token_ids)


def _nonnegative_int(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _is_complete_json_root_object(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)


def _matched_sequence(
    token_ids: Sequence[int],
    sequences: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    matched: tuple[int, ...] = ()
    for sequence in sequences:
        if len(sequence) > len(token_ids):
            continue
        if tuple(int(token) for token in token_ids[-len(sequence) :]) == sequence and len(sequence) > len(matched):
            matched = sequence
    return matched


def _longest_prefix_suffix(
    token_ids: Sequence[int],
    sequences: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    best: tuple[int, ...] = ()
    tokens = tuple(int(token) for token in token_ids)
    for sequence in sequences:
        max_len = min(len(sequence) - 1, len(tokens))
        for length in range(1, max_len + 1):
            suffix = tokens[-length:]
            if suffix == sequence[:length] and len(suffix) > len(best):
                best = suffix
    return best


__all__ = [
    "ForcedTokenQueue",
    "JsonObjectConstraintState",
    "ThinkingBudgetState",
    "TokenSequenceDFAState",
    "normalize_token_sequences",
    "token_sequence_state_for_tokens",
]
