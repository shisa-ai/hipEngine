"""Session adapter exposing the control-smoke contract over ``Gemma4Runner``.

Piece (B) of the execution-profile gate build drives a session object whose
contract is ``prefill``/``step`` returning ``(.logits, .token_id)`` under a
context manager. Gemma 4 has no resident-session class -- Qwen, Laguna, Qwen4Exp
and Paro each have one and gemma 4 has only the runner -- so this adapter
supplies exactly that contract and nothing more.

The mapping is thin because the runner already treats prefill and decode as one
computation: its own docstring says "a decode step is a prefill of one token",
so ``prefill`` and ``step`` both delegate to ``forward``.

The runner exposes neither ``backend`` nor ``target_arch``, which the smoke
reads after construction, so the session carries them as caller-supplied
values rather than guessing them from the weights.

Keyword arguments the Qwen-shaped contract carries (``use_bulk``,
``bulk_attention_mode``, ``capture_hidden_seed_fp32``) are accepted and
documented as no-ops: Gemma 4 has no GDN and no bulk-attention distinction, so
there is nothing for them to select. They are not silently reinterpreted.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Gemma4ForwardResult:
    """One prefill or decode outcome, matching the smoke's duck type."""

    logits: np.ndarray
    token_id: int


class Gemma4ResidentSession:
    """Context-managed ``prefill``/``step`` view over a ``Gemma4Runner``.

    Construct it around an already-built runner so that weight loading stays
    with the caller that owns the model; this class only owns the session
    contract and its lifecycle.
    """

    def __init__(
        self,
        runner: Any,
        *,
        backend: str,
        target_arch: str,
    ) -> None:
        if runner is None:
            raise ValueError("runner must not be None")
        self._runner = runner
        self._backend = str(backend)
        self._target_arch = str(target_arch)
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Gemma4ResidentSession":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._runner, "close", None)
        if callable(close):
            close()

    def reset(self) -> None:
        """Drop accumulated state so the next prompt starts from position 0."""

        self._require_open()
        reset = getattr(self._runner, "reset", None)
        if not callable(reset):
            raise NotImplementedError(
                f"{type(self._runner).__name__} does not implement reset(), so a "
                "second prompt cannot start a fresh sequence"
            )
        reset()

    # -- the smoke's reads -------------------------------------------------

    @property
    def runner(self) -> Any:
        return self._runner

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def target_arch(self) -> str:
        return self._target_arch

    @property
    def position(self) -> int:
        self._require_open()
        return int(self._runner.position)

    # -- the smoke's calls -------------------------------------------------

    def prefill(
        self,
        token_ids: Sequence[int],
        *,
        use_bulk: bool = True,
        bulk_attention_mode: str | None = None,
        return_logits: bool = True,
        capture_hidden_seed_fp32: bool = False,
    ) -> Gemma4ForwardResult:
        """Append a prompt and return the last row's logits plus its token.

        ``use_bulk`` and ``bulk_attention_mode`` select Qwen's bulk attention
        and GDN modes, neither of which Gemma 4 has; they are accepted so the
        call site matches the shared contract, and ignored rather than mapped
        onto something else.
        """

        del use_bulk, bulk_attention_mode, capture_hidden_seed_fp32
        self._require_open()
        tokens = [int(token) for token in token_ids]
        if not tokens:
            raise ValueError("prefill requires at least one token")
        return self._forward(tokens, return_logits=return_logits)

    def step(
        self,
        token_id: int,
        *,
        return_logits: bool = True,
    ) -> Gemma4ForwardResult:
        """Append one already-chosen token and return the next distribution."""

        self._require_open()
        return self._forward([int(token_id)], return_logits=return_logits)

    # -- internals ---------------------------------------------------------

    def _forward(self, tokens: Iterable[int], *, return_logits: bool) -> Gemma4ForwardResult:
        if not return_logits:
            raise NotImplementedError(
                "return_logits=False is not implemented for Gemma 4: the runner "
                "returns logits from forward() and the smoke only ever asks for "
                "them, so honouring False would mean discarding work rather than "
                "skipping it"
            )
        logits = self._runner.forward(list(tokens))
        token_id = self._runner.next_token(logits)
        return Gemma4ForwardResult(logits=np.ascontiguousarray(logits), token_id=int(token_id))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")


__all__ = ["Gemma4ForwardResult", "Gemma4ResidentSession"]