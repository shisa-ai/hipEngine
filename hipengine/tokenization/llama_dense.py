"""Torch-free tokenizer adapter for dense Llama-family HF checkpoints.

The checkpoint's ``tokenizer.json`` is the source of truth: the adapter validates
its vocabulary against the decoder's declared width and its chat tokens against
the config's ``eos_token_id``/``bos_token_id``, so a mismatched tokenizer fails
here instead of emitting text that decodes to the wrong bytes.

:meth:`LlamaDenseTokenizer.format_chat_prompt` renders the checkpoint's own chat
template for the no-tools case. The template is read from the checkpoint rather
than assumed, because the family spans plain Llama and ChatML fine-tunes and the
two do not share a prompt shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizers import Tokenizer

if TYPE_CHECKING:
    from collections.abc import Sequence


_CHAT_START = "<|im_start|>"
_CHAT_END = "<|im_end|>"


@dataclass(frozen=True)
class LlamaDenseTokenizer:
    """HF ``tokenizer.json`` encoder bound to a validated decoder vocabulary."""

    encoder: Tokenizer
    vocab_size: int
    eos_token_id: int
    bos_token_id: int | None
    chat_style: str

    @classmethod
    def from_model_path(
        cls,
        model_path: str | Path,
        *,
        model_vocab_size: int,
        eos_token_id: int,
        bos_token_id: int | None = None,
    ) -> "LlamaDenseTokenizer":
        directory = Path(model_path)
        path = directory / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"dense Llama generation needs a tokenizer.json beside the weights: {path}"
            )
        encoder = Tokenizer.from_file(str(path))
        tokenizer_vocab_size = int(encoder.get_vocab_size(with_added_tokens=True))
        if tokenizer_vocab_size > model_vocab_size:
            raise ValueError(
                f"tokenizer vocabulary {tokenizer_vocab_size} exceeds the decoder's "
                f"{model_vocab_size} rows"
            )
        if encoder.token_to_id(_CHAT_END) != int(eos_token_id):
            raise ValueError(
                f"tokenizer {_CHAT_END} id differs from config.eos_token_id {eos_token_id}"
            )
        chat_style = cls._detect_chat_style(directory)
        return cls(
            encoder=encoder,
            vocab_size=tokenizer_vocab_size,
            eos_token_id=int(eos_token_id),
            bos_token_id=None if bos_token_id is None else int(bos_token_id),
            chat_style=chat_style,
        )

    @staticmethod
    def _detect_chat_style(directory: Path) -> str:
        """Return the chat dialect the checkpoint declares.

        Only the ChatML dialect is implemented. A checkpoint that declares another
        template is refused rather than rendered with ChatML markers its tokenizer
        may not carry.
        """

        path = directory / "tokenizer_config.json"
        if not path.is_file():
            return "chatml"
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "chatml"
        template = config.get("chat_template")
        if isinstance(template, list):
            template = "".join(
                str(entry.get("template", "")) for entry in template if isinstance(entry, dict)
            )
        if not isinstance(template, str) or not template.strip():
            return "chatml"
        if _CHAT_START in template and _CHAT_END in template:
            return "chatml"
        raise NotImplementedError(
            f"dense Llama chat template is not ChatML; {directory / 'tokenizer_config.json'} "
            "declares a different dialect, which this adapter does not render"
        )

    # -- text --------------------------------------------------------------
    def encode(self, text: str) -> tuple[int, ...]:
        """Tokenize preformatted text without adding special tokens."""

        return tuple(
            int(token_id)
            for token_id in self.encoder.encode(str(text), add_special_tokens=False).ids
        )

    def encode_chat(
        self,
        user: str,
        *,
        system: str | None = None,
        enable_thinking: bool = True,
    ) -> tuple[int, ...]:
        return self.encode(self.format_chat_prompt(user, system=system, enable_thinking=enable_thinking))

    def decode(self, token_ids: Sequence[int], *, skip_special: bool = False) -> str:
        ids = [int(token_id) for token_id in token_ids]
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in ids):
            raise ValueError(f"token ids must be in [0, {self.vocab_size})")
        return str(self.encoder.decode(ids, skip_special_tokens=bool(skip_special)))

    def decode_token(self, token_id: int) -> str:
        """Decode exactly one token, used by tokenizer-aware sampling constraints."""

        return self.decode((int(token_id),))

    # -- prompts -----------------------------------------------------------
    def format_chat_prompt(
        self,
        user: str,
        *,
        system: str | None = None,
        enable_thinking: bool = True,
    ) -> str:
        """Render the checkpoint's ChatML prompt up to the assistant prefix."""

        prefix = "" if system is None else f"{_CHAT_START}system\n{system}{_CHAT_END}\n"
        rendered = (
            f"{prefix}{_CHAT_START}user\n{user}{_CHAT_END}\n"
            f"{_CHAT_START}assistant\n"
        )
        if enable_thinking:
            return rendered + " thinking\n"
        return rendered + " thinking\n\n</think>\n\n"


__all__ = ["LlamaDenseTokenizer"]
