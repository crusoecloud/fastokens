"""Acceleration for tiktoken-based slow ``transformers`` tokenizers.

Some models ship a slow (Python-backend) tokenizer that wraps a
``tiktoken.Encoding`` instead of a ``tokenizer.json`` -- Moonshot's Kimi
``TikTokenTokenizer`` in Kimi K3's ``tokenization_kimi.py`` is the main one. ``patch_transformers``
only swaps fast-tokenizer backends, so these kept their original encoder: its
``_encode_text_piece`` slices the text at 400k characters, then walks every
character in Python to split runs of more than 25k whitespace or non-whitespace
characters before calling tiktoken. The walk dominates rendering a long prompt
(about 80 ms for a 240k-token chat).

This module routes those methods through ``Tokenizer.from_tiktoken`` and
``encode_segments``, reproducing both splits with a regex so the token ids stay
identical: the splits move token boundaries, so they cannot be skipped.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any

# Mirrors of the constants in Kimi's ``TikTokenTokenizer._encode_text_piece``.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000

# Each alternative is anchored to the start of a run. Unanchored, a failed match
# is retried at every character of a shorter run, which is quadratic in its
# length (seconds for 400k characters of runs just under the limit).
_LONG_RUN = re.compile(
    rf"(?<!\s)\s{{{MAX_NO_WHITESPACES_CHARS + 1},}}|(?<!\S)\S{{{MAX_NO_WHITESPACES_CHARS + 1},}}"
)


def split_like_tiktoken_text_piece(text: str) -> Iterator[str]:
    """Yield the substrings Kimi's ``_encode_text_piece`` hands to tiktoken.

    The text is cut every ``TIKTOKEN_MAX_ENCODE_CHARS`` characters, then each
    chunk is cut so no piece holds more than ``MAX_NO_WHITESPACES_CHARS``
    consecutive whitespace or non-whitespace characters. Only runs longer than
    that limit are cut, so a regex over the long runs replaces the per-character
    walk; ``\\s`` matches exactly the characters for which ``str.isspace()`` is
    true.
    """
    for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS):
        chunk = text[i : i + TIKTOKEN_MAX_ENCODE_CHARS]
        start = 0
        for run in _LONG_RUN.finditer(chunk):
            cut = run.start() + MAX_NO_WHITESPACES_CHARS
            while cut < run.end():
                yield chunk[start:cut]
                start = cut
                cut += MAX_NO_WHITESPACES_CHARS
        yield chunk[start:]


def _is_tiktoken_text_piece_tokenizer(tokenizer: Any) -> bool:
    model = getattr(tokenizer, "model", None)
    return (
        callable(getattr(tokenizer, "_encode_text_piece", None))
        and type(model).__module__.startswith("tiktoken")
        and bool(getattr(tokenizer, "vocab_file", None))
        and bool(getattr(type(tokenizer), "pat_str", None))
    )


class _TiktokenEncoder:
    """Encodes text pieces for one tiktoken vocabulary with fastokens.

    A module-level class rather than closures, so an accelerated tokenizer stays
    picklable and deep-copyable: only the vocabulary path, pattern and special
    tokens are pickled, and the native tokenizer is rebuilt on first use.
    """

    def __init__(self, vocab_file: str, pat_str: str, special_tokens: dict[str, int]) -> None:
        self.vocab_file = vocab_file
        self.pat_str = pat_str
        self.special_tokens = special_tokens
        self._fast = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_fast"] = None
        return state

    def _tokenizer(self):
        if self._fast is None:
            from fastokens._native import Tokenizer

            self._fast = Tokenizer.from_tiktoken(
                self.vocab_file, pattern=self.pat_str, special_tokens=self.special_tokens
            )
        return self._fast

    def _encode_pieces(self, pieces: Iterable[tuple[str, bool]]) -> list[int]:
        return list(self._tokenizer().encode_segments(list(pieces)).ids)

    def encode_text_piece(self, text: str, allow_special_tokens: bool = True) -> list[int]:
        return self._encode_pieces(
            (piece, allow_special_tokens) for piece in split_like_tiktoken_text_piece(text)
        )

    def encode_chat_segments(self, segments) -> list[int]:
        return self._encode_pieces(
            (piece, bool(segment.allow_special))
            for segment in segments
            for piece in split_like_tiktoken_text_piece(segment.text)
        )


def accelerate_tiktoken_tokenizer(tokenizer: Any) -> bool:
    """Route a tiktoken text-piece tokenizer's encoding through fastokens.

    Replaces the instance's ``_encode_text_piece`` and, when present,
    ``_encode_chat_segments`` (a list of objects with ``text`` and
    ``allow_special``), so ``encode`` and ``apply_chat_template`` keep their
    behavior and token ids. ``allow_special`` maps to tiktoken's
    ``allowed_special="all"`` versus ``disallowed_special=()``: special-token
    text is a control token only in segments that allow it. The replacements are
    bound methods of a :class:`_TiktokenEncoder`, so the tokenizer can still be
    pickled and deep-copied, and copies stay accelerated.

    Returns whether the tokenizer was accelerated; a tokenizer without this
    interface, or whose vocabulary fastokens cannot load, is left unchanged.
    """
    if getattr(tokenizer, "_fastokens_tiktoken", None) is not None:
        return True
    if not _is_tiktoken_text_piece_tokenizer(tokenizer):
        return False

    encoder = _TiktokenEncoder(
        tokenizer.vocab_file,
        type(tokenizer).pat_str,
        dict(tokenizer.model._special_tokens),
    )
    try:
        encoder._tokenizer()
    except Exception as e:  # noqa: BLE001 - keep the original encoder on any load failure
        print(f"[fastokens] could not load {tokenizer.vocab_file} ({e!r}); keeping tiktoken.")
        return False

    tokenizer._fastokens_tiktoken = encoder
    tokenizer._encode_text_piece = encoder.encode_text_piece
    if callable(getattr(tokenizer, "_encode_chat_segments", None)):
        tokenizer._encode_chat_segments = encoder.encode_chat_segments
    return True
