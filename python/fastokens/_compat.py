"""
Compatibility shim for monkey-patching the ``transformers`` library.

Provides :class:`_TokenizerShim` (a drop-in replacement for
``tokenizers.Tokenizer``) and :class:`_Encoding` (a minimal stand-in
for ``tokenizers.Encoding``).  The shim delegates every method/attribute
it does not override to the *original* ``tokenizers.Tokenizer`` instance,
so operations we have not re-implemented (decoding, vocab lookup, ...)
keep working.
"""

from __future__ import annotations

from fastokens._native import Tokenizer


class _Encoding:
    """Minimal replacement for ``tokenizers.Encoding``."""

    __slots__ = (
        "ids",
        "type_ids",
        "attention_mask",
        "special_tokens_mask",
        "tokens",
        "offsets",
        "overflowing",
        "n_sequences",
    )

    def __init__(self, ids: list[int]) -> None:
        n = len(ids)
        self.ids = ids
        self.type_ids = [0] * n
        self.attention_mask = [1] * n
        self.special_tokens_mask = [0] * n
        self.tokens: list[str] = []
        self.offsets = [(0, 0)] * n
        self.overflowing: list[_Encoding] = []
        self.n_sequences = 1

    def __len__(self) -> int:
        return len(self.ids)

    def sequence_ids(self) -> list[int | None]:
        return [0] * len(self.ids)

    def word_ids(self) -> list[int | None]:
        return [None] * len(self.ids)

    def _truncate(self, max_length: int) -> None:
        """Truncate in-place to *max_length* tokens."""
        if len(self.ids) <= max_length:
            return
        self.ids = self.ids[:max_length]
        self.type_ids = self.type_ids[:max_length]
        self.attention_mask = self.attention_mask[:max_length]
        self.special_tokens_mask = self.special_tokens_mask[:max_length]
        self.offsets = self.offsets[:max_length]

    def _pad(self, length: int, pad_id: int) -> None:
        """Pad in-place to *length* tokens."""
        deficit = length - len(self.ids)
        if deficit <= 0:
            return
        self.ids.extend([pad_id] * deficit)
        self.type_ids.extend([0] * deficit)
        self.attention_mask.extend([0] * deficit)
        self.special_tokens_mask.extend([0] * deficit)
        self.offsets.extend([(0, 0)] * deficit)


class _TokenizerShim:
    """
    Drop-in replacement for ``tokenizers.Tokenizer``.

    Encoding is performed by our Rust :class:`Tokenizer`; everything
    else is delegated to the original ``tokenizers.Tokenizer`` that
    was created from the same JSON config.
    """

    # Set by patch_transformers() before any instances are created.
    _OrigTokenizer: type = None  # type: ignore[assignment]

    def __init__(self, original: object) -> None:
        self._original = original
        self._fast = Tokenizer.from_json_str(original.to_str())

    # -- Factory class methods ----------------------------------------

    @classmethod
    def from_str(cls, json: str) -> _TokenizerShim:
        return cls(cls._OrigTokenizer.from_str(json))

    @classmethod
    def from_file(cls, path: str) -> _TokenizerShim:
        return cls(cls._OrigTokenizer.from_file(path))

    @classmethod
    def from_pretrained(
        cls, identifier: str, *args: object, **kwargs: object
    ) -> _TokenizerShim:
        return cls(cls._OrigTokenizer.from_pretrained(identifier, *args, **kwargs))

    @classmethod
    def from_buffer(cls, buf: bytes) -> _TokenizerShim:
        return cls(cls._OrigTokenizer.from_buffer(buf))

    # -- Encoding -----------------------------------------------------

    def _encode_one(self, text: str) -> _Encoding:
        ids = self._fast.encode(text)
        enc = _Encoding(ids)
        trunc = self._original.truncation
        if trunc is not None:
            enc._truncate(trunc["max_length"])
        pad = self._original.padding
        if pad is not None and pad.get("length") is not None:
            enc._pad(pad["length"], pad.get("pad_id", 0))
        return enc

    def encode(
        self,
        sequence: str,
        pair: str | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        fast: bool = False,
    ) -> _Encoding:
        if not fast or pair is not None or is_pretokenized or not add_special_tokens:
            return self._original.encode(
                sequence, pair, is_pretokenized, add_special_tokens
            )
        return self._encode_one(sequence)

    def encode_batch(
        self,
        inputs: list,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        fast: bool = False,
    ) -> list[_Encoding]:
        if (
            not fast
            or is_pretokenized
            or not add_special_tokens
            or any(isinstance(inp, (list, tuple)) for inp in inputs)
        ):
            return self._original.encode_batch(
                inputs, is_pretokenized, add_special_tokens
            )
        return [self._encode_one(inp) for inp in inputs]

    # -- Delegation ---------------------------------------------------

    def __getattr__(self, name: str) -> object:
        return getattr(self._original, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._original, name, value)
