"""
Compatibility shim for monkey-patching the ``transformers`` library.

Provides :class:`_TokenizerShim` (a complete replacement for
``tokenizers.Tokenizer``) and :class:`_Encoding` (a minimal stand-in
for ``tokenizers.Encoding``).  All encoding, decoding, and vocabulary
operations are handled by the Rust :class:`Tokenizer` -- no reference
to the original ``tokenizers`` library is kept.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

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
    Complete replacement for ``tokenizers.Tokenizer``.

    All encoding, decoding, and vocabulary operations are performed by
    the Rust :class:`Tokenizer`.  No reference to the original
    ``tokenizers.Tokenizer`` is kept.
    """

    def __init__(self, src) -> None:
        if isinstance(src, str):
            self._json = src
            self._fast = Tokenizer.from_json_str(src)
        elif isinstance(src, _TokenizerShim):
            self._json = src._json
            self._fast = Tokenizer.from_json_str(src._json)
        elif hasattr(src, "to_str"):
            # Accept a real tokenizers.Tokenizer (e.g. from convert_slow_tokenizer).
            self._json = src.to_str()
            self._fast = Tokenizer.from_json_str(self._json)
        else:
            raise TypeError(
                f"expected JSON string, _TokenizerShim, or tokenizers.Tokenizer; "
                f"got {type(src).__name__}"
            )
        self._truncation: dict | None = None
        self._padding: dict | None = None

    def __deepcopy__(self, memo):
        new = object.__new__(_TokenizerShim)
        memo[id(self)] = new
        new._json = self._json
        new._fast = Tokenizer.from_json_str(self._json)
        new._truncation = copy.deepcopy(self._truncation, memo)
        new._padding = copy.deepcopy(self._padding, memo)
        return new

    # -- Factory class methods ----------------------------------------

    @classmethod
    def from_str(cls, json_str: str) -> _TokenizerShim:
        return cls(json_str)

    @classmethod
    def from_file(cls, path: str) -> _TokenizerShim:
        return cls(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_pretrained(
        cls, identifier: str, *args: object, **kwargs: object
    ) -> _TokenizerShim:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(identifier, "tokenizer.json")
        return cls.from_file(path)

    @classmethod
    def from_buffer(cls, buf: bytes) -> _TokenizerShim:
        return cls(buf.decode("utf-8"))

    # -- Serialization ------------------------------------------------

    def to_str(self, pretty: bool = False) -> str:
        if pretty:
            parsed = json.loads(self._json)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        return self._json

    # -- Truncation / Padding -----------------------------------------

    @property
    def truncation(self) -> dict | None:
        return self._truncation

    @property
    def padding(self) -> dict | None:
        return self._padding

    def enable_truncation(
        self,
        max_length: int,
        stride: int = 0,
        strategy: str = "longest_first",
        direction: str = "right",
    ) -> None:
        self._truncation = {
            "max_length": max_length,
            "stride": stride,
            "strategy": strategy,
            "direction": direction,
        }

    def no_truncation(self) -> None:
        self._truncation = None

    def enable_padding(
        self,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
        length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self._padding = {
            "direction": direction,
            "pad_id": pad_id,
            "pad_type_id": pad_type_id,
            "pad_token": pad_token,
            "length": length,
            "pad_to_multiple_of": pad_to_multiple_of,
        }

    def no_padding(self) -> None:
        self._padding = None

    # -- Encoding -----------------------------------------------------

    def _wrap_encoding(self, ids: list[int]) -> _Encoding:
        enc = _Encoding(ids)
        if self._truncation is not None:
            enc._truncate(self._truncation["max_length"])
        if self._padding is not None and self._padding.get("length") is not None:
            enc._pad(self._padding["length"], self._padding.get("pad_id", 0))
        return enc

    def encode(
        self,
        sequence: str,
        pair: str | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> _Encoding:
        if pair is not None:
            raise NotImplementedError("pair encoding is not supported by fastokens")
        if is_pretokenized:
            raise NotImplementedError(
                "pre-tokenized input is not supported by fastokens"
            )
        ids = self._fast.encode(sequence, add_special_tokens=add_special_tokens)
        return self._wrap_encoding(ids)

    def encode_batch(
        self,
        inputs: list,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> list[_Encoding]:
        if is_pretokenized or any(isinstance(inp, (list, tuple)) for inp in inputs):
            raise NotImplementedError(
                "pair/pre-tokenized batch encoding is not supported by fastokens"
            )
        batch_ids = self._fast.encode_batch(
            inputs, add_special_tokens=add_special_tokens
        )
        return [self._wrap_encoding(ids) for ids in batch_ids]

    def encode_batch_fast(
        self,
        inputs: list[str],
        add_special_tokens: bool = True,
    ) -> list[list[int]]:
        return self._fast.encode_batch(
            inputs, add_special_tokens=add_special_tokens
        )

    # -- Decoding -----------------------------------------------------

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._fast.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        sequences: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return self._fast.decode_batch(
            sequences, skip_special_tokens=skip_special_tokens
        )

    # -- Vocabulary ---------------------------------------------------

    def id_to_token(self, id: int) -> str | None:
        return self._fast.id_to_token(id)

    def token_to_id(self, token: str) -> int | None:
        return self._fast.token_to_id(token)

    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]:
        vocab = {}
        for i in range(self._fast.vocab_size):
            tok = self._fast.id_to_token(i)
            if tok is not None:
                vocab[tok] = i
        return vocab

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return self._fast.vocab_size

    # -- Token management (no-ops) ------------------------------------

    def add_tokens(self, tokens) -> int:
        return 0

    def add_special_tokens(self, special_tokens) -> int:
        return 0

    # -- Component accessors (stubs for transformers compatibility) ----

    @property
    def model(self):
        return _ModelStub(self)

    @property
    def normalizer(self):
        return None

    @property
    def pre_tokenizer(self):
        return None

    @property
    def post_processor(self):
        return None

    @property
    def decoder(self):
        return None


class _ModelStub:
    """Minimal stub for ``tokenizers.models.Model`` to support saving."""

    def __init__(self, shim: _TokenizerShim) -> None:
        self._shim = shim

    def save(self, folder: str, prefix: str | None = None) -> list[str]:
        name = f"{prefix}-vocab.json" if prefix else "vocab.json"
        path = Path(folder) / name
        vocab = self._shim.get_vocab()
        path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        return [str(path)]
