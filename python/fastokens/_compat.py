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


# ---------------------------------------------------------------------------
# _Encoding
# ---------------------------------------------------------------------------

class _Encoding:
    """Minimal replacement for ``tokenizers.Encoding``."""

    __slots__ = (
        "ids",
        "type_ids",
        "attention_mask",
        "special_tokens_mask",
        "_tokens",
        "_offsets",
        "overflowing",
        "n_sequences",
        "_sequence_ids",
        "_word_ids",
    )

    def __init__(self, ids: list[int]) -> None:
        n = len(ids)
        self.ids = ids
        self.type_ids = [0] * n
        self.attention_mask = [1] * n
        self.special_tokens_mask = [0] * n
        self._tokens: list[str] = []
        self._offsets = [(0, 0)] * n
        self.overflowing: list[_Encoding] = []
        self.n_sequences = 1
        self._sequence_ids: list[int | None] = [0] * n
        self._word_ids: list[int | None] = [None] * n

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        return f"Encoding(num_tokens={len(self.ids)})"

    # -- properties not tracked by fastokens ----------------------------

    @property
    def tokens(self) -> list[str]:
        raise NotImplementedError(
            "fastokens does not track token strings; "
            "use Tokenizer.id_to_token() to convert individual IDs"
        )

    @tokens.setter
    def tokens(self, value: list[str]) -> None:
        self._tokens = value

    @property
    def offsets(self) -> list[tuple[int, int]]:
        raise NotImplementedError(
            "fastokens does not track character offsets"
        )

    @offsets.setter
    def offsets(self, value: list[tuple[int, int]]) -> None:
        self._offsets = value

    @property
    def sequence_ids(self) -> list[int | None]:
        raise NotImplementedError(
            "fastokens does not track sequence IDs"
        )

    @sequence_ids.setter
    def sequence_ids(self, value: list[int | None]) -> None:
        self._sequence_ids = value

    @property
    def word_ids(self) -> list[int | None]:
        raise NotImplementedError(
            "fastokens does not track word IDs"
        )

    @word_ids.setter
    def word_ids(self, value: list[int | None]) -> None:
        self._word_ids = value

    @property
    def words(self) -> list[int | None]:
        raise NotImplementedError(
            "fastokens does not track word IDs"
        )

    @words.setter
    def words(self, value: list[int | None]) -> None:
        self._word_ids = value

    def set_sequence_id(self, sequence_id: int) -> None:
        self._sequence_ids = [sequence_id] * len(self.ids)

    # -- positional mapping (not tracked by fastokens) ------------------

    def char_to_token(self, char_pos: int, sequence_index: int = 0) -> int | None:
        raise NotImplementedError(
            "fastokens does not track character offsets"
        )

    def char_to_word(self, char_pos: int, sequence_index: int = 0) -> int | None:
        raise NotImplementedError(
            "fastokens does not track word IDs"
        )

    def token_to_chars(self, token_index: int) -> tuple[int, int] | None:
        raise NotImplementedError(
            "fastokens does not track character offsets"
        )

    def token_to_sequence(self, token_index: int) -> int | None:
        raise NotImplementedError(
            "fastokens does not track sequence IDs"
        )

    def token_to_word(self, token_index: int) -> int | None:
        raise NotImplementedError(
            "fastokens does not track word IDs"
        )

    def word_to_chars(self, word_index: int, sequence_index: int = 0) -> tuple[int, int] | None:
        raise NotImplementedError(
            "fastokens does not track character offsets"
        )

    def word_to_tokens(self, word_index: int, sequence_index: int = 0) -> tuple[int, int] | None:
        raise NotImplementedError(
            "fastokens does not track word IDs"
        )

    # -- truncate / pad (public API matching HF) ------------------------

    def truncate(self, max_length: int, stride: int = 0, direction: str = "right") -> None:
        if len(self.ids) <= max_length:
            return
        if direction == "left":
            start = len(self.ids) - max_length
            self.ids = self.ids[start:]
            self.type_ids = self.type_ids[start:]
            self.attention_mask = self.attention_mask[start:]
            self.special_tokens_mask = self.special_tokens_mask[start:]
            self._offsets = self._offsets[start:]
            self._sequence_ids = self._sequence_ids[start:]
            self._word_ids = self._word_ids[start:]
        else:
            self.ids = self.ids[:max_length]
            self.type_ids = self.type_ids[:max_length]
            self.attention_mask = self.attention_mask[:max_length]
            self.special_tokens_mask = self.special_tokens_mask[:max_length]
            self._offsets = self._offsets[:max_length]
            self._sequence_ids = self._sequence_ids[:max_length]
            self._word_ids = self._word_ids[:max_length]

    def pad(
        self,
        length: int,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
    ) -> None:
        deficit = length - len(self.ids)
        if deficit <= 0:
            return
        if direction == "left":
            self.ids = [pad_id] * deficit + self.ids
            self.type_ids = [pad_type_id] * deficit + self.type_ids
            self.attention_mask = [0] * deficit + self.attention_mask
            self.special_tokens_mask = [0] * deficit + self.special_tokens_mask
            self._offsets = [(0, 0)] * deficit + self._offsets
            self._sequence_ids = [None] * deficit + self._sequence_ids
            self._word_ids = [None] * deficit + self._word_ids
        else:
            self.ids.extend([pad_id] * deficit)
            self.type_ids.extend([pad_type_id] * deficit)
            self.attention_mask.extend([0] * deficit)
            self.special_tokens_mask.extend([0] * deficit)
            self._offsets.extend([(0, 0)] * deficit)
            self._sequence_ids.extend([None] * deficit)
            self._word_ids.extend([None] * deficit)

    # -- merge ----------------------------------------------------------

    @staticmethod
    def merge(encodings: list[_Encoding], growing_offsets: bool = True) -> _Encoding:
        ids: list[int] = []
        type_ids: list[int] = []
        attention_mask: list[int] = []
        special_tokens_mask: list[int] = []
        tokens: list[str] = []
        offsets: list[tuple[int, int]] = []
        seq_ids: list[int | None] = []
        w_ids: list[int | None] = []
        offset_shift = 0
        for enc in encodings:
            ids.extend(enc.ids)
            type_ids.extend(enc.type_ids)
            attention_mask.extend(enc.attention_mask)
            special_tokens_mask.extend(enc.special_tokens_mask)
            tokens.extend(enc._tokens)
            if growing_offsets:
                offsets.extend((s + offset_shift, e + offset_shift) for s, e in enc._offsets)
                if enc._offsets:
                    offset_shift = offsets[-1][1]
            else:
                offsets.extend(enc._offsets)
            seq_ids.extend(enc._sequence_ids)
            w_ids.extend(enc._word_ids)
        merged = _Encoding(ids)
        merged.type_ids = type_ids
        merged.attention_mask = attention_mask
        merged.special_tokens_mask = special_tokens_mask
        merged._tokens = tokens
        merged._offsets = offsets
        merged._sequence_ids = seq_ids
        merged._word_ids = w_ids
        merged.n_sequences = sum(e.n_sequences for e in encodings)
        return merged


# ---------------------------------------------------------------------------
# _TokenizerShim
# ---------------------------------------------------------------------------

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
        self._encode_special_tokens: bool = False

    # -- Pickle / copy --------------------------------------------------

    def __getstate__(self):
        return (self.to_str(), self._truncation, self._padding, self._encode_special_tokens)

    def __setstate__(self, state) -> None:
        if isinstance(state, str):
            # Backwards compat: old pickles stored just the JSON string.
            self.__init__(state)  # type: ignore[misc]
        else:
            json_str, trunc, pad, enc_special = state
            self.__init__(json_str)  # type: ignore[misc]
            self._truncation = trunc
            self._padding = pad
            self._encode_special_tokens = enc_special

    def __deepcopy__(self, memo):
        new = object.__new__(_TokenizerShim)
        memo[id(self)] = new
        new._json = self._json
        new._fast = Tokenizer.from_json_str(self._json)
        new._truncation = copy.deepcopy(self._truncation, memo)
        new._padding = copy.deepcopy(self._padding, memo)
        new._encode_special_tokens = self._encode_special_tokens
        if hasattr(self, "_special_prefix"):
            new._special_prefix = list(self._special_prefix)
            new._special_suffix = list(self._special_suffix)
        return new

    # -- Factory class methods ------------------------------------------

    @classmethod
    def from_str(cls, json_str: str) -> _TokenizerShim:
        return cls(json_str)

    @classmethod
    def from_file(cls, path: str) -> _TokenizerShim:
        return cls(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_pretrained(
        cls,
        identifier: str,
        revision: str = "main",
        token: str | None = None,
    ) -> _TokenizerShim:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            identifier, "tokenizer.json", revision=revision, token=token,
        )
        return cls.from_file(path)

    @classmethod
    def from_buffer(cls, buf: bytes) -> _TokenizerShim:
        return cls(buf.decode("utf-8"))

    # -- Serialization --------------------------------------------------

    def to_str(self, pretty: bool = False) -> str:
        cfg = json.loads(self._json)
        cfg["truncation"] = self._truncation
        cfg["padding"] = self._padding
        if pretty:
            return json.dumps(cfg, indent=2, ensure_ascii=False)
        return json.dumps(cfg, ensure_ascii=False)

    def save(self, path: str, pretty: bool = True) -> None:
        Path(path).write_text(self.to_str(pretty=pretty), encoding="utf-8")

    # -- encode_special_tokens ------------------------------------------

    @property
    def encode_special_tokens(self) -> bool:
        return self._encode_special_tokens

    @encode_special_tokens.setter
    def encode_special_tokens(self, value: bool) -> None:
        self._encode_special_tokens = value

    # -- Truncation / Padding -------------------------------------------

    @property
    def truncation(self) -> dict | None:
        return self._truncation

    @truncation.setter
    def truncation(self, value: dict | None) -> None:
        self._truncation = value

    @property
    def padding(self) -> dict | None:
        return self._padding

    @padding.setter
    def padding(self, value: dict | None) -> None:
        self._padding = value

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

    # -- Encoding -------------------------------------------------------

    def _wrap_encoding(self, ids: list[int]) -> _Encoding:
        enc = _Encoding(ids)
        if self._truncation is not None:
            enc.truncate(
                self._truncation["max_length"],
                stride=self._truncation.get("stride", 0),
                direction=self._truncation.get("direction", "right"),
            )
        if self._padding is not None and self._padding.get("length") is not None:
            enc.pad(
                self._padding["length"],
                direction=self._padding.get("direction", "right"),
                pad_id=self._padding.get("pad_id", 0),
                pad_type_id=self._padding.get("pad_type_id", 0),
                pad_token=self._padding.get("pad_token", "[PAD]"),
            )
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
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> list[_Encoding]:
        if is_pretokenized or any(isinstance(inp, (list, tuple)) for inp in inputs):
            raise NotImplementedError(
                "pair/pre-tokenized batch encoding is not supported by fastokens"
            )
        batch_ids = self._fast.encode_batch(
            inputs, add_special_tokens=add_special_tokens
        )
        return [self._wrap_encoding(ids) for ids in batch_ids]

    # -- Post-processing ------------------------------------------------

    def _get_special_token_affixes(self) -> tuple[list[int], list[int]]:
        """Return (prefix, suffix) special token IDs added by the post-processor.

        Computed once by probing the Rust encoder and cached on the instance.
        """
        if hasattr(self, "_special_prefix"):
            return (self._special_prefix, self._special_suffix)

        with_special = list(self._fast.encode("a", add_special_tokens=True))
        without_special = list(self._fast.encode("a", add_special_tokens=False))

        # Find where the without-special subsequence sits inside with-special.
        inner, outer = without_special, with_special
        for start in range(len(outer) - len(inner) + 1):
            if outer[start : start + len(inner)] == inner:
                self._special_prefix = outer[:start]
                self._special_suffix = outer[start + len(inner) :]
                return (self._special_prefix, self._special_suffix)

        # Fallback: no subsequence match — assume no affixes.
        self._special_prefix: list[int] = []
        self._special_suffix: list[int] = []
        return (self._special_prefix, self._special_suffix)

    def post_process(
        self,
        encoding: _Encoding,
        pair: _Encoding | None = None,
        add_special_tokens: bool = True,
    ) -> _Encoding:
        if pair is not None:
            raise NotImplementedError(
                "pair post-processing is not supported by fastokens"
            )
        if not add_special_tokens:
            return encoding
        prefix, suffix = self._get_special_token_affixes()
        if not prefix and not suffix:
            return encoding
        new_ids = prefix + list(encoding.ids) + suffix
        return self._wrap_encoding(new_ids)

    def _count_special_from_config(self, pp: dict, is_pair: bool) -> int:
        pp_type = pp.get("type", "")
        if pp_type == "TemplateProcessing":
            template = pp.get("pair" if is_pair else "single", [])
            return sum(1 for piece in template if "SpecialToken" in piece)
        if pp_type in ("BertProcessing", "RobertaProcessing"):
            return 3 if is_pair else 2
        if pp_type == "ByteLevel":
            return 0
        if pp_type == "Sequence":
            return sum(
                self._count_special_from_config(child, is_pair)
                for child in pp.get("processors", [])
            )
        return 0

    def num_special_tokens_to_add(self, is_pair: bool) -> int:
        try:
            cfg = json.loads(self._json)
        except (json.JSONDecodeError, TypeError):
            return 0
        pp = cfg.get("post_processor")
        if pp is None:
            return 0
        return self._count_special_from_config(pp, is_pair)

    # -- Decoding -------------------------------------------------------

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

    # -- Vocabulary -----------------------------------------------------

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

    def get_added_tokens_decoder(self) -> dict[int, object]:
        try:
            cfg = json.loads(self._json)
        except (json.JSONDecodeError, TypeError):
            return {}
        result: dict[int, object] = {}
        for entry in cfg.get("added_tokens", []):
            tid = entry.get("id")
            if tid is not None:
                result[tid] = _AddedTokenInfo(
                    content=entry.get("content", ""),
                    single_word=entry.get("single_word", False),
                    lstrip=entry.get("lstrip", False),
                    rstrip=entry.get("rstrip", False),
                    normalized=entry.get("normalized", True),
                    special=entry.get("special", False),
                )
        return result

    # -- Token management (no-ops) --------------------------------------

    def add_tokens(self, tokens) -> int:
        return 0

    def add_special_tokens(self, special_tokens) -> int:
        return 0

    # -- Component accessors --------------------------------------------

    @property
    def model(self):
        return _ModelStub(self)

    @model.setter
    def model(self, value) -> None:
        pass  # ignored — model is fixed at construction

    @property
    def normalizer(self):
        return None

    @normalizer.setter
    def normalizer(self, value) -> None:
        pass

    @property
    def pre_tokenizer(self):
        return None

    @pre_tokenizer.setter
    def pre_tokenizer(self, value) -> None:
        pass

    @property
    def post_processor(self):
        return None

    @post_processor.setter
    def post_processor(self, value) -> None:
        pass

    @property
    def decoder(self):
        return None

    @decoder.setter
    def decoder(self, value) -> None:
        pass


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------

class _AddedTokenInfo:
    """Minimal stand-in for ``tokenizers.AddedToken`` returned by
    :meth:`_TokenizerShim.get_added_tokens_decoder`."""

    __slots__ = ("content", "single_word", "lstrip", "rstrip", "normalized", "special")

    def __init__(
        self,
        content: str = "",
        single_word: bool = False,
        lstrip: bool = False,
        rstrip: bool = False,
        normalized: bool = True,
        special: bool = False,
    ) -> None:
        self.content = content
        self.single_word = single_word
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.normalized = normalized
        self.special = special

    def __repr__(self) -> str:
        return (
            f"AddedToken({self.content!r}, "
            f"rstrip={self.rstrip}, lstrip={self.lstrip}, "
            f"single_word={self.single_word}, "
            f"normalized={self.normalized}, "
            f"special={self.special})"
        )


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
