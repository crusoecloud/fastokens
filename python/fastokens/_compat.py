"""
Compatibility shim for monkey-patching the ``transformers`` library.

Provides :class:`_TokenizerShim` (a complete replacement for
``tokenizers.Tokenizer``).  All encoding, decoding, vocabulary operations,
truncation, and padding are handled by the Rust backend.  The returned
:class:`Encoding` objects are also Rust-backed ``#[pyclass]`` instances --
no Python-side wrapping is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastokens._native import Encoding, Tokenizer


# Backwards-compatibility alias used by tests and any code that imports
# _Encoding directly from this module.
_Encoding = Encoding

def _pcre2_options(
    pcre2_match_limit: int | None = None,
    pcre2_depth_limit: int | None = None,
    pcre2_heap_limit: int | None = None,
    pcre2_max_jit_stack_size: int | None = None,
) -> dict[str, int | None]:
    return {
        "pcre2_match_limit": pcre2_match_limit,
        "pcre2_depth_limit": pcre2_depth_limit,
        "pcre2_heap_limit": pcre2_heap_limit,
        "pcre2_max_jit_stack_size": pcre2_max_jit_stack_size,
    }


# ---------------------------------------------------------------------------
# _TokenizerShim
# ---------------------------------------------------------------------------

class _TokenizerShim:
    """
    Complete replacement for ``tokenizers.Tokenizer``.

    All encoding, decoding, vocabulary, truncation, and padding operations
    are delegated to the Rust :class:`Tokenizer`.  No reference to the
    original ``tokenizers.Tokenizer`` is kept.
    """

    def __init__(
        self,
        src,
        pcre2_match_limit: int | None = None,
        pcre2_depth_limit: int | None = None,
        pcre2_heap_limit: int | None = None,
        pcre2_max_jit_stack_size: int | None = None,
    ) -> None:
        self._pcre2_options = _pcre2_options(
            pcre2_match_limit,
            pcre2_depth_limit,
            pcre2_heap_limit,
            pcre2_max_jit_stack_size,
        )
        if isinstance(src, str):
            self._json = src
            self._fast = Tokenizer.from_json_str(src, **self._pcre2_options)
        elif isinstance(src, _TokenizerShim):
            self._json = src._json
            if all(value is None for value in self._pcre2_options.values()):
                self._pcre2_options = dict(src._pcre2_options)
            self._fast = Tokenizer.from_json_str(src._json, **self._pcre2_options)
            # Like a tokenizers.Tokenizer's to_str() below, follow the live
            # settings rather than those `src._json` declares.
            self.truncation = src.truncation
            self.padding = src.padding
        elif hasattr(src, "to_str"):
            # Accept a real tokenizers.Tokenizer (e.g. from convert_slow_tokenizer).
            self._json = src.to_str()
            self._fast = Tokenizer.from_json_str(self._json, **self._pcre2_options)
        else:
            raise TypeError(
                f"expected JSON string, _TokenizerShim, or tokenizers.Tokenizer; "
                f"got {type(src).__name__}"
            )
        self._encode_special_tokens: bool = False
        # Whether add_tokens has grown the vocabulary past what `_json` holds.
        self._vocab_extended: bool = False

    # -- Pickle / copy --------------------------------------------------

    def __getstate__(self):
        return (
            self.to_str(),
            self._fast.truncation,
            self._fast.padding,
            self._encode_special_tokens,
            self._pcre2_options,
        )

    def __setstate__(self, state) -> None:
        if isinstance(state, str):
            # Backwards compat: old pickles stored just the JSON string.
            self.__init__(state)  # type: ignore[misc]
        else:
            if len(state) == 4:
                json_str, trunc, pad, enc_special = state
                pcre2_options = {}
            else:
                json_str, trunc, pad, enc_special, pcre2_options = state
            self.__init__(json_str, **pcre2_options)  # type: ignore[misc]
            if trunc is not None:
                self._fast.enable_truncation(**trunc)
            if pad is not None:
                self._fast.enable_padding(**{k: v for k, v in pad.items() if v is not None})
            self._encode_special_tokens = enc_special

    def __deepcopy__(self, memo):
        new = object.__new__(_TokenizerShim)
        memo[id(self)] = new
        new._json = self._json
        new._pcre2_options = dict(self._pcre2_options)
        # Once the vocabulary has been extended the source JSON is no longer
        # the whole tokenizer, so the copy has to be built from the live state.
        source = self.to_str() if self._vocab_extended else self._json
        new._vocab_extended = self._vocab_extended
        new._fast = Tokenizer.from_json_str(source, **new._pcre2_options)
        # `source` may still declare settings that have since been disabled.
        new.truncation = self.truncation
        new.padding = self.padding
        new._encode_special_tokens = self._encode_special_tokens
        if hasattr(self, "_special_prefix"):
            new._special_prefix = list(self._special_prefix)
            new._special_suffix = list(self._special_suffix)
        return new

    # -- Factory class methods ------------------------------------------

    @classmethod
    def from_str(cls, json_str: str, **kwargs) -> _TokenizerShim:
        return cls(json_str, **kwargs)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> _TokenizerShim:
        return cls(Path(path).read_text(encoding="utf-8"), **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        identifier: str,
        revision: str = "main",
        token: str | None = None,
        **kwargs,
    ) -> _TokenizerShim:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            identifier, "tokenizer.json", revision=revision, token=token,
        )
        return cls.from_file(path, **kwargs)

    @classmethod
    def from_buffer(cls, buf: bytes, **kwargs) -> _TokenizerShim:
        return cls(buf.decode("utf-8"), **kwargs)

    # -- Serialization --------------------------------------------------

    def to_str(self, pretty: bool = False) -> str:
        cfg = json.loads(self._json)
        cfg.update(json.loads(self._fast.truncation_padding_json()))
        cfg["added_tokens"] = self._added_tokens(cfg.get("added_tokens") or [])
        if pretty:
            return json.dumps(cfg, indent=2, ensure_ascii=False)
        return json.dumps(cfg, ensure_ascii=False)

    def _added_tokens(self, source: list[dict] | None = None) -> list[dict]:
        """The effective ``added_tokens`` array: source entries plus any added
        after construction.

        Source-JSON entries are kept verbatim so a load/save round-trip does not
        rewrite fields the backend defaults differently — notably ``normalized``,
        which the backend stores as ``False`` when a source entry omits it but
        HuggingFace treats as ``True``. The live ``special`` flag is overlaid so
        promotions via :meth:`add_special_tokens` are reflected, and tokens added
        after construction are appended in id order.
        """
        if source is None:
            source = json.loads(self._json).get("added_tokens") or []
        live_by_id = {entry["id"]: entry for entry in self._fast.added_tokens()}
        result: list[dict] = []
        seen: set[int] = set()
        for entry in source:
            tid = entry.get("id")
            seen.add(tid)
            live = live_by_id.get(tid)
            if live is None:
                result.append(entry)
            else:
                # Only `special` can change for a token already declared in the
                # source JSON (a promotion); every other field is immutable.
                result.append({**entry, "special": live["special"]})
        for tid, live in live_by_id.items():
            if tid not in seen:
                result.append(live)
        return result

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
        return self._fast.truncation

    @truncation.setter
    def truncation(self, value: dict | None) -> None:
        if value is None:
            self._fast.no_truncation()
        else:
            self._fast.enable_truncation(**value)

    @property
    def padding(self) -> dict | None:
        return self._fast.padding

    @padding.setter
    def padding(self, value: dict | None) -> None:
        if value is None:
            self._fast.no_padding()
        else:
            self._fast.enable_padding(**{k: v for k, v in value.items() if v is not None})

    def enable_segment_cache(self, max_bytes: int) -> None:
        self._fast.enable_segment_cache(max_bytes)

    def segment_cache_stats(self) -> dict[str, int] | None:
        return self._fast.segment_cache_stats()

    def enable_truncation(
        self,
        max_length: int,
        stride: int = 0,
        strategy: str = "longest_first",
        direction: str = "right",
    ) -> None:
        self._fast.enable_truncation(
            max_length, stride=stride, strategy=strategy, direction=direction
        )

    def no_truncation(self) -> None:
        self._fast.no_truncation()

    def enable_padding(
        self,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
        length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self._fast.enable_padding(
            direction=direction,
            pad_id=pad_id,
            pad_type_id=pad_type_id,
            pad_token=pad_token,
            **({"length": length} if length is not None else {}),
            **({"pad_to_multiple_of": pad_to_multiple_of} if pad_to_multiple_of is not None else {}),
        )

    def no_padding(self) -> None:
        self._fast.no_padding()

    # -- Encoding -------------------------------------------------------

    def encode(
        self,
        sequence: str,
        pair: str | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> Encoding:
        if pair is not None:
            raise NotImplementedError("pair encoding is not supported by fastokens")
        if is_pretokenized:
            raise NotImplementedError("pre-tokenized input is not supported by fastokens")
        return self._fast.encode(
            sequence,
            add_special_tokens=add_special_tokens,
            split_special_tokens=self._encode_special_tokens,
        )

    def encode_batch(
        self,
        inputs: list,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> list[Encoding]:
        if is_pretokenized or any(isinstance(inp, (list, tuple)) for inp in inputs):
            raise NotImplementedError(
                "pair/pre-tokenized batch encoding is not supported by fastokens"
            )
        return self._fast.encode_batch(
            inputs,
            add_special_tokens=add_special_tokens,
            split_special_tokens=self._encode_special_tokens,
        )

    def encode_batch_fast(
        self,
        inputs: list[str],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> list[Encoding]:
        if is_pretokenized or any(isinstance(inp, (list, tuple)) for inp in inputs):
            raise NotImplementedError(
                "pair/pre-tokenized batch encoding is not supported by fastokens"
            )
        return self._fast.encode_batch(
            inputs,
            add_special_tokens=add_special_tokens,
            split_special_tokens=self._encode_special_tokens,
        )

    # -- Post-processing ------------------------------------------------

    def post_process(
        self,
        encoding: Encoding,
        pair: Encoding | None = None,
        add_special_tokens: bool = True,
    ) -> Encoding:
        return self._fast.post_process(encoding, pair, add_special_tokens)

    def num_special_tokens_to_add(self, is_pair: bool) -> int:
        return self._fast.num_special_tokens_to_add(is_pair)

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
        added = self._fast.added_tokens()
        if not with_added_tokens:
            # Exclude the added vocabulary so `transformers.get_added_vocab()`,
            # which diffs `get_vocab(False)` against `get_vocab(True)`, actually
            # sees the added tokens instead of an empty difference.
            added_ids = {entry["id"] for entry in added}
            return {
                tok: i
                for i in range(self._fast.vocab_size)
                if i not in added_ids
                and (tok := self._fast.id_to_token(i)) is not None
            }
        vocab = {}
        for i in range(self._fast.vocab_size):
            tok = self._fast.id_to_token(i)
            if tok is not None:
                vocab[tok] = i
        # An added token can sit above the vocabulary size when the declared ids
        # leave a gap, so the id scan alone can miss one.
        for entry in added:
            vocab[entry["content"]] = entry["id"]
        return vocab

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        # fastokens reports one number for both: the count of distinct token
        # strings, which is `get_vocab_size(True)` upstream. `False` should be
        # the base vocabulary alone, but the backend does not track the split,
        # and over-reporting is the safer direction for callers that size
        # embedding tables or logit masks from it.
        return self._fast.vocab_size

    def get_added_tokens_decoder(self) -> dict[int, object]:
        result: dict[int, object] = {}
        for entry in self._added_tokens():
            tid = entry.get("id")
            if tid is None:
                continue
            result[tid] = _AddedTokenInfo(
                content=entry.get("content", ""),
                single_word=entry.get("single_word", False),
                lstrip=entry.get("lstrip", False),
                rstrip=entry.get("rstrip", False),
                # HuggingFace treats a missing `normalized` as True; the backend
                # would report False, so default here from the source entry.
                normalized=entry.get("normalized", True),
                special=entry.get("special", False),
            )
        return result

    # -- Token management -----------------------------------------------

    def add_tokens(self, tokens) -> int:
        """Extend the vocabulary, returning the number of tokens added.

        Accepts plain strings and ``AddedToken``-like objects. Ids are assigned
        as HuggingFace ``tokenizers`` assigns them, so a loader that pads a
        tokenizer up to a checkpoint's embedding count lands the placeholders on
        the padded rows.
        """
        return self._add_tokens(tokens, special=False)

    def add_special_tokens(self, special_tokens) -> int:
        """:meth:`add_tokens` for special tokens.

        A token already in the vocabulary is not duplicated; it is promoted to
        special, which is what makes ``encode_special_tokens`` skip it.
        """
        return self._add_tokens(special_tokens, special=True)

    def _add_tokens(self, tokens, special: bool) -> int:
        specs = [self._coerce_token(token, special) for token in tokens]
        added = self._fast.add_tokens(specs)
        if added:
            self._vocab_extended = True
        return len(added)

    @staticmethod
    def _coerce_token(token, special: bool):
        if not hasattr(token, "content"):
            # `AddedToken.from(content, special)` leaves special tokens
            # unnormalized; mirror that for bare strings.
            return _AddedTokenInfo(content=token, normalized=not special, special=special)
        if special and not getattr(token, "special", False):
            # `Tokenizer.add_special_tokens` always marks its arguments special
            # in HuggingFace; an `AddedToken(..., special=False)` passed here
            # must still be registered special, or `encode_special_tokens` would
            # not skip it and a control-token string in untrusted input would
            # produce a real control-token id. Copy the object's flags rather
            # than mutating the caller's object.
            return _AddedTokenInfo(
                content=token.content,
                single_word=getattr(token, "single_word", False),
                lstrip=getattr(token, "lstrip", False),
                rstrip=getattr(token, "rstrip", False),
                normalized=getattr(token, "normalized", True),
                special=True,
            )
        return token

    # -- Component accessors --------------------------------------------

    @property
    def model(self):
        return _ModelStub(self)

    @model.setter
    def model(self, value) -> None:
        pass

    @property
    def normalizer(self):
        return None

    @normalizer.setter
    def normalizer(self, value) -> None:
        pass

    @property
    def decoder(self):
        return _DecoderShim(self._fast)

    @decoder.setter
    def decoder(self, value) -> None:
        pass

    @property
    def pre_tokenizer(self):
        return None

    @pre_tokenizer.setter
    def pre_tokenizer(self, value) -> None:
        pass

    @property
    def post_processor(self):
        return self._fast.post_processor

    @post_processor.setter
    def post_processor(self, value) -> None:
        self._fast.post_processor = value




# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------

class _DecoderShim:
    """
    Stand-in for ``tokenizers.decoders.Decoder``.

    ``PreTrainedTokenizerFast.convert_tokens_to_string`` calls
    ``self.backend_tokenizer.decoder.decode(tokens)`` when the decoder is not
    None.  Without this shim it falls back to ``" ".join(tokens)``, which
    leaves ByteLevel-encoded tokens (e.g. "Ġhello") undecoded in the output.
    """

    __slots__ = ("_fast",)

    def __init__(self, fast) -> None:
        self._fast = fast

    def decode(self, tokens: list[str]) -> str:
        return self._fast.decode_tokens(tokens)


class _AddedTokenInfo:
    """Minimal stand-in for ``tokenizers.AddedToken``."""

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
