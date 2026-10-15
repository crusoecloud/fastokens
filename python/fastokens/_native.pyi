from typing import Any, Optional, Protocol

class AddedTokenLike(Protocol):
    """Any object with the attributes of a ``tokenizers.AddedToken``."""

    content: str
    single_word: bool
    lstrip: bool
    rstrip: bool
    normalized: bool
    special: bool

class Encoding:
    """Rust-backed encoding returned by tokenizer encoding methods."""

    ids: list[int]
    attention_mask: list[int]
    type_ids: list[int]
    special_tokens_mask: list[int]
    n_sequences: int
    overflowing: list["Encoding"]

    def __new__(
        cls,
        ids: list[int],
        attention_mask: Optional[list[int]] = None,
    ) -> "Encoding": ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

    # Properties that raise NotImplementedError
    @property
    def tokens(self) -> list[str]: ...
    @tokens.setter
    def tokens(self, value: list[str]) -> None: ...

    @property
    def offsets(self) -> list[tuple[int, int]]: ...
    @offsets.setter
    def offsets(self, value: list[tuple[int, int]]) -> None: ...

    @property
    def sequence_ids(self) -> list[Optional[int]]: ...
    @sequence_ids.setter
    def sequence_ids(self, value: list[Optional[int]]) -> None: ...

    @property
    def word_ids(self) -> list[Optional[int]]: ...
    @word_ids.setter
    def word_ids(self, value: list[Optional[int]]) -> None: ...

    @property
    def words(self) -> list[Optional[int]]: ...
    @words.setter
    def words(self, value: list[Optional[int]]) -> None: ...

    def set_sequence_id(self, sequence_id: int) -> None: ...

    def char_to_token(self, char_pos: int, sequence_index: int = 0) -> Optional[int]: ...
    def char_to_word(self, char_pos: int, sequence_index: int = 0) -> Optional[int]: ...
    def token_to_chars(self, token_index: int) -> Optional[tuple[int, int]]: ...
    def token_to_sequence(self, token_index: int) -> Optional[int]: ...
    def token_to_word(self, token_index: int) -> Optional[int]: ...
    def word_to_chars(self, word_index: int, sequence_index: int = 0) -> Optional[tuple[int, int]]: ...
    def word_to_tokens(self, word_index: int, sequence_index: int = 0) -> Optional[tuple[int, int]]: ...

    def truncate(self, max_length: int, stride: int = 0, direction: str = "right") -> None: ...
    def pad(
        self,
        length: int,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
    ) -> None: ...

    @staticmethod
    def merge(encodings: list["Encoding"], growing_offsets: bool = True) -> "Encoding": ...


class Tokenizer:
    """An LLM tokenizer backed by ``tokenizer.json``."""

    def __new__(
        cls,
        model: str,
        pcre2_match_limit: Optional[int] = None,
        pcre2_depth_limit: Optional[int] = None,
        pcre2_heap_limit: Optional[int] = None,
        pcre2_max_jit_stack_size: Optional[int] = None,
    ) -> "Tokenizer": ...

    @staticmethod
    def from_file(
        path: str,
        pcre2_match_limit: Optional[int] = None,
        pcre2_depth_limit: Optional[int] = None,
        pcre2_heap_limit: Optional[int] = None,
        pcre2_max_jit_stack_size: Optional[int] = None,
    ) -> "Tokenizer": ...

    @staticmethod
    def from_json_str(
        json: str,
        pcre2_match_limit: Optional[int] = None,
        pcre2_depth_limit: Optional[int] = None,
        pcre2_heap_limit: Optional[int] = None,
        pcre2_max_jit_stack_size: Optional[int] = None,
    ) -> "Tokenizer": ...

    @staticmethod
    def from_model(
        model: str,
        pcre2_match_limit: Optional[int] = None,
        pcre2_depth_limit: Optional[int] = None,
        pcre2_heap_limit: Optional[int] = None,
        pcre2_max_jit_stack_size: Optional[int] = None,
    ) -> "Tokenizer": ...

    @staticmethod
    def from_tiktoken(
        path: str,
        pattern: Optional[str] = None,
        special_tokens: Optional[dict[str, int]] = None,
        encoding: Optional[str] = None,
    ) -> "Tokenizer": ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def truncation(self) -> Optional[dict]: ...

    @property
    def padding(self) -> Optional[dict]: ...

    def enable_truncation(
        self,
        max_length: int,
        stride: int = 0,
        strategy: str = "longest_first",
        direction: str = "right",
    ) -> None: ...

    def no_truncation(self) -> None: ...

    def enable_padding(
        self,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
        length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None: ...

    def no_padding(self) -> None: ...

    def truncation_padding_json(self) -> str: ...

    def post_process(
        self,
        encoding: Encoding,
        pair: Optional[Encoding] = None,
        add_special_tokens: bool = True,
    ) -> Encoding: ...

    def num_special_tokens_to_add(self, is_pair: bool) -> int: ...

    def encode(
        self,
        input: str,
        add_special_tokens: bool = False,
        split_special_tokens: bool = False,
    ) -> Encoding:
        """Encode ``input``.

        With ``split_special_tokens``, special added tokens are encoded as
        ordinary text rather than as control-token ids; the rest of the added
        vocabulary still matches."""

    def encode_ordinary(self, input: str) -> Encoding: ...
    def encode_segments(self, segments: list[tuple[str, bool]]) -> Encoding:
        """Encode a pre-segmented input, concatenating each segment's token ids.

        ``segments`` is a list of ``(text, allow_special)`` pairs. Each segment is
        tokenized independently; special/added tokens are recognized only in
        segments with ``allow_special=True``. Mirrors legacy tiktoken / Dynamo
        segmented encoding; no post-processor special tokens are inserted."""

    def encode_batch(
        self,
        inputs: list[str],
        add_special_tokens: bool = False,
        split_special_tokens: bool = False,
    ) -> list[Encoding]: ...
    def encode_batch_flat(
        self,
        inputs: list[str],
        add_special_tokens: bool = False,
        split_special_tokens: bool = False,
    ) -> tuple[bytes, bytes]:
        """Encode a batch into a single flat token buffer for high-throughput
        bulk tokenization. Returns ``(ids, offsets)``: ``ids`` is every input's
        token ids concatenated as little-endian ``uint32`` bytes, and
        ``offsets`` is ``len(inputs) + 1`` little-endian ``uint64`` values, so
        input ``i``'s tokens are ``ids[offsets[i]:offsets[i+1]]``. Decode with
        ``numpy.frombuffer(ids, numpy.uint32)`` and
        ``numpy.frombuffer(offsets, numpy.uint64)``. No per-token Python objects
        are created. Truncation applies per input; padding does not (the result
        is ragged, addressed by ``offsets``)."""

    def decode_tokens(self, tokens: list[str]) -> str: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str: ...

    def decode_batch(
        self, sentences: list[list[int]], skip_special_tokens: bool = False
    ) -> list[str]: ...

    def token_to_id(self, token: str) -> Optional[int]: ...
    def id_to_token(self, id: int) -> Optional[str]: ...

    def enable_segment_cache(self, max_bytes: int) -> None:
        """Enable the segment cache with a budget of ``max_bytes`` for segment
        text and ids, replacing any current one; ``0`` turns it off. Also
        enabled by the ``FASTOKENS_SEGMENT_CACHE=<MiB>`` environment variable
        at construction. It is for chat and agent traffic, whose requests
        repeat the text segments between added tokens of earlier requests;
        results are identical to encoding without it. A model with BPE dropout
        is never cached."""

    def segment_cache_stats(self) -> Optional[dict[str, int]]:
        """The segment cache's counters and size (``hits``, ``misses``,
        ``hit_bytes``, ``miss_bytes``, ``entries``, ``bytes``, ``max_bytes``),
        or ``None`` when it is off."""

    def added_tokens(self) -> list[dict[str, Any]]:
        """The added-vocabulary entries, as ``tokenizer.json`` serializes them,
        including any added since construction."""

    def add_tokens(self, tokens: list[AddedTokenLike]) -> list[tuple[str, int]]:
        """Extend the vocabulary, returning the ``(content, id)`` of every entry
        created or changed.

        Ids are assigned as HuggingFace ``tokenizers`` assigns them: a content
        already in the vocabulary keeps its id and only its flags can change,
        anything else is appended above the vocabulary."""
