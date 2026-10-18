"""Tests for accelerating tiktoken-based slow tokenizers (Kimi's ``TikTokenTokenizer``).

The splitter is checked against a copy of the per-character loop it replaces.
The id parity tests build a Kimi-style tokenizer over a tiny byte-level tiktoken
model, so they need ``tiktoken`` but no network.
"""

import base64
import random
from types import SimpleNamespace

import pytest

from fastokens._tiktoken import (
    MAX_NO_WHITESPACES_CHARS,
    TIKTOKEN_MAX_ENCODE_CHARS,
    accelerate_tiktoken_tokenizer,
    split_like_tiktoken_text_piece,
)


def _reference_split(s, max_consecutive_slice_len):
    """Kimi's ``TikTokenTokenizer._split_whitespaces_or_nonwhitespaces``."""
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0
    for i in range(len(s)):
        is_now_space = s[i].isspace()
        if current_slice_is_space ^ is_now_space:
            current_slice_len = 1
            current_slice_is_space = is_now_space
        else:
            current_slice_len += 1
            if current_slice_len > max_consecutive_slice_len:
                yield s[slice_start:i]
                slice_start = i
                current_slice_len = 1
    yield s[slice_start:]


def _reference_pieces(text):
    for i in range(0, len(text), TIKTOKEN_MAX_ENCODE_CHARS):
        yield from _reference_split(text[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS)


def _random_runs(rng, n):
    """Text made of runs of letters, spaces, newlines, unicode whitespace and CJK,
    some longer than the 25k split limit."""
    alphabet = ["a", "b", " ", "\n", "\t", "　", " ", "​", "日", "é"]
    out = []
    while sum(map(len, out)) < n:
        ch = rng.choice(alphabet)
        length = rng.choice([1, 3, 40, 2_000, 24_999, 25_000, 25_001, 31_000, 60_001])
        out.append(ch * length)
    return "".join(out)[:n]


@pytest.mark.parametrize("seed", range(6))
def test_split_matches_reference(seed):
    rng = random.Random(seed)
    text = _random_runs(rng, rng.choice([1_000, 120_000, 450_000, 850_000]))
    assert list(split_like_tiktoken_text_piece(text)) == list(_reference_pieces(text))


@pytest.mark.parametrize("text", ["", "a", " " * 25_001, "x" * 25_000 + " " * 25_001 + "y" * 50_002])
def test_split_matches_reference_edges(text):
    assert list(split_like_tiktoken_text_piece(text)) == list(_reference_pieces(text))


# ── id parity against a Kimi-style slow tokenizer ──────────────────────

SPECIALS = {"<|im_end|>": 300, "<|open|>": 301}


def _byte_level_model(tmp_path):
    """All 256 bytes plus a few merges, in tiktoken's ``base64 rank`` format."""
    ranks = {bytes([b]): b for b in range(256)}
    for i, merge in enumerate([b"ab", b"abc", b"  ", b"\n\n", b"he", b"hel", b"lo"]):
        ranks[merge] = 256 + i
    path = tmp_path / "tiny.model"
    path.write_text(
        "\n".join(f"{base64.b64encode(k).decode('ascii')} {v}" for k, v in ranks.items()) + "\n",
        encoding="utf-8",
    )
    return str(path), ranks


def _kimi_style_tokenizer(tiktoken, tmp_path):
    path, ranks = _byte_level_model(tmp_path)
    pat = r"\S+|\s+"
    enc = tiktoken.Encoding("tiny", pat_str=pat, mergeable_ranks=ranks, special_tokens=SPECIALS)

    class KimiStyleTokenizer:
        pat_str = pat

        def __init__(self):
            self.model = enc
            self.vocab_file = path

        # Kimi's TikTokenTokenizer._encode_text_piece / _encode_chat_segments.
        def _encode_text_piece(self, text, allow_special_tokens=True):
            ids = []
            for substr in _reference_pieces(text):
                if allow_special_tokens:
                    ids.extend(self.model.encode(substr, allowed_special="all"))
                else:
                    ids.extend(self.model.encode(substr, disallowed_special=()))
            return ids

        def _encode_chat_segments(self, segments):
            ids = []
            for segment in segments:
                ids.extend(self._encode_text_piece(segment.text, allow_special_tokens=segment.allow_special))
            return ids

    return KimiStyleTokenizer


TEXTS = [
    "hello abc  ab\n\nlo",
    "special <|im_end|> and <|open|> in text",
    "abc " * 30_000,
    "x" * 60_001 + " " * 30_000 + "y" * 25_000 + "\n" * 25_001 + "z" * 3_000,
    "　  é日 abc " * 70_000,
]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("allow_special", [True, False])
def test_accelerated_ids_match_reference(tmp_path, text, allow_special):
    tiktoken = pytest.importorskip("tiktoken")
    cls = _kimi_style_tokenizer(tiktoken, tmp_path)
    ref, fast = cls(), cls()
    assert accelerate_tiktoken_tokenizer(fast)
    assert fast._encode_text_piece(text, allow_special) == ref._encode_text_piece(text, allow_special)


def test_accelerated_chat_segments_match_reference(tmp_path):
    tiktoken = pytest.importorskip("tiktoken")
    cls = _kimi_style_tokenizer(tiktoken, tmp_path)
    ref, fast = cls(), cls()
    assert accelerate_tiktoken_tokenizer(fast)
    segments = [
        SimpleNamespace(text="<|open|>", allow_special=True),
        SimpleNamespace(text="user text with <|im_end|> inside" + " pad" * 20_000, allow_special=False),
        SimpleNamespace(text="<|im_end|>", allow_special=True),
    ]
    assert fast._encode_chat_segments(segments) == ref._encode_chat_segments(segments)


def test_other_tokenizers_are_left_alone():
    plain = SimpleNamespace(model=object(), vocab_file="x", _encode_text_piece=lambda *a: [])
    assert not accelerate_tiktoken_tokenizer(plain)


def test_patch_wraps_and_restores_slow_from_pretrained():
    pytest.importorskip("transformers")
    import fastokens

    try:
        from transformers.tokenization_python import PythonBackend as slow_cls
    except ImportError:
        from transformers.tokenization_utils import PreTrainedTokenizer as slow_cls

    try:
        fastokens.patch_transformers()
        assert "from_pretrained" in slow_cls.__dict__
    finally:
        fastokens.unpatch_transformers()
    assert "from_pretrained" not in slow_cls.__dict__


def test_split_is_linear_in_runs_under_the_limit():
    # Unanchored, the long-run regex retried at every character of a shorter run:
    # ~5 s here. Kimi's own per-character loop takes ~12 ms.
    import time

    text = ("y" * (MAX_NO_WHITESPACES_CHARS - 1) + " ") * (TIKTOKEN_MAX_ENCODE_CHARS // MAX_NO_WHITESPACES_CHARS)
    started = time.perf_counter()
    pieces = list(split_like_tiktoken_text_piece(text))
    elapsed = time.perf_counter() - started
    assert pieces == list(_reference_pieces(text))
    assert elapsed < 0.5, f"splitting took {elapsed:.2f} s"


class _PicklableKimiStyleTokenizer:
    """Module-level so instances pickle; same interface as ``_kimi_style_tokenizer``."""

    pat_str = r"\S+|\s+"

    def __init__(self, tiktoken, path, ranks):
        self.model = tiktoken.Encoding("tiny", pat_str=self.pat_str, mergeable_ranks=ranks, special_tokens=SPECIALS)
        self.vocab_file = path

    def _encode_text_piece(self, text, allow_special_tokens=True):
        ids = []
        for substr in _reference_pieces(text):
            if allow_special_tokens:
                ids.extend(self.model.encode(substr, allowed_special="all"))
            else:
                ids.extend(self.model.encode(substr, disallowed_special=()))
        return ids


@pytest.mark.parametrize("copy_fn", ["pickle", "deepcopy"])
def test_accelerated_tokenizer_survives_pickle_and_deepcopy(tmp_path, copy_fn):
    import copy
    import pickle

    tiktoken = pytest.importorskip("tiktoken")
    path, ranks = _byte_level_model(tmp_path)
    ref = _PicklableKimiStyleTokenizer(tiktoken, path, ranks)
    fast = _PicklableKimiStyleTokenizer(tiktoken, path, ranks)
    assert accelerate_tiktoken_tokenizer(fast)

    clone = pickle.loads(pickle.dumps(fast)) if copy_fn == "pickle" else copy.deepcopy(fast)

    text = TEXTS[3]
    assert clone._fastokens_tiktoken is not None
    assert clone._encode_text_piece.__self__ is clone._fastokens_tiktoken
    assert clone._encode_text_piece(text) == ref._encode_text_piece(text)
    assert fast._encode_text_piece(text) == ref._encode_text_piece(text)
