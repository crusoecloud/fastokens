"""The segment cache through the Python binding: the switches, the counters,
every entry point against HuggingFace `tokenizers`, threads sharing one
cache, the budget, and `add_tokens`. The pipeline-level coverage is in the
Rust tests; this is a synthetic byte-level BPE tokenizer with an NFC
normalizer and chat-style control tokens, so it needs no network.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from fastokens._native import Tokenizer

hf_tokenizers = pytest.importorskip("tokenizers")

MIB = 1 << 20


def tokenizer_json() -> str:
    alphabet = sorted(hf_tokenizers.pre_tokenizers.ByteLevel.alphabet())
    vocab = {c: i for i, c in enumerate(alphabet)}
    merges = ["Ġ t", "h e", "Ġt he", "i n", "in g", "Ċ Ċ", "u s", "e r", "us er"]
    for m in merges:
        vocab[m.replace(" ", "")] = len(vocab)
    n = len(vocab)

    def added(id, content, strip, special):
        return {"id": id, "content": content, "single_word": False, "lstrip": strip,
                "rstrip": strip, "normalized": False, "special": special}

    return json.dumps({
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [added(n, "<|im_start|>", False, True), added(n + 1, "<|im_end|>", False, True),
                         added(n + 2, "<|pad|>", True, False)],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": merges},
    })


SPECIALS = ["<|im_start|>", "<|im_end|>", "<|pad|>"]
PIECES = ["the", " user", " said", "Hello", "WORLD", "中文", " 123", "\n", "\n\n  \n", " /", "{", "}\n",
          "();", "\t", "é", "é", "😀", " ", "'s", "　", "   "]


def chat_requests(seed: int, turns: int = 30) -> list[str]:
    """Every request is the whole history so far; messages are wrapped in
    control tokens, some repeat, and the history is now and then rewritten
    (a message dropped, or everything replaced by a summary)."""
    rng = random.Random(seed)
    history, out = [], []
    for _ in range(turns):
        if history and rng.random() < 0.2:
            body = rng.choice(history)
        else:
            body = "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 400)))
        pad = rng.choice(["", " ", "\n", "  "])
        history.append(f"{rng.choice(SPECIALS)}{pad}{body}{pad}{rng.choice(SPECIALS)}")
        if len(history) > 2 and rng.random() < 0.1:
            del history[rng.randrange(1, len(history))]
        if len(history) > 4 and rng.random() < 0.05:
            history = [history[0], f"{SPECIALS[0]}summary{SPECIALS[1]}"]
        out.append("".join(history))
    return out


@pytest.fixture
def hf():
    return hf_tokenizers.Tokenizer.from_str(tokenizer_json())


@pytest.fixture
def plain():
    return Tokenizer.from_json_str(tokenizer_json())


def cached(max_bytes=64 * MIB):
    tok = Tokenizer.from_json_str(tokenizer_json())
    tok.enable_segment_cache(max_bytes)
    return tok


def test_off_by_default_and_enabled_by_env_var(monkeypatch, plain):
    assert plain.segment_cache_stats() is None
    monkeypatch.setenv("FASTOKENS_SEGMENT_CACHE", "3")
    stats = Tokenizer.from_json_str(tokenizer_json()).segment_cache_stats()
    assert stats["max_bytes"] == 3 * MIB
    assert stats["hits"] == stats["misses"] == stats["entries"] == stats["bytes"] == 0
    monkeypatch.setenv("FASTOKENS_SEGMENT_CACHE", "0")
    assert Tokenizer.from_json_str(tokenizer_json()).segment_cache_stats() is None
    tok = cached()
    tok.enable_segment_cache(0)
    assert tok.segment_cache_stats() is None


def test_matches_hf_and_uncached(hf, plain):
    tok = cached()
    requests = chat_requests(1)
    for i, request in enumerate(requests):
        want = hf.encode(request, add_special_tokens=False).ids
        assert plain.encode(request).ids == want, i
        assert tok.encode(request).ids == want, i
        assert tok.encode(request, split_special_tokens=True).ids == plain.encode(request, split_special_tokens=True).ids, i
        assert tok.encode_ordinary(request).ids == plain.encode_ordinary(request).ids, i
    # The longest request, cold, on a fresh cache: every segment misses at once.
    assert cached().encode(requests[-1]).ids == hf.encode(requests[-1], add_special_tokens=False).ids
    assert [e.ids for e in tok.encode_batch(requests)] == [hf.encode(r, add_special_tokens=False).ids for r in requests]
    segments = [("<|im_start|>", True), (requests[3], False), (requests[3], True)]
    assert tok.encode_segments(segments).ids == plain.encode_segments(segments).ids
    misses = tok.segment_cache_stats()["misses"]
    assert tok.encode_segments(segments).ids == plain.encode_segments(segments).ids
    assert tok.segment_cache_stats()["misses"] == misses  # the untrusted segment is cached whole
    stats = tok.segment_cache_stats()
    assert stats["hits"] > 0 and stats["misses"] > 0
    assert stats["hit_bytes"] > 0 and stats["miss_bytes"] > 0
    assert 0 < stats["bytes"] <= stats["max_bytes"]


def test_counters_count_segment_lookups():
    tok = cached()
    body = "the user said Hello WORLD\n"
    tok.encode(body)  # no control tokens: not a segment, not counted
    assert tok.segment_cache_stats()["misses"] == 0
    tok.encode(f"<|im_start|>{body}<|im_end|>")
    tok.encode(f"<|im_end|>{body}<|im_start|>{body}<|im_end|>")  # twice in one input: two hits
    stats = tok.segment_cache_stats()
    assert (stats["hits"], stats["misses"]) == (2, 1)
    assert stats["hit_bytes"] == 2 * stats["miss_bytes"] == 2 * len(body.encode())
    assert stats["entries"] == 1


def test_threads_share_one_cache(hf):
    tok = cached(256 * 1024)  # small enough to evict throughout
    streams = [chat_requests(10 + s, turns=20) for s in range(8)]
    expected = [[hf.encode(r, add_special_tokens=False).ids for r in s] for s in streams]

    def replay(stream):
        return [[tok.encode(r).ids for r in stream] for _ in range(2)]

    with ThreadPoolExecutor(8) as pool:
        for got, want in zip(pool.map(replay, streams), expected):
            assert got == [want, want]
    stats = tok.segment_cache_stats()
    assert stats["entries"] > 0 and stats["hits"] > stats["misses"]
    assert stats["bytes"] <= 256 * 1024


def test_eviction_keeps_the_cache_within_its_budget():
    tok = cached(MIB)
    rng = random.Random(3)
    # ~2 KiB segments, well under the largest a 1 MiB cache stores (a segment
    # costing more than 1/64 of the budget is not cached); 300 of them with
    # their ids are a few times the budget.
    bodies = ["".join(rng.choice(PIECES) for _ in range(600)) for _ in range(300)]
    for body in bodies:
        tok.encode(f"<|im_start|>{body}<|im_end|>")
        assert tok.segment_cache_stats()["bytes"] <= MIB
    stats = tok.segment_cache_stats()
    assert stats["misses"] == 300 and stats["hits"] == 0
    assert 0 < stats["entries"] < 300
    # The newest segments are still there, the oldest are gone.
    tok.encode(f"<|im_start|>{bodies[-1]}<|im_end|>")
    tok.encode(f"<|im_start|>{bodies[0]}<|im_end|>")
    stats = tok.segment_cache_stats()
    assert (stats["hits"], stats["misses"]) == (1, 301)


def test_add_tokens_clears_the_cache(hf):
    tok = cached()
    body = "say hi to <|new|> now and <|new|> again"
    request = f"<|im_start|>{body}<|im_end|>{body}<|im_end|>"
    assert tok.encode(request).ids == hf.encode(request, add_special_tokens=False).ids
    token = hf_tokenizers.AddedToken("<|new|>", special=True)
    hf.add_tokens([token])
    tok.add_tokens([token])
    assert tok.segment_cache_stats()["entries"] == 0
    assert tok.encode(request).ids == hf.encode(request, add_special_tokens=False).ids
    assert tok.encode(request).ids == hf.encode(request, add_special_tokens=False).ids


def test_compat_shim_exposes_the_cache(hf):
    """`patch_transformers` hands out `_TokenizerShim`; it forwards the two
    cache methods like it forwards truncation and padding."""
    from fastokens._compat import _TokenizerShim

    shim = _TokenizerShim(tokenizer_json())
    assert shim.segment_cache_stats() is None
    shim.enable_segment_cache(64 * MIB)
    request = chat_requests(3, turns=4)[-1]
    for _ in range(2):
        assert shim.encode(request, add_special_tokens=False).ids == hf.encode(request, add_special_tokens=False).ids
    stats = shim.segment_cache_stats()
    assert stats["hits"] > 0 and stats["entries"] > 0 and stats["max_bytes"] == 64 * MIB
    shim.enable_segment_cache(0)
    assert shim.segment_cache_stats() is None
