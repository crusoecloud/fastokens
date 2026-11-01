"""The segment cache through the Python binding: the switches, the counters,
every entry point against HuggingFace `tokenizers`, threads sharing one
cache, deep copies sharing the original's cache, the budget, and
`add_tokens`. The pipeline-level coverage is in the
Rust tests; this is a synthetic byte-level BPE tokenizer with an NFC
normalizer and chat-style control tokens, so it needs no network.
"""

from __future__ import annotations

import copy
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


def test_add_tokens_starts_a_fresh_cache(hf):
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


def test_a_copy_shares_the_cache_and_applies_its_own_truncation(hf):
    from fastokens._compat import _TokenizerShim

    shim = _TokenizerShim(tokenizer_json())
    shim.enable_segment_cache(64 * MIB)
    requests = chat_requests(5, turns=6)
    for request in requests[:-1]:
        shim.encode(request, add_special_tokens=False)
    before = shim.segment_cache_stats()

    clone = copy.deepcopy(shim)
    want = hf.encode(requests[-1], add_special_tokens=False).ids
    assert clone.encode(requests[-1], add_special_tokens=False).ids == want
    stats = clone.segment_cache_stats()
    assert stats["hits"] > before["hits"], "the copy did not hit what the original cached"
    assert stats["misses"] > before["misses"], "the last request brought nothing new"
    assert shim.segment_cache_stats() == stats, "the copy's counters are the original's"
    # What the copy encoded is a hit for the original.
    assert shim.encode(requests[-1], add_special_tokens=False).ids == want
    assert shim.segment_cache_stats()["misses"] == stats["misses"]

    # Truncation is the copy's own: a request neither has seen is cached in
    # full by the truncating copy, and the original gets the full ids from it.
    unseen = chat_requests(9, turns=2)[-1]
    want = hf.encode(unseen, add_special_tokens=False).ids
    clone.enable_truncation(max_length=7)
    assert clone.encode(unseen, add_special_tokens=False).ids == want[:7]
    misses = clone.segment_cache_stats()["misses"]
    assert misses > stats["misses"]
    assert shim.encode(unseen, add_special_tokens=False).ids == want
    assert shim.segment_cache_stats()["misses"] == misses
    assert shim.truncation is None


def test_add_tokens_moves_a_copy_to_its_own_cache(hf):
    from fastokens._compat import _TokenizerShim

    shim = _TokenizerShim(tokenizer_json())
    shim.enable_segment_cache(64 * MIB)
    body = "say hi to <|new|> now and <|new|> again"
    request = f"<|im_start|>{body}<|im_end|>{body}<|im_end|>"
    assert shim.encode(request, add_special_tokens=False).ids == hf.encode(request, add_special_tokens=False).ids
    entries = shim.segment_cache_stats()["entries"]
    assert entries > 0

    clone = copy.deepcopy(shim)
    hf_clone = hf_tokenizers.Tokenizer.from_str(tokenizer_json())
    token = hf_tokenizers.AddedToken("<|new|>", special=True)
    hf_clone.add_tokens([token])
    clone.add_tokens([token])
    assert clone.segment_cache_stats()["entries"] == 0
    assert clone.segment_cache_stats()["max_bytes"] == 64 * MIB
    assert shim.segment_cache_stats()["entries"] == entries, "the original lost its entries"
    for _ in range(2):
        assert clone.encode(request, add_special_tokens=False).ids == hf_clone.encode(request, add_special_tokens=False).ids
        assert shim.encode(request, add_special_tokens=False).ids == hf.encode(request, add_special_tokens=False).ids
    assert clone.segment_cache_stats()["entries"] > 0
    assert shim.segment_cache_stats()["entries"] == entries


def test_a_copy_of_an_uncached_tokenizer_has_no_cache(hf, plain):
    from fastokens._compat import _TokenizerShim

    shim = _TokenizerShim(tokenizer_json())
    clone = copy.deepcopy(shim)
    assert shim.segment_cache_stats() is None and clone.segment_cache_stats() is None
    request = chat_requests(6, turns=3)[-1]
    assert clone.encode(request, add_special_tokens=False).ids == hf.encode(request, add_special_tokens=False).ids
    # Through the binding directly: sharing with itself changes nothing, and
    # sharing from a tokenizer without a cache turns a copy's cache off.
    tok = cached()
    assert tok.encode(request).ids == plain.encode(request).ids
    stats = tok.segment_cache_stats()
    tok._share_segment_cache_from(tok)
    assert tok.segment_cache_stats() == stats
    tok._share_segment_cache_from(plain)
    assert tok.segment_cache_stats() is None
    assert tok.encode(request).ids == plain.encode(request).ids


def test_a_copy_with_a_cache_of_its_own_joins_the_original(monkeypatch, hf):
    # With the env var set, every tokenizer starts with a cache of its own; a
    # deep copy drops that one for the original's.
    from fastokens._compat import _TokenizerShim

    monkeypatch.setenv("FASTOKENS_SEGMENT_CACHE", "1")
    shim = _TokenizerShim(tokenizer_json())
    request = chat_requests(8, turns=3)[-1]
    want = hf.encode(request, add_special_tokens=False).ids
    assert shim.encode(request, add_special_tokens=False).ids == want
    before = shim.segment_cache_stats()
    assert before["max_bytes"] == MIB and before["misses"] > 0

    clone = copy.deepcopy(shim)
    assert clone.segment_cache_stats() == before, "the copy kept a cache of its own"
    assert clone.encode(request, add_special_tokens=False).ids == want
    stats = clone.segment_cache_stats()
    assert stats["misses"] == before["misses"] and stats["hits"] > before["hits"]
    assert shim.segment_cache_stats() == stats


def test_copies_on_threads_share_one_cache(hf):
    """A pool of deep copies, each used from its own thread, as a serving stack
    uses them: every result matches HF and the copies report one set of
    counters."""
    from fastokens._compat import _TokenizerShim

    shim = _TokenizerShim(tokenizer_json())
    shim.enable_segment_cache(256 * 1024)  # small enough to evict throughout
    pool = [shim, *(copy.deepcopy(shim) for _ in range(7))]
    streams = [chat_requests(10 + s, turns=20) for s in range(8)]
    expected = [[hf.encode(r, add_special_tokens=False).ids for r in s] for s in streams]

    def replay(args):
        tok, stream = args
        return [[tok.encode(r, add_special_tokens=False).ids for r in stream] for _ in range(2)]

    with ThreadPoolExecutor(8) as executor:
        for got, want in zip(executor.map(replay, zip(pool, streams)), expected):
            assert got == [want, want]
    stats = shim.segment_cache_stats()
    assert all(tok.segment_cache_stats() == stats for tok in pool), "one cache, one set of counters"
    assert stats["entries"] > 0 and stats["hits"] > stats["misses"]
    assert stats["bytes"] <= 256 * 1024


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
