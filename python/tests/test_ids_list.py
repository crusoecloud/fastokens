"""`Encoding.ids` builds long lists on the worker pool; they must hold exactly the
right ints, with exactly the right refcounts."""

import array
import gc
import json
import sys

from fastokens._native import Tokenizer


def _tokenizer():
    # Every letter and every letter pair is a token, so ids run well past 256
    # (CPython's shared small ints) into the ints this module creates itself.
    letters = "abcdefghijklmnopqrstuvwxyz"
    vocab = {c: i for i, c in enumerate(letters + " ")}
    merges = []
    for a in letters:
        for b in letters:
            vocab[a + b] = len(vocab)
            merges.append([a, b])
    config = {
        "version": "1.0",
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": " ?[a-z]+"}, "behavior": "Isolated", "invert": False},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
    return Tokenizer.from_json_str(json.dumps(config))


def _text(n_words):
    state = 12345
    words = []
    for _ in range(n_words):
        state = (state * 1103515245 + 12345) % 2**31
        k = 1 + state % 7
        words.append("".join("abcdefghijklmnopqrstuvwxyz"[(state >> (3 * j)) % 26] for j in range(k)))
    return " ".join(words)


def _refcounts(objs):
    return [sys.getrefcount(o) for o in objs]


def test_long_ids_list_is_exact_and_refcounts_balance():
    tok = _tokenizer()
    text = _text(200_000)  # far above the parallel threshold
    enc = tok.encode(text)
    flat, _ = tok.encode_batch_flat([text])
    expected = array.array("I", flat).tolist()
    ids = enc.ids
    assert len(ids) > 64 * 1024
    assert ids == expected
    # Small ints are CPython's shared (from 3.12 immortal) objects: skip them.
    probes = sorted(v for v in set(ids) if v > 256)
    assert len(probes) > 100
    counts = [ids.count(p) for p in probes]
    before = _refcounts(probes)
    kept = [enc.ids for _ in range(3)]
    after = _refcounts(probes)
    assert [a - b for a, b in zip(after, before)] == [3 * c for c in counts]
    del kept
    gc.collect()
    assert _refcounts(probes) == before


def test_batch_prebuilt_ids_lists_are_exact_and_refcounts_balance():
    tok = _tokenizer()
    texts = [_text(500 + 37 * i) for i in range(300)]  # together far above the threshold
    flat, offsets = tok.encode_batch_flat(texts)
    flat = array.array("I", flat).tolist()
    offsets = array.array("Q", offsets).tolist()
    expected = [flat[offsets[i] : offsets[i + 1]] for i in range(len(texts))]
    # The library's own int objects (not the fresh ones `tolist` made).
    own = tok.encode(texts[0] + " " + texts[1]).ids
    probes = sorted(v for v in set(own) if v > 256)
    counts = [flat.count(p) for p in probes]
    before = _refcounts(probes)
    encs = tok.encode_batch(texts)
    lists = [e.ids for e in encs]  # the prebuilt ones
    assert lists == expected
    assert [a - b for a, b in zip(_refcounts(probes), before)] == counts
    # Later reads build fresh lists, as every read does.
    again = [e.ids for e in encs]
    assert again == expected and all(a is not b for a, b in zip(again, lists))
    del lists, again, encs
    gc.collect()
    assert _refcounts(probes) == before


def test_prebuilt_ids_do_not_survive_edits():
    tok = _tokenizer()
    texts = [_text(2000) for _ in range(40)]
    encs = tok.encode_batch(texts)
    enc = encs[0]
    n = len(enc)
    enc.pad(n + 3, pad_id=7)
    assert enc.ids[-3:] == [7, 7, 7]
    assert len(enc.ids) == n + 3
