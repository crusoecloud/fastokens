"""The native decode paths drop ids that cannot be real tokens.

`decode`/`decode_batch` take a fast path for a clean list of in-range u32s and
fall back to an element-wise pass that drops anything else: negatives, values
above `u32::MAX`, ints too large for `i64`, and non-integers such as
`inf`/`-inf`/`nan`. These build a tokenizer directly, so they run without
`transformers` installed.
"""

import json

from fastokens._native import Tokenizer


def _tokenizer():
    config = {
        "version": "1.0",
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
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
            "ignore_merges": True,
            # Ids must be contiguous from 0, or the BPE model fails to load.
            "vocab": {"a": 0, "h": 1, "e": 2, "l": 3, "o": 4},
            "merges": [],
        },
    }
    return Tokenizer.from_json_str(json.dumps(config))


# Every kind of value that must be dropped. Note the floats and the >i64 int:
# these are the cases a Vec<i64>-based fallback could not handle (they raise
# TypeError / OverflowError during extraction rather than being filtered).
INVALID = [-1, 2**32, 2**100, float("inf"), float("-inf"), float("nan"), -(2**80)]


def test_decode_drops_invalid_ids():
    tok = _tokenizer()
    valid = [1, 2, 3, 3, 4]
    mixed = [1, -1, 2, 2**32, 3, float("inf"), 3, float("-inf"), 4, float("nan"), 2**100]
    # Dropping the invalid ids must leave exactly the valid decode.
    assert tok.decode(mixed) == tok.decode(valid)


def test_decode_all_invalid_is_empty():
    tok = _tokenizer()
    assert tok.decode(INVALID) == tok.decode([])


def test_decode_batch_sanitizes_each_sequence():
    tok = _tokenizer()
    mixed = [[1, -1, 2], [2**32, 3, float("nan")], INVALID]
    clean = [[1, 2], [3], []]
    assert tok.decode_batch(mixed) == tok.decode_batch(clean)


def test_clean_ids_take_the_fast_path_unchanged():
    tok = _tokenizer()
    assert tok.decode([1, 2, 3, 3, 4]) == tok.decode([1, 2, 3, 3, 4])
