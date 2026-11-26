"""Batch encoding must release the state lock before reacquiring the GIL."""

import array
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from fastokens._native import Tokenizer


def _tokenizer():
    # Local fixture: no model downloads are needed in the child processes.
    return Tokenizer.from_json_str(json.dumps({
        "version": "1.0",
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"Regex": "\\s+"},
            "behavior": "Removed",
            "invert": False,
        },
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "[PAD]": 2},
            "merges": [],
        },
    }))


# A two-thread pool in the child: a batch of a few MB is then large enough for
# `encode_batch_flat` to encode it in slabs, on another thread, while it is
# still converting the strings.
_LARGE_ENV = {"FASTOKENS_BPE_THREADS": "2"}
_LARGE_TEXTS = ["ab " * (20_000 + i) for i in range(48)]


def _child(*args, env=None):
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, **(env or {})},
    )


def _encode_while_mutating(encoder, mutator, size="small"):
    # A watchdog in the child supplies useful stacks on a GIL/state-lock cycle.
    # The parent's timeout also bounds hangs during setup or interpreter exit.
    faulthandler.dump_traceback_later(50, exit=True)
    tok = _tokenizer()
    tok.enable_padding(pad_id=2)
    tok.enable_truncation(max_length=32)
    encode = getattr(tok, encoder)
    mutate = getattr(tok, mutator)
    kwargs = {
        "enable_padding": {"pad_id": 2},
        "no_padding": {},
        "enable_truncation": {"max_length": 32},
        "no_truncation": {},
    }[mutator]
    texts = ["ab " * 20_000, "a"] if size == "small" else _LARGE_TEXTS
    rounds = 50 if size == "small" else 10
    ready = threading.Event()
    done = threading.Event()
    errors = []
    encodes = 0

    def reader():
        nonlocal encodes
        try:
            ready.set()
            for _ in range(rounds):
                encode(texts)
                encodes += 1
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=reader, daemon=True)
    worker.start()
    ready.wait()
    mutations = 0
    while not done.is_set():
        mutate(**kwargs)
        mutations += 1
        time.sleep(0.0001)
    worker.join()
    assert not errors, errors
    assert encodes == rounds
    assert mutations > 0
    faulthandler.cancel_dump_traceback_later()


def _large_flat_batch():
    faulthandler.dump_traceback_later(50, exit=True)
    tok = _tokenizer()
    ids, offsets = tok.encode_batch_flat(_LARGE_TEXTS)
    ids = array.array("I", ids).tolist()
    offsets = array.array("Q", offsets).tolist()
    assert [ids[a:b] for a, b in zip(offsets, offsets[1:])] == [
        tok.encode(text).ids for text in _LARGE_TEXTS
    ]
    # A string that cannot be converted, after the pool has started on the
    # batch: the conversion's error, and the tokenizer still works.
    bad = _LARGE_TEXTS[:40] + ["a \ud800"] + _LARGE_TEXTS[40:]
    with pytest.raises(UnicodeEncodeError):
        tok.encode_batch_flat(bad)
    assert tok.encode_batch_flat(["ab a"]) == (
        array.array("I", [0, 1, 0]).tobytes(),
        array.array("Q", [0, 3]).tobytes(),
    )
    faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("encoder", ["encode_batch", "encode_batch_flat"])
@pytest.mark.parametrize("mutator", [
    "enable_padding", "no_padding", "enable_truncation", "no_truncation",
])
def test_batch_encoding_and_mutation_concurrent(encoder, mutator):
    result = _child(encoder, mutator)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mutator", ["enable_truncation", "no_truncation"])
def test_large_flat_batch_and_mutation_concurrent(mutator):
    result = _child("encode_batch_flat", mutator, "large", env=_LARGE_ENV)
    assert result.returncode == 0, result.stdout + result.stderr


def test_large_flat_batch_matches_encode_and_raises_conversion_errors():
    result = _child("large_flat_batch", env=_LARGE_ENV)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("direction", ["left", "right"])
@pytest.mark.parametrize("length", [None, 5])
def test_batch_padding_and_truncation(direction, length):
    tok = _tokenizer()
    tok.enable_truncation(max_length=3)
    tok.enable_padding(direction=direction, pad_id=2, length=length)
    texts = ["ab ab", "a"]
    unpadded = [[0, 1, 0], [0]]
    target = length or 3
    encodings = tok.encode_batch(texts)
    for encoding, ids in zip(encodings, unpadded):
        pad = [2] * (target - len(ids))
        mask = [1] * len(ids)
        zeros = [0] * len(pad)
        assert encoding.ids == (pad + ids if direction == "left" else ids + pad)
        assert encoding.attention_mask == (
            zeros + mask if direction == "left" else mask + zeros
        )

    # The flat API still applies truncation while ignoring padding.
    ids, offsets = tok.encode_batch_flat(texts)
    assert array.array("I", ids).tolist() == [0, 1, 0, 0]
    assert array.array("Q", offsets).tolist() == [0, 3, 4]
    assert tok.encode_batch([]) == []


if __name__ == "__main__":
    if sys.argv[1] == "large_flat_batch":
        _large_flat_batch()
    else:
        _encode_while_mutating(*sys.argv[1:])
