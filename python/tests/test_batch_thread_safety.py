"""Batch encoding must release the state lock before reacquiring the GIL."""

import array
import faulthandler
import json
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


def _encode_while_mutating(encoder, mutator):
    # A watchdog in the child supplies useful stacks on a GIL/state-lock cycle.
    # The parent's timeout also bounds hangs during setup or interpreter exit.
    faulthandler.dump_traceback_later(15, exit=True)
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
    texts = ["ab " * 20_000, "a"]
    ready = threading.Event()
    done = threading.Event()
    errors = []
    encodes = 0

    def reader():
        nonlocal encodes
        try:
            ready.set()
            for _ in range(50):
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
    assert encodes == 50
    assert mutations > 0
    faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize("encoder", ["encode_batch", "encode_batch_flat"])
@pytest.mark.parametrize("mutator", [
    "enable_padding", "no_padding", "enable_truncation", "no_truncation",
])
def test_batch_encoding_and_mutation_concurrent(encoder, mutator):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), encoder, mutator],
        capture_output=True,
        text=True,
        timeout=20,
    )
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
    _encode_while_mutating(*sys.argv[1:])
