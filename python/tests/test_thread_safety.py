"""Concurrency test: `decode` (read) and `enable_truncation` (write) on the
same `Tokenizer` instance must not panic the Rust side.

`decode` releases the GIL via `py.allow_threads`, so another Python thread
can call a mutator while it is running. Before the `RwLock` refactor, that
race could surface as a PyO3 borrow-check panic ("already borrowed"); with
the lock, the mutator simply waits its turn.
"""

import threading
import time

import pytest

from fastokens._native import Tokenizer

MODEL = "Qwen/Qwen3-0.6B"


def test_decode_and_enable_truncation_concurrent():
    tok = Tokenizer.from_model(MODEL)
    ids = tok.encode("Hello, world! This is a thread-safety smoke test.").ids
    assert ids, "needed non-empty ids to decode"

    stop = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                tok.decode(ids)
        except BaseException as exc:
            errors.append(exc)

    def writer() -> None:
        try:
            i = 0
            while not stop.is_set():
                if i % 2 == 0:
                    tok.enable_truncation(max_length=8)
                else:
                    tok.no_truncation()
                i += 1
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)] + [
        threading.Thread(target=writer) for _ in range(2)
    ]

    for t in threads:
        t.start()
    time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=5)
        if t.is_alive():
            pytest.fail(f"thread {t.name} did not exit")

    assert not errors, f"got {len(errors)} error(s); first: {errors[0]!r}"
