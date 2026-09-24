"""The multi-GB corpus behind ``bench/bulk.py`` and ``bench/rust``'s ``bulk``.

One pool of real text, cut three ways at the same bytes:

  small   ~400 B per document (~100 tokens)
  medium  ~8 KB               (~2,000 tokens)
  large   ~1 MB               (~250,000 tokens)

Document lengths are uniform in [0.5, 1.5] x the target, and each cut is moved
forward to just after a space or newline (within 64 bytes), else to a UTF-8
character boundary. The three forms thus tokenize the same text and differ only
in how it is split into documents.

The pool mixes, by bytes (``MIX``): English web text (C4 ``en``), Chinese web
text (mC4 ``zh``), chat (ShareGPT, both 90k parts) and long documents
(LongBench-v2 contexts), shuffled at the document level with a fixed seed so
every prefix has the same mix. A source that runs out (ShareGPT has ~1.5 GB,
LongBench ~0.45 GB) leaves the rest of its share to the web sources. A warm-up
pool of 32 MB (disjoint documents from the same sources) is cut the same way.

Files, under ``~/.cache/fastokens-bench/bulk-<GB>GB/`` (``$FASTOKENS_BENCH_CACHE``
replaces ``~/.cache/fastokens-bench``):

  pool.txt, warm.txt             the text, UTF-8
  <form>.idx, <form>.warm.idx    document boundaries: n+1 little-endian u64 offsets
  meta.json                      parameters, composition and counts (written last)
"""

from __future__ import annotations

import array
import gzip
import json
import os
import random
import sys

#: form -> target document size in bytes (~4 bytes per token for these models).
FORMS = {"small": 400, "medium": 8_000, "large": 1_000_000}
#: source -> share of the pool's bytes.
MIX = {"web-en": 0.4, "web-zh": 0.2, "chat": 0.3, "long": 0.1}
WARM_BYTES = 32_000_000
#: Bump when the construction changes, so cached corpora are rebuilt.
VERSION = 1


def cache_root() -> str:
    return os.environ.get("FASTOKENS_BENCH_CACHE") or os.path.expanduser("~/.cache/fastokens-bench")


def corpus_dir(gb: float) -> str:
    return os.path.join(cache_root(), f"bulk-{gb:g}GB")


def _download(repo: str, name: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, name, repo_type="dataset")


def _web(pattern: str):
    for shard in range(1024):
        with gzip.open(_download("allenai/c4", pattern.format(shard)), "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)["text"]


def _chat():
    for part in ("sg_90k_part1.json", "sg_90k_part2.json"):
        with open(_download("RyokoAI/ShareGPT52K", part), encoding="utf-8") as fh:
            items = json.load(fh)
        for item in items:
            text = "\n\n".join(m["value"] for m in item.get("conversations") or [] if m.get("value"))
            if text:
                yield text


def _long():
    with open(_download("zai-org/LongBench-v2", "data.json"), encoding="utf-8") as fh:
        for d in json.load(fh):
            if d.get("context"):
                yield d["context"]


SOURCES = {
    "web-en": lambda: _web("en/c4-train.{:05d}-of-01024.json.gz"),
    "web-zh": lambda: _web("multilingual/c4-zh.tfrecord-{:05d}-of-01024.json.gz"),
    "chat": _chat,
    "long": _long,
}
#: Sources that can run out; the unlimited web sources absorb their shortfall.
FINITE = ("chat", "long")


def _cut(buf: bytes, target: int, rng: random.Random) -> array.array:
    """Boundaries of documents of ``target`` +-50% bytes, snapped as described above."""
    n, pos, lo = len(buf), 0, target // 2
    offs = array.array("Q", [0])
    find = buf.find
    while True:
        p = pos + lo + int(rng.random() * (target + 1))
        if p >= n:
            break
        sp, nl = find(b" ", p, p + 64), find(b"\n", p, p + 64)
        if sp < 0 and nl < 0:
            while p < n and buf[p] & 0xC0 == 0x80:
                p += 1
        else:
            p = (nl if sp < 0 else sp if nl < 0 else min(sp, nl)) + 1
        if p >= n:
            break
        offs.append(p)
        pos = p
    offs.append(n)
    return offs


def _write_idx(path: str, offs: array.array) -> None:
    if sys.byteorder == "big":
        offs = array.array("Q", offs)
        offs.byteswap()
    with open(path, "wb") as fh:
        offs.tofile(fh)


def build(gb: float, mix: dict[str, float] | None = None, seed: int = 0) -> str:
    """Build the ``gb``-gigabyte corpus (or reuse a matching one); returns its directory."""
    mix = dict(mix or MIX)
    d = corpus_dir(gb)
    meta_path = os.path.join(d, "meta.json")
    want = {"version": VERSION, "gb": gb, "mix": mix, "seed": seed, "forms": FORMS, "warm_bytes": WARM_BYTES}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
        if all(meta.get(k) == v for k, v in want.items()):
            return d
        os.remove(meta_path)
    os.makedirs(d, exist_ok=True)

    # Whole documents from each source until its share (plus a little slack) is met.
    total = gb * 1e9 + WARM_BYTES
    order = [s for s in mix if s in FINITE] + [s for s in mix if s not in FINITE]
    web_weight = sum(w for s, w in mix.items() if s not in FINITE)
    docs, shortfall, have = [], 0.0, {}
    for s in order:
        quota = total * mix[s] * 1.02
        if s not in FINITE and web_weight:
            quota += shortfall * mix[s] / web_weight
        n = 0
        if quota > 0:
            print(f"  collecting {s}: {quota / 1e9:.2f} GB", file=sys.stderr)
            for text in SOURCES[s]():
                b = text.encode("utf-8", "replace")
                docs.append((s, b))
                n += len(b)
                if n >= quota:
                    break
        have[s] = n
        if s in FINITE and n < quota:
            print(f"  {s} has only {n / 1e9:.2f} GB; the web sources make up the rest", file=sys.stderr)
            shortfall += quota - n

    random.Random(seed).shuffle(docs)
    it = iter(docs)
    warm, n = [], 0
    for _, b in it:
        warm.append(b)
        n += len(b) + 2
        if n >= WARM_BYTES:
            break
    pool, n, composition = [], 0, dict.fromkeys(mix, 0)
    for s, b in it:
        pool.append(b)
        composition[s] += len(b)
        n += len(b) + 2
        if n >= gb * 1e9:
            break
    del docs, it
    pool_buf, warm_buf = b"\n\n".join(pool), b"\n\n".join(warm)
    del pool, warm
    for name, buf in (("pool.txt", pool_buf), ("warm.txt", warm_buf)):
        with open(os.path.join(d, name), "wb") as fh:
            fh.write(buf)

    forms = {}
    for form, target in FORMS.items():
        offs = _cut(pool_buf, target, random.Random(f"{seed}-{form}"))
        warm_offs = _cut(warm_buf, target, random.Random(f"{seed}-{form}-warm"))
        _write_idx(os.path.join(d, f"{form}.idx"), offs)
        _write_idx(os.path.join(d, f"{form}.warm.idx"), warm_offs)
        forms[form] = {"target_bytes": target, "docs": len(offs) - 1, "warm_docs": len(warm_offs) - 1}
        print(f"  {form}: {len(offs) - 1:,} documents", file=sys.stderr)

    meta = dict(want, bytes=len(pool_buf), warm=len(warm_buf), composition=composition, counts=forms)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1)
    return d


def meta(d: str) -> dict:
    with open(os.path.join(d, "meta.json")) as fh:
        return json.load(fh)


def offsets(d: str, form: str, warm: bool = False) -> array.array:
    offs = array.array("Q")
    with open(os.path.join(d, f"{form}.warm.idx" if warm else f"{form}.idx"), "rb") as fh:
        offs.frombytes(fh.read())
    if sys.byteorder == "big":
        offs.byteswap()
    return offs


def load(d: str, form: str, warm: bool = False, only=None) -> tuple[list[str], array.array]:
    """A form's documents (or just those at the indices ``only``) and their byte offsets."""
    with open(os.path.join(d, "warm.txt" if warm else "pool.txt"), "rb") as fh:
        buf = fh.read()
    offs = offsets(d, form, warm)
    if only is not None:
        return [buf[offs[i] : offs[i + 1]].decode() for i in only], offs
    return [buf[a:b].decode() for a, b in zip(offs, offs[1:])], offs
