"""Text corpora for the comparison harness.

Three ways to supply test text, in increasing order of "wideness":

  1. Built-in categories (:func:`builtin`) — no network, deterministic, and chosen
     to exercise the parts of a BPE pipeline that actually diverge between
     implementations: case runs, contractions, CJK/script boundaries, whitespace
     runs, combining marks, ZWJ/format/control characters, numbers, URLs/paths.
  2. A directory of ``*.txt`` files (:func:`load_dir`) — drop a file in to add a
     test case. This is the easy path for "compare on my data".
  3. A cached HuggingFace dataset (:func:`load_hf_dataset`) — long real documents,
     read straight from the local hub cache (no download).

Every loader returns ``list[Sample]``; a :class:`Sample` carries a category and an
id so mismatches can be reported by kind.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    category: str
    id: str
    text: str


# ---------------------------------------------------------------------------
# Built-in categories
# ---------------------------------------------------------------------------
_BUILTIN: dict[str, list[str]] = {
    "english": [
        "The quick brown fox jumps over the lazy dog.",
        "Don't you think it's O'Brien's? They're going—wait, aren't they?",
        "HTTPRequest, HelloWorld, camelCase, ALLCAPS, iOS, getHTTPResponseCode()",
        "In 1997, roughly 3.14 million people (about 0.05%) said \"yes\".",
        "e.g. i.e. etc. vs. Dr. Smith Jr. went to Washington, D.C. at 9 a.m.",
    ],
    "code": [
        "def f(x: int) -> int:\n    return x * 2 + 1  # doubles then bumps\n",
        "let mut v: Vec<u32> = Vec::with_capacity(64);\nfor i in 0..v.len() { v[i] += 1; }",
        '{"key": "value", "n": 42, "arr": [1, 2, 3], "nested": {"a": true}}',
        "SELECT id, name FROM users WHERE age >= 18 AND status != 'banned';",
        "#include <stdio.h>\nint main(void){printf(\"%d\\n\", 1<<10);return 0;}",
    ],
    "markdown": [
        "# Title\n\nSome **bold** and _italic_ and `code`.\n\n- one\n- two\n\n> quote\n",
        "[link](https://example.com) and ![img](./a.png) and a | table | row |\n|---|---|",
    ],
    "cjk": [
        "你好，世界！这是一个中文测试。",
        "日本語のテストです。カタカナとひらがな、漢字が混ざる。",
        "한국어 토크나이저 테스트입니다.",
        "混合ABCと日本語とEnglish123と中文456。",
        "龥一 ヿ゠ 絵文字 数字123",
    ],
    "multilingual": [
        "café résumé naïve über straße señor niño Ελληνικά",
        "Привет, мир! Здравствуйте.",
        "مرحبا بالعالم — עברית شלום",
        "สวัสดีชาวโลก नमस्ते दुनिया",
    ],
    "whitespace": [
        " leading space",
        "trailing space  ",
        "a\t\tdouble\ttabs",
        "line1\n\n\nline2",
        "mixed   \n  \n   runs",
        "   2 leading spaces before a number",
        "word   \n\t 42   中 end   ",
    ],
    "unicode_adversarial": [
        "cafe\u0301\u0302 combining marks",
        "emoji \U0001f44d\u200d\U0001f44d zwj sequence \U0001f469\u200d\U0001f4bb",
        "control \x00\x07\x1b and format \u200b\u200e\u202a chars",
        "private \ue000\uf8ff use and fullwidth \uff21\uff22\uff23 \uff10\uff11",
        "surrogate-free astral \U00020000\U0001f600 \u2028line\u2029para",
    ],
    "numbers": [
        "12345678 90 007 3.14159 1,000,000 0xFF 1e-9 v1.2.3-rc.4 100000000000",
        "phone +1 (555) 123-4567, ip 192.168.0.1, hex #a1b2c3",
    ],
    "urls_paths": [
        "Visit https://example.com/path?q=1&x=2#frag or http://a.b.co.",
        "/usr/local/bin/env  C:\\Windows\\System32  ~/.config/app  ./rel/path.ext",
        "user@host.example.com mailto:x@y.z s3://bucket/key",
    ],
    "edge": [
        "",
        " ",
        "\n",
        "\t",
        "a",
        "\U0001f642",
        "aaaaaaaaaa" * 1000,
        "🙂" * 500,
        "\n" * 200,
    ],
}


def builtin(categories: list[str] | None = None) -> list[Sample]:
    cats = categories or list(_BUILTIN)
    out: list[Sample] = []
    for cat in cats:
        if cat not in _BUILTIN:
            raise SystemExit(f"unknown corpus category {cat!r}; known: {', '.join(_BUILTIN)}")
        for i, text in enumerate(_BUILTIN[cat]):
            out.append(Sample(cat, f"{cat}[{i}]", text))
    return out


def categories() -> list[str]:
    return list(_BUILTIN)


# ---------------------------------------------------------------------------
# Directory of .txt files — drop a file in to add a test case.
# ---------------------------------------------------------------------------
def load_dir(path: str) -> list[Sample]:
    out: list[Sample] = []
    for fp in sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)):
        with open(fp, encoding="utf-8") as fh:
            out.append(Sample("file", os.path.relpath(fp, path), fh.read()))
    return out


# ---------------------------------------------------------------------------
# Cached HuggingFace datasets (no download; reads the local hub cache).
# ---------------------------------------------------------------------------
_DATASET_FILES = {
    "longbench": ("zai-org/LongBench-v2", "data.json", "context"),
    "sharegpt": ("RyokoAI/ShareGPT52K", "sg_90k_part1.json", None),
}


def load_hf_dataset(spec: str, max_samples: int) -> list[Sample]:
    """``spec`` is ``name[:N]`` (e.g. ``longbench:200``). Reads the cached JSON."""
    import json

    name, _, n = spec.partition(":")
    limit = int(n) if n else max_samples
    if name not in _DATASET_FILES:
        raise SystemExit(f"unknown dataset {name!r}; known: {', '.join(_DATASET_FILES)}")
    repo, fname, field = _DATASET_FILES[name]
    path = _find_in_hub_cache(repo, fname, kind="datasets")
    if path is None:
        print(f"  [skip] dataset {name!r}: {fname} not in local HF cache")
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out: list[Sample] = []
    for i, item in enumerate(data):
        if field:
            text = item.get(field)
        else:  # sharegpt: join conversation turns
            msgs = item.get("conversations") or []
            text = "\n\n".join(m.get("value", "") for m in msgs)
        if text:
            out.append(Sample(name, f"{name}[{i}]", text))
        if len(out) >= limit:
            break
    return out


def _find_in_hub_cache(repo: str, fname: str, kind: str) -> str | None:
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    slug = f"{kind}--" + repo.replace("/", "--")
    base = os.path.join(home, "hub", slug, "snapshots")
    if not os.path.isdir(base):
        return None
    hits = glob.glob(os.path.join(base, "*", "**", fname), recursive=True)
    return hits[0] if hits else None
