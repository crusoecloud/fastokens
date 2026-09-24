#!/usr/bin/env python3
"""Compare fastokens against ``tokenizers`` / any other tokenizer library.

One command sweeps {models} × {backends} for **correctness** (token-id parity
against a reference implementation, with rich diffs on divergence) and **speed**
(throughput, with ratios). Backends, corpora, and models are pluggable — see
``backends.py``, ``corpora.py``, ``models.py``.

Examples
--------
    # Everything available, built-in corpus, both modes:
    python bench/compare.py

    # fastokens vs transformers on two models, parity only, over real docs:
    python bench/compare.py --models deepseek-v3.2,phi-4 \
        --backends fastokens,transformers --reference transformers \
        --mode correctness --dataset longbench:200

    # tokenizers 0.x vs 1.0 speed on one local tokenizer.json:
    python bench/compare.py --model-path ./tokenizer.json \
        --backends tokenizers --mode speed

    # Prove the harness runs with no libraries installed:
    python bench/compare.py --self-test
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backends as B  # noqa: E402
import corpora as C  # noqa: E402
import models as M  # noqa: E402


# ---------------------------------------------------------------------------
# Table rendering (no third-party deps)
# ---------------------------------------------------------------------------
def render_table(headers, rows, *, md=False) -> str:
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    if md:
        h = "| " + " | ".join(str(x).ljust(w) for x, w in zip(headers, widths)) + " |"
        sep = "| " + " | ".join("-" * w for w in widths) + " |"
        body = [
            "| " + " | ".join(str(x).ljust(w) for x, w in zip(r, widths)) + " |" for r in rows
        ]
        return "\n".join([h, sep] + body)
    line = "  ".join(str(x).ljust(w) for x, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(str(x).ljust(w) for x, w in zip(r, widths)) for r in rows]
    return "\n".join([line, sep] + body)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def diff_report(sample, ref_name, ref_ids, ref_tok, other_name, other_ids, other_tok) -> str:
    i = first_divergence(ref_ids, other_ids)
    lo = max(0, (i or 0) - 3)
    hi = (i or 0) + 4
    lines = [
        f"    sample {sample.id} ({sample.category}): "
        f"lengths ref={len(ref_ids)} {other_name}={len(other_ids)}, first diff @ token {i}",
        f"      {ref_name:>12} ids: {ref_ids[lo:hi]}",
        f"      {other_name:>12} ids: {other_ids[lo:hi]}",
    ]
    # Locate the divergence in the source text by detokenizing the common prefix.
    if i is not None and ref_tok is not None:
        prefix = ref_tok.decode(ref_ids[:i])
        if prefix is not None:
            off = len(prefix)
            window = sample.text[max(0, off - 20) : off + 20]
            lines.append(f"      near char {off}: {window!r}")
    for tok, name, ids in ((ref_tok, ref_name, ref_ids), (other_tok, other_name, other_ids)):
        if tok is not None and i is not None:
            dec = tok.decode(ids[lo:hi])
            if dec is not None:
                lines.append(f"      {name:>12} txt: {dec!r}")
    return "\n".join(lines)


def run_correctness(models_, loaded, ref_name, samples, add_special, args):
    """Return (matrix_rows, diagnostics). Compares each backend to the reference."""
    other_names = [n for n in loaded_backends(loaded) if n != ref_name]
    header = ["model", "samples"] + other_names
    rows = []
    diags = []
    for m in models_:
        ref = loaded.get((m.name, ref_name))
        row = [m.name, str(len(samples))]
        if ref is None:
            rows.append([m.name, "-"] + ["ref n/a"] * len(other_names))
            continue
        # Reference id cache for this model.
        ref_ids_cache = {}
        for s in samples:
            try:
                ref_ids_cache[s.id] = ref.encode(s.text, add_special)
            except Exception as e:
                ref_ids_cache[s.id] = e
        for other in other_names:
            tok = loaded.get((m.name, other))
            if tok is None:
                row.append("load fail")
                continue
            mism = 0
            errs = 0
            first = None
            for s in samples:
                ref_ids = ref_ids_cache[s.id]
                if isinstance(ref_ids, Exception):
                    continue
                try:
                    ids = tok.encode(s.text, add_special)
                except Exception:
                    errs += 1
                    continue
                if ids != ref_ids:
                    mism += 1
                    if first is None:
                        first = (s, ref_ids, ids)
            if mism == 0 and errs == 0:
                row.append("PASS")
            else:
                tag = f"{mism} diff"
                if errs:
                    tag += f" +{errs} err"
                row.append(tag)
                if first is not None:
                    s, ref_ids, ids = first
                    diags.append(
                        f"  {m.name}: {ref_name} vs {other}\n"
                        + diff_report(s, ref_name, ref_ids, ref, other, ids, tok)
                    )
        rows.append(row)
    return header, rows, diags


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------
def run_speed(models_, loaded, samples, args):
    names = loaded_backends(loaded)
    total_bytes = sum(len(s.text.encode("utf-8")) for s in samples)
    header = ["model"] + [f"{n} MB/s" for n in names] + ["fastest"]
    rows = []
    for m in models_:
        row = [m.name]
        best_name, best_rate = None, 0.0
        for n in names:
            tok = loaded.get((m.name, n))
            if tok is None:
                row.append("-")
                continue
            # warmup
            for s in samples[: min(len(samples), 3)]:
                try:
                    tok.encode(s.text, args.add_special)
                except Exception:
                    pass
            best = None
            ok = True
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                try:
                    for s in samples:
                        tok.encode(s.text, args.add_special)
                except Exception:
                    ok = False
                    break
                dt = time.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            if not ok or not best:
                row.append("err")
                continue
            rate = total_bytes / best / 1e6
            row.append(f"{rate:.1f}")
            if rate > best_rate:
                best_rate, best_name = rate, n
        row.append(best_name or "-")
        rows.append(row)
    return header, rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def loaded_backends(loaded) -> list[str]:
    seen = []
    for (_, bname) in loaded:
        if bname not in seen:
            seen.append(bname)
    return seen


def build_samples(args) -> list[C.Sample]:
    samples: list[C.Sample] = []
    if args.corpus == "all":
        samples += C.builtin()
    elif args.corpus:
        samples += C.builtin(args.corpus.split(","))
    if args.corpus_dir:
        samples += C.load_dir(args.corpus_dir)
    if args.dataset:
        samples += C.load_hf_dataset(args.dataset, args.max_samples)
    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="comma list of model names (see --list) or org/repo ids")
    ap.add_argument("--models-file", help="JSON file of model specs")
    ap.add_argument("--model-path", help="one-off local tokenizer.json to compare")
    ap.add_argument("--backends", help="comma list of backend names (default: all available)")
    ap.add_argument("--reference", help="reference backend for correctness (default: auto)")
    ap.add_argument("--corpus", default="all", help="'all', or comma list of categories; '' to skip built-ins")
    ap.add_argument("--corpus-dir", help="extra directory of .txt samples")
    ap.add_argument("--dataset", help="cached HF dataset, e.g. longbench:200")
    ap.add_argument("--max-samples", type=int, default=0, help="cap total samples (0 = no cap)")
    ap.add_argument("--mode", choices=["correctness", "speed", "both"], default="both")
    ap.add_argument("--add-special", dest="add_special", action="store_true", default=False)
    ap.add_argument("--report", choices=["table", "md", "json"], default="table")
    ap.add_argument("--repeat", type=int, default=3, help="speed: best-of-N timed passes")
    ap.add_argument("--offline", action="store_true", help="set HF_HUB_OFFLINE=1")
    ap.add_argument("--list", action="store_true", help="list backends/corpora/models and exit")
    ap.add_argument("--self-test", action="store_true", help="run against built-in mock backends (no deps)")
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if args.self_test:
        _install_mock_backends()

    if args.list:
        _print_list()
        return

    # Resolve models.
    if args.model_path:
        models_ = [B.ModelSpec(os.path.basename(os.path.dirname(args.model_path)) or "local", path=args.model_path)]
    elif args.models_file:
        models_ = M.from_file(args.models_file)
    elif args.models:
        models_ = M.by_names(args.models.split(","))
    else:
        models_ = M.DEFAULT_MODELS

    # Resolve backends (only the available ones).
    wanted = B.by_name(args.backends.split(",")) if args.backends else B.REGISTRY
    avail = []
    print("Backends:", file=sys.stderr)
    for b in wanted:
        ok, info = b.available()
        print(f"  {'ok ' if ok else 'MISS'} {b.name:<14} {info}", file=sys.stderr)
        if ok:
            avail.append(b)
    if not avail:
        raise SystemExit("no backends available")

    # Reference backend (correctness). Prefer explicit, else transformers, else tokenizers, else first.
    ref_name = args.reference
    if ref_name is None:
        for pref in ("transformers", "tokenizers", "tiktoken", avail[0].name):
            if pref in [b.name for b in avail]:
                ref_name = pref
                break

    samples = build_samples(args)
    print(f"\nModels: {len(models_)}   Backends: {len(avail)}   Samples: {len(samples)}   Reference: {ref_name}\n", file=sys.stderr)

    # Load every (model, backend) once.
    loaded: dict[tuple[str, str], B.LoadedTokenizer | None] = {}
    for m in models_:
        cells = []
        for b in avail:
            tok = None
            try:
                tok = b.load(m)
            except Exception:
                tok = None
            loaded[(m.name, b.name)] = tok
            cells.append(f"{b.name}={'y' if tok else '-'}")
        print(f"  load {m.name:<16} {'  '.join(cells)}", file=sys.stderr)
    print(file=sys.stderr)

    result: dict = {
        "reference": ref_name,
        "backends": [b.name for b in avail],
        "models": [m.name for m in models_],
        "samples": len(samples),
        "add_special": args.add_special,
    }
    if args.mode in ("correctness", "both"):
        header, rows, diags = run_correctness(models_, loaded, ref_name, samples, args.add_special, args)
        result["correctness"] = {"header": header, "rows": rows, "diagnostics": diags}
    if args.mode in ("speed", "both"):
        sheader, srows = run_speed(models_, loaded, samples, args)
        result["speed"] = {"header": sheader, "rows": srows}

    if args.report == "json":
        import json

        print(json.dumps(result, indent=2))
        return

    md = args.report == "md"
    if "correctness" in result:
        c = result["correctness"]
        print("=== Correctness (token-id parity vs reference) ===")
        print(render_table(c["header"], c["rows"], md=md))
        if c["diagnostics"]:
            print("\n--- first divergence per failing pair ---")
            print("\n".join(c["diagnostics"]))
        print()
    if "speed" in result:
        s = result["speed"]
        print("=== Speed (throughput, best-of-%d) ===" % args.repeat)
        print(render_table(s["header"], s["rows"], md=md))
        print()


# ---------------------------------------------------------------------------
# Self-test: two mock backends (one deliberately buggy) prove the pipeline.
# ---------------------------------------------------------------------------
def _install_mock_backends():
    class _Loaded(B.LoadedTokenizer):
        def __init__(self, buggy):
            self.buggy = buggy

        def encode(self, text, add_special_tokens):
            ids = [ord(c) & 0xFFFF for c in text]
            if self.buggy and "z" in text:  # inject a divergence on samples with 'z'
                ids = ids[:-1] if ids else ids
            return ids

        def decode(self, ids):
            return "".join(chr(i) for i in ids)

    class MockRef(B.Backend):
        name = "mock_ref"

        def available(self):
            return True, "self-test"

        def load(self, spec):
            return _Loaded(buggy=False)

    class MockBuggy(B.Backend):
        name = "mock_buggy"

        def available(self):
            return True, "self-test"

        def load(self, spec):
            return _Loaded(buggy=True)

    B.REGISTRY[:] = [MockRef(), MockBuggy()]
    M.DEFAULT_MODELS[:] = [B.ModelSpec("mock-model")]


def _print_list():
    print("Backends:", file=sys.stderr)
    for b in B.REGISTRY:
        ok, info = b.available()
        print(f"  {b.name:<14} {'available' if ok else 'unavailable'}  {info}")
    print("\nCorpus categories:")
    print("  " + ", ".join(C.categories()))
    print("\nDefault models:")
    for m in M.DEFAULT_MODELS:
        print(f"  {m.name:<16} hf={m.hf_id or '-'}  tiktoken={m.tiktoken_encoding or '-'}")


if __name__ == "__main__":
    main()
