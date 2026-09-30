#!/usr/bin/env python3
"""fastokens vs other tokenizers on a multi-GB corpus, in three document sizes.

One pool of real text (``bulk_corpus.py``: English and Chinese web text, chat,
long documents; 3 GB by default) cut three ways at the same bytes:

  small   ~400 B per document (~100 tokens)
  medium  ~8 KB               (~2,000 tokens)
  large   ~1 MB               (~250,000 tokens)

and each form encoded two ways (``--modes``):

  encode  one ``encode`` call per document, in order
  batch   ``encode_batch`` over consecutive documents, ~16 MB per call (``--batch-mb``)

The libraries and their calls are ``serving.py``'s (``--libs hf,gt,ft``, each
optionally ``LIB@PYTHON``; the first is the reference), with ids materialized as
Python lists. Every (library, model, form, mode) runs in a fresh subprocess that
warms up on a disjoint 32 MB of the same shape, then encodes every timed
document exactly once. Timing covers only the encode calls (per group of 64
documents in ``encode`` mode), not reading the corpus or checking the ids.

Parity: each worker hashes every document's ids; the parent checks every
library and mode against the reference library's first mode. Kimi's
disagreements are settled against tiktoken, as in ``serving.py``; the other
models' by ``--judge LIB@PYTHON`` if given, e.g. ``hf@/venv-0.x/bin/python`` for
tokenizers 0.x (1.0.0rc2 mis-merges some Chinese, e.g. GLM-5.3's "的件").

    pip install "tokenizers==1.0.0rc2" tiktoken huggingface_hub
    maturin develop --release       # the fastokens extension, from the repo root
    python bench/bulk.py --prepare  # download and build the corpus once (~2 min)
    python bench/bulk.py --libs hf,gt,ft
    python bench/bulk.py --gb 1 --forms small,large --modes batch --repeat 3
    python bench/bulk.py --libs hf,gt,ft --judge hf@/tmp/tok0/bin/python

The corpus is also what ``bench/rust``'s ``bulk`` subcommand reads.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bulk_corpus  # noqa: E402
import kimi_hf  # noqa: E402
from serving import LIBS, MODELS, TIKTOKEN_ONLY, load_lib, print_table  # noqa: E402

GROUP = 64
#: At most this many differing documents per (model, form) are checked against tiktoken.
TIKTOKEN_CHECKS = 500


def judge_worker(lib: str, model: str, form: str, args) -> dict:
    """Encode just the disputed documents (``--indices``) for the parent to settle them."""
    version, enc, _ = load_lib(lib, MODELS[model])
    idx = array.array("Q")
    with open(args.indices, "rb") as fh:
        idx.frombytes(fh.read())
    texts, _ = bulk_corpus.load(args.dir, form, only=idx)
    digs = array.array("Q", (digest(enc(t)) for t in texts))
    with open(args.digests_out, "wb") as fh:
        digs.tofile(fh)
    return {"version": version}


def digest(ids) -> int:
    return int.from_bytes(hashlib.blake2b(array.array("I", ids).tobytes(), digest_size=8).digest(), "little")


def batch_bounds(offs, budget: float) -> list[int]:
    """Document indices splitting the corpus into runs of >= ``budget`` bytes."""
    bounds, start = [0], offs[0]
    for i in range(1, len(offs) - 1):
        if offs[i] - start >= budget:
            bounds.append(i)
            start = offs[i]
    bounds.append(len(offs) - 1)
    return bounds


def worker(lib: str, model: str, form: str, mode: str, args) -> dict:
    version, enc, enc_batch = load_lib(lib, MODELS[model])
    docs, offs = bulk_corpus.load(args.dir, form)
    warm, warm_offs = bulk_corpus.load(args.dir, form, warm=True)
    budget = args.batch_mb * 1e6

    def run(texts, offs, digs):
        dt, ntok = 0.0, 0
        if mode == "encode":
            groups = [(i, i + GROUP) for i in range(0, len(texts), GROUP)]
            call = lambda grp: [enc(t) for t in grp]  # noqa: E731
        else:
            b = batch_bounds(offs, budget)
            groups = list(zip(b, b[1:]))
            call = enc_batch
        for lo, hi in groups:
            grp = texts[lo:hi]
            t0 = time.perf_counter()
            out = call(grp)
            dt += time.perf_counter() - t0
            if digs is not None:
                for ids in out:
                    digs.append(digest(ids))
                    ntok += len(ids)
            # Free the ids here, untimed: rebinding ``out`` in the timed call above
            # would otherwise charge the previous group's deallocation (a decref per
            # token) to the next one.
            out = ids = None
        return dt, ntok

    run(warm, warm_offs, None)
    digs = array.array("Q")
    dt, ntok = run(docs, offs, digs)
    with open(args.digests_out, "wb") as fh:
        digs.tofile(fh)
    nbytes = offs[-1] - offs[0]
    return {"version": version, "n": len(docs), "bytes": nbytes, "tokens": ntok, "seconds": dt, "mb_s": nbytes / dt / 1e6}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument(
        "--libs",
        default="hf,ft",
        help="comma-separated hf (tokenizers), gt (gigatoken), ft (fastokens), each optionally "
        "LIB@PYTHON to run under another interpreter; the first is the reference",
    )
    ap.add_argument("--gb", type=float, default=3.0, help="corpus size in GB (each form is the whole corpus)")
    ap.add_argument("--forms", default=",".join(bulk_corpus.FORMS))
    ap.add_argument("--modes", default="encode,batch")
    ap.add_argument("--batch-mb", type=float, default=16.0, help="bytes per encode_batch call, in MB")
    ap.add_argument("--repeat", type=int, default=1, help="run each worker N times (alternating libraries) and report medians")
    ap.add_argument("--prepare", action="store_true", help="only download and build the corpus")
    ap.add_argument("--judge", metavar="LIB@PYTHON", help="settle non-Kimi disagreements with this library, e.g. "
                    "hf@/venv-0.x/bin/python (tokenizers 0.x)")
    ap.add_argument("--json", help="also write raw results here")
    ap.add_argument("--indices", help=argparse.SUPPRESS)
    ap.add_argument("--judge-worker", nargs=3, metavar=("LIB", "MODEL", "FORM"), help=argparse.SUPPRESS)
    ap.add_argument("--dir", help=argparse.SUPPRESS)
    ap.add_argument("--digests-out", help=argparse.SUPPRESS)
    ap.add_argument("--worker", nargs=4, metavar=("LIB", "MODEL", "FORM", "MODE"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        json.dump(worker(*args.worker, args), sys.stdout)
        return
    if args.judge_worker:
        json.dump(judge_worker(*args.judge_worker, args), sys.stdout)
        return

    args.dir = bulk_corpus.build(args.gb)
    meta = bulk_corpus.meta(args.dir)
    shares = sorted(meta["composition"].items(), key=lambda kv: -kv[1])
    mix = ", ".join(f"{s} {b / meta['bytes']:.0%}" for s, b in shares)
    print(f"corpus: {meta['bytes'] / 1e9:.2f} GB ({mix}) in {args.dir}", file=sys.stderr)
    if args.prepare:
        return

    models, forms, modes = args.models.split(","), args.forms.split(","), args.modes.split(",")
    for f in forms:
        if f not in bulk_corpus.FORMS:
            sys.exit(f"unknown form {f!r} (expected one of {', '.join(bulk_corpus.FORMS)})")
    for m in modes:
        if m not in ("encode", "batch"):
            sys.exit(f"unknown mode {m!r} (expected encode, batch)")
    libs = []
    for item in args.libs.split(","):
        name, _, python = item.partition("@")
        if name not in LIBS:
            sys.exit(f"unknown library {name!r} (expected one of {', '.join(LIBS)})")
        libs.append((name, python or sys.executable))

    tmp = tempfile.mkdtemp(prefix="fastokens-bulk-")
    results, labels, notes, rows, ok = {}, None, [], [], True
    for model in models:
        repo = MODELS[model]
        for form in forms:
            digs = {}
            for mode in modes:
                runs = [[] for _ in libs]
                for rep in range(args.repeat):
                    for k, (lib, python) in enumerate(libs):
                        out_path = os.path.join(tmp, f"{k}-{mode}.u64")
                        cmd = [python, __file__, "--worker", lib, model, form, mode, f"--dir={args.dir}",
                               f"--batch-mb={args.batch_mb}", f"--digests-out={out_path}"]
                        p = subprocess.run(cmd, capture_output=True, text=True)
                        if p.returncode != 0:
                            sys.exit(f"worker {lib}/{model}/{form}/{mode} failed:\n{p.stderr}")
                        r = json.loads(p.stdout)
                        runs[k].append(r)
                        if rep == 0:
                            d = array.array("Q")
                            with open(out_path, "rb") as fh:
                                d.frombytes(fh.read())
                            digs[(k, mode)] = d
                        os.remove(out_path)
                        print(f"  done {model:<20} {form:<6} {mode:<6} {lib} {r['version']}: {r['mb_s']:.1f} MB/s",
                              file=sys.stderr)
                for k, rs in enumerate(runs):
                    merged = dict(rs[0], mb_s=statistics.median(r["mb_s"] for r in rs))
                    results[(model, form, mode, k)] = merged
            if labels is None:
                labels = [f"{LIBS[lib]} {results[(model, form, modes[0], k)]['version']}" for k, (lib, _) in enumerate(libs)]

            # Parity: every (library, mode) against the reference library's first mode.
            ref = digs[(0, modes[0])]
            bad = {}
            for key, got in digs.items():
                if key != (0, modes[0]) and got != ref:
                    bad[key] = [i for i, (x, y) in enumerate(zip(ref, got)) if x != y]
                    bad[key] += list(range(min(len(ref), len(got)), max(len(ref), len(got))))
            truth, judge = {}, None
            idx = sorted({i for v in bad.values() for i in v if i < len(ref)})[:TIKTOKEN_CHECKS]
            if bad and repo in TIKTOKEN_ONLY:
                texts, _ = bulk_corpus.load(args.dir, form, only=idx)
                tt = kimi_hf.reference(repo)
                truth = {i: digest(tt.encode(t, allowed_special="all")) for i, t in zip(idx, texts)}
                judge = "tiktoken (the model's tokenizer)"
            elif bad and args.judge:
                jlib, _, jpy = args.judge.partition("@")
                idx_path, out_path = os.path.join(tmp, "judge.idx"), os.path.join(tmp, "judge.u64")
                with open(idx_path, "wb") as fh:
                    array.array("Q", idx).tofile(fh)
                p = subprocess.run([jpy or sys.executable, __file__, "--judge-worker", jlib, model, form, f"--dir={args.dir}",
                                    f"--indices={idx_path}", f"--digests-out={out_path}"], capture_output=True, text=True)
                if p.returncode != 0:
                    sys.exit(f"judge {args.judge}/{model}/{form} failed:\n{p.stderr}")
                got = array.array("Q")
                with open(out_path, "rb") as fh:
                    got.frombytes(fh.read())
                os.remove(idx_path)
                os.remove(out_path)
                truth = dict(zip(idx, got))
                judge = f"{LIBS[jlib]} {json.loads(p.stdout)['version']} (--judge)"
            elif bad:
                notes.append(f"{model}/{form}: libraries disagree on {len(idx)} doc(s); --judge LIB@PYTHON (e.g. tokenizers 0.x) "
                             "settles which is right")
            for mode in modes:
                status = []
                for k in range(len(libs)):
                    if (k, mode) not in bad:
                        continue
                    miss, got, who = bad[(k, mode)], digs[(k, mode)], f"{labels[k]} {mode}"
                    if truth and all(i in truth for i in miss):
                        lib_ok = all(i < len(got) and got[i] == truth[i] for i in miss)
                        ref_ok = sum(ref[i] == truth[i] for i in miss)
                        notes.append(f"{model}/{form}/{mode}: {who} differs from {labels[0]} {modes[0]} on {len(miss)} doc(s); "
                                     f"against {judge} {who} {'matches' if lib_ok else 'DIFFERS'}, "
                                     f"{labels[0]} matches {ref_ok}/{len(miss)}")
                        status.append("PASS*" if lib_ok else f"{who} FAIL")
                    else:
                        status.append(f"{who} FAIL({len(miss)})")
                fails = [s for s in status if s != "PASS*"]
                ok &= not fails
                r0 = results[(model, form, mode, 0)]
                row = [model, form, mode, f"{r0['n']:,}", f"{r0['bytes'] / 1e9:.2f} GB", f"{r0['tokens'] / r0['n']:,.0f}"]
                speeds = [results[(model, form, mode, k)]["mb_s"] for k in range(len(libs))]
                row += [f"{v:.1f}" for v in speeds]
                ft = next((k for k, (lib, _) in enumerate(libs) if lib == "ft"), None)
                if ft is not None:
                    row += [f"{speeds[ft] / speeds[k]:.2f}x" for k in range(len(libs)) if k != ft]
                row.append(", ".join(fails) if fails else ("PASS*" if status else "PASS"))
                rows.append(row)
    os.rmdir(tmp)

    ft = next((k for k, (lib, _) in enumerate(libs) if lib == "ft"), None)
    hdr = ["model", "form", "mode", "docs", "size", "tok/doc"] + [f"{lab} MB/s" for lab in labels]
    if ft is not None:
        hdr += [f"ft vs {labels[k]}" for k in range(len(libs)) if k != ft]
    print(f"corpus: {meta['bytes'] / 1e9:.2f} GB ({mix}); batch mode: ~{args.batch_mb:g} MB per call")
    print_table(hdr + ["parity"], rows)
    for n in notes:
        print("  * " + n)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"|".join([m, f, md, labels[k]]): v for (m, f, md, k), v in results.items()}, fh)
    if not ok:
        sys.exit("PARITY FAILURE")


if __name__ == "__main__":
    main()
