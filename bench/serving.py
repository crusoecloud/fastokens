#!/usr/bin/env python3
"""fastokens vs other tokenizers: per-request latency and batch throughput.

What a serving stack calls, with ids materialized as Python lists:

  long   one ``encode`` per LongBench-v2 context (long-context prompts, ~0.1-10 MB)
  chat   one ``encode`` per ShareGPT conversation (chat-sized prompts)
  batch  ``encode_batch`` over ShareGPT conversations, 256 per call

Libraries (``--libs``, the first is the reference for parity and speedups):

  hf  HuggingFace ``tokenizers``: ``encode(s, add_special_tokens=False).ids``,
      ``encode_batch``
  gt  ``gigatoken``: ``encode(s).tolist()`` (its ``encode`` returns a NumPy
      array), ``encode_batch_list``
  ft  fastokens: ``encode(s, False, False).ids``, ``encode_batch``

An entry ``LIB@PYTHON`` runs that library's workers under another interpreter, so
two versions of one package can be compared in one table, e.g.
``--libs hf,hf@/venv-0.23/bin/python,gt,ft``. Every library runs with its own
default threading.

Each (library, model) runs in a fresh subprocess, so thread pools and caches never
leak between measurements. A worker first warms up on documents disjoint from the
timed ones, then encodes every timed document exactly once — nothing timed was
ever seen before, so no cache is flattered by repeats.

Parity: every worker returns a digest of each document's ids and the parent checks
each library's against the reference's. For Kimi, whose HF tokenizer is converted
from its tiktoken model (see ``kimi_hf.py``), any disagreement is settled by the
model's own tokenizer (tiktoken).

    pip install "tokenizers==1.0.0rc2" tiktoken huggingface_hub
    maturin develop --release    # the fastokens extension, from the repo root
    python bench/serving.py
    python bench/serving.py --repeat 3      # medians over 3 runs per library

The Rust-level counterpart (no Python on either side) is `bench/rust`.
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
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kimi_hf  # noqa: E402

#: name -> Hub repo. The HF side loads the repo's ``tokenizer.json`` (Kimi: converted).
MODELS = {
    "GLM-5.3": "zai-org/GLM-5.3",
    "Kimi-K3": "moonshotai/Kimi-K3",
    "DeepSeek-V4.1-Flash": "deepseek-ai/DeepSeek-V4.1-Flash",
}
TIKTOKEN_ONLY = {"moonshotai/Kimi-K3"}

#: ``--libs`` name -> label prefix in the tables.
LIBS = {"hf": "HF", "gt": "gigatoken", "ft": "fastokens"}


def hf_json(repo: str) -> str:
    if repo in TIKTOKEN_ONLY:
        return kimi_hf.hf_tokenizer_json(repo)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, "tokenizer.json")


def load_data(args):
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download("zai-org/LongBench-v2", "data.json", repo_type="dataset"), encoding="utf-8") as fh:
        long_docs = [d["context"] for d in json.load(fh) if d.get("context")]
    with open(hf_hub_download("RyokoAI/ShareGPT52K", "sg_90k_part1.json", repo_type="dataset"), encoding="utf-8") as fh:
        chats = []
        for item in json.load(fh):
            t = "\n\n".join(m["value"] for m in item.get("conversations") or [] if m.get("value"))
            if t:
                chats.append(t)
    # Timed sets first; the warm-up sets are disjoint slices after them.
    return {
        "long": long_docs[: args.long_n],
        "long_warm": long_docs[args.long_n : args.long_n + args.long_warm],
        "chat": chats[: args.chat_n],
        "batch": chats[args.chat_n : args.chat_n + args.batch_n],
        "chat_warm": chats[args.chat_n + args.batch_n :][: args.chat_warm],
    }


def digest(ids) -> str:
    return hashlib.blake2b(array.array("I", ids).tobytes(), digest_size=8).hexdigest()


def load_lib(lib: str, repo: str):
    """``(version, encode, encode_batch)`` for a library: text -> ids and texts -> ids lists."""
    import importlib.metadata

    if lib == "hf":
        import tokenizers

        tok = tokenizers.Tokenizer.from_file(hf_json(repo))
        version = tokenizers.__version__
        enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
        enc_batch = lambda b: [e.ids for e in tok.encode_batch(b, add_special_tokens=False)]  # noqa: E731
    elif lib == "gt":
        import gigatoken

        # Kimi from the model's own tiktoken vocabulary (gigatoken has no reader for
        # the converted tokenizer.json's regex); the others from the same
        # tokenizer.json HF loads.
        tok = gigatoken.Tokenizer(repo if repo in TIKTOKEN_ONLY else hf_json(repo))
        version = importlib.metadata.version("gigatoken")
        enc = lambda s: tok.encode(s).tolist()  # noqa: E731
        enc_batch = tok.encode_batch_list
    else:
        from fastokens._native import Tokenizer

        tok = Tokenizer.from_model(repo)
        version = importlib.metadata.version("fastokens")
        enc = lambda s: tok.encode(s, False, False).ids  # noqa: E731
        enc_batch = lambda b: [e.ids for e in tok.encode_batch(b, False, False)]  # noqa: E731
    return version, enc, enc_batch


def worker(lib: str, model: str, args) -> dict:
    data = load_data(args)
    version, enc, enc_batch = load_lib(lib, MODELS[model])
    for t in data["long_warm"] + data["chat_warm"]:
        enc(t)
    enc_batch(data["chat_warm"][:256])

    out = {"lib": lib, "version": version, "model": model}

    def per_request(texts):
        lat, digs = [], []
        for t in texts:
            if args.gap_ms:
                time.sleep(args.gap_ms / 1e3)  # an idle gap before each request
            t0 = time.perf_counter()
            ids = enc(t)
            lat.append(time.perf_counter() - t0)
            digs.append(digest(ids))
        nbytes = sum(len(t.encode()) for t in texts)
        return {
            "n": len(texts),
            "bytes": nbytes,
            "mb_s": nbytes / sum(lat) / 1e6,
            "p50_ms": statistics.median(lat) * 1e3,
            "p90_ms": statistics.quantiles(lat, n=10)[-1] * 1e3,
            "digests": digs,
        }

    out["long"] = per_request(data["long"])
    out["chat"] = per_request(data["chat"])

    batches = [data["batch"][i : i + 256] for i in range(0, len(data["batch"]), 256)]
    results = []
    t0 = time.perf_counter()
    for b in batches:
        results.extend(enc_batch(b))
    dt = time.perf_counter() - t0
    nbytes = sum(len(t.encode()) for t in data["batch"])
    out["batch"] = {"n": len(data["batch"]), "bytes": nbytes, "mb_s": nbytes / dt / 1e6, "digests": [digest(r) for r in results]}
    return out


def print_table(hdr, rows):
    w = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
    print("  ".join(h.ljust(n) for h, n in zip(hdr, w)))
    print("  ".join("-" * n for n in w))
    for r in rows:
        print("  ".join(str(x).ljust(n) for x, n in zip(r, w)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument(
        "--libs",
        default="hf,ft",
        help="comma-separated hf (tokenizers), gt (gigatoken), ft (fastokens), each optionally "
        "LIB@PYTHON to run under another interpreter; the first is the reference",
    )
    ap.add_argument("--long-n", type=int, default=200)
    ap.add_argument("--long-warm", type=int, default=40)
    ap.add_argument("--chat-n", type=int, default=5000)
    ap.add_argument("--batch-n", type=int, default=20000)
    ap.add_argument("--chat-warm", type=int, default=2000)
    ap.add_argument("--gap-ms", type=float, default=0.0, help="idle gap before each timed request")
    ap.add_argument("--repeat", type=int, default=1, help="run each worker N times (alternating libraries) and report medians")
    ap.add_argument("--json", help="also write raw results here")
    ap.add_argument("--worker", nargs=2, metavar=("LIB", "MODEL"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        json.dump(worker(args.worker[0], args.worker[1], args), sys.stdout)
        return

    models = args.models.split(",")
    libs = []
    for item in args.libs.split(","):
        name, _, python = item.partition("@")
        if name not in LIBS:
            sys.exit(f"unknown library {name!r} (expected one of {', '.join(LIBS)})")
        libs.append((name, python or sys.executable))
    passthru = [
        f"--long-n={args.long_n}", f"--long-warm={args.long_warm}", f"--chat-n={args.chat_n}",
        f"--batch-n={args.batch_n}", f"--chat-warm={args.chat_warm}", f"--gap-ms={args.gap_ms}",
    ]
    results = {}
    for model in models:
        runs = [[] for _ in libs]
        for _ in range(args.repeat):
            for k, (lib, python) in enumerate(libs):
                p = subprocess.run([python, __file__, "--worker", lib, model, *passthru], capture_output=True, text=True)
                if p.returncode != 0:
                    sys.exit(f"worker {lib}/{model} failed:\n{p.stderr}")
                runs[k].append(json.loads(p.stdout))
                print(f"  done {model:<20} {lib} {runs[k][-1]['version']}", file=sys.stderr)
        for k, rs in enumerate(runs):
            # Parity from the first run; speed as the median over runs (the host's
            # noise hits each run differently).
            merged = rs[0]
            for sc in ("long", "chat", "batch"):
                for key in ("mb_s", "p50_ms", "p90_ms"):
                    if key in merged[sc]:
                        merged[sc][key] = statistics.median(r[sc][key] for r in rs)
            results[(model, k)] = merged

    labels = [f"{LIBS[lib]} {results[(models[0], k)]['version']}" for k, (lib, _) in enumerate(libs)]
    ft = next((k for k, (lib, _) in enumerate(libs) if lib == "ft"), None)
    others = [k for k in range(len(libs)) if k != ft]

    rows, lat_rows, notes, ok, data = [], [], [], True, None
    for model in models:
        repo = MODELS[model]
        for sc in ("long", "chat", "batch"):
            ref = results[(model, 0)][sc]["digests"]
            status = []
            for k in range(1, len(libs)):
                got = results[(model, k)][sc]["digests"]
                bad = [i for i, (x, y) in enumerate(zip(ref, got)) if x != y] + list(range(min(len(ref), len(got)), max(len(ref), len(got))))
                if not bad:
                    continue
                if repo in TIKTOKEN_ONLY:
                    data = data or load_data(args)
                    tt = kimi_hf.reference(repo)
                    truth = [digest(tt.encode(data[sc][i], allowed_special="all")) for i in bad]
                    lib_ok = all(i < len(got) and got[i] == t for i, t in zip(bad, truth))
                    ref_ok = sum(i < len(ref) and ref[i] == t for i, t in zip(bad, truth))
                    notes.append(f"{model}/{sc}: {labels[k]} differs from {labels[0]} on {len(bad)} doc(s); against tiktoken "
                                 f"(the model's tokenizer) {labels[k]} {'matches' if lib_ok else 'DIFFERS'}, "
                                 f"{labels[0]} matches {ref_ok}/{len(bad)}")
                    status.append("PASS*" if lib_ok else f"{labels[k]} FAIL")
                else:
                    status.append(f"{labels[k]} FAIL({len(bad)})")
            fails = [s for s in status if s != "PASS*"]
            parity = ", ".join(fails) if fails else ("PASS*" if status else "PASS")
            ok &= not fails
            r0 = results[(model, 0)][sc]
            row = [model, sc, str(r0["n"]), f"{r0['bytes'] / 1e6:.0f} MB"]
            row += [f"{results[(model, k)][sc]['mb_s']:.1f}" for k in range(len(libs))]
            if ft is not None:
                row += [f"{results[(model, ft)][sc]['mb_s'] / results[(model, k)][sc]['mb_s']:.2f}x" for k in others]
            row.append(parity)
            rows.append(row)
            if sc != "batch":
                lat_rows.append([model, sc] + [f"{results[(model, k)][sc]['p50_ms'] * 1e3:.1f}" for k in range(len(libs))])

    hdr = ["model", "scenario", "docs", "size"] + [f"{lab} MB/s" for lab in labels]
    if ft is not None:
        hdr += [f"ft vs {labels[k]}" for k in others]
    print_table(hdr + ["parity"], rows)
    print()
    print_table(["model", "scenario"] + [f"{lab} p50 µs" for lab in labels], lat_rows)
    for n in notes:
        print("  * " + n)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({f"{m}|{labels[k]}": v for (m, k), v in results.items()}, fh)
    if not ok:
        sys.exit("PARITY FAILURE")


if __name__ == "__main__":
    main()
