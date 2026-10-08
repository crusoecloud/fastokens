# Tokenizer comparison harness

A single command to compare **fastokens** against HuggingFace `tokenizers` (any
version), `transformers`' `AutoTokenizer`, OpenAI `tiktoken`, or any library you
add — for **correctness** (token-id parity, with rich diffs) and **speed**
(throughput, with ratios), across a **wide, extensible set of test inputs**.

Why Python (not a Rust bench)? Because "compare to any other repo with ease" is a
packaging problem: `pip install tokenizers==1.0.0rc2` vs `==0.22.2`, `tiktoken`,
`transformers`, … all coexist behind one uniform adapter, whereas a Rust crate
cannot depend on two versions of `tokenizers` at once. For maximum-accuracy
in-process **speed** numbers on a single library, the Rust `examples/simple_bench.rs`
remains the tool; this harness is for breadth.

## Quick start

```bash
# Prove the harness runs with nothing installed (mock backends):
python bench/compare.py --self-test

# See what's available and the defaults:
python bench/compare.py --list

# fastokens vs the canonical reference on two models, parity + speed:
python bench/compare.py --models deepseek-v3.2,phi-4 \
    --backends fastokens,transformers --reference transformers

# Parity over real long documents (read from the local HF cache, no download):
python bench/compare.py --dataset longbench:200 --mode correctness

# One-off local tokenizer.json, tokenizers-vs-tiktoken speed:
python bench/compare.py --model-path ./tokenizer.json --mode speed
```

To include the **fastokens** backend, build the Python extension once:

```bash
pip install maturin
maturin develop --release           # from the repo root
```

Other backends are optional; install whichever you want to compare against:
`pip install "tokenizers==1.0.0rc2"` (or any version), `pip install transformers`,
`pip install tiktoken`. Missing backends are reported and skipped, never fatal.

## What it checks

- **Correctness** — every non-reference backend's token ids are compared to the
  reference's, per model, over every sample. Output is a PASS / `N diff` matrix.
  For each failing pair it prints the first divergence: the token index, the ids
  on both sides, the char offset located by detokenizing the common prefix, and
  the decoded text from each side — so a mismatch is diagnosable at a glance.
- **Speed** — best-of-N throughput (MB/s) per model × backend, and which backend
  is fastest. Use `examples/simple_bench.rs` when you need in-process precision.

## The wide test set

Three sources, combined freely:

1. **Built-in categories** (no network, deterministic): `english`, `code`,
   `markdown`, `cjk`, `multilingual`, `whitespace`, `unicode_adversarial`,
   `numbers`, `urls_paths`, `edge` (empty/huge/pathological). These target exactly
   the places BPE pipelines diverge — case runs, contractions, script/CJK
   boundaries, whitespace runs, combining marks, ZWJ/format/control chars.
   Select with `--corpus english,cjk` or `--corpus all`.
2. **A directory of `.txt` files** — `--corpus-dir path/`; each file is one
   sample. Drop a file in to add a test case.
3. **Cached HF datasets** — `--dataset longbench:200` reads the dataset's JSON
   straight from your local `~/.cache/huggingface` (skipped if absent).

## Extending (one file each)

- **New library to compare**: add a `Backend` subclass in `backends.py` (implement
  `available()` and `load()`, return a `LoadedTokenizer` whose `encode()` yields
  `list[int]`), then append it to `REGISTRY`.
- **New models**: edit `DEFAULT_MODELS` in `models.py`, or pass `--models org/repo`,
  or `--models-file specs.json`.
- **New corpus category**: add an entry to `_BUILTIN` in `corpora.py` (or just use
  `--corpus-dir`).

## Reports

`--report table` (default), `--report md` (paste into a PR/issue), `--report json`.
`--add-special` toggles special-token injection (default off, for cross-library
comparability — `tiktoken` never injects specials).

## Serving benchmark (`serving.py`)

`compare.py`'s speed mode re-encodes the same samples best-of-N, which flatters
any implementation with a large word cache. `serving.py` instead measures what a
serving stack sees, with ids materialized as Python lists:

- **long** — one `encode` per LongBench-v2 context (long-context prompts);
- **chat** — one `encode` per ShareGPT conversation (chat-sized prompts);
- **batch** — `encode_batch` over ShareGPT conversations, 256 per call.

Every (library, model) pair runs in a fresh process, warms up on documents
disjoint from the timed ones, and encodes each timed document exactly once. Token
ids are checked for parity on every document. The defaults are GLM-5.3, Kimi-K3
and DeepSeek-V4.1-Flash against `tokenizers==1.0.0rc2`:

```bash
pip install "tokenizers==1.0.0rc2" tiktoken huggingface_hub
python bench/serving.py                     # all three models, both libraries
python bench/serving.py --gap-ms 5          # with an idle gap before each request
```

`--libs` picks the libraries (the first is the reference for parity and
speedups): `hf` (`tokenizers`), `gt` (`gigatoken`) and `ft` (fastokens). An entry
`LIB@PYTHON` runs that library under another interpreter, so two versions of one
package can share a table — e.g. against the latest 0.x `tokenizers` too:

```bash
python -m venv /tmp/tok0 && /tmp/tok0/bin/pip install "tokenizers<1" tiktoken huggingface_hub
pip install gigatoken
python bench/serving.py --libs hf,hf@/tmp/tok0/bin/python,gt,ft --repeat 3
```

Kimi ships only a `tiktoken.model`, so `kimi_hf.py` converts it to a
`tokenizer.json` for HF (the way `transformers`' `TikTokenConverter` does). The
model's own tokenizer, tiktoken, stays the reference: a document where HF and
fastokens disagree passes only if fastokens matches tiktoken.

### Rust level (`bench/rust`)

The same scenarios, data and rules with no Python in the way: fastokens'
`Tokenizer::encode` / `encode_batch` against the `tokenizers` 1.0 crate's
`PipelineTokenizer::encode_into` (single inputs, its fastest path) and batch
`encode` (parallel), each at its default thread count. A standalone crate, so
the main workspace keeps its own `tokenizers` dev-dependency:

```bash
cd bench/rust
cargo run --release -- --repeat 3          # medians over 3 runs per library
cargo run --release -- --models GLM-5.3 --long-n 50
cargo run --release -- --libs hf,hf0,ft    # + tokenizers 0.23.2 (`encode_fast`)
```

`--libs gt --gt-worker PATH` adds gigatoken through an external worker binary
speaking the same worker protocol. gigatoken builds only on nightly Rust, so it
cannot be a dependency of this crate; `bench/gigatoken-worker` is that worker,
built over gigatoken's source release (`fetch.sh` downloads and checksums it):

```bash
(cd bench/gigatoken-worker && ./fetch.sh && cargo build --release)  # nightly via rust-toolchain.toml
cd bench/rust
cargo run --release -- --libs hf,gt,ft --repeat 3 \
    --gt-worker ../gigatoken-worker/target/release/gigatoken-worker
```

It reads Kimi's HF tokenizer from the conversion `serving.py` writes, so run
that (or `kimi_hf.hf_tokenizer_json()`) once first.

## Bulk benchmark (`bulk.py`)

Throughput over a multi-GB corpus (3 GB by default, `--gb`), in three forms that
are the **same bytes** cut into documents of different sizes:

- **small** — ~400 B per document (~100 tokens; ~7.4M documents at 3 GB);
- **medium** — ~8 KB (~2,000 tokens; ~375k documents);
- **large** — ~1 MB (~250,000 tokens; ~3,000 documents).

Each form is encoded two ways: **encode**, one call per document, and **batch**,
`encode_batch` over consecutive documents at ~16 MB per call (`--batch-mb`).
The corpus (`bulk_corpus.py`) is real text mixed by bytes: English web text (C4
`en`) 40%, Chinese web text (mC4 `zh`) 20%, chat (ShareGPT) 30%, long documents
(LongBench-v2) 10%. It is shuffled at the document level with a fixed seed, and
each document's length is drawn in ±50% of the target, cut just after a space
or newline (else at a character boundary). It is downloaded and built once
(~2 min) into `~/.cache/fastokens-bench/bulk-<GB>GB/`.

The libraries, calls, subprocess isolation, disjoint warm-up and parity rules
are `serving.py`'s. Only the encode calls are timed (per group of 64 documents in
encode mode), and every library and mode is checked against the reference's
first mode:

```bash
python bench/bulk.py --prepare                   # build the corpus
python bench/bulk.py --libs hf,gt,ft             # all three models, forms and modes
python bench/bulk.py --gb 1 --forms small,large --modes batch --repeat 3
python bench/bulk.py --libs hf,gt,ft --judge hf@/tmp/tok0/bin/python
```

Documents where the libraries disagree are settled by a judge: tiktoken for
Kimi (as in `serving.py`), and for the other models `--judge LIB@PYTHON` — e.g.
tokenizers 0.x in a second venv, as above. This matters on this corpus:
`tokenizers` 1.0.0rc2 mis-merges some Chinese on GLM-5.3 (`"的件"` →
`çļ` + `Ħä»¶`, a token straddling both characters, where BPE's ranks, 0.23.2,
fastokens and gigatoken all give `的` + `件`), on ~25 documents per form.
Without `--judge` those rows report the disagreement and the run exits non-zero.

The same corpus at the Rust level (build it with `--prepare` first):

```bash
cd bench/rust
cargo run --release -- bulk --libs hf,gt,ft \
    --gt-worker ../gigatoken-worker/target/release/gigatoken-worker
```

The Rust side settles disputed documents itself, with tokenizers 0.23.2
in-process (Kimi's are flagged `*` for `bulk.py`'s tiktoken check).

A full three-model run takes about 55 minutes in Python and 20 in Rust with
`hf,gt,ft`, per `--repeat`. The multi-threaded batch rows are noisy on a shared
host (a single run here once dropped to a third of its median), so prefer
`--repeat 3`. `hf0` (tokenizers 0.x, ~3 MB/s here) would take hours at 3 GB;
use a smaller `--gb` for it.
