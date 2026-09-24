# ⚡ fastokens

fastokens is a fast [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer for use with
popular open-weight LLMs, built on top of a high-performance Rust backend. It loads both the
HuggingFace `tokenizer.json` format and [tiktoken](https://github.com/openai/tiktoken) model files.

`fastokens` publishes prebuilt wheels (Linux, macOS, and Windows; Python 3.9+), so the simplest way
to install it is from PyPI:
```
uv pip install fastokens        # or: pip install fastokens
```

To build from source instead (requires a [Rust toolchain](https://rustup.rs)), clone the repository
and install its **root** directory. The `pyproject.toml` (a [maturin](https://github.com/PyO3/maturin)
project) lives at the repo root:
```
git clone https://github.com/crusoecloud/fastokens
uv pip install ./fastokens
```

To use `fastokens` as a drop-in replacement with
[transformers](https://github.com/huggingface/transformers), or with [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo), see the
[usage examples](#usage) below.

## Performance

`fastokens` on average achieves a 10x+ faster tokenization compared to the `tokenizers` library.
The gap widens as prompt sizes scale, as shown in the graphs below.

![OSS Speedup on various processors](assets/speedup_oss.png)

![Average Speedup](assets/speedup_average.png)

Faster tokenization directly impacts live workloads. Tested using SGLang's benchmark suite, `fastokens` reduces time-to-first-token (TTFT) across prompt sizes:

![TTFT P50 comparison](assets/ttft_p50.png)

Note that `fastokens` is focused on inference and does not support all features of `tokenizers`.
In particular, additional encoding outputs, and some normalizers/pretokenizers are not available.

## Tested models

Generally fastokens supports all models (Qwen's, Kimi's, Minimax, Gemma, DeepSeek were all families that verified and tested).
There might be exceptions though - we suggest to verify your model with the verification script.


## Usage

### Using with transformers

Supports transformers v4 (e.g. 4.57.1 used by current sglang) and v5+ (e.g. 5.3.0).

```python
import fastokens
fastokens.patch_transformers()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
tokens = tokenizer("Hello, world!")
assert tokens["input_ids"] == [22177, 1044, 4304, 1033]
```

#### Extending the vocabulary

`add_tokens` / `add_special_tokens` work as they do on a `tokenizers` backend,
assigning the same ids. This is what serving stacks call to reconcile a
checkpoint whose embedding matrix is padded above the tokenizer's token count:
placeholder tokens are appended until the two agree.

```python
tokenizer.add_tokens([f"<|padding_token_{i}|>" for i in range(num_embeddings - len(tokenizer))])
assert len(tokenizer) == num_embeddings
```

Passing `split_special_tokens=True` encodes special tokens found in the text as
ordinary text instead of as control-token ids, so a control token appearing in
untrusted input cannot be injected. The rest of the added vocabulary still
matches; re-register those tokens as special first to cover them too.

### Standalone usage

```python
from fastokens._native import Tokenizer
tokenizer = Tokenizer.from_model("deepseek-ai/DeepSeek-V3.2")
tokens = tokenizer.encode("A very long prompt that is now lightning fast.")
```

### Loading a tiktoken model

`fastokens` can also load [tiktoken](https://github.com/openai/tiktoken)
model files (`tiktoken.model`, or OpenAI's `.tiktoken` files) in addition to
`tokenizer.json`.

Hub repositories that ship a bare `tiktoken.model` instead of a
`tokenizer.json` — such as Moonshot's Kimi — need nothing special:
`from_model` detects the layout and resolves the pattern and special tokens from
the repository's `tokenizer_config.json`.

```python
from fastokens import Tokenizer

tok = Tokenizer.from_model("moonshotai/Kimi-K2.6")
tok.encode("Hello, world!").ids
tok.token_to_id("<|im_end|>")   # 163586, as declared by the model
```

The same applies in Rust via `Tokenizer::from_model` /
`from_model_with_token`.

#### Loading from a file

A tiktoken model file only contains the byte-level BPE ranks — the
pre-tokenization regex (`pat_str`) and the special tokens live in companion
code, so they are supplied separately. For OpenAI's standard encodings, pass
`encoding=` to use the built-in defaults:

```python
# cl100k_base (GPT-3.5 / GPT-4) or o200k_base (GPT-4o):
tok = Tokenizer.from_tiktoken("cl100k_base.tiktoken", encoding="cl100k_base")
tok.encode("Hello, world!").ids  # matches tiktoken's encode_ordinary
```

For a model whose pattern is not a known preset, pass it explicitly:

```python
tok = Tokenizer.from_tiktoken(
    "tiktoken.model",
    pattern=r"...the model's pat_str...",
    special_tokens={"<|im_end|>": 163586, "<|im_user|>": 163587},
)
```

The same is available in Rust via `Tokenizer::from_tiktoken_file`,
`from_tiktoken_str`, and `from_tiktoken_ranks`, with `TiktokenConfig::cl100k_base()`
/ `o200k_base()` presets and `TiktokenConfig::kimi()`. Special tokens are treated
like HuggingFace added tokens (split out before the model, skippable on decode);
`from_tiktoken_ranks_with_added_tokens` takes fully-specified added tokens when
per-token `lstrip` / `rstrip` / `special` flags matter.

Kimi's special tokens are derived from the vocabulary size rather than being a
fixed table: it reserves 256 ids after the mergeable ranks, names the ones its
`tokenizer_config.json` declares, and fills the rest with
`<|reserved_token_{id}|>`. `TiktokenConfig::kimi(num_ranks, named)` reproduces
that, which is why Kimi is not a `from_preset` name — a name alone is not enough.

For the `o200k` and Kimi pattern families, pre-tokenization uses a hand-written,
parallelized Unicode scanner instead of a regex engine (its classification
tables are built from the same `regex-syntax` data the reference matcher uses,
so results are identical). This makes single-document ("1M context")
tokenization several times faster than the regex path on those models.

### Prefix cache (shared system prompts)

For serving workloads where many requests share a long prefix — a common system
prompt or a large shared context — an opt-in prefix cache tokenizes the shared
prefix once and reuses its token ids, tokenizing only each request's unique
tail. Enable it with `FASTOKENS_INPUT_CACHE=<capacity>` (number of recent
prefixes to retain) or `Tokenizer::enable_input_cache(capacity)` in Rust; it is
off by default. Reuse cuts are only ever made at hard pretoken boundaries, so
results are bit-identical to tokenizing from scratch. On a ~1M-token shared
prefix this takes per-request encoding from ~2.9 ms to ~0.6 ms; an exact repeat
reuses the whole encoding.

### Threads

A single long input is split into chunks encoded in parallel, and
`encode_batch` spreads its inputs the same way, on one shared pool of worker
threads (the calling thread takes part too). The pool defaults to one thread
per logical core, capped at 32 — past that, extra threads add cold caches and
SMT/NUMA contention rather than speed. `FASTOKENS_BPE_THREADS=<n>` sets the
size (`1` disables parallelism). Idle workers poll for new work for 50 µs
before sleeping, which bridges back-to-back calls; `FASTOKENS_SPIN_US=<µs>`
changes that (`0` sleeps at once).

### PCRE2 resource limits

PCRE2 resource limits can be set when constructing a tokenizer to guard against
pathological regex/input combinations in `tokenizer.json`:

```python
from fastokens._native import Tokenizer

tokenizer = Tokenizer.from_model(
    "deepseek-ai/DeepSeek-V3.2",
    pcre2_match_limit=1_000_000,
)
```

If a limit is reached during pre-tokenization, encoding returns an error instead
of continuing an expensive regex match. The same keyword arguments are accepted
by `Tokenizer(...)`, `Tokenizer.from_file(...)`, `Tokenizer.from_json_str(...)`,
and `fastokens.patch_transformers(...)`.

### Dynamo usage

`fastokens` is integrated with NVIDIA Dynamo's frontend, and can be used by passing the flag `--tokenizer fastokens` to the latest version (either build from source or wait for the official release, coming in the next few days).

## Acknowledgements

This library builds on the well-known and widely used Hugging Face tokenizers library and uses code written for HF tokenizers in several flows.
