# ⚡ fastokens

fastokens is a fast [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokenizer for use with
popular open-weight LLMs, built on top of a high-performance Rust backend.

```python
from fastokens import Tokenizer

tokenizer = Tokenizer.from_model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
tokens = tokenizer.encode("Hello, world!")
assert tokens == [22177, 1044, 4304, 1033]
```

`fastokens` can be installed from source:
```
git clone https://github.com/atero-ai/fast-tokens
uv pip install fast-tokens/python
```

The Python API lives in the `python` directory. To use `fastokens` as a drop-in replacement with
[transformers](https://github.com/huggingface/transformers), see the
[patching example](#using-with-transformers) below.


## Performance

`fastokens` on average achieves a 4.5x faster tokenization compared to the `tokenizers` library.
The gap widens as prompt sizes scale, as shown in the graphs below.

![NVIDIA Nemotron buckets](assets/nvidia_nemotron_buckets_plots.png)

![DeepSeek V3 buckets](assets/deepseekv3_buckets_plot.png)

Faster tokenization directly impacts live workloads. Tested using SGLang's benchmark suite, `fastokens` reduces time-to-first-token (TTFT) across prompt sizes:

![sglang benchmark TTFT comparison](assets/sglang_benchmark_ttft_comparison.png)

The speedup comes from four categories of optimization:

1) **An alternative BPE algorithm** from [GitHub's rust-gems](https://github.com/github/rust-gems/tree/main/crates/bpe).
   The standard BPE implementation uses a priority-queue-based merge algorithm that iteratively
   merges the highest-priority pair in a doubly-linked list — O(n log n) per word with significant
   bookkeeping. `fastokens` replaces this with a greedy left-to-right scan backed by an
   Aho-Corasick automaton over the full vocabulary. A precomputed compatibility check determines
   whether two adjacent tokens are consistent with what BPE would produce, and a next-prefix map
   allows O(1) backtracking when the greedy choice fails. The result: no priority queue, no
   linked-list mutations, and cache-friendly memory access.

2) **Parallelization of tokenization across CPU cores.**
   Pre-tokenization often produces thousands of splits (one per regex match). `fastokens` processes
   these splits in parallel using Rayon, chunking them into groups of 16,384 for a good balance
   between parallelism overhead and work granularity. A callback-based `for_each_match` API on the
   `Pattern` trait avoids collecting intermediate results into a `Vec`. Where possible, a standard
   `regex` DFA is used in place of `fancy-regex`, avoiding the overhead of lookahead/lookbehind
   support when the pattern doesn't need it.

3) **A faster cache.**
   The standard BPE caches intermediate `Word` representations and must iterate through a
   linked-list structure to extract token IDs on every hit. `fastokens` caches the final
   `Vec<u32>` of token IDs directly — a cache hit returns exactly what the caller needs with no
   post-processing.

4) **Reduced heap string allocations.**
   All normalised content is stored in a single `String` buffer, with splits represented as
   `Range<usize>` indices rather than separately heap-allocated strings. For a document that
   produces 1,000 splits, that's 1 allocation instead of 1,000. The normalization step returns
   `Cow<'a, str>`, so when normalization is a no-op (the common case), zero additional allocations
   are made.

Note that `fastokens` is focused on inference and does not support all features of `tokenizers`.
In particular, decoding (converting tokens back to text), additional encoding outputs, and some
normalizers/pretokenizers are not available. The original `tokenizers` package is used as a
fallback for such cases, which we plan to make faster in the future.


## Tested models

The following models have been tested, but `fastokens` should generally work with most BPE tokenizers supported by the `transformers` library:

- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `openai/gpt-oss-120b`
- `deepseek-ai/DeepSeek-V3.2`
- `deepseek-ai/DeepSeek-V3`
- `deepseek-ai/DeepSeek-R1`
- `Qwen/Qwen3-Next-80B-A3B-Thinking`
- `Qwen/Qwen3-Next-80B-A3B-Instruct`
- `Qwen/Qwen3-235B-A22B-Instruct-2507`
- `Qwen/Qwen3.5-397B-A17B`
- `MiniMaxAI/MiniMax-M2.1`
- `MiniMaxAI/MiniMax-M2.5`
- `mistralai/Devstral-Small-2-24B-Instruct-2512`
- `zai-org/GLM-4.7`
- `zai-org/GLM-5`


## Usage

### Using with transformers

Note that it currently works with transformers 4.57.1 (the version used by current sglang).

```python
import fastokens
fastokens.patch_transformers()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
tokens = tokenizer("Hello, world!", fast=True)
assert tokens["input_ids"] == [22177, 1044, 4304, 1033]
```


### Using with sglang

Note that it is currently only supported with the `/completions` API. We plan to add support for more features in the future.

```shell
# Clone sglang v0.5.9. You can also use v0.5.8 here.
git clone https://github.com/sgl-project/sglang --branch v0.5.9
cd sglang
git apply /path/to/fastokens/patches/sglang-patch-minimal.patch
```
