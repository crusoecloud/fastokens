# fastokens - fast tokenizer library

**fastokens** implements tokenizers for some widely used LLMs, with
significantly higher performance than
[tokenizers](https://github.com/huggingface/tokenizers), although with a reduced
feature set focused on inference.

The backend is implemented in Rust for high performance. Python bindings are
included under the `python` directory.


## Features

Models that are known to work:

 * `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
 * `openai/gpt-oss-120b`
 * `deepseek-ai/DeepSeek-V3.2`
 * `Qwen/Qwen3-Next-80B-A3B-Thinking`
 * `MiniMaxAI/MiniMax-M2.1`

Here are some of the performance optimizations relative to `tokenizers`:

* An implementation of the alternative BPE algorithm from [GitHub's rust-gems](https://github.com/github/rust-gems/tree/main/crates/bpe)
* Parallelization of tokenization across CPU cores
* A faster cache
* Reduced heap string allocations

These add up to speed savings of up to 5x. The difference is more pronounced on
longer prompts (50k+ tokens).

Features of `tokenizers` that aren't available here:

* Additional encoding outputs other than the tokens themselves
* Decoding (converting tokens back to text)
* Some of the normalizers, pretokenizers etc. used by some models

The original `tokenizers` package can be used as a fallback for unsupported
features.


## Usage

To install:

```shell
git clone https://github.com/atero-ai/fastokens
uv pip install fastokens/python
```

Example usage:

```python
from fastokens import Tokenizer

tokenizer = Tokenizer.from_model("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
tokens = tokenizer.encode("Hello, world!")
assert tokens == [22177, 1044, 4304, 1033]
```

To use with [transformers](https://github.com/huggingface/transformers):

```python
# Do this before calling AutoTokenizer.from_pretrained().
#
# Note that it currently works with transformers 4.57.1 (the
# version used by current sglang).
import fastokens
fastokens.patch_transformers()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
# Pass fast=True to enable fastokens
tokens = tokenizer("Hello, world!", fast=True)
assert tokens["input_ids"] == [22177, 1044, 4304, 1033]
```


## Using with sglang

```shell
# Clone sglang v0.5.9. You can also use v0.5.8 here.
git clone https://github.com/sgl-project/sglang --branch v0.5.9
cd sglang
git apply /path/to/fastokens/patches/sglang-patch-minimal.patch
```
