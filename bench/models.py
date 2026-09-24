"""Default model set for the comparison harness.

Each entry is a :class:`~backends.ModelSpec`. Override on the command line with
``--models name1,name2`` (names below) or ``--models-file models.json`` (a JSON
list of ``{"name", "hf_id", "path", "tiktoken_encoding", "tags"}`` objects), or
``--model-path /path/to/tokenizer.json`` for a one-off local file.

The defaults span the pre-tokenizer families that matter for correctness and
speed: cl100k (Phi-4), o200k (GPT-4o-class), the DeepSeek 3-split sequence, Kimi,
a Metaspace model (Llama-2), and pure tiktoken encodings.
"""

from __future__ import annotations

from backends import ModelSpec

DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec("deepseek-v3.2", hf_id="deepseek-ai/DeepSeek-V3.2", tags={"byte_level", "deepseek"}),
    ModelSpec("phi-4", hf_id="microsoft/phi-4", tags={"byte_level", "cl100k"}),
    ModelSpec("kimi-k2", hf_id="moonshotai/Kimi-K2-Instruct", tags={"byte_level", "kimi"}),
    ModelSpec("qwen2.5", hf_id="Qwen/Qwen2.5-7B", tags={"byte_level", "o200k_like"}),
    ModelSpec("llama-2", hf_id="NousResearch/Llama-2-7b-hf", tags={"metaspace"}),
    ModelSpec("gemma-2", hf_id="google/gemma-2-2b", tags={"metaspace"}),
    ModelSpec("gpt2", hf_id="openai-community/gpt2", tiktoken_encoding="gpt2", tags={"byte_level"}),
    # Pure tiktoken encodings (only the tiktoken backend loads these).
    ModelSpec("cl100k", tiktoken_encoding="cl100k_base", tags={"byte_level", "cl100k"}),
    ModelSpec("o200k", tiktoken_encoding="o200k_base", tags={"byte_level", "o200k"}),
]


def by_names(names: list[str]) -> list[ModelSpec]:
    idx = {m.name: m for m in DEFAULT_MODELS}
    out = []
    for n in names:
        if n in idx:
            out.append(idx[n])
        elif "/" in n:  # treat as a raw HF id
            out.append(ModelSpec(n.split("/")[-1], hf_id=n))
        else:
            raise SystemExit(f"unknown model {n!r}; known: {', '.join(idx)} (or pass an org/repo id)")
    return out


def from_file(path: str) -> list[ModelSpec]:
    import json

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [
        ModelSpec(
            name=o["name"],
            hf_id=o.get("hf_id"),
            path=o.get("path"),
            tiktoken_encoding=o.get("tiktoken_encoding"),
            tags=set(o.get("tags", [])),
        )
        for o in raw
    ]
