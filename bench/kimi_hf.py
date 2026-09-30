"""Kimi (tiktoken-only) models for HuggingFace ``tokenizers`` benchmarks.

Moonshot ships Kimi as a bare ``tiktoken.model`` plus ``tokenization_kimi.py``, which
``tokenizers`` cannot load. :func:`hf_tokenizer_json` converts it to a
``tokenizer.json`` the way ``transformers``' ``TikTokenConverter`` does, so HF can
be benchmarked on the same vocabulary; :func:`reference` builds the model's own
tokenizer (tiktoken), which remains the ground truth for parity.

The conversion is byte-level BPE whose merges are every (left, right) split of each
rank with both halves ranks, ordered by the merged rank, with ``ignore_merges``
(tiktoken looks a whole piece up before merging). Special tokens follow
``tokenization_kimi.py``: 256 reserved ids after the ranks, named from
``tokenizer_config.json`` or ``<|reserved_token_{id}|>``.
"""

from __future__ import annotations

import json
import os

#: ``TikTokenTokenizer.pat_str`` from ``tokenization_kimi.py``, verbatim.
PAT_STR = "|".join(
    [
        r"""[\p{Han}]+""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

RESERVED = 256


def _files(repo: str) -> tuple[str, str]:
    from huggingface_hub import hf_hub_download

    return (
        hf_hub_download(repo, "tiktoken.model"),
        hf_hub_download(repo, "tokenizer_config.json"),
    )


def _specials(n_ranks: int, config_path: str) -> dict[str, int]:
    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    named = {int(k): v["content"] for k, v in cfg.get("added_tokens_decoder", {}).items()}
    return {named.get(i, f"<|reserved_token_{i}|>"): i for i in range(n_ranks, n_ranks + RESERVED)}


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def hf_tokenizer_json(repo: str = "moonshotai/Kimi-K3") -> str:
    """Path to a converted ``tokenizer.json`` for ``repo`` (built once, then cached)."""
    from tiktoken.load import load_tiktoken_bpe

    cache = os.path.join(os.path.expanduser("~/.cache/fastokens-bench"), repo.replace("/", "--"))
    out = os.path.join(cache, "tokenizer.json")
    if os.path.exists(out):
        return out
    model_path, config_path = _files(repo)
    ranks = load_tiktoken_bpe(model_path)
    enc = _bytes_to_unicode()

    def s(b: bytes) -> str:
        return "".join(enc[x] for x in b)

    vocab, merges = {}, []
    for tok, rank in ranks.items():
        vocab[s(tok)] = rank
        local = [(tok[:i], tok[i:], rank) for i in range(1, len(tok)) if tok[:i] in ranks and tok[i:] in ranks]
        local.sort(key=lambda x: (ranks[x[0]], ranks[x[1]]))
        merges.extend(local)
    merges.sort(key=lambda x: x[2])
    added = [
        {"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        for c, i in _specials(len(ranks), config_path).items()
    ]
    doc = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": PAT_STR}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
            ],
        },
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
            "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
            "vocab": vocab, "merges": [[s(l), s(r)] for l, r, _ in merges],
        },
    }
    os.makedirs(cache, exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)
    os.replace(tmp, out)
    return out


def reference(repo: str = "moonshotai/Kimi-K3"):
    """The model's own tokenizer: tiktoken with Kimi's pattern and special tokens.
    Encode with ``allowed_special="all"`` to match ``tokenizers``' added-token handling."""
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    model_path, config_path = _files(repo)
    ranks = load_tiktoken_bpe(model_path)
    return tiktoken.Encoding(
        name=repo, pat_str=PAT_STR, mergeable_ranks=ranks, special_tokens=_specials(len(ranks), config_path)
    )
