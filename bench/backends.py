"""Tokenizer backends for the comparison harness.

A *backend* wraps one tokenizer library (fastokens, HuggingFace ``tokenizers``,
``transformers``' ``AutoTokenizer``, OpenAI ``tiktoken``, …). The harness treats
them uniformly, so comparing fastokens to any other implementation — or two other
implementations to each other — is the same code path.

Adding a new library is deliberately a one-file change:

    1. Write a ``Backend`` subclass with ``available()`` and ``load()``.
    2. Return a ``LoadedTokenizer`` whose ``encode()`` yields a ``list[int]``.
    3. Append an instance to ``REGISTRY`` at the bottom.

Every method is defensive: an import failure, a model a backend cannot load, or
an encode error is reported and skipped, never fatal to the run. That is what lets
one command sweep many models × many backends and still finish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Model + backend interfaces
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """One model, addressable by every backend that can represent it."""

    name: str
    """Short display name (table rows/columns use this)."""
    hf_id: Optional[str] = None
    """HuggingFace Hub repo id, for hub-loading backends (``from_pretrained``)."""
    path: Optional[str] = None
    """Local ``tokenizer.json`` path; overrides ``hf_id`` for file-loading backends."""
    tiktoken_encoding: Optional[str] = None
    """tiktoken encoding name (e.g. ``cl100k_base``) if this model maps to one."""
    tags: set[str] = field(default_factory=set)
    """Free-form labels (e.g. ``byte_level``, ``metaspace``) for filtering."""


class LoadedTokenizer:
    """A model loaded into one backend. Subclasses implement ``encode``."""

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> Optional[str]:
        """Best-effort detokenization for diff diagnostics; ``None`` if unsupported."""
        return None


class Backend:
    """A tokenizer library. Stateless; ``load`` produces per-model tokenizers."""

    name: str = "backend"

    def available(self) -> tuple[bool, str]:
        """``(True, version)`` if importable, else ``(False, reason)``."""
        raise NotImplementedError

    def load(self, spec: ModelSpec) -> Optional[LoadedTokenizer]:
        """Load ``spec`` or return ``None`` if this backend cannot represent it."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# fastokens (this repo)
# ---------------------------------------------------------------------------
class _FastokensLoaded(LoadedTokenizer):
    def __init__(self, tok):
        self._tok = tok

    def encode(self, text, add_special_tokens):
        # native signature: encode(text, add_special_tokens, split_special_tokens)
        return list(self._tok.encode(text, add_special_tokens, False).ids)

    def decode(self, ids):
        try:
            return self._tok.decode(ids, False)
        except Exception:
            return None


class FastokensBackend(Backend):
    name = "fastokens"

    def available(self):
        try:
            import fastokens._native  # noqa: F401
        except Exception as e:  # not built / not installed
            return False, f"not importable ({e}); build it with `maturin develop` at the repo root"
        try:
            import importlib.metadata as md

            return True, md.version("fastokens")
        except Exception:
            return True, "dev"

    def load(self, spec):
        from fastokens._native import Tokenizer

        try:
            if spec.path:
                return _FastokensLoaded(Tokenizer.from_file(spec.path))
            if spec.hf_id:
                return _FastokensLoaded(Tokenizer.from_model(spec.hf_id))
        except Exception:
            return None
        return None


# ---------------------------------------------------------------------------
# HuggingFace `tokenizers` (any version — 0.x or 1.0+)
# ---------------------------------------------------------------------------
class _HFTokLoaded(LoadedTokenizer):
    def __init__(self, tok):
        self._tok = tok

    def encode(self, text, add_special_tokens):
        try:
            enc = self._tok.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            # Older/newer signatures without the kwarg.
            enc = self._tok.encode(text)
        return list(enc.ids)

    def decode(self, ids):
        try:
            return self._tok.decode(ids)
        except Exception:
            return None


class HFTokenizersBackend(Backend):
    name = "tokenizers"

    def available(self):
        try:
            import tokenizers

            return True, getattr(tokenizers, "__version__", "?")
        except Exception as e:
            return False, str(e)

    def load(self, spec):
        from tokenizers import Tokenizer

        try:
            if spec.path:
                return _HFTokLoaded(Tokenizer.from_file(spec.path))
            if spec.hf_id:
                return _HFTokLoaded(Tokenizer.from_pretrained(spec.hf_id))
        except Exception:
            return None
        return None


# ---------------------------------------------------------------------------
# transformers AutoTokenizer (the canonical reference for HF models)
# ---------------------------------------------------------------------------
class _TransformersLoaded(LoadedTokenizer):
    def __init__(self, tok):
        self._tok = tok

    def encode(self, text, add_special_tokens):
        return list(self._tok.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, ids):
        try:
            return self._tok.decode(ids, skip_special_tokens=False)
        except Exception:
            return None


class TransformersBackend(Backend):
    name = "transformers"

    def available(self):
        try:
            import transformers

            return True, transformers.__version__
        except Exception as e:
            return False, str(e)

    def load(self, spec):
        from transformers import AutoTokenizer

        # AutoTokenizer wants a repo id or a directory (not a bare tokenizer.json).
        source = spec.hf_id
        if source is None and spec.path:
            import os

            source = os.path.dirname(spec.path)
        if source is None:
            return None
        try:
            return _TransformersLoaded(
                AutoTokenizer.from_pretrained(source, use_fast=True, trust_remote_code=False)
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# OpenAI tiktoken (byte-level BPE reference; no BOS/EOS injection)
# ---------------------------------------------------------------------------
class _TiktokenLoaded(LoadedTokenizer):
    def __init__(self, enc):
        self._enc = enc

    def encode(self, text, add_special_tokens):
        # tiktoken has no notion of add_special_tokens; encode_ordinary never
        # emits special ids and never injects BOS/EOS. Compare with
        # --no-add-special for a like-for-like result.
        return list(self._enc.encode_ordinary(text))

    def decode(self, ids):
        try:
            return self._enc.decode(ids)
        except Exception:
            return None


class TiktokenBackend(Backend):
    name = "tiktoken"

    def available(self):
        try:
            import tiktoken

            return True, getattr(tiktoken, "__version__", "?")
        except Exception as e:
            return False, str(e)

    def load(self, spec):
        if not spec.tiktoken_encoding:
            return None
        import tiktoken

        try:
            return _TiktokenLoaded(tiktoken.get_encoding(spec.tiktoken_encoding))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Registry — append your backend here.
# ---------------------------------------------------------------------------
REGISTRY: list[Backend] = [
    FastokensBackend(),
    HFTokenizersBackend(),
    TransformersBackend(),
    TiktokenBackend(),
]


def by_name(names: list[str]) -> list[Backend]:
    idx = {b.name: b for b in REGISTRY}
    out = []
    for n in names:
        if n not in idx:
            raise SystemExit(f"unknown backend {n!r}; known: {', '.join(idx)}")
        out.append(idx[n])
    return out
