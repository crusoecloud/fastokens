from fastokens._native import Tokenizer

__all__ = ["Tokenizer", "patch_transformers", "unpatch_transformers"]


_patched = False
_originals: dict = {}


def patch_transformers() -> None:
    """
    Monkey-patch ``tokenizers.Tokenizer`` so that the
    ``transformers`` library uses fastokens for encoding.

    Call this before any ``AutoTokenizer.from_pretrained``
    invocation::

        import fastokens
        fastokens.patch_transformers()

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B"
        )
    """
    global _patched
    if _patched:
        return

    from fastokens._compat import _TokenizerShim
    from fastokens._native import DecodeStream

    import transformers.tokenization_utils_fast as _tuf
    import tokenizers.decoders as _td

    _originals["TokenizerFast"] = _tuf.TokenizerFast
    _tuf.TokenizerFast = _TokenizerShim

    # Replace tokenizers.decoders.DecodeStream so that vLLM's
    # FastIncrementalDetokenizer receives a stream that accepts our
    # _TokenizerShim rather than requiring a tokenizers.Tokenizer.
    _originals["DecodeStream"] = _td.DecodeStream
    _td.DecodeStream = DecodeStream

    _patched = True


def unpatch_transformers() -> None:
    """
    Reverse the monkey-patching applied by :func:`patch_transformers`,
    restoring the ``transformers`` library to its original state.
    """
    global _patched
    if not _patched:
        return

    import transformers.tokenization_utils_fast as _tuf
    import tokenizers.decoders as _td

    _tuf.TokenizerFast = _originals["TokenizerFast"]
    _td.DecodeStream = _originals["DecodeStream"]

    _originals.clear()
    _patched = False
