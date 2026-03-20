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

    import transformers.tokenization_utils_fast as _tuf

    _originals["TokenizerFast"] = _tuf.TokenizerFast
    _tuf.TokenizerFast = _TokenizerShim
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
    from transformers import PreTrainedTokenizerFast

    _tuf.TokenizerFast = _originals["TokenizerFast"]
    if "__call__" in _originals:
        PreTrainedTokenizerFast.__call__ = _originals["__call__"]
    if "encode" in _originals:
        PreTrainedTokenizerFast.encode = _originals["encode"]
    if "encode_plus" in _originals:
        PreTrainedTokenizerFast.encode_plus = _originals["encode_plus"]
    if "batch_encode_plus" in _originals:
        PreTrainedTokenizerFast.batch_encode_plus = _originals["batch_encode_plus"]

    _originals.clear()
    _patched = False
