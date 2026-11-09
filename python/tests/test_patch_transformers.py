"""Tests for fastokens.patch_transformers / unpatch_transformers."""

import pytest

transformers = pytest.importorskip("transformers")

from fastokens._compat import _TokenizerShim  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"


def test_live_post_processor_survives_deepcopy(tmp_path):
    """BOS/EOS added after patching must survive a Transformers tokenizer copy."""
    import copy

    import fastokens
    from tokenizers import Tokenizer, models, processors

    backend = Tokenizer(models.BPE({"a": 0, "<s>": 1, "</s>": 2}, []))
    original = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>",
    )
    original.save_pretrained(tmp_path)
    fastokens.patch_transformers()
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tmp_path)
    tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    assert tokenizer.encode("a") == [1, 0, 2]
    assert copy.deepcopy(tokenizer).encode("a") == [1, 0, 2]


@pytest.fixture(autouse=True)
def _unpatch():
    """Ensure every test starts and ends in an unpatched state."""
    import fastokens

    yield
    fastokens.unpatch_transformers()


def test_patch_swaps_backend():
    """After patching, AutoTokenizer.from_pretrained should use _TokenizerShim."""
    import fastokens

    fastokens.patch_transformers()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    assert isinstance(tok._tokenizer, _TokenizerShim), (
        f"expected _TokenizerShim, got {type(tok._tokenizer).__name__}"
    )


def test_encode_decode_through_shim():
    """Encoding and decoding should round-trip through the patched backend."""
    import fastokens

    fastokens.patch_transformers()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    text = "Hello, world!"
    ids = tok(text)["input_ids"]
    assert len(ids) > 0, "encode returned empty ids"
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert "Hello" in decoded, f"unexpected decode: {decoded!r}"


def test_decode_ignores_ids_outside_uint32():
    """Fast decode should drop invalid IDs instead of raising."""
    import fastokens

    fastokens.patch_transformers()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    valid = tok("Hello, world!")["input_ids"]
    mixed = valid[:2] + [-1, 2**32] + valid[2:]

    assert tok.decode(mixed, skip_special_tokens=True) == tok.decode(
        valid, skip_special_tokens=True
    )

    batch_valid = [valid, valid]
    batch_mixed = [mixed, valid[:1] + [-1] + valid[1:]]
    assert tok.batch_decode(batch_mixed, skip_special_tokens=True) == tok.batch_decode(
        batch_valid, skip_special_tokens=True
    )


def test_unpatch_restores_backend():
    """After unpatching, from_pretrained should return the original backend."""
    import fastokens

    fastokens.patch_transformers()
    fastokens.unpatch_transformers()

    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    assert not isinstance(tok._tokenizer, _TokenizerShim), (
        "backend should be original tokenizers.Tokenizer after unpatch"
    )


PADDING_TOKENS = [f"<|padding_token_{i}|>" for i in range(4)]


def test_add_tokens_matches_unpatched_backend():
    """Padding a tokenizer up to a checkpoint's embedding count.

    Checkpoints whose embedding matrix is padded above the tokenizer's token
    count are squared up by appending placeholder tokens until the two agree.
    The loader then asserts the vocabulary actually grew, so a backend that
    ignores the request cannot serve the model at all.
    """

    def pad(tokenizer):
        added = tokenizer.add_tokens(PADDING_TOKENS)
        ids = [tokenizer.convert_tokens_to_ids(t) for t in PADDING_TOKENS]
        return added, len(tokenizer), ids

    import fastokens

    reference = pad(transformers.AutoTokenizer.from_pretrained(MODEL))

    fastokens.patch_transformers()
    patched = pad(transformers.AutoTokenizer.from_pretrained(MODEL))

    assert patched == reference, f"expected {reference}, got {patched}"


def test_split_special_tokens_matches_unpatched_backend():
    """Control tokens in untrusted text must not become control-token ids."""
    import fastokens

    text = "<|im_start|>user\n<think>hello<|im_end|>"

    def encode(tokenizer, split):
        return tokenizer(text, add_special_tokens=False, split_special_tokens=split)[
            "input_ids"
        ]

    unpatched = transformers.AutoTokenizer.from_pretrained(MODEL)
    reference = {split: encode(unpatched, split) for split in (False, True)}
    assert reference[False] != reference[True], "model must have special tokens to split"

    fastokens.patch_transformers()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)

    for split, expected in reference.items():
        assert encode(tok, split) == expected, f"mismatch for split_special_tokens={split}"
