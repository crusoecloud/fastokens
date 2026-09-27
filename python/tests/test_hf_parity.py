"""Parity of the native binding's length handling with HuggingFace `tokenizers`.

`transformers` drives truncation, padding and the special-tokens mask through
these methods, so each must produce what the HF backend it replaces produces.
Uses a synthetic tokenizer with a BOS/EOS template, so no network is needed.
"""

import json

import pytest

from fastokens._native import Tokenizer

hf_tokenizers = pytest.importorskip("tokenizers")

VOCAB = {chr(code): code - 32 for code in range(32, 127)}  # " " .. "~"
BOS, EOS, PAD = len(VOCAB), len(VOCAB) + 1, len(VOCAB) + 2


def special(id, content):
    return {
        "id": id,
        "content": content,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
        "normalized": False,
        "special": True,
    }


def tokenizer_json():
    return json.dumps({
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [special(BOS, "<s>"), special(EOS, "</s>"), special(PAD, "<pad>")],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
            ],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [BOS], "tokens": ["<s>"]},
                "</s>": {"id": "</s>", "ids": [EOS], "tokens": ["</s>"]},
            },
        },
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": VOCAB,
            "merges": [],
        },
    })


@pytest.fixture
def pair():
    return Tokenizer.from_json_str(tokenizer_json()), hf_tokenizers.Tokenizer.from_str(
        tokenizer_json()
    )


# FIELDS leaves out special_tokens_mask, which has its own tests, so a mask
# bug cannot hide a truncation/padding regression (and vice versa).
FIELDS = ("ids", "attention_mask", "type_ids")
TEXTS = ["", "a", "hello", "hello world, this is longer"]


def fields(encoding, names=FIELDS):
    return {name: list(getattr(encoding, name)) for name in names}


@pytest.mark.parametrize("add_special_tokens", [False, True])
@pytest.mark.parametrize("text", TEXTS)
def test_special_tokens_mask_matches_hf(pair, text, add_special_tokens):
    """HF flags template-inserted BOS/EOS in the mask."""
    ours, hf = pair
    names = FIELDS + ("special_tokens_mask",)
    assert fields(ours.encode(text, add_special_tokens=add_special_tokens), names) == fields(
        hf.encode(text, add_special_tokens=add_special_tokens), names
    )


def test_special_tokens_mask_flags_padding_like_hf(pair):
    """HF flags padding positions in the mask too."""
    ours, hf = pair
    for tok in pair:
        tok.enable_padding(pad_id=PAD, pad_token="<pad>")
    names = FIELDS + ("special_tokens_mask",)
    for add in (False, True):
        got = ours.encode_batch(TEXTS, add_special_tokens=add)
        want = hf.encode_batch(TEXTS, add_special_tokens=add)
        assert [fields(e, names) for e in got] == [fields(e, names) for e in want]


@pytest.mark.parametrize("direction", ["right", "left"])
@pytest.mark.parametrize("max_length", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_truncation_matches_hf(pair, direction, max_length, add_special_tokens):
    """HF truncates the content and then adds BOS/EOS, so they survive."""
    ours, hf = pair
    ours.enable_truncation(max_length, direction=direction)
    hf.enable_truncation(max_length, direction=direction)
    for text in TEXTS:
        got = ours.encode(text, add_special_tokens=add_special_tokens)
        want = hf.encode(text, add_special_tokens=add_special_tokens)
        assert fields(got) == fields(want), text


@pytest.mark.parametrize("direction", ["right", "left"])
@pytest.mark.parametrize("length,multiple", [(None, None), (12, None), (None, 8), (3, 4)])
def test_padding_matches_hf(pair, direction, length, multiple):
    ours, hf = pair
    kwargs = {"direction": direction, "pad_id": PAD, "pad_token": "<pad>"}
    if length is not None:
        kwargs["length"] = length
    if multiple is not None:
        kwargs["pad_to_multiple_of"] = multiple
    ours.enable_padding(**kwargs)
    hf.enable_padding(**kwargs)
    for add in (False, True):
        for text in TEXTS:
            assert fields(ours.encode(text, add_special_tokens=add)) == fields(
                hf.encode(text, add_special_tokens=add)
            ), text
        got = ours.encode_batch(TEXTS, add_special_tokens=add)
        want = hf.encode_batch(TEXTS, add_special_tokens=add)
        assert [fields(e) for e in got] == [fields(e) for e in want]


def test_truncation_and_padding_together_match_hf(pair):
    ours, hf = pair
    for tok in pair:
        tok.enable_truncation(6)
        tok.enable_padding(pad_id=PAD, pad_token="<pad>")
    for add in (False, True):
        got = ours.encode_batch(TEXTS, add_special_tokens=add)
        want = hf.encode_batch(TEXTS, add_special_tokens=add)
        assert [fields(e) for e in got] == [fields(e) for e in want]


def test_num_special_tokens_to_add_matches_hf(pair):
    ours, hf = pair
    assert ours.num_special_tokens_to_add(False) == hf.num_special_tokens_to_add(False)


@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_encode_batch_flat_matches_hf_with_truncation(pair, add_special_tokens):
    np = pytest.importorskip("numpy")
    ours, hf = pair
    for tok in pair:
        tok.enable_truncation(4)
    ids, offsets = ours.encode_batch_flat(TEXTS, add_special_tokens=add_special_tokens)
    ids = np.frombuffer(ids, np.uint32)
    offsets = np.frombuffer(offsets, np.uint64)
    got = [ids[offsets[i] : offsets[i + 1]].tolist() for i in range(len(TEXTS))]
    want = [e.ids for e in hf.encode_batch(TEXTS, add_special_tokens=add_special_tokens)]
    assert got == want


def with_config(**changes):
    config = json.loads(tokenizer_json())
    config.update(changes)
    return json.dumps(config)


@pytest.mark.parametrize(
    "changes",
    [
        {
            "truncation": {
                "direction": "Right",
                "max_length": 4,
                "strategy": "LongestFirst",
                "stride": 0,
            }
        },
        {
            "padding": {
                "strategy": {"Fixed": 12},
                "direction": "Left",
                "pad_to_multiple_of": None,
                "pad_id": PAD,
                "pad_type_id": 0,
                "pad_token": "<pad>",
            }
        },
    ],
    ids=["truncation", "padding"],
)
def test_truncation_and_padding_declared_in_tokenizer_json_match_hf(changes):
    """HF applies `truncation` / `padding` blocks from tokenizer.json."""
    ours = Tokenizer.from_json_str(with_config(**changes))
    hf = hf_tokenizers.Tokenizer.from_str(with_config(**changes))
    for add in (False, True):
        for text in TEXTS:
            assert fields(ours.encode(text, add_special_tokens=add)) == fields(
                hf.encode(text, add_special_tokens=add)
            ), text


def test_template_type_ids_match_hf():
    """TemplateProcessing's `type_id`s end up in the encoding's type_ids."""
    config = json.loads(tokenizer_json())
    for piece in config["post_processor"]["single"]:
        next(iter(piece.values()))["type_id"] = 1
    ours = Tokenizer.from_json_str(json.dumps(config))
    hf = hf_tokenizers.Tokenizer.from_str(json.dumps(config))
    for text in TEXTS:
        assert fields(ours.encode(text, add_special_tokens=True)) == fields(
            hf.encode(text, add_special_tokens=True)
        ), text


def test_only_second_truncation_matches_hf(pair):
    """With one sequence, HF only fails when `only_second` must truncate."""
    for tok in pair:
        tok.enable_truncation(4, strategy="only_second")
    for tok in pair:
        assert tok.encode("abcd", add_special_tokens=False).ids == [VOCAB[c] for c in "abcd"]
        with pytest.raises(Exception, match="(?i)second sequence"):
            tok.encode("hello", add_special_tokens=False)


def test_to_str_writes_truncation_and_padding_like_hf():
    """`to_str` writes them in tokenizer.json's format, which loads again."""
    from fastokens._compat import _TokenizerShim

    shim = _TokenizerShim(tokenizer_json())
    hf = hf_tokenizers.Tokenizer.from_str(tokenizer_json())
    for tok in (shim, hf):
        tok.enable_truncation(4, direction="left")
        tok.enable_padding(length=8, pad_id=PAD, pad_token="<pad>")
    ours, want = json.loads(shim.to_str()), json.loads(hf.to_str())
    assert (ours["truncation"], ours["padding"]) == (want["truncation"], want["padding"])
    reloaded = _TokenizerShim(shim.to_str())
    assert (reloaded.truncation, reloaded.padding) == (shim.truncation, shim.padding)


def test_copies_keep_truncation_and_padding_disabled():
    """Copies follow the live settings, not the ones tokenizer.json declares."""
    import copy

    from fastokens._compat import _TokenizerShim

    declared = with_config(
        truncation={"direction": "Right", "max_length": 4, "strategy": "LongestFirst", "stride": 0},
        padding={
            "strategy": {"Fixed": 12},
            "direction": "Right",
            "pad_to_multiple_of": None,
            "pad_id": PAD,
            "pad_type_id": 0,
            "pad_token": "<pad>",
        },
    )
    shim = _TokenizerShim(declared)
    shim.no_truncation()
    shim.no_padding()
    for clone in (copy.deepcopy(shim), _TokenizerShim(shim)):
        assert (clone.truncation, clone.padding) == (None, None)
        assert clone.encode("hello world", add_special_tokens=False).ids == [
            VOCAB[c] for c in "hello world"
        ]


def test_reads_pickles_from_fastokens_0_3_3():
    """0.3.3 wrote truncation/padding into the JSON with HF's Python names."""
    from fastokens._compat import _TokenizerShim

    trunc = {"max_length": 4, "stride": 0, "strategy": "longest_first", "direction": "right"}
    pad = {
        "direction": "left",
        "pad_id": PAD,
        "pad_type_id": 0,
        "pad_token": "<pad>",
        "length": 12,
        "pad_to_multiple_of": None,
    }
    shim = object.__new__(_TokenizerShim)
    shim.__setstate__((with_config(truncation=trunc, padding=pad), trunc, pad, False, {}))
    hf = hf_tokenizers.Tokenizer.from_str(tokenizer_json())
    hf.enable_truncation(4)
    hf.enable_padding(direction="left", pad_id=PAD, pad_token="<pad>", length=12)
    for text in TEXTS:
        assert fields(shim.encode(text)) == fields(hf.encode(text)), text


def test_pair_encoding_is_rejected_not_ignored():
    """Pair inputs are unsupported; they must raise rather than drop the pair."""
    from fastokens._compat import _TokenizerShim

    shim = _TokenizerShim(tokenizer_json())
    with pytest.raises(NotImplementedError):
        shim.encode("hello", "world")
    with pytest.raises(NotImplementedError):
        shim.encode_batch([("hello", "world")])
    ours = Tokenizer.from_json_str(tokenizer_json())
    enc = ours.encode("hello")
    with pytest.raises(NotImplementedError):
        ours.post_process(enc, enc)
