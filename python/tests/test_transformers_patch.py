import json
import tempfile
import unittest
from pathlib import Path

import fastokens

try:
    from transformers import PreTrainedTokenizerFast
except ImportError:  # pragma: no cover - optional dependency
    PreTrainedTokenizerFast = None


TOKENIZER_JSON = {
    "version": "1.0",
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": None,
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": None,
        "continuing_subword_prefix": "",
        "end_of_word_suffix": "",
        "fuse_unk": False,
        "byte_fallback": False,
        "vocab": {
            "h": 0,
            "e": 1,
            "l": 2,
            "o": 3,
            " ": 4,
            "w": 5,
            "r": 6,
            "d": 7,
            "he": 8,
            "hel": 9,
            "hell": 10,
            "hello": 11,
            "wo": 12,
            "wor": 13,
            "worl": 14,
            "world": 15,
        },
        "merges": [
            "h e",
            "he l",
            "hel l",
            "hell o",
            "w o",
            "wo r",
            "wor l",
            "worl d",
        ],
    },
}

INPUTS = [
    "hello",
    "hello world",
    "world",
    "hello  world",
    "",
]


@unittest.skipUnless(PreTrainedTokenizerFast is not None, "transformers is not installed")
class TransformersPatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.addCleanup(fastokens.unpatch_transformers)

        tokenizer_path = Path(self._tmpdir.name) / "tokenizer.json"
        tokenizer_path.write_text(json.dumps(TOKENIZER_JSON), encoding="utf-8")
        self.tokenizer_path = tokenizer_path

    def _make_tokenizer(self, patched: bool) -> PreTrainedTokenizerFast:
        if patched:
            fastokens.patch_transformers()
        else:
            fastokens.unpatch_transformers()
        return PreTrainedTokenizerFast(tokenizer_file=str(self.tokenizer_path))

    def test_transformers_patch_matches_unpatched_behavior(self) -> None:
        baseline = self._make_tokenizer(patched=False)
        patched = self._make_tokenizer(patched=True)

        baseline_ids = [
            baseline.encode(text, add_special_tokens=False) for text in INPUTS
        ]
        patched_ids = [
            patched.encode(text, add_special_tokens=False) for text in INPUTS
        ]

        self.assertEqual(baseline_ids, patched_ids)
        self.assertEqual(
            [baseline.decode(ids) for ids in baseline_ids],
            [patched.decode(ids) for ids in patched_ids],
        )
        self.assertEqual(
            baseline.batch_decode(baseline_ids),
            patched.batch_decode(patched_ids),
        )
