import inspect
import unittest

from fastokens._compat import _TokenizerShim
from fastokens._native import Tokenizer


TOKENIZER_JSON = """
{
  "version": "1.0",
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": "",
    "end_of_word_suffix": "",
    "fuse_unk": false,
    "byte_fallback": false,
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
      "world": 15
    },
    "merges": [
      "h e",
      "he l",
      "hel l",
      "hell o",
      "w o",
      "wo r",
      "wor l",
      "worl d"
    ]
  }
}
"""

INPUTS = ["hello", "hello world", "world"]


def _ids(encodings):
    return [encoding.ids for encoding in encodings]


class NativeAsyncStubTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_encode_batch_matches_sync(self) -> None:
        tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)

        sync_encodings = tokenizer.encode_batch(INPUTS)
        awaitable = tokenizer.async_encode_batch(INPUTS)
        async_encodings = await awaitable

        self.assertTrue(inspect.isawaitable(awaitable))
        self.assertEqual(_ids(sync_encodings), _ids(async_encodings))
        self.assertEqual(sync_encodings[1].ids, [11, 4, 15])

    async def test_async_decode_batch_matches_sync(self) -> None:
        tokenizer = Tokenizer.from_json_str(TOKENIZER_JSON)
        encoded = tokenizer.encode_batch(INPUTS)
        batch_ids = [encoding.ids for encoding in encoded]

        sync_decoded = tokenizer.decode_batch(batch_ids)
        awaitable = tokenizer.async_decode_batch(batch_ids)
        async_decoded = await awaitable

        self.assertTrue(inspect.isawaitable(awaitable))
        self.assertEqual(sync_decoded, async_decoded)
        self.assertEqual(async_decoded, INPUTS)


class CompatAsyncStubTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_shim_methods_are_usable(self) -> None:
        tokenizer = _TokenizerShim.from_str(TOKENIZER_JSON)

        encode_awaitable = tokenizer.async_encode_batch(INPUTS)
        async_encodings = await encode_awaitable
        batch_ids = [encoding.ids for encoding in async_encodings]
        decode_awaitable = tokenizer.async_decode_batch(batch_ids)
        async_decoded = await decode_awaitable

        self.assertTrue(inspect.isawaitable(encode_awaitable))
        self.assertTrue(inspect.isawaitable(decode_awaitable))
        self.assertEqual(batch_ids, [[11], [11, 4, 15], [15]])
        self.assertEqual(async_decoded, INPUTS)


if __name__ == "__main__":
    unittest.main()
