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

    from transformers import BatchEncoding, PreTrainedTokenizerFast

    _orig_call = PreTrainedTokenizerFast.__call__
    _orig_encode = PreTrainedTokenizerFast.encode
    _orig_encode_plus = PreTrainedTokenizerFast.encode_plus
    _orig_batch_encode_plus = PreTrainedTokenizerFast.batch_encode_plus

    _originals["__call__"] = _orig_call
    _originals["encode"] = _orig_encode
    _originals["encode_plus"] = _orig_encode_plus
    _originals["batch_encode_plus"] = _orig_batch_encode_plus

    def _patched_call(
        self,
        text=None,
        text_pair=None,
        text_target=None,
        text_pair_target=None,
        **kwargs,
    ):
        fast = kwargs.pop("fast", True)

        if (
            fast
            and isinstance(self._tokenizer, _TokenizerShim)
            and text is not None
            and text_pair is None
            and text_target is None
            and text_pair_target is None
        ):
            if isinstance(text, (list, tuple)):
                return _patched_batch_encode_plus(self, text, fast=True, **kwargs)
            if isinstance(text, str):
                return _patched_encode_plus(self, text, fast=True, **kwargs)

        return _orig_call(
            self,
            text,
            text_pair=text_pair,
            text_target=text_target,
            text_pair_target=text_pair_target,
            **kwargs,
        )

    def _patched_encode(self, text, text_pair=None, add_special_tokens=True, **kwargs):
        fast = kwargs.pop("fast", True)
        if (
            fast
            and text_pair is None
            and isinstance(self._tokenizer, _TokenizerShim)
        ):
            return self._tokenizer.encode(
                text, add_special_tokens=add_special_tokens, fast=True
            )
        return _orig_encode(
            self,
            text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

    def _patched_encode_plus(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        **kwargs,
    ):
        fast = kwargs.pop("fast", True)
        if not (
            fast
            and isinstance(text, str)
            and text_pair is None
            and not is_split_into_words
            and stride == 0
            and not return_overflowing_tokens
            and not return_special_tokens_mask
            and not return_offsets_mapping
            and isinstance(self._tokenizer, _TokenizerShim)
        ):
            return _orig_encode_plus(
                self,
                text,
                text_pair=text_pair,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )

        ids = self._tokenizer.encode(
            text, add_special_tokens=add_special_tokens, fast=True
        ).ids
        if truncation:
            limit = max_length if max_length is not None else self.model_max_length
            if limit is not None:
                ids = ids[:limit]
        n = len(ids)

        data = {"input_ids": ids}
        if return_attention_mask is not False:
            data["attention_mask"] = [1] * n
        if return_token_type_ids:
            data["token_type_ids"] = [0] * n

        if padding == "max_length" and max_length is not None:
            target = max_length
            if pad_to_multiple_of is not None:
                target = -(-target // pad_to_multiple_of) * pad_to_multiple_of
            if n < target:
                pad_id = self.pad_token_id if self.pad_token_id is not None else 0
                d = target - n
                if self.padding_side != "left":
                    data["input_ids"] = data["input_ids"] + [pad_id] * d
                    if "attention_mask" in data:
                        data["attention_mask"] = data["attention_mask"] + [0] * d
                    if "token_type_ids" in data:
                        data["token_type_ids"] = [0] * target
                else:
                    data["input_ids"] = [pad_id] * d + data["input_ids"]
                    if "attention_mask" in data:
                        data["attention_mask"] = [0] * d + data["attention_mask"]
                    if "token_type_ids" in data:
                        data["token_type_ids"] = [0] * target

        if return_length:
            data["length"] = len(data["input_ids"])

        return BatchEncoding(data, tensor_type=return_tensors)

    def _patched_batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        **kwargs,
    ):
        fast = kwargs.pop("fast", True)
        if not (
            fast
            and not is_split_into_words
            and stride == 0
            and not return_overflowing_tokens
            and not return_special_tokens_mask
            and not return_offsets_mapping
            and isinstance(self._tokenizer, _TokenizerShim)
            and all(isinstance(t, str) for t in batch_text_or_text_pairs)
        ):
            return _orig_batch_encode_plus(
                self,
                batch_text_or_text_pairs,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                is_split_into_words=is_split_into_words,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors=return_tensors,
                return_token_type_ids=return_token_type_ids,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_length=return_length,
                verbose=verbose,
                **kwargs,
            )

        encodings = [
            self._tokenizer.encode(
                t, add_special_tokens=add_special_tokens, fast=True
            )
            for t in batch_text_or_text_pairs
        ]
        batch_ids = [enc.ids for enc in encodings]

        if truncation:
            limit = max_length if max_length is not None else self.model_max_length
            if limit is not None:
                batch_ids = [ids[:limit] for ids in batch_ids]

        # Determine target padding length.
        pad_to = None
        if padding == "max_length" and max_length is not None:
            pad_to = max_length
        elif padding is True or padding == "longest":
            pad_to = max(len(ids) for ids in batch_ids) if batch_ids else 0
        if pad_to is not None and pad_to_multiple_of is not None:
            pad_to = -(-pad_to // pad_to_multiple_of) * pad_to_multiple_of

        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        pad_right = self.padding_side != "left"

        all_input_ids = []
        all_attention_mask = []
        all_token_type_ids = []
        all_lengths = []
        for ids in batch_ids:
            n = len(ids)
            mask = [1] * n
            if pad_to is not None and n < pad_to:
                d = pad_to - n
                if pad_right:
                    ids = ids + [pad_id] * d
                    mask = mask + [0] * d
                else:
                    ids = [pad_id] * d + ids
                    mask = [0] * d + mask
            all_input_ids.append(ids)
            all_attention_mask.append(mask)
            all_token_type_ids.append([0] * len(ids))
            all_lengths.append(len(ids))

        data = {"input_ids": all_input_ids}
        if return_attention_mask is not False:
            data["attention_mask"] = all_attention_mask
        if return_token_type_ids:
            data["token_type_ids"] = all_token_type_ids
        if return_length:
            data["length"] = all_lengths

        return BatchEncoding(data, tensor_type=return_tensors)

    PreTrainedTokenizerFast.__call__ = _patched_call
    PreTrainedTokenizerFast.encode = _patched_encode
    PreTrainedTokenizerFast.encode_plus = _patched_encode_plus
    PreTrainedTokenizerFast.batch_encode_plus = _patched_batch_encode_plus

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
    PreTrainedTokenizerFast.__call__ = _originals["__call__"]
    PreTrainedTokenizerFast.encode = _originals["encode"]
    PreTrainedTokenizerFast.encode_plus = _originals["encode_plus"]
    PreTrainedTokenizerFast.batch_encode_plus = _originals["batch_encode_plus"]

    _originals.clear()
    _patched = False
