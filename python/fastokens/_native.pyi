class Tokenizer:
    """An LLM tokenizer backed by ``tokenizer.json``."""

    def __new__(cls, model: str) -> Tokenizer:
        """
        Download ``tokenizer.json`` from HuggingFace Hub for the given
        model (e.g. ``"meta-llama/Llama-3.1-8B"``) and create a tokenizer
        with it.

        (This is an alias for Tokenizer.from_model)
        """
        ...

    @staticmethod
    def from_file(path: str) -> Tokenizer:
        """Create a tokenizer from a ``tokenizer.json`` file."""
        ...

    @staticmethod
    def from_json_str(json: str) -> Tokenizer:
        """
        Create a tokenizer from a raw JSON string for
        ``tokenizer.json``.
        """
        ...

    @staticmethod
    def from_model(model: str) -> Tokenizer:
        """
        Download ``tokenizer.json`` from HuggingFace Hub for the given
        model (e.g. ``"meta-llama/Llama-3.1-8B"``) and create a tokenizer
        with it.
        """
        ...

    def encode(self, input: str) -> list[int]:
        """
        Run the full encoding pipeline: split added tokens, normalize,
        pre-tokenize, tokenize and post-process the input string.
        """
        ...
