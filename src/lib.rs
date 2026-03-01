pub mod added_tokens;
pub mod decoders;
pub mod json_structs;
pub mod models;
pub mod normalizers;
pub mod post_processors;
pub mod pre_tokenized;
pub mod pre_tokenizers;

use std::{borrow::Cow, fs, path::Path};

use hf_hub::api::sync::Api;
use rayon::prelude::*;
use serde_json::Value;

pub use self::{
    added_tokens::AddedTokens,
    json_structs::{
        AddedTokenConfig, DecoderConfig, DecoderKind, ModelConfig, ModelKind, NormalizerConfig,
        NormalizerKind, PostProcessorConfig, PostProcessorKind, PreTokenizerConfig,
        PreTokenizerKind, TokenizerJson,
    },
    models::Model,
    normalizers::{Nfc, Normalizer},
    post_processors::PostProcessor,
    pre_tokenizers::{ByteLevel, PreTokenizer, Split, SplitBehavior},
};

use self::{
    added_tokens::Segment,
    decoders::Decoder,
    pre_tokenized::{PreTokenizedString, Split as PtSplit},
};

/// Errors that can occur when constructing a [`Tokenizer`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to download tokenizer files: {0}")]
    Hub(#[from] hf_hub::api::sync::ApiError),

    #[error("failed to read tokenizer files: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to parse tokenizer files: {0}")]
    Json(#[from] serde_json::Error),

    #[error("normalizer error: {0}")]
    Normalizer(#[from] normalizers::Error),

    #[error("pre-tokenizer error: {0}")]
    PreTokenizer(#[from] pre_tokenizers::Error),

    #[error("tokenizer model building error: {0}")]
    ModelBuild(#[from] models::BuildError),

    #[error("tokenizer model error: {0}")]
    ModelTokenize(#[from] models::TokenizeError),

    #[error("post-processor error: {0}")]
    PostProcessor(#[from] post_processors::Error),

    #[error("decoder error: {0}")]
    Decoder(#[from] decoders::Error),

    #[error("decode error: {0}")]
    Decode(String),
}

/// An LLM tokenizer backed by `tokenizer.json`.
pub struct Tokenizer {
    added_tokens: Option<AddedTokens>,
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<PreTokenizer>,
    model: Model,
    post_processor: Option<PostProcessor>,
    decoder: Option<Decoder>,
}

impl Tokenizer {
    /// Build the pipeline steps from a parsed JSON config.
    fn build(json: TokenizerJson) -> Result<Self, Error> {
        let added_tokens =
            AddedTokens::from_configs(&json.added_tokens).map_err(Error::ModelBuild)?;
        let normalizer = json.normalizer.map(Normalizer::from_config).transpose()?;
        let pre_tokenizer = json
            .pre_tokenizer
            .map(PreTokenizer::from_config)
            .transpose()?;
        let model = Model::from_config(json.model).map_err(Error::ModelBuild)?;
        let post_processor = json
            .post_processor
            .map(PostProcessor::from_config)
            .transpose()?;
        let decoder = json.decoder.map(Decoder::from_config).transpose()?;

        Ok(Self {
            added_tokens,
            normalizer,
            pre_tokenizer,
            model,
            post_processor,
            decoder,
        })
    }

    /// Create a tokenizer from a raw JSON value for `tokenizer.json`.
    pub fn from_json(json: Value) -> Result<Self, Error> {
        let json: TokenizerJson = serde_json::from_value(json)?;
        Self::build(json)
    }

    /// Create a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let json: TokenizerJson = serde_json::from_str(&fs::read_to_string(path)?)?;
        Self::build(json)
    }

    /// Download `tokenizer.json` from HuggingFace Hub for the given model (e.g.
    /// `"meta-llama/Llama-3.1-8B"`) and create a tokenizer with it.
    pub fn from_model(model: &str) -> Result<Self, Error> {
        let api = Api::new()?;
        let repo = api.model(model.to_string());
        let json_path = repo.get("tokenizer.json")?;
        let json: TokenizerJson = serde_json::from_str(&fs::read_to_string(json_path)?)?;
        Self::build(json)
    }

    /// Return the normalizer, if any.
    pub fn normalizer(&self) -> Option<&Normalizer> {
        self.normalizer.as_ref()
    }

    /// Return the pre-tokenizer, if any.
    pub fn pre_tokenizer(&self) -> Option<&PreTokenizer> {
        self.pre_tokenizer.as_ref()
    }

    /// Return the post-processor, if any.
    pub fn post_processor(&self) -> Option<&PostProcessor> {
        self.post_processor.as_ref()
    }

    /// Return the tokenization model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Return the decoder, if any.
    pub fn decoder(&self) -> Option<&Decoder> {
        self.decoder.as_ref()
    }

    // ── Encoding ─────────────────────────────────────────────────────

    /// Run the full encoding pipeline: split added tokens, normalize,
    /// pre-tokenize, tokenize and post-process the input string.
    pub fn encode(&self, input: &str) -> Result<Vec<u32>, Error> {
        self.encode_with_special_tokens(input, false)
    }

    /// Run the full encoding pipeline with control over special token insertion.
    ///
    /// When `add_special_tokens` is true, the post-processor inserts special
    /// tokens (e.g. BOS/EOS) as configured in the tokenizer's post-processor.
    pub fn encode_with_special_tokens(
        &self,
        input: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, Error> {
        if input.is_empty() {
            return if add_special_tokens {
                Ok(self.post_process(Vec::new(), true))
            } else {
                Ok(Vec::new())
            };
        }

        // 1. Split on added tokens + normalize into a single buffer.
        let mut pts = self.build_pre_tokenized(input);

        // 2. Pre-tokenize (refine splits in place).
        if let Some(ref pt) = self.pre_tokenizer {
            pt.pre_tokenize(&mut pts)?;
        }

        // 3. Tokenize each text split with the model.
        let ids = pts
            .tokenize(|text| self.model.tokenize(text))
            .map_err(Error::ModelTokenize)?;

        // 4. Post-process.
        Ok(self.post_process(ids, add_special_tokens))
    }

    /// Encode a batch of inputs in parallel.
    pub fn encode_batch<S: AsRef<str> + Sync>(
        &self,
        inputs: &[S],
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<u32>>, Error> {
        inputs
            .par_iter()
            .map(|input| self.encode_with_special_tokens(input.as_ref(), add_special_tokens))
            .collect()
    }

    fn post_process(&self, ids: Vec<u32>, add_special_tokens: bool) -> Vec<u32> {
        match &self.post_processor {
            Some(pp) => pp.post_process_single(ids, add_special_tokens),
            None => ids,
        }
    }

    // ── Decoding ─────────────────────────────────────────────────────

    /// Decode token IDs back into text.
    ///
    /// If `skip_special_tokens` is true, added tokens marked as special
    /// are omitted from the output.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        let mut tokens = Vec::with_capacity(ids.len());
        for &id in ids {
            if skip_special_tokens {
                if let Some(ref at) = self.added_tokens {
                    if at.is_special(id) {
                        continue;
                    }
                }
            }
            let token_str = self
                .id_to_token(id)
                .ok_or_else(|| Error::Decode(format!("unknown token ID: {id}")))?;
            tokens.push(token_str.to_string());
        }

        match &self.decoder {
            Some(dec) => dec.decode(tokens).map_err(Error::Decoder),
            None => Ok(tokens.join("")),
        }
    }

    /// Decode a batch of token ID sequences.
    pub fn decode_batch(
        &self,
        sentences: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>, Error> {
        sentences
            .iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }

    // ── Vocabulary access ────────────────────────────────────────────

    /// Look up the string for a token ID, checking added tokens first,
    /// then the model vocabulary.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        if let Some(ref at) = self.added_tokens {
            if let Some(s) = at.id_to_token(id) {
                return Some(s);
            }
        }
        self.model.id_to_token(id)
    }

    /// Look up the token ID for a string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.token_to_id(token)
    }

    /// Return the vocabulary size (model tokens + added tokens).
    pub fn vocab_size(&self) -> usize {
        let model_size = self.model.vocab_size();
        let added_size = self.added_tokens.as_ref().map_or(0, |at| at.len());
        model_size + added_size
    }

    // ── Internal helpers ─────────────────────────────────────────────

    /// Build a [`PreTokenizedString`] by splitting on added tokens and
    /// normalizing text segments into a single contiguous buffer.
    fn build_pre_tokenized(&self, input: &str) -> PreTokenizedString {
        let segments = match &self.added_tokens {
            Some(at) => at.split(input),
            None => vec![Segment::Text(input)],
        };

        let mut buffer = String::with_capacity(input.len());
        let mut splits = Vec::new();

        for seg in &segments {
            match seg {
                Segment::Token(id) => {
                    // Added token: store its text in the buffer with a
                    // pre-assigned ID.
                    let start = buffer.len();
                    // Added tokens don't need text in the buffer (the model
                    // never sees it), so use an empty range.
                    splits.push(PtSplit {
                        range: start..start,
                        token_id: Some(*id),
                    });
                }
                Segment::Text(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    let normalized = match &self.normalizer {
                        Some(n) => n.normalize(text),
                        None => Cow::Borrowed(*text),
                    };
                    let start = buffer.len();
                    buffer.push_str(&normalized);
                    let end = buffer.len();
                    splits.push(PtSplit {
                        range: start..end,
                        token_id: None,
                    });
                }
            }
        }

        PreTokenizedString::new(buffer, splits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HF_MODELS: &[&str] = &[
        "Qwen/Qwen3-0.6B",
        "zai-org/GLM-4.7",
        "deepseek-ai/DeepSeek-V3.2",
        "MiniMaxAI/MiniMax-M2.1",
        "openai/gpt-oss-120b",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nvidia/Qwen3-Nemotron-235B-A22B-GenRM",
    ];

    /// Verify that `TokenizerJson` deserializes successfully for a range of
    /// HuggingFace models. This tests the JSON parsing layer only, not the
    /// pipeline construction (which may fail for unsupported step types).
    #[test]
    fn parse_hf_json() {
        let api = Api::new().unwrap();
        for model in HF_MODELS {
            let repo = api.model(model.to_string());
            let json_path = repo
                .get("tokenizer.json")
                .unwrap_or_else(|e| panic!("{model}: {e}"));
            let json: TokenizerJson = serde_json::from_str(&fs::read_to_string(json_path).unwrap())
                .unwrap_or_else(|e| panic!("{model}: {e}"));
            assert!(
                !matches!(json.model, ModelConfig::Other(_)),
                "{model}: model parsed as Other",
            );
        }
    }

    /// Verify that our encoding output matches the HuggingFace `tokenizers`
    /// crate for MiniMax-M2.1 across a variety of inputs.
    #[test]
    fn encode_matches_hf_tokenizers() {
        let model = "MiniMaxAI/MiniMax-M2.1";

        let hf = tokenizers::Tokenizer::from_pretrained(model, None)
            .unwrap_or_else(|e| panic!("{model}: {e}"));
        let ours = Tokenizer::from_model(model).unwrap_or_else(|e| panic!("{model}: {e}"));

        let inputs = &[
            "",
            " ",
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "  multiple   spaces   everywhere  ",
            "MiniMax-M2.1 is a large language model.",
            "Line one\nLine two\nLine three",
            "Tabs\there\tand\tthere",
            "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
            "Unicode: \u{00e9}\u{00e0}\u{00fc}\u{00f1}\u{00f6}",
            "CJK: \u{4f60}\u{597d}\u{4e16}\u{754c}",
            "Emoji: \u{1f600}\u{1f680}\u{2764}\u{fe0f}",
            "Numbers 1234567890 and mixed ABC123def",
            "JSON: {\"key\": \"value\", \"n\": 42}",
            "a",
            "ab",
            "abc",
            "Code: fn main() { println!(\"hello\"); }",
            "URLs: https://example.com/path?q=1&r=2",
            "Repeated: aaaaaaaaaa bbbbbbbbbb",
        ];

        for input in inputs {
            let hf_ids = hf
                .encode(*input, false)
                .unwrap_or_else(|e| panic!("{model}: encode({input:?}): {e}"))
                .get_ids()
                .to_vec();
            let our_ids = ours
                .encode(input)
                .unwrap_or_else(|e| panic!("{model}: encode({input:?}): {e}"));
            assert_eq!(our_ids, hf_ids, "mismatch for {input:?}");
        }
    }

    /// Verify that inputs containing added-token patterns (like `<filename>`)
    /// are handled correctly and match HF output.
    #[test]
    fn encode_with_added_tokens() {
        let model = "MiniMaxAI/MiniMax-M2.1";

        let hf = tokenizers::Tokenizer::from_pretrained(model, None)
            .unwrap_or_else(|e| panic!("{model}: {e}"));
        let ours = Tokenizer::from_model(model).unwrap_or_else(|e| panic!("{model}: {e}"));

        let inputs = &[
            // Single added token.
            "<filename>",
            // Added token embedded in regular text.
            "open <filename> for reading",
            // Multiple added tokens.
            "<filename><reponame>",
            // Added token adjacent to code.
            "printf(\"%s <filename>\\n\")",
            // Non-special added tokens.
            "<think>Let me reason about this.</think>",
            // Mixed special and non-special.
            "<think>load <filename> from <reponame></think>",
            // Added tokens that look similar to but don't match.
            "<file> is not <filename>",
            // Adjacent added tokens with text between.
            "<fim_prefix>code here<fim_suffix>more code<fim_middle>",
        ];

        for input in inputs {
            let hf_ids = hf
                .encode(*input, false)
                .unwrap_or_else(|e| panic!("{model}: encode({input:?}): {e}"))
                .get_ids()
                .to_vec();
            let our_ids = ours
                .encode(input)
                .unwrap_or_else(|e| panic!("{model}: encode({input:?}): {e}"));
            assert_eq!(our_ids, hf_ids, "mismatch for {input:?}");
        }
    }

    /// Verify that decode produces the same output as HuggingFace.
    #[test]
    fn decode_matches_hf_tokenizers() {
        let model = "MiniMaxAI/MiniMax-M2.1";

        let hf = tokenizers::Tokenizer::from_pretrained(model, None)
            .unwrap_or_else(|e| panic!("{model}: {e}"));
        let ours = Tokenizer::from_model(model).unwrap_or_else(|e| panic!("{model}: {e}"));

        let inputs = &[
            "Hello, world!",
            "The quick brown fox.",
            "Unicode: \u{00e9}\u{00e0}\u{00fc}",
            "CJK: \u{4f60}\u{597d}",
            "Emoji: \u{1f600}",
            "Code: fn main() { println!(\"hello\"); }",
            " ",
            "  multiple   spaces  ",
        ];

        for input in inputs {
            let hf_enc = hf.encode(*input, false).unwrap();
            let ids = hf_enc.get_ids();
            let hf_decoded = hf.decode(ids, false).unwrap();
            let our_decoded = ours
                .decode(ids, false)
                .unwrap_or_else(|e| panic!("{model}: decode({input:?}): {e}"));
            assert_eq!(our_decoded, hf_decoded, "decode mismatch for {input:?}");
        }
    }

    /// Verify that encode_batch matches sequential encodes.
    #[test]
    fn encode_batch_matches_sequential() {
        let model = "MiniMaxAI/MiniMax-M2.1";
        let ours = Tokenizer::from_model(model).unwrap();

        let inputs = &["Hello, world!", "The quick brown fox", "Test", ""];
        let batch_results = ours.encode_batch(inputs, false).unwrap();

        for (input, batch_result) in inputs.iter().zip(&batch_results) {
            let sequential_result = ours.encode(input).unwrap();
            assert_eq!(batch_result, &sequential_result, "batch mismatch for {input:?}");
        }
    }

    /// Verify that vocab access methods work correctly.
    #[test]
    fn vocab_access() {
        let model = "MiniMaxAI/MiniMax-M2.1";
        let ours = Tokenizer::from_model(model).unwrap();

        // vocab_size should be non-zero
        assert!(ours.vocab_size() > 0);

        // id_to_token and token_to_id should roundtrip for model tokens
        let token_str = ours.id_to_token(0).expect("token 0 should exist");
        let id = ours.token_to_id(token_str).expect("reverse lookup should work");
        assert_eq!(id, 0);
    }
}
