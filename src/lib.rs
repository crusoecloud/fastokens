pub mod added_tokens;
pub mod json_structs;
pub mod models;
pub mod normalizers;
pub mod post_processors;
pub mod pre_tokenized;
pub mod pre_tokenizers;

use std::{borrow::Cow, fs, path::Path};

use hf_hub::api::sync::Api;
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
}

/// An LLM tokenizer backed by `tokenizer.json`.
pub struct Tokenizer {
    added_tokens: Option<AddedTokens>,
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<PreTokenizer>,
    model: Model,
    post_processor: Option<PostProcessor>,
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

        Ok(Self {
            added_tokens,
            normalizer,
            pre_tokenizer,
            model,
            post_processor,
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

    /// Run the full encoding pipeline: split added tokens, normalize,
    /// pre-tokenize, tokenize and post-process the input string.
    pub fn encode(&self, input: &str) -> Result<Vec<u32>, Error> {
        if input.is_empty() {
            return Ok(Vec::new());
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

        // All currently "implemented" post-processors don't actually do
        // anything to the output tokens, so we don't actually have to call
        // them.

        Ok(ids)
    }

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
}
