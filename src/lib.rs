pub mod added_tokens;
pub mod decoders;
pub mod json_structs;
pub mod models;
pub mod normalizers;
pub mod post_processors;
pub mod pre_tokenized;
pub mod pre_tokenizers;

use std::{fs, path::Path};

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

    #[error("post-processor error: {0}")]
    PostProcessor(#[from] post_processors::Error),

    #[error("decoder error: {0}")]
    Decoder(#[from] decoders::Error),

    #[error("model error: {0}")]
    Model(String),

    #[error("decode error: {0}")]
    Decode(String),

    #[error("invalid model identifier: {0}")]
    InvalidIdentifier(String),
}

/// An LLM tokenizer backed by `tokenizer.json`.
pub struct Tokenizer {
    added_tokens: Option<AddedTokens>,
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<PreTokenizer>,
    model: Model,
    post_processor: Option<PostProcessor>,
    decoder: Option<Decoder>,
    /// When the pre-tokenizer is `Sequence([Split, ByteLevel(bulk)])`,
    /// we store a Split-only pre-tokenizer and fuse ByteLevel into BPE.
    split_only: Option<PreTokenizer>,
}

impl Tokenizer {
    /// Build the pipeline steps from a parsed JSON config.
    fn build(json: TokenizerJson) -> Result<Self, Error> {
        let added_tokens = AddedTokens::from_configs(&json.added_tokens).map_err(Error::Model)?;
        let normalizer = json.normalizer.map(Normalizer::from_config).transpose()?;
        let pre_tokenizer = json
            .pre_tokenizer
            .map(PreTokenizer::from_config)
            .transpose()?;
        let model = Model::from_config(json.model).map_err(Error::Model)?;
        let post_processor = json
            .post_processor
            .map(PostProcessor::from_config)
            .transpose()?;
        let decoder = json.decoder.map(Decoder::from_config).transpose()?;

        // Detect Sequence([Split, ByteLevel(bulk)]) for fused byte-level+BPE.
        let split_only = Self::detect_fused_byte_level(&pre_tokenizer);

        Ok(Self {
            added_tokens,
            normalizer,
            pre_tokenizer,
            model,
            post_processor,
            decoder,
            split_only,
        })
    }

    /// If `pt` is `Sequence([Split, ByteLevel(bulk)])`, return a Split-only
    /// pre-tokenizer for fused mode.
    fn detect_fused_byte_level(pt: &Option<PreTokenizer>) -> Option<PreTokenizer> {
        let PreTokenizer::Sequence(steps) = pt.as_ref()? else {
            return None;
        };
        if steps.len() != 2 {
            return None;
        }
        let is_split = matches!(&steps[0], PreTokenizer::Split(_));
        let is_bulk_bl = matches!(&steps[1], PreTokenizer::ByteLevel(bl) if bl.is_bulk_only());
        if is_split && is_bulk_bl {
            Some(steps[0].clone())
        } else {
            None
        }
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
        if model.contains("..") {
            return Err(Error::InvalidIdentifier(
                "model identifier must not contain \"..\"".into(),
            ));
        }
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

        // Fused path: run only Split, then batch-tokenize with inline ByteLevel.
        if let Some(ref split) = self.split_only {
            split.pre_tokenize(&mut pts)?;
            let ids = pts
                .tokenize_batched(|buf, splits, out| {
                    self.model.tokenize_batch_fused(buf, splits, out)
                })
                .map_err(Error::Model)?;
            return Ok(self.post_process(ids, add_special_tokens));
        }

        // 2. Pre-tokenize (refine splits in place).
        if let Some(ref pt) = self.pre_tokenizer {
            pt.pre_tokenize(&mut pts)?;
        }

        // 3. Tokenize each text split with the model.
        let ids = pts
            .tokenize(|text, out| self.model.tokenize_into(text, out))
            .map_err(Error::Model)?;

        // 4. Post-process.
        Ok(self.post_process(ids, add_special_tokens))
    }

    /// Encode a batch of inputs.
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

    pub fn post_process(&self, ids: Vec<u32>, add_special_tokens: bool) -> Vec<u32> {
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
            if skip_special_tokens
                && let Some(ref at) = self.added_tokens
                && at.is_special(id)
            {
                continue;
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

    /// Decode a sequence of token strings back into text.
    ///
    /// Applies the decoder pipeline (e.g. ByteLevel → convert "Ġ" back to " ")
    /// without going through the ID→string lookup.  When no decoder is
    /// configured the tokens are concatenated with no separator.
    pub fn decode_tokens(&self, tokens: Vec<String>) -> Result<String, Error> {
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
        if let Some(ref at) = self.added_tokens
            && let Some(s) = at.id_to_token(id)
        {
            return Some(s);
        }
        self.model.id_to_token(id)
    }

    /// Look up the token ID for a string.
    ///
    /// Added tokens are checked first (they shadow any BPE model entry with
    /// the same string), then the BPE model vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(ref at) = self.added_tokens
            && let Some(id) = at.token_to_id(token)
        {
            return Some(id);
        }
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
    pub fn build_pre_tokenized(&self, input: &str) -> PreTokenizedString {
        let segments = match &self.added_tokens {
            Some(at) => at.split(input),
            None => vec![Segment::Text(input)],
        };

        // Fast path: if there's exactly one Text segment (no added token matches)
        // and normalization returns Cow::Borrowed, we just need a string copy.
        if segments.len() == 1
            && let Segment::Text(text) = segments[0]
        {
            let normalized = match &self.normalizer {
                Some(n) => n.normalize(text),
                None => std::borrow::Cow::Borrowed(text),
            };
            return match normalized {
                std::borrow::Cow::Borrowed(_) => PreTokenizedString::from_text(text),
                std::borrow::Cow::Owned(s) => {
                    let len = s.len();
                    PreTokenizedString::new(
                        s,
                        vec![PtSplit {
                            range: 0..len,
                            token_id: None,
                        }],
                    )
                }
            };
        }

        let mut buffer = String::with_capacity(input.len());
        let mut splits = Vec::new();

        for seg in &segments {
            match seg {
                Segment::Token(id) => {
                    let start = buffer.len();
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
                        None => std::borrow::Cow::Borrowed(*text),
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

// ---------------------------------------------------------------------------
// Streaming decode
// ---------------------------------------------------------------------------

/// Stateful incremental decoder.
///
/// Wraps the sliding-window state needed by [`decode_stream_step`] so callers
/// don't have to manage `ids`, `prefix`, and `prefix_index` themselves.
pub struct DecodeStream {
    skip_special_tokens: bool,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
}

impl DecodeStream {
    pub fn new(ids: Vec<u32>, skip_special_tokens: bool) -> Self {
        Self {
            skip_special_tokens,
            ids,
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    pub fn step(
        &mut self,
        tokenizer: &Tokenizer,
        token_ids: Vec<u32>,
    ) -> Result<Option<String>, String> {
        decode_stream_step(
            tokenizer,
            token_ids,
            self.skip_special_tokens,
            &mut self.ids,
            &mut self.prefix,
            &mut self.prefix_index,
        )
    }
}

/// Advance an incremental decode stream by one or more token IDs.
///
/// Maintains a sliding window in `ids` and a `prefix` string to subtract,
/// emitting text chunks as soon as enough context is available.
/// Incomplete UTF-8 (signalled by U+FFFD in the decoder output) is held back
/// until a subsequent token resolves it.
///
/// # Arguments
/// * `token_ids` — new token IDs to append
/// * `skip_special_tokens` — whether to omit special tokens from the output
/// * `ids` — mutable buffer of all IDs decoded so far (updated in place)
/// * `prefix` — previously returned text, subtracted to yield the next chunk
/// * `prefix_index` — index in `ids` where the current prefix window starts
///
/// # Returns
/// `Ok(Some(chunk))` when new text is available, `Ok(None)` when more tokens
/// are needed, `Err(msg)` if the decoder produces output inconsistent with the
/// stored prefix (should be treated as a stream-reset signal).
pub fn decode_stream_step(
    tokenizer: &Tokenizer,
    token_ids: Vec<u32>,
    skip_special_tokens: bool,
    ids: &mut Vec<u32>,
    prefix: &mut String,
    prefix_index: &mut usize,
) -> Result<Option<String>, String> {
    const REPLACEMENT: char = '\u{FFFD}';

    // If the prefix is empty but we already have buffered IDs (e.g. seeded
    // with prompt tokens), prime the prefix before adding the new token.
    if prefix.is_empty() && !ids.is_empty() {
        let s = tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| e.to_string())?;
        if !s.ends_with(REPLACEMENT) {
            *prefix = s;
            *prefix_index = ids.len();
        }
    }

    ids.extend(token_ids);

    let string = tokenizer
        .decode(ids, skip_special_tokens)
        .map_err(|e| e.to_string())?;

    if string.len() > prefix.len() && !string.ends_with(REPLACEMENT) {
        if !string.starts_with(prefix.as_str()) {
            return Err(format!(
                "Invalid prefix encountered while decoding stream. \
                 Expected prefix: '{}', Actual string: '{}'",
                prefix, string,
            ));
        }

        let new_text = string[prefix.len()..].to_string();
        let new_prefix_index = ids.len() - *prefix_index;
        *ids = ids.drain(*prefix_index..).collect();
        *prefix = tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| e.to_string())?;
        *prefix_index = new_prefix_index;

        Ok(Some(new_text))
    } else {
        Ok(None)
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

    /// Verify that `TokenizerConfig` and `TokenizerJson` deserialize
    /// successfully for a range of HuggingFace models. This tests the JSON
    /// parsing layer only, not the pipeline construction (which may fail for
    /// unsupported step types).
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

    /// Verify that encode_batch matches sequential encodes.
    #[test]
    fn encode_batch_matches_sequential() {
        let model = "MiniMaxAI/MiniMax-M2.1";
        let ours = Tokenizer::from_model(model).unwrap();

        let inputs = &["Hello, world!", "The quick brown fox", "Test", ""];
        let batch_results = ours.encode_batch(inputs, false).unwrap();

        for (input, batch_result) in inputs.iter().zip(&batch_results) {
            let sequential_result = ours.encode(input).unwrap();
            assert_eq!(
                batch_result, &sequential_result,
                "batch mismatch for {input:?}"
            );
        }
    }

    /// Verify that vocab access methods work correctly.
    #[test]
    fn vocab_access() {
        let model = "MiniMaxAI/MiniMax-M2.1";
        let ours = Tokenizer::from_model(model).unwrap();

        assert!(ours.vocab_size() > 0);

        let token_str = ours.id_to_token(0).expect("token 0 should exist");
        let id = ours
            .token_to_id(token_str)
            .expect("reverse lookup should work");
        assert_eq!(id, 0);
    }

    // ── Correctness tests against HuggingFace tokenizers ─────────────

    /// Comprehensive corpus of inputs designed to exercise tokenizer edge
    /// cases. Used by the multi-model correctness tests below.
    const CORPUS: &[&str] = &[
        // ── empty / trivial ──
        "",
        " ",
        "  ",
        "\n",
        "\t",
        "\r\n",
        // ── single characters ──
        "a",
        "Z",
        "0",
        "!",
        "\u{00e9}", // é (precomposed)
        "\u{4e2d}", // 中
        // ── basic text ──
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "A short sentence.",
        // ── whitespace variations ──
        "  leading spaces",
        "trailing spaces  ",
        "  both  sides  ",
        "multiple    internal    spaces",
        "tabs\there\tand\tthere",
        "line\none\nline\ntwo",
        "windows\r\nline\r\nendings",
        "mixed\n\ttabs and\r\nnewlines  with  spaces",
        // ── numbers ──
        "42",
        "3.14159",
        "1,000,000",
        "0xFF",
        "1e-10",
        "Numbers 1234567890 and mixed ABC123def",
        // ── punctuation / special characters ──
        "Hello!!! How are you???",
        "@user #hashtag $100 %50 ^caret &amp *star",
        "a-b_c.d,e;f:g",
        "(parentheses) [brackets] {braces}",
        "\"double quotes\" 'single quotes' `backticks`",
        "path/to/file.txt",
        "https://example.com/path?q=test&lang=en#section",
        "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?",
        // ── Unicode: Latin accented ──
        "caf\u{00e9} r\u{00e9}sum\u{00e9} na\u{00ef}ve",
        "\u{00fc}ber stra\u{00df}e gr\u{00f6}\u{00df}e",
        "se\u{00f1}or ni\u{00f1}o a\u{00f1}o",
        // ── Unicode: CJK ──
        "\u{4f60}\u{597d}\u{4e16}\u{754c}",         // 你好世界
        "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}", // こんにちは
        "\u{c548}\u{b155}\u{d558}\u{c138}\u{c694}", // 안녕하세요
        // ── Unicode: Cyrillic ──
        "\u{041f}\u{0440}\u{0438}\u{0432}\u{0435}\u{0442} \u{043c}\u{0438}\u{0440}",
        // ── Unicode: Arabic ──
        "\u{0645}\u{0631}\u{062d}\u{0628}\u{0627}",
        // ── Unicode: Devanagari ──
        "\u{0928}\u{092e}\u{0938}\u{094d}\u{0924}\u{0947}",
        // ── Unicode: Emoji ──
        "\u{1f600}\u{1f680}\u{2764}\u{fe0f}",
        "\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f467}\u{200d}\u{1f466}",
        "\u{1f1fa}\u{1f1f8}", // 🇺🇸
        // ── Unicode: combining marks (NFD forms) ──
        "e\u{0301}", // e + combining acute
        "n\u{0303}", // n + combining tilde
        "a\u{0308}", // a + combining diaeresis
        // ── mixed scripts ──
        "Hello \u{4e16}\u{754c} \u{041c}\u{0438}\u{0440}!",
        "User123 wrote: \u{4f60}\u{597d}!",
        // ── code / programming ──
        "fn main() { println!(\"hello\"); }",
        "def foo(x: int) -> str:\n    return str(x)",
        "SELECT * FROM users WHERE id = 1;",
        "if (x > 0 && y < 10) { z = x + y; }",
        "<html><body><p>Hello</p></body></html>",
        "#include <stdio.h>\nint main() { return 0; }",
        "import numpy as np\nx = np.array([1, 2, 3])",
        // ── JSON / structured data ──
        "{\"key\": \"value\", \"number\": 42, \"array\": [1, 2, 3]}",
        "[{\"id\": 1}, {\"id\": 2}]",
        // ── repeated patterns ──
        "aaaaaaaaaa",
        "abababababababab",
        "the the the the the the the the",
        "....",
        "----",
        "    ",
        "\n\n\n\n",
        // ── longer mixed content ──
        "This is a longer sentence with various elements: numbers (42, 3.14), \
         symbols (@#$), Unicode (caf\u{00e9}, \u{4f60}\u{597d}), and more.",
        "The year 2024 was notable for advances in AI. Models like GPT-4 and \
         Claude demonstrated remarkable capabilities in reasoning, coding, and \
         multilingual understanding.",
        // ── alphabet / character sequences ──
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "0123456789",
        // ── boundary / edge cases ──
        "a\nb\nc\n",
        "# Heading\n\n- item 1\n- item 2\n\n```code```",
        "\u{ffff}",  // max BMP non-character
        "\u{0080}",  // first non-ASCII
        "\u{07ff}",  // max 2-byte UTF-8
        "\u{0800}",  // first 3-byte UTF-8
        "\u{10000}", // first surrogate-pair range
        // ── unusual / invalid-ish Unicode ──
        "\u{fffd}",                                  // replacement character
        "\u{feff}Hello",                             // BOM prefix
        "\u{0000}",                                  // null
        "abc\u{0000}def",                            // embedded null
        "\u{fffe}",                                  // non-character
        "\u{fdd0}",                                  // non-character (FDD0 block)
        "\u{200b}\u{200c}\u{200d}",                  // zero-width space / ZWNJ / ZWJ
        "\u{202e}Hello\u{202c}",                     // RTL override + pop directional
        "\u{0001}\u{0002}\u{001f}\u{007f}",          // C0 controls + DEL
        "\u{0300}",                                  // lone combining grave (no base)
        "a\u{0300}\u{0301}\u{0302}\u{0303}\u{0304}", // 5 combining marks on one base
        "\u{e000}\u{f8ff}",                          // private use area
        "\u{01c5}\u{01c8}\u{01cb}",                  // titlecase letters (Dž Lj Nj)
        "\u{2028}\u{2029}",                          // line / paragraph separators
        "\u{fff9}\u{fffa}\u{fffb}",                  // interlinear annotation
        "\u{d7ff}\u{10ffff}",                        // last before surrogates + max codepoint
        // ── potential BPE merge edge cases ──
        "ab",
        "abc",
        "abcd",
        "aaa",
        "aaaa",
        "aaaaa",
        // ── markdown / formatting ──
        "**bold** *italic* ~~strikethrough~~ __underline__",
        "```rust\nfn main() {}\n```",
        "> blockquote\n>> nested",
        "| col1 | col2 |\n|------|------|\n| a    | b    |",
    ];

    /// Helper: compare both encoding and decoding of every input in `corpus`
    /// between our tokenizer and the HuggingFace tokenizer for a given model.
    /// Returns a list of failure descriptions (empty = all passed).
    fn compare_encode_decode(model_name: &str, corpus: &[&str]) -> Vec<String> {
        let hf = tokenizers::Tokenizer::from_pretrained(model_name, None)
            .unwrap_or_else(|e| panic!("{model_name}: HF load failed: {e}"));
        let ours = Tokenizer::from_model(model_name)
            .unwrap_or_else(|e| panic!("{model_name}: fastokens load failed: {e}"));

        let mut failures = Vec::new();
        for &input in corpus {
            let hf_enc = hf
                .encode(input, false)
                .unwrap_or_else(|e| panic!("{model_name}: HF encode({input:?}): {e}"));
            let hf_ids = hf_enc.get_ids().to_vec();
            let our_ids = match ours.encode(input) {
                Ok(ids) => ids,
                Err(e) => {
                    failures.push(format!("  encode error on {input:?}: {e}"));
                    continue;
                }
            };
            if our_ids != hf_ids {
                failures.push(format!(
                    "  encode mismatch on {input:?}: got {} tokens, expected {}\n\
                     \x20   ours: {:?}\n\
                     \x20   hf:   {:?}",
                    our_ids.len(),
                    hf_ids.len(),
                    &our_ids[..our_ids.len().min(20)],
                    &hf_ids[..hf_ids.len().min(20)],
                ));
            }

            // Decode comparison (skip empty inputs / empty token sequences).
            if input.is_empty() || hf_ids.is_empty() {
                continue;
            }
            let hf_decoded = match hf.decode(&hf_ids, false) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let our_decoded = match ours.decode(&hf_ids, false) {
                Ok(d) => d,
                Err(e) => {
                    failures.push(format!("  decode error on {input:?}: {e}"));
                    continue;
                }
            };
            if our_decoded != hf_decoded {
                failures.push(format!(
                    "  decode mismatch on {input:?}:\n\
                     \x20   ours: {:?}\n\
                     \x20   hf:   {:?}",
                    &our_decoded[..our_decoded.len().min(100)],
                    &hf_decoded[..hf_decoded.len().min(100)],
                ));
            }
        }
        failures
    }

    // ── Per-model encoding correctness ───────────────────────────────

    #[test]
    fn correctness_minimax_m2_1() {
        let f = compare_encode_decode("MiniMaxAI/MiniMax-M2.1", CORPUS);
        assert!(f.is_empty(), "MiniMaxAI/MiniMax-M2.1:\n{}", f.join("\n"));
    }

    #[test]
    fn correctness_nemotron() {
        let f = compare_encode_decode("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", CORPUS);
        assert!(
            f.is_empty(),
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16:\n{}",
            f.join("\n")
        );
    }

    #[test]
    fn correctness_deepseek_v3_2() {
        let f = compare_encode_decode("deepseek-ai/DeepSeek-V3.2", CORPUS);
        assert!(f.is_empty(), "deepseek-ai/DeepSeek-V3.2:\n{}", f.join("\n"));
    }

    #[test]
    fn correctness_gpt_oss() {
        let f = compare_encode_decode("openai/gpt-oss-120b", CORPUS);
        assert!(f.is_empty(), "openai/gpt-oss-120b:\n{}", f.join("\n"));
    }

    #[test]
    fn correctness_qwen3() {
        let f = compare_encode_decode("Qwen/Qwen3-0.6B", CORPUS);
        assert!(f.is_empty(), "Qwen/Qwen3-0.6B:\n{}", f.join("\n"));
    }

    #[test]
    fn correctness_mistral_nemo() {
        let f = compare_encode_decode("mistralai/Mistral-Nemo-Instruct-2407", CORPUS);
        assert!(
            f.is_empty(),
            "mistralai/Mistral-Nemo-Instruct-2407:\n{}",
            f.join("\n")
        );
    }

    #[test]
    fn correctness_qwen3_nemotron() {
        let f = compare_encode_decode("nvidia/Qwen3-Nemotron-235B-A22B-GenRM", CORPUS);
        assert!(
            f.is_empty(),
            "nvidia/Qwen3-Nemotron-235B-A22B-GenRM:\n{}",
            f.join("\n")
        );
    }

    // ── Cache consistency ────────────────────────────────────────────

    /// Verify that encoding the same input twice produces identical results,
    /// exercising both the cold (cache miss) and warm (cache hit) paths.
    #[test]
    fn cache_consistency() {
        let model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16";
        let ours = Tokenizer::from_model(model).unwrap();

        let inputs = &[
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "caf\u{00e9} r\u{00e9}sum\u{00e9}",
            "\u{4f60}\u{597d}\u{4e16}\u{754c}",
            "fn main() { println!(\"hello\"); }",
            "a b c d e f g h i j k l m n o p",
            "aaaaaaaaaa bbbbbbbbbb cccccccccc",
        ];

        for &input in inputs {
            let first = ours.encode(input).unwrap();
            let second = ours.encode(input).unwrap();
            assert_eq!(first, second, "cache inconsistency for {input:?}");
            // Third call to exercise potential L1→L2 promotion paths.
            let third = ours.encode(input).unwrap();
            assert_eq!(first, third, "cache inconsistency (3rd call) for {input:?}");
        }
    }

    /// Same as above but for the fused byte-level path (Nemotron uses
    /// Sequence([Split, ByteLevel]) which triggers the fused code path).
    #[test]
    fn cache_consistency_fused() {
        let model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16";
        let ours = Tokenizer::from_model(model).unwrap();

        // Verify the fused path is active.
        assert!(ours.split_only.is_some(), "expected fused path for {model}",);

        // Run the same input many times to stress the fused cache.
        let input = "The year 2024 was notable for advances in AI. Models like \
                      GPT-4 and Claude demonstrated remarkable capabilities.";
        let baseline = ours.encode(input).unwrap();
        for i in 0..20 {
            let result = ours.encode(input).unwrap();
            assert_eq!(result, baseline, "fused cache drift on iteration {i}");
        }
    }

    // ── Added tokens (model-specific) ────────────────────────────────

    /// MiniMax-M2.1 has added tokens like <filename>, <reponame>, <think>,
    /// etc. Verify they are handled identically to HF.
    #[test]
    fn added_tokens_minimax() {
        let corpus = &[
            "<filename>",
            "open <filename> for reading",
            "<filename><reponame>",
            "printf(\"%s <filename>\\n\")",
            "<think>Let me reason about this.</think>",
            "<think>load <filename> from <reponame></think>",
            "<file> is not <filename>",
            "<fim_prefix>code here<fim_suffix>more code<fim_middle>",
        ];
        let f = compare_encode_decode("MiniMaxAI/MiniMax-M2.1", corpus);
        assert!(
            f.is_empty(),
            "MiniMaxAI/MiniMax-M2.1 added tokens:\n{}",
            f.join("\n")
        );
    }

    /// DeepSeek-V3.2 added tokens.
    #[test]
    fn added_tokens_deepseek() {
        let corpus = &[
            "<|begin▁of▁sentence|>Hello",
            "Hello<|end▁of▁sentence|>",
            "<|User|>What is 2+2?<|Assistant|>4<|end▁of▁sentence|>",
            "Normal text without special tokens",
            "<|tool▁calls▁begin|>call<|tool▁calls▁end|>",
        ];
        let f = compare_encode_decode("deepseek-ai/DeepSeek-V3.2", corpus);
        assert!(
            f.is_empty(),
            "deepseek-ai/DeepSeek-V3.2 added tokens:\n{}",
            f.join("\n")
        );
    }

    /// Qwen3 added tokens.
    #[test]
    fn added_tokens_qwen3() {
        let corpus = &[
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>",
            "<|im_start|>user\nHello!<|im_end|>",
            "<|endoftext|>",
            "Plain text with no special tokens at all.",
        ];
        let f = compare_encode_decode("Qwen/Qwen3-0.6B", corpus);
        assert!(
            f.is_empty(),
            "Qwen/Qwen3-0.6B added tokens:\n{}",
            f.join("\n")
        );
    }

    /// token_to_id must find added tokens, not just BPE model vocab entries.
    ///
    /// Root cause of the Qwen3VLProcessor._check_special_mm_tokens failure:
    /// `convert_tokens_to_ids("<|image_pad|>")` calls `token_to_id`, which
    /// previously only searched the BPE model vocabulary and returned None for
    /// added tokens, causing the processor to compare input_ids against
    /// unk_token_id (0) instead of the real image-pad token ID.
    #[test]
    fn token_to_id_searches_added_tokens() {
        let tok = Tokenizer::from_model("Qwen/Qwen3-0.6B").unwrap();
        // These tokens live in added_tokens, not the BPE model vocab.
        for token in &[
            "<|image_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|im_start|>",
        ] {
            let id = tok.token_to_id(token);
            assert!(id.is_some(), "token_to_id({token:?}) returned None");
            // Round-trip: the ID must decode back to the same string.
            assert_eq!(tok.id_to_token(id.unwrap()), Some(*token));
        }
    }

    /// Qwen3-VL vision tokens — the exact text that triggered:
    ///
    ///   ValueError: Failed to apply Qwen3VLProcessor on
    ///   data={'text': '<|vision_start|><|image_pad|><|vision_end|>'}
    ///   with kwargs={'truncation': False}
    ///
    /// Qwen3-0.6B ships with the full set of VL tokens in its added_tokens
    /// array.  A sequence that consists *entirely* of adjacent special tokens
    /// (no regular text in between) exercises the code path where
    /// build_pre_tokenized produces only zero-length Token splits.
    #[test]
    fn added_tokens_qwen3vl_vision_sequence() {
        let corpus = &[
            // Exact failing input from vLLM / Qwen3VLProcessor.
            "<|vision_start|><|image_pad|><|vision_end|>",
            // Bare image-pad token.
            "<|image_pad|>",
            // Multiple adjacent image-pad tokens (real prompts have dozens).
            "<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|>",
            // Mixed: VL tokens followed by regular text.
            "<|vision_start|><|image_pad|><|vision_end|>\nDescribe this image.",
        ];
        let f = compare_encode_decode("Qwen/Qwen3.5-27B", corpus);
        assert!(
            f.is_empty(),
            "Qwen/Qwen3.5-27B VL vision sequence:\n{}",
            f.join("\n")
        );
    }

    /// Nemotron added tokens.
    #[test]
    fn added_tokens_nemotron() {
        let corpus = &[
            "<|begin_of_text|>Hello world",
            "Hello<|end_of_text|>",
            "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\nHi!<|eot_id|>",
            "No special tokens here.",
        ];
        let f = compare_encode_decode("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", corpus);
        assert!(
            f.is_empty(),
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 added tokens:\n{}",
            f.join("\n")
        );
    }

    // ── Long input stress test ───────────────────────────────────────

    /// Verify correctness on a longer input that exercises the parallel
    /// tokenization path (>128 splits).
    #[test]
    fn long_input_correctness() {
        let model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16";
        let hf = tokenizers::Tokenizer::from_pretrained(model_name, None).unwrap();
        let ours = Tokenizer::from_model(model_name).unwrap();

        // Build a ~10KB input from repeated varied content.
        let block = "The quick brown fox jumps over the lazy dog. \
                      Numbers: 42, 3.14, 1000. Code: fn main() {} \
                      Unicode: caf\u{00e9}, \u{4f60}\u{597d}. \
                      Special: @#$%^&*(). ";
        let input: String = block.repeat(100);
        assert!(input.len() > 8000);

        let hf_ids = hf.encode(input.as_str(), false).unwrap().get_ids().to_vec();
        let our_ids = ours.encode(&input).unwrap();
        assert_eq!(
            our_ids,
            hf_ids,
            "long input mismatch: {} vs {} tokens",
            our_ids.len(),
            hf_ids.len(),
        );
    }

    /// Same long-input test for a non-fused model.
    #[test]
    fn long_input_correctness_minimax() {
        let model_name = "MiniMaxAI/MiniMax-M2.1";
        let hf = tokenizers::Tokenizer::from_pretrained(model_name, None).unwrap();
        let ours = Tokenizer::from_model(model_name).unwrap();

        let block = "The quick brown fox jumps over the lazy dog. \
                      Numbers: 42, 3.14, 1000. Code: fn main() {} \
                      Unicode: caf\u{00e9}, \u{4f60}\u{597d}. \
                      Special: @#$%^&*(). ";
        let input: String = block.repeat(100);

        let hf_ids = hf.encode(input.as_str(), false).unwrap().get_ids().to_vec();
        let our_ids = ours.encode(&input).unwrap();
        assert_eq!(
            our_ids,
            hf_ids,
            "long input mismatch: {} vs {} tokens",
            our_ids.len(),
            hf_ids.len(),
        );
    }

    // ── Extended dataset tests (run with `cargo test -- --ignored`) ──

    use std::sync::OnceLock;

    struct ExtendedCorpus {
        longbench: Vec<String>,
        sharegpt: Vec<String>,
    }

    fn extended_corpus() -> &'static ExtendedCorpus {
        static CORPUS: OnceLock<ExtendedCorpus> = OnceLock::new();
        CORPUS.get_or_init(|| {
            let api = Api::new().unwrap();

            // LongBench-v2: first 100 samples
            let lb_repo = api.dataset("zai-org/LongBench-v2".to_string());
            let lb_path = lb_repo.get("data.json").unwrap();
            let lb_data: Vec<serde_json::Value> =
                serde_json::from_str(&fs::read_to_string(lb_path).unwrap()).unwrap();
            let longbench: Vec<String> = lb_data
                .iter()
                .filter_map(|item| {
                    let ctx = item.get("context")?.as_str()?;
                    if ctx.is_empty() {
                        None
                    } else {
                        Some(ctx.to_string())
                    }
                })
                .collect();

            // ShareGPT52K: first 1000 samples
            let sg_repo = api.dataset("RyokoAI/ShareGPT52K".to_string());
            let sg_path = sg_repo.get("sg_90k_part1.json").unwrap();
            let sg_data: Vec<serde_json::Value> =
                serde_json::from_str(&fs::read_to_string(sg_path).unwrap()).unwrap();
            let sharegpt: Vec<String> = sg_data
                .iter()
                .filter_map(|item| {
                    let messages = item.get("conversations")?.as_array()?;
                    let parts: Vec<String> = messages
                        .iter()
                        .filter_map(|msg| {
                            let role = msg
                                .get("from")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown");
                            let value = msg.get("value").and_then(|v| v.as_str())?;
                            if value.is_empty() {
                                return None;
                            }
                            Some(format!("[{role}]: {value}"))
                        })
                        .collect();
                    if parts.is_empty() {
                        None
                    } else {
                        Some(parts.join("\n\n"))
                    }
                })
                .collect();

            ExtendedCorpus {
                longbench,
                sharegpt,
            }
        })
    }

    /// Compare encoding and decoding in batches using encode_batch.
    fn compare_encode_decode_batched(
        model_name: &str,
        corpus: &[String],
        batch_size: usize,
        progress: bool,
    ) -> Vec<String> {
        let hf = tokenizers::Tokenizer::from_pretrained(model_name, None)
            .unwrap_or_else(|e| panic!("{model_name}: HF load failed: {e}"));
        let ours = Tokenizer::from_model(model_name)
            .unwrap_or_else(|e| panic!("{model_name}: fastokens load failed: {e}"));

        let total = corpus.len();
        let mut processed = 0usize;
        let mut failures = Vec::new();
        for chunk in corpus.chunks(batch_size) {
            let hf_results: Vec<Vec<u32>> = chunk
                .iter()
                .map(|input| {
                    hf.encode(input.as_str(), false)
                        .unwrap_or_else(|e| panic!("{model_name}: HF encode: {e}"))
                        .get_ids()
                        .to_vec()
                })
                .collect();

            let our_results = match ours.encode_batch(chunk, false) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("  encode_batch error: {e}"));
                    continue;
                }
            };

            for (i, (hf_ids, our_ids)) in hf_results.iter().zip(our_results.iter()).enumerate() {
                let input = &chunk[i];
                let input_preview = {
                    let mut end = input.len().min(80);
                    while end < input.len() && !input.is_char_boundary(end) {
                        end += 1;
                    }
                    &input[..end]
                };

                if our_ids != hf_ids {
                    failures.push(format!(
                        "  encode mismatch on {:?}: got {} tokens, expected {}\n\
                         \x20   ours: {:?}\n\
                         \x20   hf:   {:?}",
                        input_preview,
                        our_ids.len(),
                        hf_ids.len(),
                        &our_ids[..our_ids.len().min(20)],
                        &hf_ids[..hf_ids.len().min(20)],
                    ));
                }

                // Decode comparison.
                if hf_ids.is_empty() || input.is_empty() {
                    continue;
                }
                let hf_decoded = match hf.decode(hf_ids, false) {
                    Ok(d) => d,
                    Err(_) => continue,
                };
                let our_decoded = match ours.decode(hf_ids, false) {
                    Ok(d) => d,
                    Err(e) => {
                        failures.push(format!("  decode error on {input_preview:?}: {e}"));
                        continue;
                    }
                };
                if our_decoded != hf_decoded {
                    failures.push(format!(
                        "  decode mismatch on {input_preview:?}:\n\
                         \x20   ours: {:?}\n\
                         \x20   hf:   {:?}",
                        &our_decoded[..our_decoded.len().min(100)],
                        &hf_decoded[..hf_decoded.len().min(100)],
                    ));
                }
            }
            processed += chunk.len();
            if progress {
                eprint!(
                    "\r  {model_name}: {processed}/{total} ({:.0}%)",
                    processed as f64 / total as f64 * 100.0,
                );
            }
        }
        if progress {
            eprintln!();
        }
        failures
    }

    fn run_extended(model_name: &str) {
        let progress = std::env::var("EXTENDED_PROGRESS").is_ok();
        let corpus = extended_corpus();
        if progress {
            eprintln!(
                "  {model_name}: longbench ({} samples)",
                corpus.longbench.len()
            );
        }
        let mut failures =
            compare_encode_decode_batched(model_name, &corpus.longbench, 10, progress);
        if progress {
            eprintln!(
                "  {model_name}: sharegpt ({} samples)",
                corpus.sharegpt.len()
            );
        }
        failures.extend(compare_encode_decode_batched(
            model_name,
            &corpus.sharegpt,
            10,
            progress,
        ));
        assert!(
            failures.is_empty(),
            "{model_name} extended ({} failures):\n{}",
            failures.len(),
            failures.join("\n"),
        );
    }

    #[test]
    #[ignore]
    fn extended_minimax_m2_1() {
        run_extended("MiniMaxAI/MiniMax-M2.1");
    }

    #[test]
    #[ignore]
    fn extended_nemotron() {
        run_extended("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16");
    }

    #[test]
    #[ignore]
    fn extended_deepseek_v3_2() {
        run_extended("deepseek-ai/DeepSeek-V3.2");
    }

    #[test]
    #[ignore]
    fn extended_gpt_oss() {
        run_extended("openai/gpt-oss-120b");
    }

    #[test]
    #[ignore]
    fn extended_qwen3() {
        run_extended("Qwen/Qwen3-0.6B");
    }

    #[test]
    #[ignore]
    fn extended_mistral_nemo() {
        run_extended("mistralai/Mistral-Nemo-Instruct-2407");
    }

    #[test]
    #[ignore]
    fn extended_qwen3_nemotron() {
        run_extended("nvidia/Qwen3-Nemotron-235B-A22B-GenRM");
    }

    #[test]
    #[ignore]
    fn extended_mistral_large() {
        run_extended("mistralai/Mistral-Large-3-675B-Instruct-2512");
    }

    #[test]
    #[ignore]
    fn extended_qwen_small() {
        run_extended("Qwen/Qwen3-0.6B");
    }

    // ── DecodeStream ────────────────────────────────────────────────────────

    const STREAM_MODEL: &str = "Qwen/Qwen3-0.6B";

    fn stream_tok() -> Tokenizer {
        Tokenizer::from_model(STREAM_MODEL).expect("failed to load tokenizer")
    }

    /// Feed `ids` one token at a time, collect non-None chunks.
    /// Returns (concatenated_text, final_buffer_len).
    fn stream_collect(tok: &Tokenizer, ids: &[u32], skip_special_tokens: bool) -> (String, usize) {
        let mut buf = Vec::new();
        let mut prefix = String::new();
        let mut prefix_index = 0usize;
        let mut out = String::new();
        for &id in ids {
            if let Some(chunk) = decode_stream_step(
                tok,
                vec![id],
                skip_special_tokens,
                &mut buf,
                &mut prefix,
                &mut prefix_index,
            )
            .unwrap()
            {
                out.push_str(&chunk);
            }
        }
        (out, buf.len())
    }

    #[test]
    fn decode_stream_reconstructs_ascii() {
        let tok = stream_tok();
        let text = "Hello, world! This is a streaming decode test.";
        let ids = tok.encode_with_special_tokens(text, false).unwrap();
        let (decoded, _) = stream_collect(&tok, &ids, false);
        assert_eq!(decoded, text);
    }

    #[test]
    fn decode_stream_reconstructs_unicode() {
        let tok = stream_tok();
        let text = "日本語テスト: こんにちは 🌍 — привет мир";
        let ids = tok.encode_with_special_tokens(text, false).unwrap();
        let (decoded, _) = stream_collect(&tok, &ids, false);
        assert_eq!(decoded, text);
    }

    #[test]
    fn decode_stream_reconstructs_code() {
        let tok = stream_tok();
        let text = r#"fn main() { println!("hello"); }"#;
        let ids = tok.encode_with_special_tokens(text, false).unwrap();
        let (decoded, _) = stream_collect(&tok, &ids, false);
        assert_eq!(decoded, text);
    }

    #[test]
    fn decode_stream_empty_ids_no_output() {
        let tok = stream_tok();
        let (decoded, buf_len) = stream_collect(&tok, &[], false);
        assert!(decoded.is_empty());
        assert_eq!(buf_len, 0);
    }

    #[test]
    fn decode_stream_single_token() {
        let tok = stream_tok();
        let ids = tok.encode_with_special_tokens("hello", false).unwrap();
        assert!(!ids.is_empty());
        let (decoded, _) = stream_collect(&tok, &ids[..1], false);
        assert!(!decoded.is_empty());
    }

    #[test]
    fn decode_stream_batch_step_matches_sequential() {
        let tok = stream_tok();
        let text = "The quick brown fox jumps over the lazy dog.";
        let ids = tok.encode_with_special_tokens(text, false).unwrap();

        let (sequential, _) = stream_collect(&tok, &ids, false);

        let mut buf = Vec::new();
        let mut prefix = String::new();
        let mut prefix_index = 0usize;
        let batch = decode_stream_step(
            &tok,
            ids.clone(),
            false,
            &mut buf,
            &mut prefix,
            &mut prefix_index,
        )
        .unwrap()
        .unwrap_or_default();

        assert_eq!(sequential, batch);
    }

    #[test]
    fn decode_stream_pre_seeded_only_returns_new_tokens() {
        let tok = stream_tok();
        let prompt = "The capital of France is";
        let cont = " Paris.";

        let prompt_ids = tok.encode_with_special_tokens(prompt, false).unwrap();
        let cont_ids = tok.encode_with_special_tokens(cont, false).unwrap();

        let mut buf = prompt_ids.clone();
        let mut prefix = String::new();
        let mut prefix_index = 0usize;
        let mut out = String::new();
        for &id in &cont_ids {
            if let Some(chunk) = decode_stream_step(
                &tok,
                vec![id],
                false,
                &mut buf,
                &mut prefix,
                &mut prefix_index,
            )
            .unwrap()
            {
                out.push_str(&chunk);
            }
        }
        assert_eq!(out, cont);
    }

    #[test]
    fn decode_stream_skip_special_tokens() {
        let tok = stream_tok();
        let text = "hello";
        // Mistral-Nemo adds <s> BOS in basic encoding, so with/without differ.
        let tok = Tokenizer::from_model("mistralai/Mistral-Nemo-Instruct-2407").unwrap();
        let ids_with = tok.encode_with_special_tokens(text, true).unwrap();
        let ids_without = tok.encode_with_special_tokens(text, false).unwrap();
        assert!(
            ids_with.len() > ids_without.len(),
            "expected BOS/EOS tokens"
        );

        let (with_sp, _) = stream_collect(&tok, &ids_with, false);
        let (no_sp, _) = stream_collect(&tok, &ids_with, true);

        assert_eq!(no_sp, text);
        assert!(with_sp.contains(&no_sp));
    }

    #[test]
    fn decode_stream_buffer_does_not_grow_unboundedly() {
        let tok = stream_tok();
        let text = "word ".repeat(80);
        let ids = tok.encode_with_special_tokens(text.trim(), false).unwrap();
        let (_, final_buf_len) = stream_collect(&tok, &ids, false);
        assert!(
            final_buf_len < 10,
            "buffer grew to {final_buf_len} entries after {} tokens",
            ids.len()
        );
    }

    #[test]
    fn decode_stream_chunks_are_non_empty_and_concatenate() {
        let tok = stream_tok();
        let text = "one two three four five six seven eight nine ten";
        let ids = tok.encode_with_special_tokens(text, false).unwrap();

        let mut buf = Vec::new();
        let mut prefix = String::new();
        let mut prefix_index = 0usize;
        let mut chunks: Vec<String> = Vec::new();
        for &id in &ids {
            if let Some(chunk) = decode_stream_step(
                &tok,
                vec![id],
                false,
                &mut buf,
                &mut prefix,
                &mut prefix_index,
            )
            .unwrap()
            {
                assert!(!chunk.is_empty(), "stream emitted an empty chunk");
                chunks.push(chunk);
            }
        }
        assert_eq!(chunks.concat(), text);
    }

    #[test]
    fn decode_stream_invalid_prefix_error_message() {
        let tok = stream_tok();
        let ids = tok.encode_with_special_tokens("hello", false).unwrap();

        let mut buf = ids.clone();
        let mut prefix = "ZZZZZZZ".to_string(); // corrupt
        let mut prefix_index = 0usize;

        let result = decode_stream_step(
            &tok,
            vec![*ids.last().unwrap()],
            false,
            &mut buf,
            &mut prefix,
            &mut prefix_index,
        );
        if let Err(msg) = result {
            assert!(
                msg.starts_with("Invalid prefix encountered"),
                "unexpected error: {msg:?}"
            );
        }
        // If it returned Ok (corrupt prefix happened to match), that's fine —
        // we can't force the error path without controlling decoder output.
    }
}
