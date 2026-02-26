use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;
use strum::EnumDiscriminants;

use crate::{models, pre_tokenizers};

/// An entry in the `added_tokens` array of `tokenizer.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct AddedTokenConfig {
    /// Token ID.
    pub id: u32,
    /// Literal text content that triggers this token.
    pub content: String,
    /// Whether the token should only match as a whole word.
    #[serde(default)]
    pub single_word: bool,
    /// Whether to strip whitespace on the left when matching.
    #[serde(default)]
    pub lstrip: bool,
    /// Whether to strip whitespace on the right when matching.
    #[serde(default)]
    pub rstrip: bool,
    /// Whether the content should be matched against normalized
    /// text.
    #[serde(default)]
    pub normalized: bool,
    /// Whether this is a "special" token (e.g. BOS/EOS). All added
    /// tokens are matched regardless of this flag; it only affects
    /// post-processing.
    #[serde(default)]
    pub special: bool,
}

/// Parsed `tokenizer.json`.
///
/// The five pipeline steps are deserialized into typed enums. Any other
/// top-level keys (e.g. `version`) are captured in [`extra`](Self::extra).
#[derive(Debug, Deserialize)]
pub struct TokenizerJson {
    #[serde(default)]
    pub added_tokens: Vec<AddedTokenConfig>,
    pub normalizer: Option<NormalizerConfig>,
    pub pre_tokenizer: Option<PreTokenizerConfig>,
    pub model: ModelConfig,
    pub post_processor: Option<PostProcessorConfig>,
    pub decoder: Option<DecoderConfig>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Normalizer pipeline step from `tokenizer.json`.
///
/// Unrecognised types are stored as [`Other`](Self::Other).
#[derive(Clone, Debug, Deserialize, EnumDiscriminants)]
#[strum_discriminants(name(NormalizerKind), derive(strum::Display, Hash, Ord, PartialOrd))]
#[serde(tag = "type")]
pub enum NormalizerConfig {
    Sequence {
        #[serde(default)]
        normalizers: Vec<NormalizerConfig>,
    },
    #[serde(rename = "NFC")]
    #[strum_discriminants(strum(to_string = "NFC"))]
    Nfc,
    #[serde(rename = "NFD")]
    #[strum_discriminants(strum(to_string = "NFD"))]
    Nfd,
    #[serde(rename = "NFKC")]
    #[strum_discriminants(strum(to_string = "NFKC"))]
    Nfkc,
    #[serde(rename = "NFKD")]
    #[strum_discriminants(strum(to_string = "NFKD"))]
    Nfkd,
    Lowercase,
    StripAccents,
    Strip {
        #[serde(default)]
        strip_left: bool,
        #[serde(default)]
        strip_right: bool,
    },
    Prepend {
        #[serde(default)]
        prepend: String,
    },
    Replace {
        pattern: Value,
        #[serde(default)]
        content: String,
    },
    BertNormalizer {
        #[serde(default)]
        clean_text: bool,
        #[serde(default)]
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        #[serde(default)]
        lowercase: bool,
    },
    ByteLevel,
    #[serde(untagged)]
    Other(Value),
}

/// Pre-tokenizer pipeline step from `tokenizer.json`.
#[derive(Clone, Debug, Deserialize, EnumDiscriminants)]
#[strum_discriminants(name(PreTokenizerKind), derive(strum::Display, Hash, Ord, PartialOrd))]
#[serde(tag = "type")]
pub enum PreTokenizerConfig {
    Sequence {
        #[serde(default)]
        pretokenizers: Vec<PreTokenizerConfig>,
    },
    ByteLevel(pre_tokenizers::ByteLevel),
    Whitespace,
    WhitespaceSplit,
    Split(pre_tokenizers::Split),
    Punctuation {
        #[serde(default)]
        behavior: String,
    },
    Metaspace {
        #[serde(default = "default_meta_replacement")]
        replacement: char,
        prepend_scheme: Option<String>,
        add_prefix_space: Option<bool>,
    },
    Digits {
        #[serde(default)]
        individual_digits: bool,
    },
    BertPreTokenizer,
    UnicodeScripts,
    #[serde(untagged)]
    Other(Value),
}

fn default_meta_replacement() -> char {
    '\u{2581}' // ▁
}

/// Tokenization model from `tokenizer.json`.
#[derive(Clone, Debug, Deserialize, EnumDiscriminants)]
#[strum_discriminants(name(ModelKind), derive(strum::Display, Hash, Ord, PartialOrd))]
#[serde(tag = "type")]
pub enum ModelConfig {
    #[serde(rename = "BPE")]
    #[strum_discriminants(strum(to_string = "BPE"))]
    Bpe(models::bpe::Bpe),
    WordPiece {
        #[serde(default)]
        vocab: HashMap<String, u32>,
        #[serde(default)]
        unk_token: String,
        #[serde(default)]
        continuing_subword_prefix: String,
        max_input_chars_per_word: Option<u64>,
    },
    WordLevel {
        #[serde(default)]
        vocab: HashMap<String, u32>,
        #[serde(default)]
        unk_token: String,
    },
    Unigram {
        #[serde(default)]
        vocab: Vec<(String, f64)>,
        unk_id: Option<u32>,
        #[serde(default)]
        byte_fallback: bool,
    },
    #[serde(untagged)]
    Other(Value),
}

/// Post-processor pipeline step from `tokenizer.json`.
#[derive(Clone, Debug, Deserialize, EnumDiscriminants)]
#[strum_discriminants(name(PostProcessorKind), derive(strum::Display, Hash, Ord, PartialOrd))]
#[serde(tag = "type")]
pub enum PostProcessorConfig {
    Sequence {
        #[serde(default)]
        processors: Vec<PostProcessorConfig>,
    },
    ByteLevel {
        #[serde(default)]
        add_prefix_space: bool,
        #[serde(default)]
        trim_offsets: bool,
        #[serde(default)]
        use_regex: bool,
    },
    TemplateProcessing {
        #[serde(default)]
        single: Value,
        #[serde(default)]
        pair: Value,
        #[serde(default)]
        special_tokens: Value,
    },
    BertProcessing {
        sep: (String, u32),
        cls: (String, u32),
    },
    RobertaProcessing {
        sep: (String, u32),
        cls: (String, u32),
        #[serde(default)]
        trim_offsets: bool,
        #[serde(default)]
        add_prefix_space: bool,
    },
    #[serde(untagged)]
    Other(Value),
}

/// Decoder pipeline step from `tokenizer.json`.
#[derive(Clone, Debug, Deserialize, EnumDiscriminants)]
#[strum_discriminants(name(DecoderKind), derive(strum::Display, Hash, Ord, PartialOrd))]
#[serde(tag = "type")]
pub enum DecoderConfig {
    Sequence {
        #[serde(default)]
        decoders: Vec<DecoderConfig>,
    },
    ByteLevel,
    WordPiece {
        #[serde(default = "default_wp_prefix")]
        prefix: String,
        #[serde(default)]
        cleanup: bool,
    },
    Metaspace {
        #[serde(default = "default_meta_replacement")]
        replacement: char,
        prepend_scheme: Option<String>,
        add_prefix_space: Option<bool>,
    },
    #[serde(rename = "BPEDecoder")]
    #[strum_discriminants(strum(to_string = "BPEDecoder"))]
    BpeDecoder {
        #[serde(default)]
        suffix: String,
    },
    #[serde(rename = "CTC")]
    #[strum_discriminants(strum(to_string = "CTC"))]
    Ctc {
        #[serde(default)]
        pad_token: String,
        word_delimiter_token: Option<String>,
        #[serde(default)]
        cleanup: bool,
    },
    Strip {
        #[serde(default = "default_space")]
        content: char,
        #[serde(default)]
        start: usize,
        #[serde(default)]
        stop: usize,
    },
    Replace {
        pattern: Value,
        #[serde(default)]
        content: String,
    },
    Fuse,
    ByteFallback,
    #[serde(untagged)]
    Other(Value),
}

fn default_wp_prefix() -> String {
    "##".to_string()
}

fn default_space() -> char {
    ' '
}
