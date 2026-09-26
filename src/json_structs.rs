use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use strum::EnumDiscriminants;

use crate::{models, pre_tokenizers};

/// An entry in the `added_tokens` array of `tokenizer.json`.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
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

/// Parsed `tokenizer_config.json`.
#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    #[serde(default)]
    pub added_tokens_decoder: HashMap<String, Value>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl TokenizerConfig {
    /// Convert `added_tokens_decoder` into the same shape as `tokenizer.json`'s
    /// `added_tokens` array.
    pub fn added_token_configs(&self) -> Result<Vec<AddedTokenConfig>, String> {
        let mut entries = Vec::with_capacity(self.added_tokens_decoder.len());
        for (id, value) in &self.added_tokens_decoder {
            let id = id
                .parse::<u32>()
                .map_err(|e| format!("invalid added token id {id:?}: {e}"))?;
            let mut value = value.clone();
            let object = value
                .as_object_mut()
                .ok_or_else(|| format!("added token {id} is not a JSON object"))?;
            object.insert("id".into(), Value::from(id));
            let entry: AddedTokenConfig = serde_json::from_value(value)
                .map_err(|e| format!("failed to parse added token {id}: {e}"))?;
            entries.push(entry);
        }
        entries.sort_by_key(|entry| entry.id);
        Ok(entries)
    }

    /// The declared `tokenizer_class`, if any (e.g. `"TikTokenTokenizer"`).
    #[must_use]
    pub fn tokenizer_class(&self) -> Option<&str> {
        self.extra.get("tokenizer_class")?.as_str()
    }

    /// The implementing class named by `auto_map.AutoTokenizer`, if any.
    ///
    /// `auto_map.AutoTokenizer` is `[slow_class, fast_class]`; this returns the
    /// slow (first) entry, e.g. `"tokenization_kimi.TikTokenTokenizer"`. Its
    /// module prefix identifies which model family implements the tokenizer.
    #[must_use]
    pub fn auto_map_tokenizer(&self) -> Option<&str> {
        self.extra
            .get("auto_map")?
            .get("AutoTokenizer")?
            .get(0)?
            .as_str()
    }
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
    #[serde(default)]
    pub truncation: Option<TruncationConfig>,
    #[serde(default)]
    pub padding: Option<PaddingConfig>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Side of a sequence that truncation removes from or padding adds to.
///
/// Serialized by variant name (`Left`), as in `tokenizer.json`; `FromStr` and
/// `Into<&str>` use HF's Python names (`left`), which fastokens 0.3.3 also wrote
/// to `tokenizer.json` and are still read.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    strum::EnumString,
    strum::IntoStaticStr,
)]
#[strum(serialize_all = "snake_case")]
pub enum Direction {
    #[serde(alias = "left")]
    Left,
    #[default]
    #[serde(alias = "right")]
    Right,
}

/// How truncation distributes removals across a pair of sequences. fastokens
/// only encodes single sequences, and `OnlySecond` fails to truncate those, as
/// in HF. Serialized by variant name (`OnlyFirst`); `FromStr` and
/// `Into<&str>` use HF's Python names (`only_first`), which are also read, as
/// for [`Direction`].
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    Deserialize,
    Serialize,
    strum::EnumString,
    strum::IntoStaticStr,
)]
#[strum(serialize_all = "snake_case")]
pub enum TruncationStrategy {
    #[default]
    #[serde(alias = "longest_first")]
    LongestFirst,
    #[serde(alias = "only_first")]
    OnlyFirst,
    #[serde(alias = "only_second")]
    OnlySecond,
}

/// The `truncation` block of `tokenizer.json`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct TruncationConfig {
    #[serde(default)]
    pub direction: Direction,
    pub max_length: usize,
    #[serde(default)]
    pub strategy: TruncationStrategy,
    /// Overlap of overflowing pieces. Kept for round-tripping only: fastokens
    /// returns no overflow.
    #[serde(default)]
    pub stride: usize,
}

/// Target length of padding: the longest encoding in the batch, or fixed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub enum PaddingStrategy {
    #[default]
    BatchLongest,
    Fixed(usize),
}

/// The `padding` block of `tokenizer.json`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(from = "RawPaddingConfig")]
pub struct PaddingConfig {
    pub strategy: PaddingStrategy,
    pub direction: Direction,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: String,
}

/// [`PaddingConfig`] as read, also accepting fastokens 0.3.3's format, which
/// gave a fixed length as `length` (`null` for the batch's longest) instead of
/// a `strategy`.
#[derive(Deserialize)]
struct RawPaddingConfig {
    #[serde(default)]
    strategy: Option<PaddingStrategy>,
    #[serde(default)]
    length: Option<usize>,
    #[serde(default)]
    direction: Direction,
    #[serde(default)]
    pad_to_multiple_of: Option<usize>,
    #[serde(default)]
    pad_id: u32,
    #[serde(default)]
    pad_type_id: u32,
    #[serde(default = "default_pad_token")]
    pad_token: String,
}

impl From<RawPaddingConfig> for PaddingConfig {
    fn from(raw: RawPaddingConfig) -> Self {
        let strategy = raw.strategy.unwrap_or(match raw.length {
            Some(n) => PaddingStrategy::Fixed(n),
            None => PaddingStrategy::BatchLongest,
        });
        Self {
            strategy,
            direction: raw.direction,
            pad_to_multiple_of: raw.pad_to_multiple_of,
            pad_id: raw.pad_id,
            pad_type_id: raw.pad_type_id,
            pad_token: raw.pad_token,
        }
    }
}

impl TruncationConfig {
    /// Truncate a single sequence's model ids as HF does, leaving room for the
    /// `n_added` special tokens that post-processing adds next.
    pub fn truncate(&self, ids: &mut Vec<u32>, n_added: usize) -> Result<(), String> {
        // HF subtracts without checking, so a `max_length` below `n_added`
        // wraps around and nothing is truncated.
        let room = self.max_length.wrapping_sub(n_added);
        if ids.len() <= room {
            return Ok(());
        }
        // HF only checks the strategy when `room` is nonzero.
        if self.strategy == TruncationStrategy::OnlySecond && room > 0 {
            return Err("truncation strategy OnlySecond needs a second sequence".into());
        }
        match self.direction {
            Direction::Right => ids.truncate(room),
            Direction::Left => {
                ids.drain(..ids.len() - room);
            }
        }
        Ok(())
    }
}

fn default_pad_token() -> String {
    "[PAD]".to_string()
}

impl PaddingConfig {
    /// Length to pad to, as HF computes it: the fixed length, or the longest
    /// encoding's, rounded up to `pad_to_multiple_of`.
    pub fn target(&self, longest: usize) -> usize {
        let base = match self.strategy {
            PaddingStrategy::Fixed(n) => n,
            PaddingStrategy::BatchLongest => longest,
        };
        match self.pad_to_multiple_of {
            Some(m) if m > 0 => base.div_ceil(m) * m,
            _ => base,
        }
    }

    /// Pad `ids` with `pad_id` up to `target` (never shortening).
    pub fn pad(&self, ids: &mut Vec<u32>, target: usize) {
        let Some(n) = target.checked_sub(ids.len()).filter(|&n| n > 0) else {
            return;
        };
        match self.direction {
            Direction::Right => ids.resize(target, self.pad_id),
            Direction::Left => {
                ids.splice(0..0, std::iter::repeat_n(self.pad_id, n));
            }
        }
    }
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
    Split(pre_tokenizers::SplitConfig),
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
    Bpe(Box<models::bpe::Bpe>),
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn reads_truncation_and_padding_written_by_fastokens_0_3_3() {
        let truncation: TruncationConfig = serde_json::from_value(json!({
            "max_length": 8, "stride": 1, "strategy": "only_first", "direction": "left",
        }))
        .unwrap();
        assert_eq!(
            truncation,
            TruncationConfig {
                direction: Direction::Left,
                max_length: 8,
                strategy: TruncationStrategy::OnlyFirst,
                stride: 1,
            }
        );

        let padding = |length: Value| -> PaddingConfig {
            serde_json::from_value(json!({
                "direction": "left", "pad_id": 3, "pad_type_id": 1, "pad_token": "<pad>",
                "length": length, "pad_to_multiple_of": null,
            }))
            .unwrap()
        };
        assert_eq!(
            padding(json!(12)),
            PaddingConfig {
                strategy: PaddingStrategy::Fixed(12),
                direction: Direction::Left,
                pad_to_multiple_of: None,
                pad_id: 3,
                pad_type_id: 1,
                pad_token: "<pad>".into(),
            }
        );
        assert_eq!(padding(Value::Null).strategy, PaddingStrategy::BatchLongest);
    }

    #[test]
    fn padding_round_trips_in_tokenizer_json_format() {
        let padding = PaddingConfig {
            strategy: PaddingStrategy::Fixed(12),
            direction: Direction::Left,
            pad_to_multiple_of: Some(4),
            pad_id: 3,
            pad_type_id: 1,
            pad_token: "<pad>".into(),
        };
        let value = serde_json::to_value(&padding).unwrap();
        assert_eq!(value["strategy"], json!({"Fixed": 12}));
        assert_eq!(
            serde_json::from_value::<PaddingConfig>(value).unwrap(),
            padding
        );
    }
}
