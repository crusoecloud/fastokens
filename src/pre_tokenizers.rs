pub(crate) mod byte_level;
pub(crate) mod scan;
mod split;
pub(crate) mod unicode_class;

use crate::{
    json_structs::{PreTokenizerConfig, PreTokenizerKind},
    pre_tokenized::PreTokenizedString,
};

pub use self::{
    byte_level::ByteLevel,
    split::{Pattern, Pcre2Limits, Split, SplitBehavior, SplitConfig},
};

pub(crate) use self::byte_level::BYTE_TO_CHAR;

/// Errors from constructing or running a pre-tokenizer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A JSON value could not be deserialized into the expected type (e.g. an
    /// unrecognized pattern format or behavior string).
    #[error("invalid config value: {0}")]
    Json(#[from] serde_json::Error),

    /// The regex pattern failed to compile or exceeded backtracking limits at
    /// runtime.
    #[error("regex error: {0}")]
    Regex(#[from] fancy_regex::Error),

    /// The pre-tokenizer type is not yet implemented.
    #[error("unsupported pre-tokenizer type: {0}")]
    Unsupported(String),
}

/// A compiled pre-tokenizer ready for use.
#[derive(Clone, Debug)]
pub enum PreTokenizer {
    ByteLevel(ByteLevel),
    Split(Split),
    Sequence(Vec<PreTokenizer>),
}

impl PreTokenizer {
    /// Build a pre-tokenizer from its JSON configuration.
    pub fn from_config(config: PreTokenizerConfig) -> Result<Self, Error> {
        Self::from_config_with_limits(config, Pcre2Limits::default())
    }

    pub fn from_config_with_limits(
        config: PreTokenizerConfig,
        limits: Pcre2Limits,
    ) -> Result<Self, Error> {
        match config {
            PreTokenizerConfig::ByteLevel(bl) => Ok(Self::ByteLevel(bl)),
            PreTokenizerConfig::Split(s) => Ok(Self::Split(Split::from_split_config_with_limits(
                s, limits,
            )?)),
            PreTokenizerConfig::Sequence { pretokenizers } => {
                let steps = pretokenizers
                    .into_iter()
                    .map(|config| Self::from_config_with_limits(config, limits))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Sequence(steps))
            }
            // HF `Digits` splits on `char::is_numeric()`, i.e. Unicode
            // Nd | Nl | No. `individual_digits: false` uses
            // SplitDelimiterBehavior::Contiguous so a run of digits stays in
            // one piece; `true` uses Isolated so every digit becomes its own
            // piece. Both are expressible as a Split, so reuse that engine
            // rather than adding a separate runtime variant.
            PreTokenizerConfig::Digits { individual_digits } => {
                let behavior = if individual_digits {
                    SplitBehavior::Isolated
                } else {
                    SplitBehavior::Contiguous
                };
                let config = SplitConfig {
                    pattern: Pattern::Regex(r"[\p{Nd}\p{Nl}\p{No}]".to_string()),
                    behavior,
                    invert: false,
                };
                Ok(Self::Split(Split::from_split_config_with_limits(
                    config, limits,
                )?))
            }
            PreTokenizerConfig::Other(v) => {
                let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("unknown");
                Err(Error::Unsupported(typ.to_string()))
            }
            other => {
                let kind = PreTokenizerKind::from(&other);
                Err(Error::Unsupported(kind.to_string()))
            }
        }
    }

    /// Refine the splits of `pts` in place.
    pub fn pre_tokenize(&self, pts: &mut PreTokenizedString) -> Result<(), Error> {
        match self {
            Self::ByteLevel(bl) => bl.pre_tokenize(pts),
            Self::Split(s) => s.pre_tokenize(pts),
            Self::Sequence(steps) => {
                for step in steps {
                    step.pre_tokenize(pts)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn pieces(config: serde_json::Value, text: &str) -> Vec<String> {
        let config: PreTokenizerConfig = serde_json::from_value(config).unwrap();
        let pt = PreTokenizer::from_config(config).unwrap();
        let mut pts = PreTokenizedString::from_text(text);
        pt.pre_tokenize(&mut pts).unwrap();
        pts.splits()
            .iter()
            .map(|s| pts.split_text(s).to_string())
            .collect()
    }

    #[test]
    fn digits_contiguous_keeps_runs_together() {
        assert_eq!(
            pieces(json!({"type": "Digits"}), "abc123def45"),
            vec!["abc", "123", "def", "45"]
        );
    }

    #[test]
    fn digits_individual_isolates_every_digit() {
        assert_eq!(
            pieces(
                json!({"type": "Digits", "individual_digits": true}),
                "abc123"
            ),
            vec!["abc", "1", "2", "3"]
        );
    }

    #[test]
    fn digits_covers_non_ascii_numerics() {
        // `char::is_numeric()` is Nd | Nl | No, so Arabic-Indic digits and
        // Roman numerals count too -- matching HF's behavior.
        assert_eq!(
            pieces(json!({"type": "Digits"}), "a\u{0661}\u{0662}b\u{2171}c"),
            vec!["a", "\u{0661}\u{0662}", "b", "\u{2171}", "c"]
        );
    }
}
