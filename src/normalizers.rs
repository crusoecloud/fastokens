mod lowercase;
mod nfc;
mod prepend;
mod replace;

use std::borrow::Cow;

pub use self::lowercase::Lowercase;
pub use self::nfc::Nfc;
pub use self::prepend::Prepend;
pub use self::replace::Replace;
use crate::json_structs::{NormalizerConfig, NormalizerKind};

/// Errors from constructing a normalizer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid config value: {0}")]
    Json(#[from] serde_json::Error),

    #[error("regex error: {0}")]
    Regex(#[from] fancy_regex::Error),

    #[error("unsupported normalizer type: {0}")]
    Unsupported(String),
}

/// A compiled normalizer ready for use.
#[derive(Debug)]
pub enum Normalizer {
    Lowercase(Lowercase),
    Nfc(Nfc),
    Prepend(Prepend),
    Replace(Replace),
    Sequence(Vec<Normalizer>),
}

impl Normalizer {
    /// Build a normalizer from its JSON configuration.
    pub fn from_config(config: NormalizerConfig) -> Result<Self, Error> {
        match config {
            NormalizerConfig::Lowercase => Ok(Self::Lowercase(Lowercase)),
            NormalizerConfig::Nfc => Ok(Self::Nfc(Nfc)),
            NormalizerConfig::Prepend { prepend } => Ok(Self::Prepend(Prepend::new(prepend))),
            NormalizerConfig::Replace { pattern, content } => {
                Ok(Self::Replace(Replace::from_config(pattern, content)?))
            }
            NormalizerConfig::Sequence { normalizers } => {
                let steps = normalizers
                    .into_iter()
                    .map(Self::from_config)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Sequence(steps))
            }
            NormalizerConfig::Other(v) => {
                let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("unknown");
                Err(Error::Unsupported(typ.to_string()))
            }
            other => {
                let kind = NormalizerKind::from(&other);
                Err(Error::Unsupported(kind.to_string()))
            }
        }
    }

    /// Whether normalizing `input` provably leaves it unchanged, checked without
    /// normalizing. Conservative: `false` means "maybe changes". NFC (on ASCII or
    /// already-composed text) and Lowercase (on text with nothing to lower) are
    /// the steps that can be the identity on real text; a sequence is the
    /// identity when each of its steps is.
    pub fn is_identity_on(&self, input: &str) -> bool {
        match self {
            Self::Lowercase(lowercase) => lowercase.is_normalized(input),
            Self::Nfc(nfc) => nfc.is_normalized(input),
            Self::Prepend(_) | Self::Replace(_) => false,
            Self::Sequence(steps) => steps.iter().all(|s| s.is_identity_on(input)),
        }
    }

    /// Normalize `input`, returning `Cow::Borrowed` when unchanged.
    pub fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        match self {
            Self::Lowercase(lowercase) => lowercase.normalize(input),
            Self::Nfc(nfc) => nfc.normalize(input),
            Self::Prepend(prepend) => prepend.normalize(input),
            Self::Replace(replace) => replace.normalize(input),
            Self::Sequence(steps) => {
                let mut current = Cow::Borrowed(input);
                for step in steps {
                    current = match current {
                        Cow::Borrowed(s) => step.normalize(s),
                        Cow::Owned(s) => Cow::Owned(step.normalize(&s).into_owned()),
                    };
                }
                current
            }
        }
    }
}
