mod nfc;

use std::borrow::Cow;

pub use self::nfc::Nfc;
use crate::json_structs::{NormalizerConfig, NormalizerKind};

/// Errors from constructing a normalizer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported normalizer type: {0}")]
    Unsupported(String),
}

/// A compiled normalizer ready for use.
#[derive(Debug)]
pub enum Normalizer {
    Nfc(Nfc),
    Sequence(Vec<Normalizer>),
}

impl Normalizer {
    /// Build a normalizer from its JSON configuration.
    pub fn from_config(config: NormalizerConfig) -> Result<Self, Error> {
        match config {
            NormalizerConfig::Nfc => Ok(Self::Nfc(Nfc)),
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

    /// Normalize `input`, returning `Cow::Borrowed` when unchanged.
    pub fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        match self {
            Self::Nfc(nfc) => nfc.normalize(input),
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
