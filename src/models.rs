pub mod bpe;

use self::bpe::Bpe;
use crate::json_structs::{ModelConfig, ModelKind};

#[derive(Clone, Debug, thiserror::Error)]
#[error("Error building model: {0}")]
pub struct BuildError(pub String);

#[derive(Clone, Debug, thiserror::Error)]
#[error("Tokenization failed")]
pub struct TokenizeError(pub String);

/// A constructed tokenization model ready for encoding.
#[derive(Debug)]
pub enum Model {
    Bpe(Bpe),
}

impl Model {
    /// Build a model from its JSON configuration.
    ///
    /// Takes the config by value to avoid cloning large structures like the BPE
    /// automaton.
    pub fn from_config(config: ModelConfig) -> Result<Self, BuildError> {
        match config {
            ModelConfig::Bpe(bpe) => Ok(Self::Bpe(bpe.clone())),
            other => {
                let kind = ModelKind::from(&other);
                Err(BuildError(format!("unsupported model type {kind}")))
            }
        }
    }

    /// Tokenize a pre-tokenized piece of text into token IDs.
    pub fn tokenize(&self, input: &str) -> Result<Vec<u32>, TokenizeError> {
        match self {
            Self::Bpe(bpe) => bpe.tokenize(input),
        }
    }

    /// Look up the string representation of a token ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        match self {
            Self::Bpe(bpe) => bpe.id_to_token(id),
        }
    }

    /// Look up the token ID for a string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::Bpe(bpe) => bpe.token_to_id(token),
        }
    }

    /// Return the vocabulary size (number of model tokens).
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Bpe(bpe) => bpe.vocab_size(),
        }
    }
}
