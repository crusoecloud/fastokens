use crate::json_structs::{PostProcessorConfig, PostProcessorKind};

/// Errors from constructing a post-processor.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The post-processor type is not yet implemented.
    #[error("unsupported post-processor type: {0}")]
    Unsupported(String),
}

/// A constructed post-processor.
///
/// Currently only [`ByteLevel`](Self::ByteLevel) is supported.
/// Since this tokenizer only produces token IDs (not offset
/// information), the ByteLevel post-processor is a no-op — its
/// role in the HuggingFace pipeline is to trim whitespace from
/// offsets, which we do not track.
#[derive(Debug)]
pub enum PostProcessor {
    ByteLevel,
    Sequence(Vec<PostProcessor>),
}

impl PostProcessor {
    /// Build a post-processor from its JSON configuration.
    pub fn from_config(config: PostProcessorConfig) -> Result<Self, Error> {
        match config {
            PostProcessorConfig::ByteLevel { .. } => Ok(Self::ByteLevel),
            PostProcessorConfig::Sequence { processors } => {
                let steps = processors
                    .into_iter()
                    .map(Self::from_config)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Sequence(steps))
            }
            PostProcessorConfig::Other(v) => {
                let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("unknown");
                Err(Error::Unsupported(typ.to_string()))
            }
            other => {
                let kind = PostProcessorKind::from(&other);
                Err(Error::Unsupported(kind.to_string()))
            }
        }
    }
}
