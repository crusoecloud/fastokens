use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use crate::json_structs::{PostProcessorConfig, PostProcessorKind};

/// Errors from constructing a post-processor.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The post-processor type is not yet implemented.
    #[error("unsupported post-processor type: {0}")]
    Unsupported(String),

    /// A configuration value could not be parsed.
    #[error("invalid post-processor config: {0}")]
    InvalidConfig(String),
}

// ── TemplateProcessing types ─────────────────────────────────────────────

/// Which input sequence a template piece refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum SequenceId {
    A,
    B,
}

/// A single piece in a TemplateProcessing template.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub enum TemplatePiece {
    Sequence {
        id: SequenceId,
        #[allow(dead_code)]
        type_id: u32,
    },
    SpecialToken {
        id: String,
        #[allow(dead_code)]
        type_id: u32,
    },
}

/// Special token definition as stored in the tokenizer JSON.
#[derive(Debug, Deserialize)]
struct SpecialTokenDef {
    #[allow(dead_code)]
    id: String,
    ids: Vec<u32>,
    #[allow(dead_code)]
    tokens: Vec<String>,
}

/// Compiled TemplateProcessing post-processor.
///
/// For single-sequence encoding, iterates over the `single` template:
/// `Sequence` pieces are replaced by the encoded token IDs, `SpecialToken`
/// pieces are replaced by the looked-up IDs from `special_tokens`.
#[derive(Debug)]
pub struct TemplateProcessing {
    single: Vec<TemplatePiece>,
    #[allow(dead_code)]
    pair: Vec<TemplatePiece>,
    special_tokens: HashMap<String, Vec<u32>>,
}

impl TemplateProcessing {
    /// Build from the raw JSON values in `PostProcessorConfig`.
    pub fn from_config(
        single: Value,
        pair: Value,
        special_tokens_val: Value,
    ) -> Result<Self, Error> {
        let single: Vec<TemplatePiece> = serde_json::from_value(single)
            .map_err(|e| Error::InvalidConfig(format!("single template: {e}")))?;
        let pair: Vec<TemplatePiece> = serde_json::from_value(pair)
            .map_err(|e| Error::InvalidConfig(format!("pair template: {e}")))?;

        let special_tokens_raw: HashMap<String, SpecialTokenDef> =
            serde_json::from_value(special_tokens_val)
                .map_err(|e| Error::InvalidConfig(format!("special_tokens: {e}")))?;

        let special_tokens: HashMap<String, Vec<u32>> = special_tokens_raw
            .into_iter()
            .map(|(k, v)| (k, v.ids))
            .collect();

        Ok(Self {
            single,
            pair,
            special_tokens,
        })
    }

    /// Apply the single-sequence template, inserting special token IDs
    /// around the encoded sequence.
    pub fn apply_single(&self, encoded: Vec<u32>) -> Vec<u32> {
        let mut result = Vec::with_capacity(encoded.len() + 4);
        for piece in &self.single {
            match piece {
                TemplatePiece::Sequence {
                    id: SequenceId::A, ..
                } => {
                    result.extend_from_slice(&encoded);
                }
                TemplatePiece::SpecialToken { id, .. } => {
                    if let Some(ids) = self.special_tokens.get(id) {
                        result.extend_from_slice(ids);
                    }
                }
                // Sequence B in a single template is ignored.
                _ => {}
            }
        }
        result
    }
}

// ── PostProcessor enum ───────────────────────────────────────────────────

/// A constructed post-processor.
///
/// Since this tokenizer only produces token IDs (not offset information),
/// `ByteLevel` is a no-op. `TemplateProcessing` inserts special tokens
/// (BOS/EOS/CLS/SEP) when `add_special_tokens` is true.
#[derive(Debug)]
pub enum PostProcessor {
    ByteLevel,
    TemplateProcessing(TemplateProcessing),
    Sequence(Vec<PostProcessor>),
}

impl PostProcessor {
    /// Build a post-processor from its JSON configuration.
    pub fn from_config(config: PostProcessorConfig) -> Result<Self, Error> {
        match config {
            PostProcessorConfig::ByteLevel { .. } => Ok(Self::ByteLevel),
            PostProcessorConfig::TemplateProcessing {
                single,
                pair,
                special_tokens,
            } => Ok(Self::TemplateProcessing(TemplateProcessing::from_config(
                single,
                pair,
                special_tokens,
            )?)),
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

    /// Apply post-processing to a single-sequence encoding.
    ///
    /// Only has an effect when `add_special_tokens` is true and the processor
    /// adds special tokens (e.g. `TemplateProcessing`).
    pub fn post_process_single(&self, encoded: Vec<u32>, add_special_tokens: bool) -> Vec<u32> {
        if !add_special_tokens {
            return encoded;
        }
        match self {
            Self::ByteLevel => encoded,
            Self::TemplateProcessing(tp) => tp.apply_single(encoded),
            Self::Sequence(steps) => steps.iter().fold(encoded, |acc, step| {
                step.post_process_single(acc, add_special_tokens)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_processing_bos_only() {
        let tp = TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "<s>".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([("<s>".to_string(), vec![1])]),
        };
        assert_eq!(tp.apply_single(vec![100, 200, 300]), vec![1, 100, 200, 300]);
    }

    #[test]
    fn template_processing_cls_sep() {
        let tp = TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "[CLS]".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
                TemplatePiece::SpecialToken {
                    id: "[SEP]".into(),
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([
                ("[CLS]".to_string(), vec![101]),
                ("[SEP]".to_string(), vec![102]),
            ]),
        };
        assert_eq!(tp.apply_single(vec![50, 60]), vec![101, 50, 60, 102]);
    }

    #[test]
    fn template_processing_empty_input() {
        let tp = TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "<s>".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([("<s>".to_string(), vec![1])]),
        };
        assert_eq!(tp.apply_single(vec![]), vec![1]);
    }

    #[test]
    fn parse_from_json() {
        let single = serde_json::json!([
            {"SpecialToken": {"id": "<s>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}}
        ]);
        let pair = serde_json::json!([]);
        let special_tokens = serde_json::json!({
            "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}
        });
        let tp = TemplateProcessing::from_config(single, pair, special_tokens).unwrap();
        assert_eq!(tp.apply_single(vec![10, 20]), vec![1, 10, 20]);
    }

    #[test]
    fn post_process_single_respects_flag() {
        let pp = PostProcessor::TemplateProcessing(TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "<s>".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([("<s>".to_string(), vec![1])]),
        });

        // With special tokens
        assert_eq!(pp.post_process_single(vec![10, 20], true), vec![1, 10, 20]);
        // Without special tokens
        assert_eq!(pp.post_process_single(vec![10, 20], false), vec![10, 20]);
    }
}
