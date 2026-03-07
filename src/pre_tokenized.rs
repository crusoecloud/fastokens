use std::{ops::Range, sync::OnceLock};

use rayon::prelude::*;

/// Minimum number of splits before switching to parallel tokenization. Below
/// this threshold the rayon overhead exceeds the parallelism gain.
const PARALLEL_THRESHOLD: usize = 128;

/// Dedicated rayon thread pool for BPE tokenization.
/// Using a fixed-size pool ensures the same threads are reused across calls,
/// keeping their thread-local caches warm.
fn bpe_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("failed to build BPE thread pool")
    })
}

/// A split within a [`PreTokenizedString`]'s buffer.
///
/// Each split is either a text segment to be tokenized by the model, or a
/// pre-assigned token ID (from added tokens).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    /// Byte range into the parent buffer.
    pub range: Range<usize>,
    /// If `Some`, this split is an added token and should emit this ID directly
    /// rather than being passed to the model.
    pub token_id: Option<u32>,
}

/// A single-buffer representation of pre-tokenized text.
///
/// Stores all normalized/transformed text in one contiguous `String` and tracks
/// splits as byte ranges into that buffer. This avoids per-segment `String`
/// allocations during pre-tokenization.
#[derive(Debug, Clone)]
pub struct PreTokenizedString {
    buffer: String,
    splits: Vec<Split>,
}

impl PreTokenizedString {
    /// Create from a single text span (no pre-assigned tokens).
    ///
    /// If `text` is empty, the resulting `PreTokenizedString` has no splits.
    pub fn from_text(text: &str) -> Self {
        let splits = if text.is_empty() {
            Vec::new()
        } else {
            vec![Split {
                range: 0..text.len(),
                token_id: None,
            }]
        };
        Self {
            buffer: text.to_string(),
            splits,
        }
    }

    /// Create with a pre-built buffer and splits.
    pub fn new(buffer: String, splits: Vec<Split>) -> Self {
        Self { buffer, splits }
    }

    /// The underlying buffer.
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// The current splits.
    pub fn splits(&self) -> &[Split] {
        &self.splits
    }

    /// Text content of a split.
    pub fn split_text(&self, split: &Split) -> &str {
        &self.buffer[split.range.clone()]
    }

    /// Replace the buffer and splits entirely.
    ///
    /// Used by pre-tokenizers that transform content (e.g. ByteLevel byte
    /// encoding).
    pub fn set_buffer(&mut self, buffer: String, splits: Vec<Split>) {
        self.buffer = buffer;
        self.splits = splits;
    }

    /// Replace only the splits, keeping the buffer unchanged.
    ///
    /// Used by pre-tokenizers that only re-slice without transforming content
    /// (e.g. Split).
    pub fn refine_splits(&mut self, splits: Vec<Split>) {
        self.splits = splits;
    }

    /// Tokenize all splits, using rayon parallelism for large inputs.
    ///
    /// For each text split, calls `tokenize_fn` to append token IDs directly
    /// into the output buffer. Added-token splits emit their pre-assigned ID
    /// directly. When there are enough splits, chunks are processed in
    /// parallel.
    pub fn tokenize<F>(&self, tokenize_fn: F) -> Result<Vec<u32>, String>
    where
        F: Fn(&str, &mut Vec<u32>) -> Result<(), String> + Sync,
    {
        if self.splits.len() < PARALLEL_THRESHOLD {
            return self.tokenize_sequential(&tokenize_fn);
        }

        let pool = bpe_pool();
        let chunk_size = (self.splits.len() + pool.current_num_threads() - 1) / pool.current_num_threads();

        pool.install(|| {
            let chunk_results: Result<Vec<Vec<u32>>, String> = self
                .splits
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut ids = Vec::with_capacity(chunk.len() * 3);
                    for split in chunk {
                        if let Some(id) = split.token_id {
                            ids.push(id);
                        } else if !split.range.is_empty() {
                            let text = &self.buffer[split.range.clone()];
                            tokenize_fn(text, &mut ids)?;
                        }
                    }
                    Ok(ids)
                })
                .collect();

            let chunks = chunk_results?;
            let total: usize = chunks.iter().map(Vec::len).sum();
            let mut ids = Vec::with_capacity(total);
            for chunk_ids in chunks {
                ids.extend(chunk_ids);
            }
            Ok(ids)
        })
    }

    /// Batched tokenization: the callback receives the full buffer and a chunk
    /// of splits, allowing it to amortize per-call overhead (e.g. thread-local
    /// cache access) across the entire chunk.
    pub fn tokenize_batched<F>(&self, tokenize_fn: F) -> Result<Vec<u32>, String>
    where
        F: Fn(&str, &[Split], &mut Vec<u32>) -> Result<(), String> + Sync,
    {
        if self.splits.len() < PARALLEL_THRESHOLD {
            let mut ids = Vec::with_capacity(self.splits.len() * 2);
            tokenize_fn(&self.buffer, &self.splits, &mut ids)?;
            return Ok(ids);
        }

        let pool = bpe_pool();
        let chunk_size = (self.splits.len() + pool.current_num_threads() - 1) / pool.current_num_threads();

        pool.install(|| {
            let chunk_results: Result<Vec<Vec<u32>>, String> = self
                .splits
                .par_chunks(chunk_size)
                .map(|chunk| {
                    let mut ids = Vec::with_capacity(chunk.len() * 3);
                    tokenize_fn(&self.buffer, chunk, &mut ids)?;
                    Ok(ids)
                })
                .collect();

            let chunks = chunk_results?;
            let total: usize = chunks.iter().map(Vec::len).sum();
            let mut ids = Vec::with_capacity(total);
            for chunk_ids in chunks {
                ids.extend(chunk_ids);
            }
            Ok(ids)
        })
    }

    /// Sequential tokenization (public, for profiling).
    pub fn tokenize_sequential_pub<F>(&self, tokenize_fn: F) -> Result<Vec<u32>, String>
    where
        F: Fn(&str, &mut Vec<u32>) -> Result<(), String>,
    {
        self.tokenize_sequential(&tokenize_fn)
    }

    /// Sequential tokenization (used for small inputs).
    fn tokenize_sequential<F>(&self, tokenize_fn: &F) -> Result<Vec<u32>, String>
    where
        F: Fn(&str, &mut Vec<u32>) -> Result<(), String>,
    {
        let mut ids = Vec::with_capacity(self.splits.len() * 2);
        for split in &self.splits {
            if let Some(id) = split.token_id {
                ids.push(id);
            } else {
                let text = self.split_text(split);
                if !text.is_empty() {
                    tokenize_fn(text, &mut ids)?;
                }
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_text_empty() {
        let pts = PreTokenizedString::from_text("");
        assert!(pts.splits().is_empty());
        assert!(pts.buffer().is_empty());
    }

    #[test]
    fn from_text_single_span() {
        let pts = PreTokenizedString::from_text("hello world");
        assert_eq!(pts.splits().len(), 1);
        assert_eq!(pts.split_text(&pts.splits()[0]), "hello world");
        assert_eq!(pts.splits()[0].token_id, None);
    }

    #[test]
    fn new_with_mixed_splits() {
        let buffer = "hello<sep>world".to_string();
        let splits = vec![
            Split {
                range: 0..5,
                token_id: None,
            },
            Split {
                range: 5..10,
                token_id: Some(42),
            },
            Split {
                range: 10..15,
                token_id: None,
            },
        ];
        let pts = PreTokenizedString::new(buffer, splits);
        assert_eq!(pts.split_text(&pts.splits()[0]), "hello");
        assert_eq!(pts.split_text(&pts.splits()[1]), "<sep>");
        assert_eq!(pts.splits()[1].token_id, Some(42));
        assert_eq!(pts.split_text(&pts.splits()[2]), "world");
    }

    #[test]
    fn set_buffer_replaces() {
        let mut pts = PreTokenizedString::from_text("old");
        pts.set_buffer(
            "new text".to_string(),
            vec![Split {
                range: 0..3,
                token_id: None,
            }],
        );
        assert_eq!(pts.buffer(), "new text");
        assert_eq!(pts.split_text(&pts.splits()[0]), "new");
    }

    #[test]
    fn refine_splits_keeps_buffer() {
        let mut pts = PreTokenizedString::from_text("hello world");
        pts.refine_splits(vec![
            Split {
                range: 0..5,
                token_id: None,
            },
            Split {
                range: 5..11,
                token_id: None,
            },
        ]);
        assert_eq!(pts.buffer(), "hello world");
        assert_eq!(pts.split_text(&pts.splits()[0]), "hello");
        assert_eq!(pts.split_text(&pts.splits()[1]), " world");
    }

    #[test]
    fn tokenize_text_splits() {
        let pts = PreTokenizedString::from_text("ab");
        let ids = pts
            .tokenize(|text, out| {
                out.extend(text.bytes().map(u32::from));
                Ok(())
            })
            .unwrap();
        assert_eq!(ids, vec![97, 98]);
    }

    #[test]
    fn tokenize_mixed_splits() {
        let buffer = "helloXworld".to_string();
        let splits = vec![
            Split {
                range: 0..5,
                token_id: None,
            },
            Split {
                range: 5..6,
                token_id: Some(99),
            },
            Split {
                range: 6..11,
                token_id: None,
            },
        ];
        let pts = PreTokenizedString::new(buffer, splits);
        let ids = pts
            .tokenize(|text, out| {
                out.push(text.len() as u32);
                Ok(())
            })
            .unwrap();
        // text "hello" -> [5], token 99, text "world" -> [5]
        assert_eq!(ids, vec![5, 99, 5]);
    }

    #[test]
    fn tokenize_empty() {
        let pts = PreTokenizedString::from_text("");
        let ids = pts
            .tokenize(|_, out| {
                out.push(1);
                Ok(())
            })
            .unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn tokenize_propagates_error() {
        let pts = PreTokenizedString::from_text("x");
        let err = pts
            .tokenize(|_, _out| Err("boom".to_string()))
            .unwrap_err();
        assert_eq!(err, "boom");
    }
}
