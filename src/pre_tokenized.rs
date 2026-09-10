use std::{ops::Range, sync::OnceLock};

use rayon::prelude::*;

fn parallel_job_count(split_count: usize, byte_len: usize, pool_threads: usize) -> usize {
    if byte_len < SCAN_FUSED_PARALLEL_MIN {
        // Below 64 KiB total, Rayon overhead is not worthwhile.
        return 1;
    }
    // One job per ~32 KiB of work, but never more jobs than splits or threads.
    pool_threads.min(split_count).min(byte_len / (32 * 1024))
}

fn job_range(split_count: usize, job_count: usize, job_index: usize) -> Range<usize> {
    let base = split_count / job_count;
    let remainder = split_count % job_count;
    let start = job_index * base + job_index.min(remainder);
    let len = base + usize::from(job_index < remainder);
    start..start + len
}

/// On Apple Silicon, the number of performance (P) cores
/// (`hw.perflevel0.logicalcpu`). BPE tokenization is a barrier-synchronized
/// parallel stage, so scheduling work on the slower efficiency cores only adds
/// straggler latency — the fastest point is exactly the P-core count.
#[cfg(target_os = "macos")]
fn perf_core_count() -> Option<usize> {
    let name = c"hw.perflevel0.logicalcpu";
    let mut value: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            &mut value as *mut libc::c_int as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && value > 0).then_some(value as usize)
}

/// Default BPE worker-thread count. On homogeneous CPUs this is every logical
/// core; on Apple Silicon's hybrid CPUs it is the performance-core count (the
/// measured optimum — efficiency-core threads only slow the barrier down).
fn default_bpe_threads() -> usize {
    #[cfg(target_os = "macos")]
    if let Some(p) = perf_core_count() {
        return p;
    }
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Dedicated rayon thread pool for BPE tokenization.
///
/// A fixed-size pool reuses the same threads across calls, keeping their
/// thread-local caches warm. The size is [`default_bpe_threads`] capped by
/// available parallelism, overridable with the `FASTOKENS_BPE_THREADS`
/// environment variable.
fn bpe_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let avail = std::thread::available_parallelism().map_or(1, |n| n.get());
        let n = std::env::var("FASTOKENS_BPE_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(default_bpe_threads)
            .min(avail);
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
    /// directly. For sufficiently large buffers with multiple splits, chunks
    /// are processed in parallel.
    pub fn tokenize<F>(&self, tokenize_fn: F) -> Result<Vec<u32>, String>
    where
        F: Fn(&str, &mut Vec<u32>) -> Result<(), String> + Sync,
    {
        let pool = bpe_pool();
        let job_count = parallel_job_count(
            self.splits.len(),
            self.buffer.len(),
            pool.current_num_threads(),
        );
        if job_count < 2 {
            return self.tokenize_sequential(&tokenize_fn);
        }

        pool.install(|| {
            let chunk_results: Result<Vec<Vec<u32>>, String> = (0..job_count)
                .into_par_iter()
                .map(|job_index| {
                    let chunk = &self.splits[job_range(self.splits.len(), job_count, job_index)];
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
        let pool = bpe_pool();
        let job_count = parallel_job_count(
            self.splits.len(),
            self.buffer.len(),
            pool.current_num_threads(),
        );
        if job_count < 2 {
            let mut ids = Vec::with_capacity(self.splits.len() * 2);
            tokenize_fn(&self.buffer, &self.splits, &mut ids)?;
            return Ok(ids);
        }

        pool.install(|| {
            let chunk_results: Result<Vec<Vec<u32>>, String> = (0..job_count)
                .into_par_iter()
                .map(|job_index| {
                    let chunk = &self.splits[job_range(self.splits.len(), job_count, job_index)];
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

/// Minimum buffer size before the fused scan+BPE encode splits across threads.
const SCAN_FUSED_PARALLEL_MIN: usize = 64 * 1024;

/// Fused scan+BPE driver for the scanner fast path: split the buffer at newline
/// boundaries and run `per_chunk` (scan a segment into pretokens *and* BPE them)
/// on each segment in parallel, concatenating results in order.
///
/// This is a single pass over the buffer — each segment's bytes are scanned and
/// tokenized while still hot in cache — and never materializes a whole-document
/// range list, unlike scanning to a `Vec<(u32,u32)>` then tokenizing it.
pub fn tokenize_scanned<F>(buffer: &str, per_chunk: F) -> Result<Vec<u32>, String>
where
    F: Fn(&str) -> Result<Vec<u32>, String> + Sync,
{
    let bytes = buffer.as_bytes();
    let pool = bpe_pool();
    let threads = pool.current_num_threads();
    if bytes.len() < SCAN_FUSED_PARALLEL_MIN || threads < 2 {
        return per_chunk(buffer);
    }

    let n_chunks = threads.min(bytes.len() / (32 * 1024)).max(2);
    let segments = crate::pre_tokenizers::scan::newline_chunk_bounds(buffer, n_chunks);
    if segments.len() <= 1 {
        return per_chunk(buffer);
    }

    pool.install(|| {
        let parts: Result<Vec<Vec<u32>>, String> = segments
            .par_iter()
            .map(|&(s, e)| per_chunk(&buffer[s..e]))
            .collect();
        let parts = parts?;
        let total: usize = parts.iter().map(Vec::len).sum();
        let mut ids = Vec::with_capacity(total);
        for part in parts {
            ids.extend(part);
        }
        Ok(ids)
    })
}

/// A chunk's token ids together with its `(byte_offset, token_index)` reuse
/// boundaries — the payload a bounded scan produces for the prefix cache.
type IdsWithBounds = (Vec<u32>, Vec<(u32, u32)>);

/// Like [`tokenize_scanned`], but `per_chunk` also yields fine-grained reuse
/// boundaries (local `(byte_offset, token_index)` within the chunk). This
/// returns the concatenated ids and the boundaries mapped to global offsets,
/// ascending, where `ids[..token_index]` is exactly the encoding of
/// `buffer[..byte_offset]`. These are the offsets the prefix cache may cut a
/// reused prefix at.
pub fn tokenize_scanned_with_bounds<F>(buffer: &str, per_chunk: F) -> Result<IdsWithBounds, String>
where
    F: Fn(&str) -> Result<IdsWithBounds, String> + Sync,
{
    let bytes = buffer.as_bytes();
    let pool = bpe_pool();
    let threads = pool.current_num_threads();
    if bytes.len() < SCAN_FUSED_PARALLEL_MIN || threads < 2 {
        return per_chunk(buffer);
    }

    let n_chunks = threads.min(bytes.len() / (32 * 1024)).max(2);
    let segments = crate::pre_tokenizers::scan::newline_chunk_bounds(buffer, n_chunks);
    if segments.len() <= 1 {
        return per_chunk(buffer);
    }

    pool.install(|| {
        let parts: Result<Vec<IdsWithBounds>, String> = segments
            .par_iter()
            .map(|&(s, e)| per_chunk(&buffer[s..e]))
            .collect();
        let parts = parts?;
        let total: usize = parts.iter().map(|(ids, _)| ids.len()).sum();
        let n_bounds: usize = parts.iter().map(|(_, b)| b.len()).sum();
        let mut ids = Vec::with_capacity(total);
        let mut bounds = Vec::with_capacity(n_bounds);
        for ((part_ids, part_bounds), &(s, _e)) in parts.iter().zip(segments.iter()) {
            // Chunk-local (byte, token) → global by adding the chunk's byte start
            // and the running token count before this chunk.
            let base_tok = ids.len() as u32;
            let base_byte = s as u32;
            for &(bo, tk) in part_bounds {
                bounds.push((base_byte + bo, base_tok + tk));
            }
            ids.extend_from_slice(part_ids);
        }
        Ok((ids, bounds))
    })
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
        let err = pts.tokenize(|_, _out| Err("boom".to_string())).unwrap_err();
        assert_eq!(err, "boom");
    }

    #[test]
    fn parallel_jobs_follow_byte_size_and_caps() {
        assert_eq!(parallel_job_count(0, 0, 176), 1);
        assert_eq!(parallel_job_count(10_000, 64 * 1024 - 1, 176), 1);
        assert_eq!(parallel_job_count(10_000, 64 * 1024, 176), 2);
        assert_eq!(parallel_job_count(10_000, 96 * 1024 - 1, 176), 2);
        assert_eq!(parallel_job_count(10_000, 96 * 1024, 176), 3);
        assert_eq!(parallel_job_count(1, 96 * 1024, 176), 1);
        assert_eq!(parallel_job_count(2, 96 * 1024, 176), 2);
        assert_eq!(parallel_job_count(10_000, 96 * 1024, 1), 1);
        assert_eq!(parallel_job_count(10_000, 96 * 1024, 2), 2);
    }

    #[test]
    fn job_ranges_distribute_splits_evenly() {
        let ranges: Vec<_> = (0..156)
            .map(|index| job_range(10_000, 156, index))
            .collect();
        assert_eq!(ranges.first(), Some(&(0..65)));
        assert_eq!(ranges.last(), Some(&(9_936..10_000)));
        assert!(ranges.iter().all(|range| range.len() >= 64));
    }
}
