use std::cell::RefCell;

use fancy_regex::Regex;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;

use crate::pre_tokenized::{PreTokenizedString, Split as PtSplit};

use super::Error;

// Thread-local cache of previous Split results for incremental re-use.
thread_local! {
    static SPLIT_CACHE: RefCell<SplitCache> = RefCell::new(SplitCache::default());
}

#[derive(Default)]
struct SplitCache {
    prev_input: Vec<u8>,
    prev_matches: Vec<(usize, usize)>,
}

/// Minimum shared prefix length (bytes) before incremental re-use kicks in.
const INCREMENTAL_MIN_PREFIX: usize = 4096;

/// Wrapper around a JIT-compiled PCRE2 regex for the Llama-3 pattern.
struct Pcre2Regex(pcre2::bytes::Regex);

// Safety: PCRE2 JIT-compiled regexes are thread-safe for matching.
// Each thread uses independent match data internally via pcre2 crate.
unsafe impl Send for Pcre2Regex {}
unsafe impl Sync for Pcre2Regex {}

impl std::fmt::Debug for Pcre2Regex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Pcre2Regex(...)")
    }
}

impl Clone for Pcre2Regex {
    fn clone(&self) -> Self {
        // Re-compile for independent match state.
        Self(
            pcre2::bytes::RegexBuilder::new()
                .utf(true)
                .ucp(true)
                .jit_if_available(true)
                .build(self.0.as_str())
                .expect("re-compile PCRE2 regex"),
        )
    }
}

/// Minimum chunk size (bytes) for parallel regex matching.
const MIN_CHUNK_SIZE: usize = 64 * 1024;

/// Maximum number of parallel chunks (and pre-compiled regex copies).
const MAX_PARALLEL: usize = 16;

/// A pattern for a Split pre-tokenizer: either a literal string or a regex.
#[derive(Clone, Debug, Deserialize)]
pub enum Pattern {
    /// Literal string match (will be regex-escaped).
    String(std::string::String),
    /// Regular expression (used as-is).
    Regex(std::string::String),
}

impl Pattern {
    /// Return the regex source (escaping literals).
    fn source(&self) -> std::string::String {
        match self {
            Self::String(s) => fancy_regex::escape(s).to_string(),
            Self::Regex(r) => r.clone(),
        }
    }
}

/// How matched delimiters are handled in the output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
pub enum SplitBehavior {
    /// Discard matched segments entirely.
    Removed,
    /// Each matched segment becomes its own token.
    #[default]
    Isolated,
    /// Matched segments attach to the preceding token.
    MergedWithPrevious,
    /// Matched segments attach to the following token.
    MergedWithNext,
    /// Adjacent segments of the same kind (both matched or both non-matched)
    /// are grouped into one token.
    Contiguous,
}

/// Raw deserialization helper for [`Split`].
#[derive(Deserialize)]
struct SplitRaw {
    pattern: Pattern,
    #[serde(default)]
    behavior: SplitBehavior,
    #[serde(default)]
    invert: bool,
}

/// A compiled Split pre-tokenizer.
///
/// Constructed once from a pattern, behavior and invert flag (typically from
/// [`PreTokenizerConfig::Split`]), then reused across many inputs. Implements
/// [`Deserialize`] so it can be built directly from the JSON representation.
///
/// [`PreTokenizerConfig::Split`]: crate::PreTokenizerConfig::Split
#[derive(Clone, Debug, Deserialize)]
#[serde(try_from = "SplitRaw")]
pub struct Split {
    /// Pre-compiled regex copies for parallel matching. Index 0 is the
    /// "primary" used for sequential matching; the rest are independent
    /// copies (each with its own DFA cache) used by parallel threads.
    regexes: Vec<Regex>,
    behavior: SplitBehavior,
    invert: bool,
    /// PCRE2 JIT-compiled regex copies for parallel matching (one per thread).
    /// Compiled opportunistically for all patterns; `None` only if PCRE2
    /// cannot handle the pattern syntax.
    pcre2_regexes: Option<Vec<Pcre2Regex>>,
}

/// Compile PCRE2 JIT regexes from `source`, returning `None` if PCRE2 cannot
/// handle the pattern (e.g. unsupported syntax).
fn try_compile_pcre2_regexes(source: &str, n: usize) -> Option<Vec<Pcre2Regex>> {
    let mut regexes = Vec::with_capacity(n);
    for _ in 0..n {
        let re = pcre2::bytes::RegexBuilder::new()
            .utf(true)
            .ucp(true)
            .jit_if_available(true)
            .build(source)
            .ok()?;
        regexes.push(Pcre2Regex(re));
    }
    Some(regexes)
}


/// Compile `n` independent copies of a regex from `source`.
fn compile_regexes(source: &str, n: usize) -> Result<Vec<Regex>, Error> {
    let mut regexes = Vec::with_capacity(n);
    for _ in 0..n {
        regexes.push(Regex::new(source)?);
    }
    Ok(regexes)
}

impl TryFrom<SplitRaw> for Split {
    type Error = Error;

    fn try_from(raw: SplitRaw) -> Result<Self, Error> {
        let source = raw.pattern.source();
        let pcre2_regexes = try_compile_pcre2_regexes(&source, MAX_PARALLEL);
        let regexes = compile_regexes(&source, MAX_PARALLEL)?;
        Ok(Self {
            regexes,
            behavior: raw.behavior,
            invert: raw.invert,
            pcre2_regexes,
        })
    }
}

impl Split {
    /// Build a [`Split`] from raw JSON fields.
    ///
    /// `pattern` must be a JSON object with either a `"String"` key (literal,
    /// will be regex-escaped) or a `"Regex"` key (used as-is).
    pub fn from_config(pattern: &Value, behavior: &str, invert: bool) -> Result<Self, Error> {
        let pattern: Pattern = serde_json::from_value(pattern.clone())?;
        let source = pattern.source();
        let regexes = compile_regexes(&source, MAX_PARALLEL)?;
        let behavior: SplitBehavior = serde_json::from_value(Value::String(behavior.to_string()))?;
        let pcre2_regexes = try_compile_pcre2_regexes(&source, MAX_PARALLEL);
        Ok(Self {
            regexes,
            behavior,
            invert,
            pcre2_regexes,
        })
    }

    /// Refine the splits of a [`PreTokenizedString`] in place.
    ///
    /// Since Split only re-slices text (no content transformation), this is
    /// zero-copy: the buffer stays unchanged and only the split ranges are
    /// replaced.
    pub fn pre_tokenize(&self, pts: &mut PreTokenizedString) -> Result<(), Error> {
        // Fast path: PCRE2 JIT + Isolated + !invert + single text split.
        // Every match and gap becomes its own split, so we can skip the
        // segments/behavior abstraction entirely. Only applies when the PTS
        // has exactly one text split (the common case on first pre-tokenize
        // step). Multi-split inputs (e.g. from an earlier Sequence step)
        // go through the generic per-split path which still uses PCRE2 via
        // find_segments.
        if self.pcre2_regexes.is_some()
            && self.behavior == SplitBehavior::Isolated
            && !self.invert
            && pts.splits().len() == 1
            && pts.splits()[0].token_id.is_none()
        {
            return self.pre_tokenize_pcre2_isolated(pts);
        }

        let mut new_splits = Vec::with_capacity(pts.splits().len() * 2);

        for split in pts.splits() {
            if split.token_id.is_some() {
                new_splits.push(split.clone());
                continue;
            }

            let text = pts.split_text(split);
            if text.is_empty() {
                continue;
            }

            let base = split.range.start;
            let segments = self.find_segments(text)?;
            let ranges = self.apply_behavior(&segments);
            for (s, e) in ranges {
                if s < e {
                    new_splits.push(PtSplit {
                        range: (base + s)..(base + e),
                        token_id: None,
                    });
                }
            }
        }

        pts.refine_splits(new_splits);
        Ok(())
    }

    /// Fast path for any pattern with PCRE2 JIT + Isolated behavior.
    ///
    /// Uses PCRE2 JIT-compiled regex with parallel matching and incremental
    /// caching. Since behavior is Isolated, every match and every gap between
    /// matches becomes its own split.
    fn pre_tokenize_pcre2_isolated(&self, pts: &mut PreTokenizedString) -> Result<(), Error> {
        let buffer = pts.buffer();
        let bytes = buffer.as_bytes();
        let pcre2 = self.pcre2_regexes.as_ref().unwrap();

        let split = &pts.splits()[0];
        let base = split.range.start;
        let text = &buffer[split.range.clone()];

        // Probe the cache: if the input shares a large prefix with the
        // previous input, take the cached matches and the restart position.
        let (mut matches, restart_pos) = SPLIT_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            let common_len = common_prefix_len(&cache.prev_input, bytes);

            if common_len >= INCREMENTAL_MIN_PREFIX && !cache.prev_matches.is_empty() {
                let reuse_count = cache
                    .prev_matches
                    .partition_point(|&(_, end)| end <= common_len);
                let restart = if reuse_count > 0 {
                    cache.prev_matches[reuse_count - 1].1
                } else {
                    0
                };
                // Take the cached vec to avoid cloning; truncate to reuse portion.
                let mut m = std::mem::take(&mut cache.prev_matches);
                m.truncate(reuse_count);
                (m, restart)
            } else {
                (Vec::new(), 0)
            }
        });

        // Run PCRE2 on the portion after the reusable prefix.
        let suffix = &text[restart_pos..];
        if suffix.len() >= MIN_CHUNK_SIZE * 2 {
            let suffix_matches =
                self.find_matches_pcre2_parallel(suffix, base + restart_pos)?;
            matches.extend(suffix_matches);
        } else if !suffix.is_empty() {
            let suffix_matches =
                find_matches_pcre2(suffix, base + restart_pos, &pcre2[0])?;
            matches.extend(suffix_matches);
        }

        // Build splits from matches before moving them into the cache.
        let mut new_splits = Vec::with_capacity(matches.len() * 2);
        let mut prev = base;
        for &(s, e) in &matches {
            if s > prev {
                new_splits.push(PtSplit { range: prev..s, token_id: None });
            }
            new_splits.push(PtSplit { range: s..e, token_id: None });
            prev = e;
        }
        if prev < base + text.len() {
            new_splits.push(PtSplit { range: prev..(base + text.len()), token_id: None });
        }

        // Update the cache for next call: move matches (no clone).
        SPLIT_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            let input_buf = std::mem::take(&mut cache.prev_input);
            if input_buf.len() == bytes.len() {
                cache.prev_input = input_buf;
                cache.prev_input.copy_from_slice(bytes);
            } else {
                cache.prev_input = bytes.to_vec();
            }
            cache.prev_matches = matches;
        });

        pts.refine_splits(new_splits);
        Ok(())
    }

    /// Run PCRE2 matching in parallel on a text segment.
    fn find_matches_pcre2_parallel(
        &self,
        text: &str,
        base: usize,
    ) -> Result<Vec<(usize, usize)>, Error> {
        let bytes = text.as_bytes();
        let pcre2 = self.pcre2_regexes.as_ref().unwrap();

        let n_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let n_chunks = n_cpus
            .min(text.len() / MIN_CHUNK_SIZE)
            .min(pcre2.len())
            .max(2);
        let chunk_size = text.len() / n_chunks;

        let mut boundaries = vec![0usize];
        for i in 1..n_chunks {
            let target = i * chunk_size;
            if let Some(b) = find_safe_boundary_in(bytes, target) {
                if b > *boundaries.last().unwrap() && b < text.len() {
                    boundaries.push(b);
                }
            }
        }

        if boundaries.len() < 2 {
            return find_matches_pcre2(text, base, &pcre2[0]);
        }

        let chunk_results: Vec<Result<Vec<(usize, usize)>, Error>> = boundaries
            .par_iter()
            .enumerate()
            .map(|(idx, &start)| {
                let end = boundaries.get(idx + 1).copied().unwrap_or(text.len());
                let chunk_str = &text[start..end];
                find_matches_pcre2(chunk_str, base + start, &pcre2[idx])
            })
            .collect();

        let total: usize = chunk_results
            .iter()
            .map(|r| r.as_ref().map_or(0, Vec::len))
            .sum();
        let mut all_matches = Vec::with_capacity(total);
        for result in chunk_results {
            all_matches.extend(result?);
        }
        Ok(all_matches)
    }

    /// Split `input` into segments according to the compiled pattern and
    /// behavior.
    ///
    /// Returns borrowed slices into `input` to avoid allocation. Empty segments
    /// are never included.
    #[cfg(test)]
    fn split<'a>(&self, input: &'a str) -> Result<Vec<&'a str>, Error> {
        let segments = self.find_segments(input)?;
        let ranges = self.apply_behavior(&segments);
        Ok(ranges
            .into_iter()
            .filter(|&(s, e)| s < e)
            .map(|(s, e)| &input[s..e])
            .collect())
    }

    /// Phase 1: find all regex matches and build an interleaved list of
    /// `(start, end, is_match)` segments.
    fn find_segments(&self, input: &str) -> Result<Vec<(usize, usize, bool)>, Error> {
        // Prefer PCRE2 JIT when available.
        if let Some(pcre2) = &self.pcre2_regexes {
            let matches = if input.len() >= MIN_CHUNK_SIZE * 2 && pcre2.len() >= 2 {
                self.find_matches_pcre2_parallel(input, 0)?
            } else {
                find_matches_pcre2(input, 0, &pcre2[0])?
            };
            return Ok(matches_to_segments(&matches, input.len(), self.invert));
        }

        if input.len() >= MIN_CHUNK_SIZE * 2 && self.regexes.len() >= 2 {
            if let Some(segments) = self.find_segments_parallel(input)? {
                return Ok(segments);
            }
        }
        self.find_segments_seq(input)
    }

    /// Sequential regex matching (fancy_regex fallback).
    fn find_segments_seq(&self, input: &str) -> Result<Vec<(usize, usize, bool)>, Error> {
        let regex = &self.regexes[0];
        let mut segments = Vec::new();
        let mut prev_end = 0;

        for m in regex.find_iter(input) {
            let m = m?;
            if m.start() == m.end() {
                continue;
            }
            if m.start() > prev_end {
                segments.push((prev_end, m.start(), false));
            }
            segments.push((m.start(), m.end(), true));
            prev_end = m.end();
        }
        if prev_end < input.len() {
            segments.push((prev_end, input.len(), false));
        }

        if self.invert {
            for seg in &mut segments {
                seg.2 = !seg.2;
            }
        }

        Ok(segments)
    }

    /// Parallel regex matching: split input into chunks at safe boundaries,
    /// use pre-compiled independent Regex copies per thread.
    fn find_segments_parallel(
        &self,
        input: &str,
    ) -> Result<Option<Vec<(usize, usize, bool)>>, Error> {
        let bytes = input.as_bytes();
        let n_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let n_chunks = n_cpus
            .min(input.len() / MIN_CHUNK_SIZE)
            .min(self.regexes.len())
            .max(2);
        let chunk_size = input.len() / n_chunks;

        let mut boundaries = vec![0usize];
        for i in 1..n_chunks {
            let target = i * chunk_size;
            if let Some(b) = find_safe_boundary(bytes, target) {
                if b > *boundaries.last().unwrap() && b < input.len() {
                    boundaries.push(b);
                }
            }
        }

        if boundaries.len() < 2 {
            return Ok(None);
        }

        let regexes = &self.regexes;
        let results: Vec<Result<Vec<(usize, usize, bool)>, fancy_regex::Error>> = boundaries
            .par_iter()
            .enumerate()
            .map(|(i, &start)| {
                let end = boundaries.get(i + 1).copied().unwrap_or(input.len());
                let regex = &regexes[i];
                let chunk = &input[start..end];
                let mut segments = Vec::new();
                let mut prev_end = 0;
                for m in regex.find_iter(chunk) {
                    let m = m?;
                    if m.start() == m.end() {
                        continue;
                    }
                    if m.start() > prev_end {
                        segments.push((start + prev_end, start + m.start(), false));
                    }
                    segments.push((start + m.start(), start + m.end(), true));
                    prev_end = m.end();
                }
                if prev_end < chunk.len() {
                    segments.push((start + prev_end, start + chunk.len(), false));
                }
                Ok(segments)
            })
            .collect();

        let total: usize = results
            .iter()
            .map(|r| r.as_ref().map_or(0, Vec::len))
            .sum();
        let mut segments = Vec::with_capacity(total);
        for result in results {
            segments.extend(result?);
        }

        if self.invert {
            for seg in &mut segments {
                seg.2 = !seg.2;
            }
        }

        Ok(Some(segments))
    }

    /// Phase 2: merge / remove / isolate segments according to the configured
    /// [`SplitBehavior`].
    fn apply_behavior(&self, segments: &[(usize, usize, bool)]) -> Vec<(usize, usize)> {
        match self.behavior {
            SplitBehavior::Removed => segments
                .iter()
                .filter(|&&(_, _, is_match)| !is_match)
                .map(|&(s, e, _)| (s, e))
                .collect(),

            SplitBehavior::Isolated => segments.iter().map(|&(s, e, _)| (s, e)).collect(),

            SplitBehavior::Contiguous => {
                let mut result: Vec<(usize, usize)> = Vec::new();
                let mut prev_match = None;
                for &(s, e, is_match) in segments {
                    if prev_match == Some(is_match) {
                        if let Some(last) = result.last_mut() {
                            last.1 = e;
                        }
                    } else {
                        result.push((s, e));
                    }
                    prev_match = Some(is_match);
                }
                result
            }

            SplitBehavior::MergedWithPrevious => {
                let mut result: Vec<(usize, usize)> = Vec::new();
                let mut prev_was_match = false;
                for &(s, e, is_match) in segments {
                    if is_match && !prev_was_match {
                        if let Some(last) = result.last_mut() {
                            last.1 = e;
                        } else {
                            result.push((s, e));
                        }
                    } else {
                        result.push((s, e));
                    }
                    prev_was_match = is_match;
                }
                result
            }

            SplitBehavior::MergedWithNext => {
                let mut result: Vec<(usize, usize)> = Vec::new();
                let mut prev_was_match = false;
                for &(s, e, is_match) in segments.iter().rev() {
                    if is_match && !prev_was_match {
                        if let Some(last) = result.last_mut() {
                            last.0 = s;
                        } else {
                            result.push((s, e));
                        }
                    } else {
                        result.push((s, e));
                    }
                    prev_was_match = is_match;
                }
                result.reverse();
                result
            }
        }
    }
}

/// Convert a list of `(start, end)` match ranges into interleaved
/// `(start, end, is_match)` segments covering the full input.
fn matches_to_segments(
    matches: &[(usize, usize)],
    input_len: usize,
    invert: bool,
) -> Vec<(usize, usize, bool)> {
    let mut segments = Vec::with_capacity(matches.len() * 2 + 1);
    let mut prev = 0;
    for &(s, e) in matches {
        if s > prev {
            segments.push((prev, s, invert));
        }
        segments.push((s, e, !invert));
        prev = e;
    }
    if prev < input_len {
        segments.push((prev, input_len, invert));
    }
    segments
}

/// Find the length of the common prefix between two byte slices.
///
/// Compares 8 bytes at a time for speed on large inputs.
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    let chunks = min_len / 8;
    for i in 0..chunks {
        let off = i * 8;
        let wa = u64::from_ne_bytes(a[off..off + 8].try_into().unwrap());
        let wb = u64::from_ne_bytes(b[off..off + 8].try_into().unwrap());
        if wa != wb {
            let diff = wa ^ wb;
            return off + (diff.trailing_zeros() / 8) as usize;
        }
    }
    let tail_start = chunks * 8;
    for i in tail_start..min_len {
        if a[i] != b[i] {
            return i;
        }
    }
    min_len
}

/// Find a safe chunk boundary within a local byte slice (offsets relative to
/// the start of `bytes`).
fn find_safe_boundary_in(bytes: &[u8], target: usize) -> Option<usize> {
    let search_range = 4096.min(bytes.len() / 4);
    for i in target..bytes.len().min(target + search_range) {
        if i > 0 && bytes[i - 1].is_ascii_alphanumeric() && is_ascii_ws(bytes[i]) {
            return Some(i);
        }
    }
    for i in (target.saturating_sub(search_range)..target).rev() {
        if i > 0 && bytes[i - 1].is_ascii_alphanumeric() && is_ascii_ws(bytes[i]) {
            return Some(i);
        }
    }
    None
}

/// Find all pattern matches using PCRE2 JIT.
fn find_matches_pcre2(
    input: &str,
    base: usize,
    regex: &Pcre2Regex,
) -> Result<Vec<(usize, usize)>, Error> {
    let mut matches = Vec::with_capacity(input.len() / 3);
    let bytes = input.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        match regex.0.find_at(bytes, pos) {
            Ok(Some(m)) => {
                if m.start() == m.end() {
                    pos = m.end() + 1;
                    continue;
                }
                matches.push((base + m.start(), base + m.end()));
                pos = m.end();
            }
            Ok(None) => break,
            Err(e) => return Err(Error::Unsupported(format!("PCRE2: {e}"))),
        }
    }
    Ok(matches)
}

/// Find a safe chunk boundary near `target` for parallel regex matching.
///
/// Returns a byte offset where the preceding byte is ASCII alphanumeric and the
/// byte at the offset is ASCII whitespace. For tokenizer patterns (Llama-3,
/// GPT-4, etc.), no regex alternative spans across such a boundary.
fn find_safe_boundary(bytes: &[u8], target: usize) -> Option<usize> {
    let search_range = 4096.min(bytes.len() / 4);
    for i in target..bytes.len().min(target + search_range) {
        if i > 0 && bytes[i - 1].is_ascii_alphanumeric() && is_ascii_ws(bytes[i]) {
            return Some(i);
        }
    }
    for i in (target.saturating_sub(search_range)..target).rev() {
        if i > 0 && bytes[i - 1].is_ascii_alphanumeric() && is_ascii_ws(bytes[i]) {
            return Some(i);
        }
    }
    None
}

fn is_ascii_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    // ── Behavior tests ──────────────────────────────────

    #[test]
    fn split_removed() {
        let s = Split::from_config(&json!({"String": "-"}), "Removed", false).unwrap();
        assert_eq!(
            s.split("the-final--countdown").unwrap(),
            vec!["the", "final", "countdown"],
        );
    }

    #[test]
    fn split_isolated() {
        let s = Split::from_config(&json!({"String": "-"}), "Isolated", false).unwrap();
        assert_eq!(
            s.split("the-final--countdown").unwrap(),
            vec!["the", "-", "final", "-", "-", "countdown"],
        );
    }

    #[test]
    fn split_merged_with_previous() {
        let s = Split::from_config(&json!({"String": "-"}), "MergedWithPrevious", false).unwrap();
        assert_eq!(
            s.split("the-final--countdown").unwrap(),
            vec!["the-", "final-", "-", "countdown"],
        );
    }

    #[test]
    fn split_merged_with_next() {
        let s = Split::from_config(&json!({"String": "-"}), "MergedWithNext", false).unwrap();
        assert_eq!(
            s.split("the-final--countdown").unwrap(),
            vec!["the", "-final", "-", "-countdown"],
        );
    }

    #[test]
    fn split_contiguous() {
        let s = Split::from_config(&json!({"String": "-"}), "Contiguous", false).unwrap();
        assert_eq!(
            s.split("the-final--countdown").unwrap(),
            vec!["the", "-", "final", "--", "countdown"],
        );
    }

    // ── Invert tests ────────────────────────────────────

    #[test]
    fn split_invert_removed() {
        let s = Split::from_config(&json!({"Regex": "\\d+"}), "Removed", true).unwrap();
        assert_eq!(s.split("abc123def456").unwrap(), vec!["123", "456"]);
    }

    #[test]
    fn split_invert_isolated() {
        let s = Split::from_config(&json!({"Regex": "\\d+"}), "Isolated", true).unwrap();
        assert_eq!(
            s.split("abc123def456").unwrap(),
            vec!["abc", "123", "def", "456"],
        );
    }

    // ── Edge cases ──────────────────────────────────────

    #[test]
    fn split_empty_input() {
        let s = Split::from_config(&json!({"String": "-"}), "Isolated", false).unwrap();
        assert!(s.split("").unwrap().is_empty());
    }

    #[test]
    fn split_no_matches() {
        let s = Split::from_config(&json!({"String": "-"}), "Isolated", false).unwrap();
        assert_eq!(s.split("hello world").unwrap(), vec!["hello world"]);
    }

    #[test]
    fn split_all_delimiters() {
        let s = Split::from_config(&json!({"String": "-"}), "Removed", false).unwrap();
        assert!(s.split("---").unwrap().is_empty());
    }

    #[test]
    fn split_delimiter_at_start() {
        let s = Split::from_config(&json!({"String": "-"}), "MergedWithPrevious", false).unwrap();
        assert_eq!(s.split("-hello").unwrap(), vec!["-", "hello"]);
    }

    #[test]
    fn split_delimiter_at_end() {
        let s = Split::from_config(&json!({"String": "-"}), "MergedWithNext", false).unwrap();
        assert_eq!(s.split("hello-").unwrap(), vec!["hello", "-"]);
    }

    #[test]
    fn split_default_behavior() {
        let s = Split::from_config(&json!({"String": " "}), "Isolated", false).unwrap();
        assert_eq!(s.split("a b c").unwrap(), vec!["a", " ", "b", " ", "c"]);
    }

    #[test]
    fn split_string_pattern_not_treated_as_regex() {
        let s = Split::from_config(&json!({"String": "[a]"}), "Isolated", false).unwrap();
        assert_eq!(s.split("a[a]b").unwrap(), vec!["a", "[a]", "b"]);
    }

    #[test]
    fn split_regex_whitespace() {
        let s = Split::from_config(&json!({"Regex": "\\s+"}), "Isolated", false).unwrap();
        assert_eq!(
            s.split("hello  world").unwrap(),
            vec!["hello", "  ", "world"],
        );
    }

    // ── Deserialize test ────────────────────────────────

    #[test]
    fn split_deserialize() {
        let s: Split = serde_json::from_value(json!({
            "pattern": {"Regex": "\\s+"},
            "behavior": "Isolated",
        }))
        .unwrap();
        assert_eq!(
            s.split("hello  world").unwrap(),
            vec!["hello", "  ", "world"],
        );
    }

    // ── Error tests ─────────────────────────────────────

    #[test]
    fn error_invalid_pattern() {
        let err = Split::from_config(&json!({"Foo": "bar"}), "Isolated", false).unwrap_err();
        assert!(matches!(err, Error::Json(_)));
    }

    #[test]
    fn error_bad_regex() {
        let err =
            Split::from_config(&json!({"Regex": "(unclosed"}), "Isolated", false).unwrap_err();
        assert!(matches!(err, Error::Regex(_)));
    }

    #[test]
    fn error_unknown_behavior() {
        let err = Split::from_config(&json!({"String": "-"}), "Foobar", false).unwrap_err();
        assert!(matches!(err, Error::Json(_)));
    }
}
