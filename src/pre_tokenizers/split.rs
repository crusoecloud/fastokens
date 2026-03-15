#[cfg(feature = "pcre2")]
use std::cell::RefCell;
#[cfg(feature = "pcre2")]
use std::sync::atomic::{AtomicUsize, Ordering};

use fancy_regex::Regex;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;

use crate::pre_tokenized::{PreTokenizedString, Split as PtSplit};

use super::Error;

// Thread-local cache of previous Split results for incremental re-use.
#[cfg(feature = "pcre2")]
thread_local! {
    static SPLIT_CACHE: RefCell<SplitCache> = RefCell::new(SplitCache::default());
}

#[cfg(feature = "pcre2")]
struct SplitCache {
    /// Identity of the Split pre-tokenizer that populated this cache.
    /// Prevents cross-model contamination when different models share a thread.
    split_id: usize,
    prev_input: Vec<u8>,
    prev_matches: Vec<(usize, usize)>,
}

#[cfg(feature = "pcre2")]
impl Default for SplitCache {
    fn default() -> Self {
        Self {
            split_id: 0,
            prev_input: Vec::new(),
            prev_matches: Vec::new(),
        }
    }
}

/// Minimum shared prefix length (bytes) before incremental re-use kicks in.
#[cfg(feature = "pcre2")]
const INCREMENTAL_MIN_PREFIX: usize = 4096;

/// Wrapper around a JIT-compiled PCRE2 regex for the Llama-3 pattern.
#[cfg(feature = "pcre2")]
struct Pcre2Regex(pcre2::bytes::Regex);

// Safety: PCRE2 JIT-compiled regexes are thread-safe for matching.
// Each thread uses independent match data internally via pcre2 crate.
#[cfg(feature = "pcre2")]
unsafe impl Send for Pcre2Regex {}
#[cfg(feature = "pcre2")]
unsafe impl Sync for Pcre2Regex {}

#[cfg(feature = "pcre2")]
impl std::fmt::Debug for Pcre2Regex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Pcre2Regex(...)")
    }
}

#[cfg(feature = "pcre2")]
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
/// Parallel matching triggers when the input is >= 2 × this value (i.e. 16 KB).
const MIN_CHUNK_SIZE: usize = 8 * 1024;

/// Number of pre-compiled regex copies (one per potential parallel thread).
/// Sized to the machine's available parallelism so that regex matching can
/// scale across all cores.
fn max_parallel() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    })
}

/// Overlap (bytes) appended to each parallel regex chunk.
///
/// Each chunk extends past its "authority zone" by this many bytes so that any
/// match starting near the boundary is still found in full. Matches whose
/// start falls outside a chunk's authority zone are discarded — the adjacent
/// chunk owns them. Must exceed the longest possible single regex match.
/// For tokenizer patterns (letter/number/whitespace runs, contractions, etc.)
/// individual matches are at most a few hundred bytes, so 1 KB is generous.
const CHUNK_OVERLAP: usize = 1024;

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

/// Monotonic counter for unique Split instance IDs.
#[cfg(feature = "pcre2")]
static SPLIT_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

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
    /// Unique identity for cache invalidation across different Split instances.
    #[serde(skip)]
    #[cfg(feature = "pcre2")]
    id: usize,
    /// Pre-compiled regex copies for parallel matching. Index 0 is the
    /// "primary" used for sequential matching; the rest are independent
    /// copies (each with its own DFA cache) used by parallel threads.
    regexes: Vec<Regex>,
    behavior: SplitBehavior,
    invert: bool,
    /// PCRE2 JIT-compiled regex copies for parallel matching (one per thread).
    /// Compiled opportunistically for all patterns; `None` only if PCRE2
    /// cannot handle the pattern syntax or the feature is disabled.
    #[cfg(feature = "pcre2")]
    pcre2_regexes: Option<Vec<Pcre2Regex>>,
}

/// Compile PCRE2 JIT regexes from `source`, returning `None` if PCRE2 cannot
/// handle the pattern (e.g. unsupported syntax).
#[cfg(feature = "pcre2")]
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
        let regexes = compile_regexes(&source, max_parallel())?;
        Ok(Self {
            #[cfg(feature = "pcre2")]
            id: SPLIT_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            regexes,
            behavior: raw.behavior,
            invert: raw.invert,
            #[cfg(feature = "pcre2")]
            pcre2_regexes: try_compile_pcre2_regexes(&source, max_parallel()),
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
        let regexes = compile_regexes(&source, max_parallel())?;
        let behavior: SplitBehavior = serde_json::from_value(Value::String(behavior.to_string()))?;
        Ok(Self {
            #[cfg(feature = "pcre2")]
            id: SPLIT_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            regexes,
            behavior,
            invert,
            #[cfg(feature = "pcre2")]
            pcre2_regexes: try_compile_pcre2_regexes(&source, max_parallel()),
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
        #[cfg(feature = "pcre2")]
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
    #[cfg(feature = "pcre2")]
    fn pre_tokenize_pcre2_isolated(&self, pts: &mut PreTokenizedString) -> Result<(), Error> {
        let buffer = pts.buffer();
        let bytes = buffer.as_bytes();
        let pcre2 = self.pcre2_regexes.as_ref().unwrap();

        let split = &pts.splits()[0];
        let base = split.range.start;
        let text = &buffer[split.range.clone()];

        // Probe the cache: if the input shares a large prefix with the
        // previous input (from the SAME Split instance), take the cached
        // matches and the restart position.
        let split_id = self.id;
        let (mut matches, restart_pos) = SPLIT_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            if cache.split_id != split_id {
                cache.split_id = split_id;
                cache.prev_input.clear();
                cache.prev_matches.clear();
                return (Vec::new(), 0);
            }
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
            let suffix_matches = self.find_matches_pcre2_parallel(suffix, base + restart_pos)?;
            matches.extend(suffix_matches);
        } else if !suffix.is_empty() {
            let suffix_matches = find_matches_pcre2(suffix, base + restart_pos, &pcre2[0])?;
            matches.extend(suffix_matches);
        }

        // Build splits from matches before moving them into the cache.
        let mut new_splits = Vec::with_capacity(matches.len() * 2);
        let mut prev = base;
        for &(s, e) in &matches {
            if s > prev {
                new_splits.push(PtSplit {
                    range: prev..s,
                    token_id: None,
                });
            }
            new_splits.push(PtSplit {
                range: s..e,
                token_id: None,
            });
            prev = e;
        }
        if prev < base + text.len() {
            new_splits.push(PtSplit {
                range: prev..(base + text.len()),
                token_id: None,
            });
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

    /// Run PCRE2 matching in parallel using overlapping chunks.
    ///
    /// The input is divided into N authority zones of roughly equal size.  Each
    /// thread matches its zone *plus* a [`CHUNK_OVERLAP`] tail so that matches
    /// starting near a boundary are still found in full.  Only matches whose
    /// start falls within a thread's authority zone are kept; the rest are
    /// discarded (the adjacent chunk owns them).
    #[cfg(feature = "pcre2")]
    fn find_matches_pcre2_parallel(
        &self,
        text: &str,
        base: usize,
    ) -> Result<Vec<(usize, usize)>, Error> {
        let pcre2 = self.pcre2_regexes.as_ref().unwrap();

        let n_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let n_chunks = n_cpus
            .min(text.len() / MIN_CHUNK_SIZE)
            .min(pcre2.len())
            .max(2);
        let nominal = text.len() / n_chunks;

        // Authority boundaries — each chunk owns matches starting in
        // [auth[i], auth[i+1]).
        let mut auth = vec![0usize];
        for i in 1..n_chunks {
            let b = snap_char_ceil(text, i * nominal);
            if b > *auth.last().unwrap() && b < text.len() {
                auth.push(b);
            }
        }
        auth.push(text.len());

        let actual = auth.len() - 1;
        if actual < 2 {
            return find_matches_pcre2(text, base, &pcre2[0]);
        }

        let chunk_results: Result<Vec<Vec<(usize, usize)>>, Error> = (0..actual)
            .into_par_iter()
            .map(|i| {
                let auth_start = auth[i];
                let auth_end = auth[i + 1];
                // Extend past authority zone so cross-boundary matches are
                // found in full.
                let chunk_end = snap_char_ceil(text, (auth_end + CHUNK_OVERLAP).min(text.len()));
                let chunk = &text[auth_start..chunk_end];
                let all = find_matches_pcre2(chunk, base + auth_start, &pcre2[i])?;
                // Keep only matches whose start is in our authority zone.
                let abs_auth_end = base + auth_end;
                Ok(all.into_iter().filter(|&(s, _)| s < abs_auth_end).collect())
            })
            .collect();

        let chunks = chunk_results?;

        // Merge parallel results, repairing gaps at chunk boundaries where
        // cross-boundary matches cause the next chunk to mis-align.
        let regex = &pcre2[0];
        let bytes = text.as_bytes();
        merge_chunk_matches(chunks, base, |pos_rel| {
            let mut p = pos_rel;
            loop {
                if p >= bytes.len() {
                    return Ok(None);
                }
                match regex.0.find_at(bytes, p) {
                    Ok(Some(m)) => {
                        if m.start() == m.end() {
                            p = m.end() + 1;
                            continue;
                        }
                        return Ok(Some((base + m.start(), base + m.end())));
                    }
                    Ok(None) => return Ok(None),
                    Err(e) => return Err(Error::Unsupported(format!("PCRE2: {e}"))),
                }
            }
        })
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
        #[cfg(feature = "pcre2")]
        if let Some(pcre2) = &self.pcre2_regexes {
            let matches = if input.len() >= MIN_CHUNK_SIZE * 2 && pcre2.len() >= 2 {
                self.find_matches_pcre2_parallel(input, 0)?
            } else {
                find_matches_pcre2(input, 0, &pcre2[0])?
            };
            return Ok(matches_to_segments(&matches, input.len(), self.invert));
        }

        if input.len() >= MIN_CHUNK_SIZE * 2 && self.regexes.len() >= 2 {
            if let Some(matches) = self.find_matches_fancy_parallel(input)? {
                return Ok(matches_to_segments(&matches, input.len(), self.invert));
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

    /// Parallel regex matching (fancy_regex) using overlapping chunks.
    ///
    /// Same overlap strategy as [`find_matches_pcre2_parallel`]: each thread
    /// matches its authority zone plus a [`CHUNK_OVERLAP`] tail, keeping only
    /// matches whose start is within its authority zone.
    fn find_matches_fancy_parallel(
        &self,
        input: &str,
    ) -> Result<Option<Vec<(usize, usize)>>, Error> {
        let n_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let n_chunks = n_cpus
            .min(input.len() / MIN_CHUNK_SIZE)
            .min(self.regexes.len())
            .max(2);
        let nominal = input.len() / n_chunks;

        let mut auth = vec![0usize];
        for i in 1..n_chunks {
            let b = snap_char_ceil(input, i * nominal);
            if b > *auth.last().unwrap() && b < input.len() {
                auth.push(b);
            }
        }
        auth.push(input.len());

        let actual = auth.len() - 1;
        if actual < 2 {
            return Ok(None);
        }

        let regexes = &self.regexes;
        let chunk_results: Result<Vec<Vec<(usize, usize)>>, Error> = (0..actual)
            .into_par_iter()
            .map(|i| {
                let auth_start = auth[i];
                let auth_end = auth[i + 1];
                let chunk_end = snap_char_ceil(input, (auth_end + CHUNK_OVERLAP).min(input.len()));
                let regex = &regexes[i];
                let chunk = &input[auth_start..chunk_end];
                let mut matches = Vec::new();
                for m in regex.find_iter(chunk) {
                    let m = m?;
                    if m.start() == m.end() {
                        continue;
                    }
                    let abs_start = auth_start + m.start();
                    if abs_start >= auth_end {
                        break; // Past our authority — stop early.
                    }
                    matches.push((abs_start, auth_start + m.end()));
                }
                Ok(matches)
            })
            .collect();

        let chunks = chunk_results?;

        let regex = &self.regexes[0];
        let result = merge_chunk_matches(chunks, 0, |pos_rel| {
            let slice = &input[pos_rel..];
            for m in regex.find_iter(slice) {
                let m = m?;
                if m.start() == m.end() {
                    continue;
                }
                return Ok(Some((pos_rel + m.start(), pos_rel + m.end())));
            }
            Ok(None)
        })?;

        Ok(Some(result))
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

/// Merge per-chunk parallel match results, repairing boundary misalignments.
///
/// After parallel matching with overlapping chunks, each boundary may have
/// "ghost" matches — the right chunk re-matches text already covered by the
/// left chunk's cross-boundary match.  Dropping ghosts is necessary but not
/// sufficient: the right chunk's *subsequent* matches may also be wrong because
/// its regex engine scanned from `auth_start` instead of from the end of the
/// cross-boundary match.
///
/// Additionally, when `CHUNK_OVERLAP` is smaller than the longest match, a
/// chunk may produce a **truncated** match that ended at its readable boundary
/// rather than at the match's true end.  This is detected when a dropped
/// ghost's end exceeds `prev_end`: the last accepted match is popped and
/// re-matched on the full text to obtain the correct extent.
///
/// `find_from(pos_rel)` must return the next non-empty regex match starting at
/// or after byte offset `pos_rel` (relative to the text start), with positions
/// expressed as absolute offsets (`base + byte_offset`).
fn merge_chunk_matches(
    chunks: Vec<Vec<(usize, usize)>>,
    base: usize,
    mut find_from: impl FnMut(usize) -> Result<Option<(usize, usize)>, Error>,
) -> Result<Vec<(usize, usize)>, Error> {
    let total: usize = chunks.iter().map(Vec::len).sum();
    let mut flat: Vec<(usize, usize)> = Vec::with_capacity(total);
    for c in chunks {
        flat.extend(c);
    }

    if flat.is_empty() {
        return Ok(flat);
    }

    let mut result = Vec::with_capacity(flat.len());
    let mut prev_end = base;
    let mut idx = 0;

    while idx < flat.len() {
        if flat[idx].0 >= prev_end {
            // Normal match — accept.
            result.push(flat[idx]);
            prev_end = flat[idx].1;
            idx += 1;
        } else {
            // Ghost match(es) — skip all that start before prev_end, but
            // track the maximum end position among dropped ghosts.
            let mut max_ghost_end = 0usize;
            while idx < flat.len() && flat[idx].0 < prev_end {
                max_ghost_end = max_ghost_end.max(flat[idx].1);
                idx += 1;
            }

            // If a ghost extends past prev_end, the last accepted match was
            // truncated at a chunk boundary.  Re-match from the truncated
            // match's start on the full text to obtain the correct extent.
            if max_ghost_end > prev_end {
                if let Some(&(trunc_start, _)) = result.last() {
                    result.pop();
                    match find_from(trunc_start - base)? {
                        Some((ms, me)) => {
                            result.push((ms, me));
                            prev_end = me;
                        }
                        None => {
                            prev_end = result.last().map_or(base, |&(_, e)| e);
                        }
                    }
                }
                // The re-matched match may be longer, making additional flat
                // entries into ghosts — skip those too.
                while idx < flat.len() && flat[idx].0 < prev_end {
                    idx += 1;
                }
            }

            // After ghost handling (and possible truncation repair), check
            // for a gap that needs repair.  A gap means the next surviving
            // match starts *after* prev_end — the region may contain matches
            // that the right chunk missed because its regex engine was
            // scanning from a different origin.
            if idx < flat.len() && flat[idx].0 > prev_end {
                let remaining = &flat[idx..];
                let mut pos_rel = prev_end - base;

                loop {
                    match find_from(pos_rel)? {
                        Some((ms, me)) => {
                            // Check convergence with remaining parallel matches.
                            let limit = remaining.len().min(64);
                            if let Some(j) = remaining[..limit].iter().position(|&m| m == (ms, me))
                            {
                                // Converged — resume from this point in flat.
                                idx += j;
                                break;
                            }
                            result.push((ms, me));
                            prev_end = me;
                            pos_rel = me - base;
                        }
                        None => {
                            // No more matches in the text — done.
                            idx = flat.len();
                            break;
                        }
                    }
                }
            }
            // If flat[idx].0 == prev_end, the next iteration accepts it.
        }
    }

    Ok(result)
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
#[cfg(feature = "pcre2")]
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

/// Find all pattern matches using PCRE2 JIT.
#[cfg(feature = "pcre2")]
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

/// Round a byte offset up to the nearest UTF-8 char boundary.
fn snap_char_ceil(s: &str, pos: usize) -> usize {
    let mut p = pos;
    while p < s.len() && !s.is_char_boundary(p) {
        p += 1;
    }
    p
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

    // ── Real-world tokenizer patterns ──────────────────────

    /// The Llama-3 / GPT-4 tokenizer pre-tokenization pattern.
    const LLAMA3_PATTERN: &str = concat!(
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*",
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
        r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+",
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
        r"|\p{N}",
        r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"|\s*[\r\n]+",
        r"|\s+(?!\S)",
        r"|\s+",
    );

    /// The GPT-2 tokenizer pre-tokenization pattern.
    const GPT2_PATTERN: &str = concat!(
        r"'s|'t|'re|'ve|'m|'ll|'d",
        r"| ?\p{L}+",
        r"| ?\p{N}+",
        r"| ?[^\s\p{L}\p{N}]+",
        r"|\s+(?!\S)",
        r"|\s+",
    );

    /// Assert that splitting with `Isolated` behavior produces pieces that
    /// concatenate back to the original input (no bytes lost or duplicated).
    fn assert_full_coverage(split: &Split, input: &str) {
        let pieces = split.split(input).unwrap();
        assert_eq!(
            pieces.join(""),
            input,
            "coverage gap on {input:?}: pieces={pieces:?}",
        );
    }

    #[test]
    fn llama3_full_coverage() {
        let s = Split::from_config(&json!({"Regex": LLAMA3_PATTERN}), "Isolated", false).unwrap();
        let inputs = [
            "Hello, world!",
            "fn main() { println!(\"hello\"); }",
            "café résumé naïve",
            "你好世界 こんにちは 안녕하세요",
            "user@example.com https://test.com/path?q=1",
            "2024-01-15T10:30:00Z",
            "  \t\n  leading whitespace and trailing   ",
            "CamelCase snake_case kebab-case",
            "\u{1f680}\u{1f4a1}\u{2728} emoji text \u{1f389}",
            "\u{0300}\u{0301}\u{0302}",
            "\u{200b}\u{200c}\u{200d}",
            "I'm don't we're they've I'll he'd",
            "ALLCAPS mixedCase Title",
            "",
            "  ",
            "100,000.50 + 3.14e-10 = ?",
        ];
        for &input in &inputs {
            assert_full_coverage(&s, input);
        }
    }

    #[test]
    fn llama3_digits_are_individual() {
        let s = Split::from_config(&json!({"Regex": LLAMA3_PATTERN}), "Isolated", false).unwrap();
        // \p{N} matches one digit at a time, not runs.
        let result = s.split("12345").unwrap();
        assert_eq!(result, vec!["1", "2", "3", "4", "5"]);
    }

    #[test]
    fn gpt2_contractions() {
        let s = Split::from_config(&json!({"Regex": GPT2_PATTERN}), "Isolated", false).unwrap();
        let input = "I'm don't we're they've I'll he'd";
        let result = s.split(input).unwrap();
        assert!(result.contains(&"'m"));
        assert!(result.contains(&"'t"));
        assert!(result.contains(&"'re"));
        assert!(result.contains(&"'ve"));
        assert!(result.contains(&"'ll"));
        assert!(result.contains(&"'d"));
        assert_eq!(result.join(""), input);
    }

    #[test]
    fn gpt2_full_coverage() {
        let s = Split::from_config(&json!({"Regex": GPT2_PATTERN}), "Isolated", false).unwrap();
        let inputs = [
            "Hello, world!",
            "  multiple   spaces  ",
            "12345 numbers",
            "@#$%^&*()",
            "mixed 123 text!!! with... stuff",
            "",
        ];
        for &input in &inputs {
            assert_full_coverage(&s, input);
        }
    }

    // ── Gap-producing patterns ─────────────────────────────

    #[test]
    fn digits_only_leaves_gaps() {
        let s = Split::from_config(&json!({"Regex": "\\d+"}), "Isolated", false).unwrap();
        let result = s.split("abc123def456ghi").unwrap();
        assert_eq!(result, vec!["abc", "123", "def", "456", "ghi"]);
    }

    #[test]
    fn letters_only_with_gaps() {
        let s = Split::from_config(&json!({"Regex": "\\p{L}+"}), "Isolated", false).unwrap();
        let result = s.split("hello---world...test").unwrap();
        assert_eq!(result, vec!["hello", "---", "world", "...", "test"]);
    }

    #[test]
    fn alternation_with_gaps() {
        let s = Split::from_config(&json!({"Regex": "\\p{L}+|\\d+"}), "Isolated", false).unwrap();
        let result = s.split("abc@123#def!456").unwrap();
        assert_eq!(result, vec!["abc", "@", "123", "#", "def", "!", "456"]);
    }

    #[test]
    fn gaps_with_removed_behavior() {
        let s = Split::from_config(&json!({"Regex": "\\d+"}), "Removed", false).unwrap();
        // Matches (digits) are removed; only gaps remain.
        let result = s.split("abc123def456ghi").unwrap();
        assert_eq!(result, vec!["abc", "def", "ghi"]);
    }

    #[test]
    fn gaps_merged_with_next() {
        let s = Split::from_config(&json!({"Regex": "\\s+"}), "MergedWithNext", false).unwrap();
        let result = s.split("hello world test").unwrap();
        assert_eq!(result, vec!["hello", " world", " test"]);
    }

    #[test]
    fn contiguous_adjacent_single_matches() {
        let s = Split::from_config(&json!({"Regex": "[a-c]"}), "Contiguous", false).unwrap();
        // Each letter is a separate match; Contiguous merges adjacent matches.
        let result = s.split("abcxabc").unwrap();
        assert_eq!(result, vec!["abc", "x", "abc"]);
    }

    // ── Unicode category patterns ──────────────────────────

    #[test]
    fn titlecase_letter_pattern() {
        let s = Split::from_config(&json!({"Regex": "\\p{Lt}"}), "Isolated", false).unwrap();
        // U+01C5 (Dž), U+01C8 (Lj), U+01CB (Nj) are titlecase letters.
        let input = "a\u{01c5}b\u{01c8}c\u{01cb}d";
        let result = s.split(input).unwrap();
        assert_eq!(result.join(""), input);
        assert!(result.contains(&"\u{01c5}"));
        assert!(result.contains(&"\u{01c8}"));
        assert!(result.contains(&"\u{01cb}"));
    }

    #[test]
    fn camelcase_pattern() {
        let s = Split::from_config(&json!({"Regex": "[\\p{Lu}][\\p{Ll}]*"}), "Isolated", false)
            .unwrap();
        let result = s.split("CamelCaseHTTPServer").unwrap();
        assert_eq!(result.join(""), "CamelCaseHTTPServer");
        assert!(result.contains(&"Camel"));
        assert!(result.contains(&"Case"));
        assert!(result.contains(&"Server"));
    }

    #[test]
    fn unicode_script_with_gaps() {
        let s = Split::from_config(
            &json!({"Regex": "\\p{Han}+|\\p{Latin}+|\\s+"}),
            "Isolated",
            false,
        )
        .unwrap();
        // @, #, ! are not \p{Han}, \p{Latin}, or \s — they become gap segments.
        let input = "Hello# \u{4f60}\u{597d}@ World!";
        let result = s.split(input).unwrap();
        assert_eq!(result.join(""), input);
        assert!(result.contains(&"#"));
        assert!(result.contains(&"@"));
        assert!(result.contains(&"!"));
    }

    // ── Edge-case patterns ─────────────────────────────────

    #[test]
    fn pattern_matches_nothing_in_input() {
        let s = Split::from_config(&json!({"Regex": "ZZZZZ"}), "Isolated", false).unwrap();
        // No matches → entire input is one gap segment.
        let result = s.split("hello world").unwrap();
        assert_eq!(result, vec!["hello world"]);
    }

    #[test]
    fn pattern_single_char_matches() {
        let s = Split::from_config(&json!({"Regex": "."}), "Isolated", false).unwrap();
        // Every char is a match, no gaps at all.
        let result = s.split("abc").unwrap();
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    // ── Invert with gap-producing patterns ─────────────────

    #[test]
    fn invert_removes_non_matching() {
        let s = Split::from_config(&json!({"Regex": "\\p{L}+"}), "Removed", true).unwrap();
        // Invert flips match/gap, Removed drops the new "matches" (which
        // are the original gaps). Only letter runs survive.
        let result = s.split("hello 123 world 456").unwrap();
        assert_eq!(result, vec!["hello", "world"]);
    }

    #[test]
    fn invert_contiguous_vowels() {
        let s = Split::from_config(&json!({"Regex": "[aeiou]+"}), "Contiguous", true).unwrap();
        // Inverted: vowels become gaps, consonant runs become matches.
        // Contiguous merges adjacent same-type segments.
        let result = s.split("hello").unwrap();
        // h(gap), e(match), ll(gap), o(match)
        assert_eq!(result, vec!["h", "e", "ll", "o"]);
    }

    // ── Sequential pre-tokenizers (Sequence) ───────────────

    /// Helper: collect split texts from a PreTokenizedString.
    fn pts_texts(pts: &PreTokenizedString) -> Vec<String> {
        pts.splits()
            .iter()
            .filter(|s| s.token_id.is_none())
            .map(|s| pts.split_text(s).to_string())
            .filter(|t| !t.is_empty())
            .collect()
    }

    #[test]
    fn sequential_two_splits() {
        // First: remove whitespace.  Then: isolate digits.
        let s1 = Split::from_config(&json!({"Regex": "\\s+"}), "Removed", false).unwrap();
        let s2 = Split::from_config(&json!({"Regex": "\\d+"}), "Isolated", false).unwrap();

        let mut pts = PreTokenizedString::from_text("hello 123world 456test");
        s1.pre_tokenize(&mut pts).unwrap();
        s2.pre_tokenize(&mut pts).unwrap();

        assert_eq!(
            pts_texts(&pts),
            vec!["hello", "123", "world", "456", "test"],
        );
    }

    #[test]
    fn sequential_three_splits() {
        // Whitespace → digits → uppercase letters.
        let s1 = Split::from_config(&json!({"Regex": "\\s+"}), "Removed", false).unwrap();
        let s2 = Split::from_config(&json!({"Regex": "\\d+"}), "Isolated", false).unwrap();
        let s3 = Split::from_config(&json!({"Regex": "[\\p{Lu}]+"}), "Isolated", false).unwrap();

        let mut pts = PreTokenizedString::from_text("helloWORLD 123ABCdef");
        s1.pre_tokenize(&mut pts).unwrap();
        s2.pre_tokenize(&mut pts).unwrap();
        s3.pre_tokenize(&mut pts).unwrap();

        assert_eq!(pts_texts(&pts), vec!["hello", "WORLD", "123", "ABC", "def"],);
    }

    #[test]
    fn sequential_preserves_added_tokens() {
        let s = Split::from_config(&json!({"Regex": "\\s+"}), "Removed", false).unwrap();

        let buffer = "hello world".to_string();
        let splits = vec![
            PtSplit {
                range: 0..5,
                token_id: None,
            },
            PtSplit {
                range: 5..5,
                token_id: Some(42),
            },
            PtSplit {
                range: 5..11,
                token_id: None,
            },
        ];
        let mut pts = PreTokenizedString::new(buffer, splits);
        s.pre_tokenize(&mut pts).unwrap();

        let has_added = pts.splits().iter().any(|s| s.token_id == Some(42));
        assert!(has_added, "added token lost after pre_tokenize");
    }

    #[test]
    fn sequential_mixed_behaviors() {
        // First split: isolate punctuation. Second split: merge whitespace with next.
        let s1 = Split::from_config(&json!({"Regex": "[,.!?]+"}), "Isolated", false).unwrap();
        let s2 = Split::from_config(&json!({"Regex": "\\s+"}), "MergedWithNext", false).unwrap();

        let mut pts = PreTokenizedString::from_text("hello, world! test");
        s1.pre_tokenize(&mut pts).unwrap();
        s2.pre_tokenize(&mut pts).unwrap();

        let texts = pts_texts(&pts);
        // After s1: ["hello", ",", " world", "!", " test"]
        // After s2 on each piece:
        //   "hello" → "hello"
        //   "," → ","
        //   " world" → " world" (space merged with next = "world" → " world")
        //   "!" → "!"
        //   " test" → " test"
        assert!(texts.contains(&"hello".to_string()));
        assert!(texts.contains(&",".to_string()));
        assert!(texts.contains(&"!".to_string()));
        // Verify full coverage: all text accounted for.
        let all_text: String = pts
            .splits()
            .iter()
            .filter(|s| s.token_id.is_none())
            .map(|s| pts.split_text(s))
            .collect();
        assert_eq!(all_text, "hello, world! test");
    }

    // ── merge_chunk_matches: truncated match repair ──────────────────

    /// Helper: simulate merge_chunk_matches with a known full-text oracle.
    fn merge_with_oracle(
        chunks: Vec<Vec<(usize, usize)>>,
        full_matches: &[(usize, usize)],
    ) -> Vec<(usize, usize)> {
        let full = full_matches.to_vec();
        merge_chunk_matches(chunks, 0, |pos_rel| {
            Ok(full.iter().copied().find(|&(s, _)| s >= pos_rel))
        })
        .unwrap()
    }

    #[test]
    fn merge_truncated_match_at_boundary() {
        // Chunk 0 produced a truncated match (0, 10) — the true match is (0, 15).
        // Chunk 1 produced a ghost (8, 15) that starts inside the truncated match
        // but extends past it, signaling truncation.
        let chunks = vec![vec![(0, 10)], vec![(8, 15), (20, 25)]];
        let full = [(0, 15), (20, 25)];
        let result = merge_with_oracle(chunks, &full);
        assert_eq!(result, vec![(0, 15), (20, 25)]);
    }

    #[test]
    fn merge_truncated_match_spans_three_chunks() {
        // A single match spans three chunks.
        // Chunk 0: truncated (0, 10). Chunk 1: ghost (8, 20). Chunk 2: ghost (18, 30).
        // True match: (0, 30). Then a normal match at (35, 40).
        let chunks = vec![vec![(0, 10)], vec![(8, 20)], vec![(18, 30), (35, 40)]];
        let full = [(0, 30), (35, 40)];
        let result = merge_with_oracle(chunks, &full);
        assert_eq!(result, vec![(0, 30), (35, 40)]);
    }

    #[test]
    fn merge_truncated_match_last_entry() {
        // Truncated match is the last entry in flat — no matches follow.
        let chunks = vec![vec![(0, 10)], vec![(8, 15)]];
        let full = [(0, 15)];
        let result = merge_with_oracle(chunks, &full);
        assert_eq!(result, vec![(0, 15)]);
    }

    #[test]
    fn merge_no_truncation_ghost_ends_at_prev_end() {
        // Ghost ends exactly at prev_end — NOT truncation.
        let chunks = vec![vec![(0, 10)], vec![(8, 10), (15, 20)]];
        let full = [(0, 10), (15, 20)];
        let result = merge_with_oracle(chunks, &full);
        assert_eq!(result, vec![(0, 10), (15, 20)]);
    }

    // ── pcre2 feature gate ────────────────────────────────────────────

    /// When the `pcre2` feature is on, the Split struct should have a
    /// `pcre2_regexes` field that is `Some` for compilable patterns.
    #[test]
    #[cfg(feature = "pcre2")]
    fn pcre2_regexes_populated_when_feature_on() {
        let s = Split::from_config(&json!({"Regex": "\\p{L}+"}), "Isolated", false).unwrap();
        assert!(
            s.pcre2_regexes.is_some(),
            "pcre2_regexes should be Some when pcre2 feature is enabled",
        );
    }

    /// When the `pcre2` feature is off, the fancy-regex fallback must
    /// produce identical results to the pcre2 path.  This test uses
    /// a complex real-world pattern (Llama-3) on non-trivial Unicode
    /// input to verify correctness of the fallback.
    #[test]
    #[cfg(not(feature = "pcre2"))]
    fn fancy_regex_fallback_matches_expected_output() {
        let s = Split::from_config(&json!({"Regex": LLAMA3_PATTERN}), "Isolated", false).unwrap();
        // Verify on a mix of scripts, emoji, whitespace, digits, and
        // contractions -- the same inputs that the pcre2 path handles.
        let inputs = [
            "Hello, world!",
            "café résumé naïve",
            "你好世界 こんにちは 안녕하세요",
            "\u{1f680}\u{1f4a1}\u{2728} emoji text \u{1f389}",
            "CamelCase snake_case kebab-case",
            "I'm don't we're they've I'll he'd",
            "100,000.50 + 3.14e-10 = ?",
        ];
        for &input in &inputs {
            let pieces = s.split(input).unwrap();
            assert_eq!(
                pieces.join(""),
                input,
                "fancy-regex fallback lost bytes on {input:?}: pieces={pieces:?}",
            );
        }
    }

    // ── integration: long match crossing parallel chunk boundary ─────

    #[test]
    fn long_match_crosses_parallel_boundary() {
        // Build an input large enough for parallel matching (>= 2 * MIN_CHUNK_SIZE = 16KB).
        // Place a long run of lowercase letters that exceeds CHUNK_OVERLAP (1KB)
        // and spans the authority zone boundary.
        let chunk = super::MIN_CHUNK_SIZE;
        // Put 5KB of 'a' right before the expected boundary, extending 3KB past it.
        // Total: 8KB match that crosses the boundary.
        let pre_gap = chunk.saturating_sub(5 * 1024);
        let match_len = 8 * 1024; // 8KB match, well over 1KB overlap
        let post_pad = chunk; // pad to ensure total >= 2 * chunk

        let mut input = String::with_capacity(pre_gap + match_len + post_pad);
        // Non-matching prefix: digits (won't match [a-z]+)
        for _ in 0..pre_gap {
            input.push('0');
        }
        let match_start = input.len();
        // The long match: all lowercase letters
        for _ in 0..match_len {
            input.push('a');
        }
        let match_end = input.len();
        // Non-matching suffix: digits
        for _ in 0..post_pad {
            input.push('0');
        }
        assert!(input.len() >= 2 * chunk, "input must trigger parallel path");

        let split =
            Split::from_config(&serde_json::json!({"Regex": "[a-z]+"}), "Isolated", false).unwrap();

        let pieces = split.split(&input).unwrap();
        // There should be exactly 3 pieces: digits, long 'a' run, digits.
        // The 'a' run must be ONE piece, not split across chunks.
        let long_piece = pieces.iter().find(|p| p.starts_with('a')).unwrap();
        assert_eq!(
            long_piece.len(),
            match_len,
            "long match should be {match_len} bytes, got {}; \
             match was split across chunk boundaries",
            long_piece.len(),
        );
        // Also verify via byte offsets
        let a_pieces: Vec<&str> = pieces
            .iter()
            .copied()
            .filter(|p| p.starts_with('a'))
            .collect();
        assert_eq!(
            a_pieces.len(),
            1,
            "should be exactly one 'a' piece, got {a_pieces:?}"
        );
        assert_eq!(&input[match_start..match_end], *long_piece,);
    }
}
