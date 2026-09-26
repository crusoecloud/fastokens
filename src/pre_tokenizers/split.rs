use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use fancy_regex::Regex;
use memchr::{memchr, memchr2};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Value;

use crate::pre_tokenized::{PreTokenizedString, Split as PtSplit};

use super::Error;
use super::scan::{self, ScanKind};

// Thread-local cache of previous Split results for incremental re-use.
thread_local! {
    static SPLIT_CACHE: RefCell<SplitCache> = RefCell::new(SplitCache::default());
}

#[derive(Default)]
struct SplitCache {
    /// Identity of the Split pre-tokenizer that populated this cache.
    /// Prevents cross-model contamination when different models share a thread.
    split_id: usize,
    prev_input: Vec<u8>,
    prev_matches: Vec<(usize, usize)>,
}

/// Minimum shared prefix length (bytes) before incremental re-use kicks in.
const INCREMENTAL_MIN_PREFIX: usize = 4096;

/// Slack before the first unknown byte: only matches ending at least this far
/// before it are reused, counted after backing off any whitespace run there
/// (see [`lookahead_safe_limit`]). Covers the short fixed lookahead of
/// tokenizer patterns past a match's end, e.g. a contraction suffix `(?i:'ll)`
/// probed after a letter run, or the `(?!\S)` check after a whitespace run.
const LOOKAHEAD_MARGIN: usize = 64;

/// PCRE2 resource limits applied while compiling Split regexes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pcre2Limits {
    pub match_limit: Option<u32>,
    pub depth_limit: Option<u32>,
    /// Heap limit in KiB, matching PCRE2's `(*LIMIT_HEAP=...)` units.
    pub heap_limit: Option<u32>,
    pub max_jit_stack_size: Option<usize>,
}

impl Pcre2Limits {
    fn has_any(self) -> bool {
        self.match_limit.is_some()
            || self.depth_limit.is_some()
            || self.heap_limit.is_some()
            || self.max_jit_stack_size.is_some()
    }

    fn apply_to_source(self, source: &str) -> std::string::String {
        if self.match_limit.is_none() && self.depth_limit.is_none() && self.heap_limit.is_none() {
            return source.to_string();
        }

        let mut prefixed = String::new();
        if let Some(limit) = self.heap_limit {
            prefixed.push_str(&format!("(*LIMIT_HEAP={limit})"));
        }
        if let Some(limit) = self.match_limit {
            prefixed.push_str(&format!("(*LIMIT_MATCH={limit})"));
        }
        if let Some(limit) = self.depth_limit {
            prefixed.push_str(&format!("(*LIMIT_DEPTH={limit})"));
        }
        prefixed.push_str(source);
        prefixed
    }
}

/// Wrapper around a JIT-compiled PCRE2 regex for a PCRE2-compatible Split
/// pattern. `source` is retained so [`Clone`] can recompile (the JIT regex is
/// not itself cloneable).
struct Pcre2Regex {
    regex: pcre2::bytes::Regex,
    source: std::string::String,
    max_jit_stack_size: Option<usize>,
}

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
        let mut builder = pcre2::bytes::RegexBuilder::new();
        builder
            .utf(true)
            .ucp(true)
            .jit_if_available(true)
            .max_jit_stack_size(self.max_jit_stack_size);
        Self {
            // Infallible in practice: `source` compiled successfully when this
            // instance was first built, and `Clone` cannot surface an error.
            regex: builder
                .build(&self.source)
                .expect("PCRE2 source recompiles: it already compiled once"),
            source: self.source.clone(),
            max_jit_stack_size: self.max_jit_stack_size,
        }
    }
}

/// Minimum chunk size (bytes) for parallel regex matching.
/// Parallel matching triggers when the input is >= 2 × this value (i.e. 64 KiB).
const MIN_CHUNK_SIZE: usize = 32 * 1024;

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
/// Each chunk scans this many bytes past the start of the next chunk, so the
/// next chunk's scan has already resynchronized with the sequential scan where
/// this chunk's trusted matches end (see [`stitch_chunk_matches`]). Only speed
/// depends on it: a match or whitespace run longer than the overlap is still
/// found, by scanning that stretch sequentially.
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
#[derive(Clone, Debug, Deserialize)]
pub struct SplitConfig {
    pub pattern: Pattern,
    #[serde(default)]
    pub behavior: SplitBehavior,
    #[serde(default)]
    pub invert: bool,
}

/// Monotonic counter for unique Split instance IDs.
static SPLIT_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// A compiled Split pre-tokenizer.
///
/// Constructed once from a pattern, behavior and invert flag (typically from
/// [`PreTokenizerConfig::Split`]), then reused across many inputs. Implements
/// [`Deserialize`] so it can be built directly from the JSON representation.
///
/// [`PreTokenizerConfig::Split`]: crate::PreTokenizerConfig::Split
#[derive(Clone, Debug, Deserialize)]
#[serde(try_from = "SplitConfig")]
pub struct Split {
    /// Unique identity for cache invalidation across different Split instances.
    #[serde(skip)]
    id: usize,
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
    /// Set when the pattern is a recognized tiktoken family, enabling the
    /// hand-written scanner fast path (see [`Self::scan_kind`]).
    scan: Option<ScanKind>,
    /// The regex source, retained so a multi-`Split` sequence (e.g. DeepSeek's)
    /// can be recognized after construction. One small `String` per Split.
    source: std::string::String,
}

/// Compile PCRE2 JIT regexes from `source`, returning `None` if PCRE2 cannot
/// handle the pattern (e.g. unsupported syntax).
fn try_compile_pcre2_regexes(
    source: &str,
    n: usize,
    limits: Pcre2Limits,
) -> Result<Option<Vec<Pcre2Regex>>, Error> {
    if has_class_intersection(source) {
        if limits.has_any() {
            return Err(Error::Unsupported(
                "PCRE2 limits require a PCRE2-compatible regex pattern".into(),
            ));
        }
        return Ok(None);
    }
    let source = limits.apply_to_source(source);
    let mut regexes = Vec::with_capacity(n);
    for _ in 0..n {
        let mut builder = pcre2::bytes::RegexBuilder::new();
        builder
            .utf(true)
            .ucp(true)
            .jit_if_available(true)
            .max_jit_stack_size(limits.max_jit_stack_size);
        let re = match builder.build(&source) {
            Ok(re) => re,
            Err(e) if limits.has_any() => {
                return Err(Error::Unsupported(format!(
                    "PCRE2 limits require a PCRE2-compatible regex pattern: {e}"
                )));
            }
            Err(_) => return Ok(None),
        };
        regexes.push(Pcre2Regex {
            regex: re,
            source: source.clone(),
            max_jit_stack_size: limits.max_jit_stack_size,
        });
    }
    Ok(Some(regexes))
}

fn has_class_intersection(source: &str) -> bool {
    // && inside a regex character class is the fancy-regex set-intersection syntax, e.g. [a&&b],
    // which PCRE2 does not support.
    let bytes = source.as_bytes();

    if memchr::memmem::find(bytes, b"&&").is_none() {
        return false;
    }

    let mut search_start = 0;
    'classes: while let Some(open_rel) = memchr(b'[', &bytes[search_start..]) {
        let open = search_start + open_rel;
        if is_escaped(bytes, open) {
            search_start = open + 1;
            continue;
        }

        let mut pos = open + 1;
        loop {
            let Some(rel) = memchr2(b'&', b']', &bytes[pos..]) else {
                return false;
            };
            let idx = pos + rel;
            if is_escaped(bytes, idx) {
                pos = idx + 1;
                continue;
            }

            match bytes[idx] {
                b'&' if bytes.get(idx + 1) == Some(&b'&') => return true,
                b']' if idx > open + 1 => {
                    search_start = idx + 1;
                    continue 'classes;
                }
                _ => pos = idx + 1,
            }
        }
    }

    false
}

fn is_escaped(bytes: &[u8], pos: usize) -> bool {
    let mut slash_count = 0;
    let mut i = pos;
    while i > 0 && bytes[i - 1] == b'\\' {
        slash_count += 1;
        i -= 1;
    }
    slash_count % 2 == 1
}

/// Compile `n` independent copies of a regex from `source`.
fn compile_regexes(source: &str, n: usize) -> Result<Vec<Regex>, Error> {
    let mut regexes = Vec::with_capacity(n);
    for _ in 0..n {
        regexes.push(Regex::new(source)?);
    }
    Ok(regexes)
}

impl TryFrom<SplitConfig> for Split {
    type Error = Error;

    fn try_from(raw: SplitConfig) -> Result<Self, Error> {
        Self::from_split_config_with_limits(raw, Pcre2Limits::default())
    }
}

impl Split {
    fn from_parts(
        source: std::string::String,
        behavior: SplitBehavior,
        invert: bool,
        limits: Pcre2Limits,
    ) -> Result<Self, Error> {
        let scan = scan::recognize(&source);
        let regexes = compile_regexes(&source, max_parallel())?;
        let pcre2_regexes = try_compile_pcre2_regexes(&source, max_parallel(), limits)?;
        Ok(Self {
            id: SPLIT_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            regexes,
            behavior,
            invert,
            pcre2_regexes,
            scan,
            source,
        })
    }

    /// This Split's `Isolated`, non-inverted regex source, if it is one; used to
    /// recognize a multi-`Split` pre-tokenizer sequence (DeepSeek). Returns `None`
    /// for any other behavior/invert combination so callers need not re-check.
    pub(crate) fn isolated_source(&self) -> Option<&str> {
        (self.behavior == SplitBehavior::Isolated && !self.invert).then_some(self.source.as_str())
    }

    /// The recognized scanner family for this Split, if it is eligible for the
    /// scanner fast path. Used by the fused scan+BPE encode path.
    ///
    /// Eligible when the Split keeps every pretoken and nothing else — either
    /// `Isolated` and not inverted, or `Removed` and inverted. The recognized
    /// patterns (`scan::recognize`) are *total*: their trailing `\s+` makes
    /// every byte fall in exactly one match, so there are no gaps between
    /// matches. With no gaps, "keep the matches and the (empty) gaps" (`Isolated`)
    /// and "keep the matches, drop the (empty) inverse" (`Removed` + `invert`)
    /// produce the identical covering pretokens the scanner emits. cl100k configs
    /// in the wild (e.g. Phi-4) use the `Removed`+`invert` spelling; o200k / Kimi
    /// use `Isolated`. The equivalence is checked against the regex engine in
    /// `scan::tests::scanner_matches_regex_engine`.
    pub fn scan_kind(&self) -> Option<ScanKind> {
        let kind = self.scan?;
        let isolated = self.behavior == SplitBehavior::Isolated && !self.invert;
        let removed_inverted = self.behavior == SplitBehavior::Removed && self.invert;
        (isolated || removed_inverted).then_some(kind)
    }

    /// Build a [`Split`] from a deserialized [`SplitConfig`] with explicit PCRE2
    /// resource limits.
    pub fn from_split_config_with_limits(
        config: SplitConfig,
        limits: Pcre2Limits,
    ) -> Result<Self, Error> {
        Self::from_parts(
            config.pattern.source(),
            config.behavior,
            config.invert,
            limits,
        )
    }

    /// Build a [`Split`] from raw JSON fields.
    ///
    /// `pattern` must be a JSON object with either a `"String"` key (literal,
    /// will be regex-escaped) or a `"Regex"` key (used as-is).
    pub fn from_config(pattern: &Value, behavior: &str, invert: bool) -> Result<Self, Error> {
        Self::from_config_with_limits(pattern, behavior, invert, Pcre2Limits::default())
    }

    /// [`Self::from_config`] with explicit PCRE2 resource limits.
    pub fn from_config_with_limits(
        pattern: &Value,
        behavior: &str,
        invert: bool,
        limits: Pcre2Limits,
    ) -> Result<Self, Error> {
        let pattern: Pattern = serde_json::from_value(pattern.clone())?;
        let source = pattern.source();
        let behavior: SplitBehavior = serde_json::from_value(Value::String(behavior.to_string()))?;
        Self::from_parts(source, behavior, invert, limits)
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
                let limit = lookahead_safe_limit(buffer, common_len);
                let reuse_count = cache.prev_matches.partition_point(|&(_, end)| end <= limit);
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
        if suffix.len() >= MIN_CHUNK_SIZE * 2 && pcre2.len() >= 2 {
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
    /// The input is divided into N zones of roughly equal size. Each thread
    /// scans its zone *plus* a [`CHUNK_OVERLAP`] tail, and
    /// [`stitch_chunk_matches`] joins the chunks' matches into exactly the
    /// sequential scan's result.
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

        // Zone boundaries: chunk i starts scanning at auth[i].
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

        let chunk_results: Result<Vec<ChunkScan>, Error> = (0..actual)
            .into_par_iter()
            .map(|i| {
                let auth_start = auth[i];
                let chunk_end = snap_char_ceil(text, (auth[i + 1] + CHUNK_OVERLAP).min(text.len()));
                // Search the text itself from `auth_start` (not a slice starting
                // there) so a search from any position sees the same subject
                // as the sequential scan, except for its end.
                let matches =
                    find_matches_pcre2_from(&text[..chunk_end], auth_start, base, &pcre2[i])?;
                Ok(ChunkScan {
                    start: base + auth_start,
                    matches,
                    trusted_end: base + chunk_trusted_end(text, chunk_end),
                })
            })
            .collect();

        let chunks = chunk_results?;

        let regex = &pcre2[0];
        let bytes = text.as_bytes();
        stitch_chunk_matches(&chunks, base, |pos| {
            let mut p = pos - base;
            loop {
                if p >= bytes.len() {
                    return Ok(None);
                }
                match regex.regex.find_at(bytes, p) {
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
        if let Some(pcre2) = &self.pcre2_regexes {
            let matches = if input.len() >= MIN_CHUNK_SIZE * 2 && pcre2.len() >= 2 {
                self.find_matches_pcre2_parallel(input, 0)?
            } else {
                find_matches_pcre2(input, 0, &pcre2[0])?
            };
            return Ok(matches_to_segments(&matches, input.len(), self.invert));
        }

        if input.len() >= MIN_CHUNK_SIZE * 2
            && self.regexes.len() >= 2
            && let Some(matches) = self.find_matches_fancy_parallel(input)?
        {
            return Ok(matches_to_segments(&matches, input.len(), self.invert));
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
    /// Same strategy as [`find_matches_pcre2_parallel`]: each thread scans its
    /// zone plus a [`CHUNK_OVERLAP`] tail, and [`stitch_chunk_matches`] joins
    /// the chunks.
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
        let chunk_results: Result<Vec<ChunkScan>, Error> = (0..actual)
            .into_par_iter()
            .map(|i| {
                let auth_start = auth[i];
                let chunk_end =
                    snap_char_ceil(input, (auth[i + 1] + CHUNK_OVERLAP).min(input.len()));
                let regex = &regexes[i];
                let chunk = &input[auth_start..chunk_end];
                let mut matches = Vec::new();
                for m in regex.find_iter(chunk) {
                    let m = m?;
                    if m.start() == m.end() {
                        continue;
                    }
                    matches.push((auth_start + m.start(), auth_start + m.end()));
                }
                Ok(ChunkScan {
                    start: auth_start,
                    matches,
                    trusted_end: chunk_trusted_end(input, chunk_end),
                })
            })
            .collect();

        let chunks = chunk_results?;

        let regex = &self.regexes[0];
        let result = stitch_chunk_matches(&chunks, 0, |pos| {
            let slice = &input[pos..];
            for m in regex.find_iter(slice) {
                let m = m?;
                if m.start() == m.end() {
                    continue;
                }
                return Ok(Some((pos + m.start(), pos + m.end())));
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

/// One parallel chunk's scan, in absolute offsets.
struct ChunkScan {
    /// Where the chunk's scan started searching.
    start: usize,
    /// Every match the chunk's scan found, in order.
    matches: Vec<(usize, usize)>,
    /// Largest match end that cannot depend on bytes past the end of the
    /// chunk, which the scan could not see (see [`chunk_trusted_end`]).
    trusted_end: usize,
}

/// Stitch per-chunk parallel scans into exactly the sequential scan's matches.
///
/// A chunk's scan may start inside a match, so its first matches can be wrong.
/// But a scan is a chain of searches, each starting where the previous match
/// ended, so once the sequential scan searches from a position the chunk's scan
/// also searched from (its start or one of its match ends), the two agree up to
/// the chunk's `trusted_end`. Between chunks, this runs the sequential scan
/// (`find_from`) until it reaches such a position. That is usually immediate:
/// the previous chunk's trusted matches end inside the overlap, where the next
/// chunk's scan has already resynchronized.
///
/// `find_from(pos)` must return the sequential scan's next non-empty match when
/// searching from absolute offset `pos`, or `None` past the last match.
fn stitch_chunk_matches(
    chunks: &[ChunkScan],
    base: usize,
    mut find_from: impl FnMut(usize) -> Result<Option<(usize, usize)>, Error>,
) -> Result<Vec<(usize, usize)>, Error> {
    let total: usize = chunks.iter().map(|c| c.matches.len()).sum();
    let mut result = Vec::with_capacity(total);
    let mut pos = base;
    for chunk in chunks {
        let synced = loop {
            if pos == chunk.start {
                break Some(0);
            }
            if let Ok(i) = chunk.matches.binary_search_by_key(&pos, |&(_, e)| e) {
                break Some(i + 1);
            }
            if pos >= chunk.trusted_end {
                break None;
            }
            match find_from(pos)? {
                Some(m) => {
                    result.push(m);
                    pos = m.1;
                }
                None => return Ok(result),
            }
        };
        if let Some(first) = synced {
            for &m in &chunk.matches[first..] {
                if m.1 > chunk.trusted_end {
                    break;
                }
                result.push(m);
                pos = m.1;
            }
        }
    }
    while let Some(m) = find_from(pos)? {
        result.push(m);
        pos = m.1;
    }
    Ok(result)
}

/// Trusted end (relative to `text`) for a parallel chunk scanned up to
/// `chunk_end`: all of it at the true end of `text`, otherwise
/// [`lookahead_safe_limit`] of the end the scan saw.
fn chunk_trusted_end(text: &str, chunk_end: usize) -> usize {
    if chunk_end == text.len() {
        chunk_end
    } else {
        lookahead_safe_limit(text, chunk_end)
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

/// Largest end of a match found with only `text[..end]` known that cannot
/// depend on bytes at or past `end` (where the input diverges from the cached
/// one, or where a parallel chunk's scan stopped).
///
/// A match can depend on bytes well past its end: `\s+(?!\S)` and `\s*[\r\n]+`
/// read the whole whitespace run and backtrack, so in `"x   y"` the match `"  "`
/// depends on the `y`, and a `\n` there would make the whole run one match. So
/// back off over the whitespace run reaching `end`, then by
/// [`LOOKAHEAD_MARGIN`] for bounded lookahead. This assumes that, outside
/// whitespace runs, no match depends on more than that many bytes past its
/// end, which holds for tokenizer patterns but not for arbitrary regexes.
fn lookahead_safe_limit(text: &str, end: usize) -> usize {
    let mut limit = end;
    while !text.is_char_boundary(limit) {
        limit -= 1;
    }
    for c in text[..limit].chars().rev() {
        // PCRE2's `\s` also includes U+180E, which is not White_Space.
        if !c.is_whitespace() && c != '\u{180e}' {
            break;
        }
        limit -= c.len_utf8();
    }
    limit.saturating_sub(LOOKAHEAD_MARGIN)
}

/// Find the length of the common prefix between two byte slices.
///
/// Compares 8 bytes at a time for speed on large inputs.
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    let chunks = min_len / 8;
    for i in 0..chunks {
        let off = i * 8;
        let wa = u64::from_le_bytes(a[off..off + 8].try_into().unwrap());
        let wb = u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
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
fn find_matches_pcre2(
    input: &str,
    base: usize,
    regex: &Pcre2Regex,
) -> Result<Vec<(usize, usize)>, Error> {
    find_matches_pcre2_from(input, 0, base, regex)
}

/// [`find_matches_pcre2`] scanning `input` from byte offset `start`.
fn find_matches_pcre2_from(
    input: &str,
    start: usize,
    base: usize,
    regex: &Pcre2Regex,
) -> Result<Vec<(usize, usize)>, Error> {
    let mut matches = Vec::with_capacity((input.len() - start) / 3);
    let bytes = input.as_bytes();
    let mut pos = start;
    if pos >= bytes.len() {
        return Ok(matches);
    }
    // Reuse one workspace within this scan for the default JIT stack. Custom
    // stacks use find_at's pool so they are not allocated and freed per scan.
    let mut locations = regex
        .max_jit_stack_size
        .is_none()
        .then(|| regex.regex.capture_locations());
    while pos < bytes.len() {
        let found = match &mut locations {
            Some(locations) => regex.regex.captures_read_at(locations, bytes, pos),
            None => regex.regex.find_at(bytes, pos),
        };
        match found {
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

    use crate::tiktoken::QWEN2_PATTERN;

    use super::*;

    #[test]
    fn scratch_reuse_matches_stock_offsets_and_errors() {
        fn stock(
            input: &str,
            base: usize,
            regex: &Pcre2Regex,
        ) -> Result<Vec<(usize, usize)>, String> {
            let mut out = Vec::new();
            let mut pos = 0;
            while pos < input.len() {
                match regex.regex.find_at(input.as_bytes(), pos) {
                    Ok(Some(m)) if m.start() == m.end() => pos = m.end() + 1,
                    Ok(Some(m)) => {
                        out.push((base + m.start(), base + m.end()));
                        pos = m.end();
                    }
                    Ok(None) => break,
                    Err(e) => return Err(format!("PCRE2: {e}")),
                }
            }
            Ok(out)
        }
        let mut inputs = vec![String::new()];
        let mut level = vec![String::new()];
        for _ in 0..4 {
            level = level
                .iter()
                .flat_map(|s| ["a", "b", " ", "\n", "é", "🙂"].map(|c| format!("{s}{c}")))
                .collect();
            inputs.extend(level.clone());
        }
        inputs.push("aaaaaaaaaaaaaaaa!".into());
        inputs.push("hello 世界 123\n".repeat(MIN_CHUNK_SIZE));
        for pattern in [
            r"a|(?=b)|$(?R)",
            r"\w+|\s+|[^\w\s]",
            r"a*",
            r"",
            r"^|$",
            r"\b",
            r"(?=a)",
            r"(?<=a)b",
            r"a|(?=b)|$",
            r"\A.+\z",
            r"\s+(?!\S)",
            r"never_matches",
            r"(*LIMIT_MATCH=1)^(a+)+$",
            r"a\K",
            r"(?=é)|.",
        ] {
            for max_jit_stack_size in [None, Some(1 << 20)] {
                let limits = Pcre2Limits {
                    max_jit_stack_size,
                    ..Default::default()
                };
                let regexes = try_compile_pcre2_regexes(pattern, 1, limits)
                    .unwrap()
                    .unwrap();
                let regex = &regexes[0];
                for input in &inputs {
                    let got = find_matches_pcre2(input, 17, regex).map_err(|e| match e {
                        Error::Unsupported(s) => s,
                        e => panic!("unexpected error variant: {e:?}"),
                    });
                    assert_eq!(
                        got,
                        stock(input, 17, regex),
                        "pattern={pattern:?}, stack={max_jit_stack_size:?}, input={input:?}"
                    );
                }
            }
        }
    }

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

    #[test]
    fn pcre2_match_limit_is_enforced() {
        let split = Split::from_config_with_limits(
            &json!({"Regex": "^(a+)+$"}),
            "Isolated",
            false,
            Pcre2Limits {
                match_limit: Some(1),
                ..Default::default()
            },
        )
        .unwrap();

        let err = split.split("aaaaaaaaaaaaaaaa!").unwrap_err();
        assert!(
            err.to_string().contains("match limit"),
            "expected match limit error, got {err}"
        );
    }

    #[test]
    fn pcre2_limits_require_pcre2_compatible_pattern() {
        let err = Split::from_config_with_limits(
            &json!({"Regex": "[a&&b]"}),
            "Isolated",
            false,
            Pcre2Limits {
                match_limit: Some(1_000),
                ..Default::default()
            },
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("PCRE2 limits require"),
            "expected PCRE2 compatibility error, got {err}"
        );
    }

    #[test]
    fn pcre2_limits_allow_literal_ampersands() {
        let split = Split::from_config_with_limits(
            &json!({"String": "&&"}),
            "Isolated",
            false,
            Pcre2Limits {
                match_limit: Some(1_000),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(split.split("a&&b").unwrap(), vec!["a", "&&", "b"]);
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

    // ── stitch_chunk_matches ─────────────────────────────────────────

    /// Stitch `chunks` with `full` as the sequential scan.
    fn stitch_with_oracle(chunks: &[ChunkScan], full: &[(usize, usize)]) -> Vec<(usize, usize)> {
        stitch_chunk_matches(chunks, 0, |pos| {
            Ok(full.iter().copied().find(|&(s, _)| s >= pos))
        })
        .unwrap()
    }

    fn chunk(start: usize, trusted_end: usize, matches: &[(usize, usize)]) -> ChunkScan {
        ChunkScan {
            start,
            matches: matches.to_vec(),
            trusted_end,
        }
    }

    #[test]
    fn stitch_drops_untrusted_truncated_match() {
        // Chunk 0 saw (0, 10) cut short by its end; the true match is (0, 15).
        // It is past chunk 0's trusted end, so the sequential scan finds it,
        // lands on chunk 1's match end 15 and takes chunk 1 from there.
        let chunks = [chunk(0, 8, &[(0, 10)]), chunk(8, 25, &[(8, 15), (20, 25)])];
        let full = [(0, 15), (20, 25)];
        assert_eq!(stitch_with_oracle(&chunks, &full), full);
    }

    #[test]
    fn stitch_match_spanning_three_chunks() {
        let chunks = [
            chunk(0, 8, &[(0, 10)]),
            chunk(8, 18, &[(8, 20)]),
            chunk(18, 40, &[(18, 30), (35, 40)]),
        ];
        let full = [(0, 30), (35, 40)];
        assert_eq!(stitch_with_oracle(&chunks, &full), full);
    }

    #[test]
    fn stitch_resyncs_misaligned_chunk() {
        // Chunk 1 started inside the true match (8, 12), so its first matches
        // are misaligned; it agrees again from its match ending at 20.
        let chunks = [
            chunk(0, 14, &[(0, 4), (4, 8), (8, 12)]),
            chunk(10, 40, &[(10, 13), (13, 20), (20, 30), (30, 40)]),
        ];
        let full = [(0, 4), (4, 8), (8, 12), (12, 20), (20, 30), (30, 40)];
        assert_eq!(stitch_with_oracle(&chunks, &full), full);
    }

    #[test]
    fn stitch_skips_chunk_inside_one_match() {
        // One true match covers all of chunk 1.
        let chunks = [
            chunk(0, 5, &[(0, 4), (4, 9)]),
            chunk(10, 18, &[(10, 20)]),
            chunk(20, 40, &[(20, 30), (30, 40)]),
        ];
        let full = [(0, 4), (4, 30), (30, 40)];
        assert_eq!(stitch_with_oracle(&chunks, &full), full);
    }

    #[test]
    fn parallel_scan_matches_sequential_with_long_whitespace_runs() {
        // Whitespace runs longer than CHUNK_OVERLAP, which a chunk's scan
        // cannot see the end of.
        let ws = [" ", "\n", "\t", "\u{3000}", " \n", "\r\n", "  "];
        let words = ["word", "x", "-", "12345", "héllo", "网友", "!!", "I'LL"];
        let mut state: u64 = 0x1234_5678;
        let mut rnd = |n: usize| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % n
        };
        let mut inputs = Vec::new();
        for _ in 0..300 {
            let target = 2 * MIN_CHUNK_SIZE + rnd(48 * 1024);
            let mut t = String::new();
            while t.len() < target {
                if rnd(10) < 3 {
                    let w = ws[rnd(ws.len())];
                    for _ in 0..1 + rnd(3000) {
                        t.push_str(w);
                    }
                } else {
                    for _ in 0..1 + rnd(200) {
                        t.push_str(words[rnd(words.len())]);
                        t.push_str(ws[rnd(ws.len())]);
                    }
                }
            }
            inputs.push(t);
        }

        for pattern in [QWEN2_PATTERN, LLAMA3_PATTERN] {
            let mut split =
                Split::from_config(&json!({"Regex": pattern}), "Isolated", false).unwrap();
            // Force several matchers even when the test runs with few CPUs.
            let pcre2 = split.pcre2_regexes.as_mut().unwrap();
            while pcre2.len() < 4 {
                pcre2.push(pcre2[0].clone());
            }
            while split.regexes.len() < 4 {
                split.regexes.push(split.regexes[0].clone());
            }
            for (i, t) in inputs.iter().enumerate() {
                let sequential =
                    find_matches_pcre2(t, 0, &split.pcre2_regexes.as_ref().unwrap()[0]).unwrap();
                let parallel = split.find_matches_pcre2_parallel(t, 0).unwrap();
                assert!(parallel == sequential, "PCRE2 input {i}, len {}", t.len());
                if i % 10 == 0 {
                    let fancy = split.find_matches_fancy_parallel(t).unwrap();
                    if let Some(fancy) = fancy {
                        assert!(fancy == sequential, "fancy input {i}, len {}", t.len());
                    }
                }
            }
        }
    }

    // ── integration: long match crossing parallel chunk boundary ─────

    #[test]
    fn long_input_with_single_pcre2_matcher_falls_back_to_sequential() {
        let mut split = Split::from_config(&json!({"Regex": "\\s+"}), "Isolated", false).unwrap();
        split.pcre2_regexes.as_mut().unwrap().truncate(1);

        let input = format!(
            "{} {}",
            "a".repeat(MIN_CHUNK_SIZE),
            "b".repeat(MIN_CHUNK_SIZE),
        );
        let mut pts = PreTokenizedString::from_text(&input);

        split.pre_tokenize(&mut pts).unwrap();

        assert_eq!(pts_texts(&pts).concat(), input);
    }

    #[test]
    fn long_match_crosses_parallel_boundary() {
        const PATTERN: &str = "[a-z]+";

        // Build an input large enough for parallel matching (>= 2 * MIN_CHUNK_SIZE = 64 KiB).
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

        for max_jit_stack_size in [None, Some(1 << 20)] {
            let limits = Pcre2Limits {
                max_jit_stack_size,
                ..Default::default()
            };
            let mut split = Split::from_config_with_limits(
                &json!({"Regex": PATTERN}),
                "Isolated",
                false,
                limits,
            )
            .unwrap();
            // Force two matchers even when the test runs with one available CPU.
            split.pcre2_regexes = Some(
                try_compile_pcre2_regexes(PATTERN, 2, limits)
                    .unwrap()
                    .expect("PCRE2 unavailable; boundary repair path not covered"),
            );

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

    // ── Incremental cache: lookahead at divergence boundary ──

    #[test]
    fn cache_reuse_boundary_lookahead() {
        // Regression test: the split cache must not reuse a match whose
        // end sits exactly at `common_len`, because lookahead patterns
        // like `\s+(?!\S)` inspect the byte *at* that position — which
        // differs between the cached input and the new input.
        //
        // Setup:
        //   input_a = <5 KB filler> + trailing spaces  (short, ends with ws)
        //   input_b = <same filler> + same spaces + "more text"
        //
        // Without the fix (`end <= common_len`), the `\s+(?!\S)` match
        // from input_a (trailing ws at end-of-string) is wrongly reused
        // for input_b, where those spaces are followed by "more text".
        //
        // Must go through `pre_tokenize()` (not the test-only `split()`
        // helper) because the incremental cache lives exclusively in the
        // `pre_tokenize_pcre2_isolated` fast-path.

        // Build a filler prefix > INCREMENTAL_MIN_PREFIX (4096 bytes).
        let filler: String = "abcdefgh ".repeat(600); // 5400 bytes
        assert!(filler.len() > INCREMENTAL_MIN_PREFIX);

        let input_a = format!("{filler}   "); // ends with trailing spaces
        let input_b = format!("{filler}   more text after spaces");

        // Reference: pre_tokenize input_b with a fresh Split (clean cache).
        let fresh =
            Split::from_config(&json!({"Regex": LLAMA3_PATTERN}), "Isolated", false).unwrap();
        let mut pts_expected = PreTokenizedString::from_text(&input_b);
        fresh.pre_tokenize(&mut pts_expected).unwrap();
        let expected = pts_texts(&pts_expected);

        // Cached path: pre_tokenize input_a first (populates cache),
        // then input_b (reuses cache).
        let cached =
            Split::from_config(&json!({"Regex": LLAMA3_PATTERN}), "Isolated", false).unwrap();
        let mut pts_a = PreTokenizedString::from_text(&input_a);
        cached.pre_tokenize(&mut pts_a).unwrap();

        let mut pts_actual = PreTokenizedString::from_text(&input_b);
        cached.pre_tokenize(&mut pts_actual).unwrap();
        let actual = pts_texts(&pts_actual);

        assert_eq!(
            actual, expected,
            "cache reuse at boundary produced wrong splits; \
             lookahead context was stale"
        );
    }

    #[test]
    fn cache_reuse_whitespace_run_lookahead() {
        // A cached whitespace match can depend on bytes past where A and B
        // diverge (see `lookahead_safe_limit`), so it must not be reused for B.
        let words = "word ".repeat(1000);
        let spaces = " ".repeat(5000);
        let newline_spaces = format!("{words}\n{}", " ".repeat(3000));
        let cases: &[(&str, &str, &str)] = &[
            (&words, "   x", "   \n"),
            (&words, "  -", "  \n"),
            (&words, "\n  x", "\n  \n"),
            (&words, "   x", "    \r\n"),
            (&words, "    \u{2}", "     \n"),
            (&spaces, "x", "\n"),
            (&spaces, "\u{3000}x", "\u{3000}\n"),
            (&newline_spaces, "x", "\n"),
        ];
        for pattern in [QWEN2_PATTERN, LLAMA3_PATTERN] {
            let fresh = Split::from_config(&json!({"Regex": pattern}), "Isolated", false).unwrap();
            let cached = Split::from_config(&json!({"Regex": pattern}), "Isolated", false).unwrap();
            for &(prefix, tail_a, tail_b) in cases {
                let input_a = format!("{prefix}{tail_a}");
                let input_b = format!("{prefix}{tail_b}");

                let mut pts_expected = PreTokenizedString::from_text(&input_b);
                fresh.pre_tokenize(&mut pts_expected).unwrap();

                let mut pts_a = PreTokenizedString::from_text(&input_a);
                cached.pre_tokenize(&mut pts_a).unwrap();
                let mut pts_actual = PreTokenizedString::from_text(&input_b);
                cached.pre_tokenize(&mut pts_actual).unwrap();

                assert_eq!(
                    pts_texts(&pts_actual),
                    pts_texts(&pts_expected),
                    "stale cached match: tail_a={tail_a:?} tail_b={tail_b:?} \
                     prefix_len={}",
                    prefix.len()
                );
            }
        }
    }
}
