use std::collections::{HashMap, HashSet};
use std::fmt;

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};

use crate::json_structs::AddedTokenConfig;

/// A compiled set of added tokens that can be matched against input text.
///
/// The HuggingFace `tokenizer.json` format includes an `added_tokens` array of
/// literal patterns that are matched *before* the normal tokenization pipeline.
/// Matched spans are assigned their token IDs directly; unmatched spans pass
/// through normalization, pre-tokenization and the model as usual.
/// Inputs at least this long have their added-token candidates checked in
/// parallel (see [`AddedTokens::split_with`]).
const PARALLEL_SPLIT_MIN: usize = 256 * 1024;

pub struct AddedTokens {
    daac: DoubleArrayAhoCorasick<u32>,
    /// The same patterns, leftmost-longest, for the prefiltered paths: a text
    /// short enough to split sequentially is one search over it (the automaton's
    /// own prefilter and verification in one pass), and a parallel split's
    /// candidate is one anchored search — the longest token starting exactly
    /// there, a transition per byte of it.
    ac: aho_corasick::AhoCorasick,
    /// `ac`'s pattern index -> token id.
    ac_ids: Vec<u32>,
    /// Split a sequential text with one `ac` search rather than prefilter
    /// candidates + anchored checks: for a large token set (DeepSeek's 1283,
    /// Kimi's 256), whose few shared prefixes make most candidates real matches.
    /// A small set (GLM's 36 — `<|…|>`, `[MASK]`, `/nothink`) keeps the
    /// prefilter, whose fingerprints reject its frequent false candidates
    /// (`<|` of other templates, `/` in paths) more cheaply.
    ac_scan: bool,
    /// The entries this set was compiled from. Kept because compilation is
    /// lossy — `single_word` and `normalized` are not represented in the
    /// matcher — so an extended set cannot be rebuilt from the fields below.
    configs: Vec<AddedTokenConfig>,
    /// Token lengths (in bytes) indexed by token ID, for matched tokens only.
    /// Non-added token IDs map to 0.
    token_lens: Vec<usize>,
    /// Per-ID `lstrip`/`rstrip` flags. When set, a match absorbs adjacent
    /// Unicode whitespace (left/right) into the token span, matching HF's
    /// `AddedToken` behavior. Indexed by token ID; `false` for non-added IDs.
    lstrip: Vec<bool>,
    rstrip: Vec<bool>,
    /// First bytes of the added tokens not covered by [`Self::finders`]: a SIMD
    /// memchr skips positions that cannot start one of those (at most 3).
    start_bytes: Vec<u8>,
    /// Longest added token in bytes. Limits the DAAC scan window.
    max_token_len: usize,
    /// Bit `b0 << 8 | b1` set iff some added token starts with bytes `b0 b1`
    /// (or is exactly the single byte `b0`, which then admits every `b1`). A
    /// prefilter candidate failing it cannot start a match, so the DAAC probe is
    /// skipped — first bytes alone are weak for common ones like `/` or `<`.
    pair_ok: Box<[u64]>,
    /// First bytes of single-byte added tokens (the only ones that can match at
    /// the last input byte).
    single_ok: [bool; 256],
    /// The distinct first-4-byte prefixes of the tokens, as `(value, mask)` over a
    /// little-endian `u32` (shorter tokens mask their missing bytes), when there
    /// are few enough to test each candidate against — a sharper filter than
    /// `pair_ok` when tokens share a common multibyte first char (Llama-2's
    /// normalized `▁<s>`, `▁</s>`: every `▁` passes the 2-byte test). Empty: off.
    quad: Vec<(u32, u32)>,
    /// The tokens grouped by first byte: a finder for each group's longest common
    /// prefix when it is 2+ bytes (`<|` of ChatML specials, `▁<` of Llama-2's
    /// normalized ones, DeepSeek's lone `｜DSML｜` next to its `<…>` tokens), so
    /// candidates come from a SIMD substring search instead of a first-byte
    /// `memchr` that would stop at every occurrence of a common byte (`/`, or
    /// the lead byte of CJK full-width punctuation). At most 3.
    finders: Vec<memchr::memmem::Finder<'static>>,
    /// Whether [`Self::candidates`] applies (else the automaton scans everything).
    prefilter: bool,
    /// A packed SIMD (Teddy) searcher over the distinct token prefixes of
    /// [`Self::quad`], when it builds (a supported CPU): the candidates come from
    /// its fingerprint search, which on chat text (HTML, code, Markdown: a `<`
    /// or `[` every ~50 bytes) rarely stops at one that begins no token prefix —
    /// where a first-byte `memchr` would stop and restart at every one.
    teddy: Option<aho_corasick::packed::Searcher>,
    /// Mapping from token ID to token content string.
    id_to_content: HashMap<u32, String>,
    /// Reverse mapping: token content string → token ID.
    content_to_id: HashMap<String, u32>,
    /// Set of token IDs marked as special (e.g. BOS/EOS).
    special_ids: HashSet<u32>,
}

/// Most distinct 4-byte token prefixes [`AddedTokens::quad`] holds (GLM-5.3 has
/// 21): as many as its packed searcher takes.
const QUAD_MAX: usize = 64;

/// Added-token count from which a sequential split is one automaton search
/// (see [`AddedTokens::ac_scan`]).
const AC_SCAN_MIN_TOKENS: usize = 128;

/// A segment of the input after added-token splitting.
#[derive(Debug, PartialEq, Eq)]
pub enum Segment<'a> {
    /// A span that matched an added token. The `u32` is the token ID to emit
    /// directly.
    Token(u32),
    /// A span that did not match any added token. The `&str` should be run
    /// through the normal pipeline.
    Text(&'a str),
}

/// Public view of one added-token entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddedTokenInfo<'a> {
    pub id: u32,
    pub content: &'a str,
    pub special: bool,
}

impl AddedTokens {
    /// Build from the `added_tokens` array in `tokenizer.json`.
    ///
    /// Returns `None` if there are no added tokens.
    pub fn from_configs(configs: &[AddedTokenConfig]) -> Result<Option<Self>, String> {
        if configs.is_empty() {
            return Ok(None);
        }

        let max_id = configs.iter().map(|c| c.id).max().unwrap_or(0);
        let mut token_lens = vec![0usize; (max_id + 1) as usize];
        let mut lstrip = vec![false; (max_id + 1) as usize];
        let mut rstrip = vec![false; (max_id + 1) as usize];

        let mut id_to_content = HashMap::with_capacity(configs.len());
        let mut special_ids = HashSet::new();

        let mut content_to_id = HashMap::with_capacity(configs.len());

        let patterns: Vec<(&str, u32)> = configs
            .iter()
            .map(|c| {
                token_lens[c.id as usize] = c.content.len();
                lstrip[c.id as usize] = c.lstrip;
                rstrip[c.id as usize] = c.rstrip;
                id_to_content.insert(c.id, c.content.clone());
                content_to_id.insert(c.content.clone(), c.id);
                if c.special {
                    special_ids.insert(c.id);
                }
                (c.content.as_str(), c.id)
            })
            .collect();

        let ac = aho_corasick::AhoCorasick::builder()
            .match_kind(aho_corasick::MatchKind::LeftmostLongest)
            .start_kind(aho_corasick::StartKind::Both)
            .build(patterns.iter().map(|&(p, _)| p))
            .map_err(|e| format!("error building added-tokens automaton: {e}"))?;
        let ac_ids: Vec<u32> = patterns.iter().map(|&(_, id)| id).collect();
        let ac_scan = patterns.len() >= AC_SCAN_MIN_TOKENS;
        let daac = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(daachorse::MatchKind::LeftmostLongest)
            .build_with_values(patterns)
            .map_err(|e| format!("error building added-tokens DAAC: {e}"))?;

        let max_token_len = configs.iter().map(|c| c.content.len()).max().unwrap_or(0);
        let mut pair_ok = vec![0u64; 1024].into_boxed_slice();
        let mut single_ok = [false; 256];
        for c in configs {
            match *c.content.as_bytes() {
                [] => {}
                [b0] => {
                    single_ok[b0 as usize] = true;
                    for b1 in 0..256usize {
                        let k = (b0 as usize) << 8 | b1;
                        pair_ok[k >> 6] |= 1 << (k & 63);
                    }
                }
                [b0, b1, ..] => {
                    let k = (b0 as usize) << 8 | b1 as usize;
                    pair_ok[k >> 6] |= 1 << (k & 63);
                }
            }
        }
        let mut quad: Vec<(u32, u32)> = Vec::new();
        for c in configs {
            let b = c.content.as_bytes();
            if b.is_empty() {
                continue;
            }
            let k = b.len().min(4);
            let mut v = [0u8; 4];
            v[..k].copy_from_slice(&b[..k]);
            let mask = if k == 4 {
                u32::MAX
            } else {
                (1u32 << (8 * k)) - 1
            };
            let e = (u32::from_le_bytes(v), mask);
            if !quad.contains(&e) {
                quad.push(e);
            }
        }
        if quad.len() > QUAD_MAX {
            quad.clear();
        }
        // Group by first byte; a group whose common prefix spans 2+ bytes gets a
        // finder, the others' first bytes a memchr.
        let mut groups: std::collections::BTreeMap<u8, Vec<&[u8]>> = Default::default();
        for c in configs {
            let b = c.content.as_bytes();
            if let Some(&b0) = b.first() {
                groups.entry(b0).or_default().push(b);
            }
        }
        let mut finders = Vec::new();
        let mut start_bytes = Vec::new();
        for (&b0, members) in &groups {
            let first = members[0];
            let mut n = first.len();
            for b in &members[1..] {
                n = n.min(
                    first
                        .iter()
                        .zip(b.iter())
                        .take_while(|(x, y)| x == y)
                        .count(),
                );
            }
            // Char-aligned so the finder never starts inside a UTF-8 sequence of
            // a token (every match position is then a char boundary of the input).
            while n > 0 && std::str::from_utf8(&first[..n]).is_err() {
                n -= 1;
            }
            if n >= 2 {
                finders.push(memchr::memmem::Finder::new(&first[..n]).into_owned());
            } else {
                start_bytes.push(b0);
            }
        }
        if finders.len() > 3 {
            // Too many streams to merge: first bytes only.
            finders.clear();
            start_bytes = groups.keys().copied().collect();
        }
        let prefilter = start_bytes.len() <= 3 && !(finders.is_empty() && start_bytes.is_empty());
        let teddy = if quad.is_empty() {
            None
        } else {
            let mut b = aho_corasick::packed::Config::new()
                .match_kind(aho_corasick::packed::MatchKind::LeftmostFirst)
                .builder();
            for &(v, m) in &quad {
                b.add(&v.to_le_bytes()[..(m.count_ones() / 8) as usize]);
            }
            b.build()
        };

        Ok(Some(Self {
            daac,
            ac,
            ac_ids,
            ac_scan,
            configs: configs.to_vec(),
            token_lens,
            lstrip,
            rstrip,
            start_bytes,
            max_token_len,
            pair_ok,
            single_ok,
            quad,
            finders,
            prefilter,
            teddy,
            id_to_content,
            content_to_id,
            special_ids,
        }))
    }

    /// The entries this set was built from, in declaration order.
    ///
    /// Callers that extend the vocabulary append to this and rebuild with
    /// [`Self::from_configs`].
    pub fn configs(&self) -> &[AddedTokenConfig] {
        &self.configs
    }

    /// Look up the string content of an added token by ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_content.get(&id).map(String::as_str)
    }

    /// Look up the token ID for a content string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.content_to_id.get(token).copied()
    }

    /// Iterate the distinct content strings of the added tokens.
    ///
    /// Unlike [`Self::iter`], which yields one entry per ID, this yields one
    /// entry per string: two added tokens sharing a content collapse to one.
    /// That is the granularity a vocabulary count needs, since a vocabulary is a
    /// token -> ID map and cannot hold the same string twice.
    pub fn contents(&self) -> impl Iterator<Item = &str> {
        self.content_to_id.keys().map(String::as_str)
    }

    /// Check if a token ID is a special added token.
    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    /// Return the number of added tokens.
    pub fn len(&self) -> usize {
        self.id_to_content.len()
    }

    /// Return whether there are no added tokens.
    pub fn is_empty(&self) -> bool {
        self.id_to_content.is_empty()
    }

    /// Iterate over added-token entries.
    ///
    /// The iteration order is unspecified. Callers that need a stable order
    /// should sort by `id` themselves.
    pub fn iter(&self) -> impl Iterator<Item = AddedTokenInfo<'_>> {
        self.id_to_content
            .iter()
            .map(|(&id, content)| AddedTokenInfo {
                id,
                content: content.as_str(),
                special: self.special_ids.contains(&id),
            })
    }

    /// Split `input` into segments: spans matching added tokens and spans of
    /// regular text.
    ///
    /// Added tokens are matched leftmost-longest. Non-overlapping matches are
    /// emitted as [`Segment::Token`]; the gaps between them as
    /// [`Segment::Text`].
    pub fn split<'a>(&self, input: &'a str) -> Vec<Segment<'a>> {
        self.split_with(input, false)
    }

    /// Split `input`, optionally leaving special tokens as ordinary text.
    ///
    /// With `skip_special`, an entry flagged `special` is consumed by the scan
    /// but emitted as part of the surrounding [`Segment::Text`] instead of as a
    /// [`Segment::Token`], so control-token strings in untrusted input cannot
    /// produce control-token IDs. Non-special added tokens still match. This is
    /// what HuggingFace `tokenizers` does under `encode_special_tokens`, which
    /// `transformers` sets for `split_special_tokens=True`.
    pub fn split_with<'a>(&self, input: &'a str, skip_special: bool) -> Vec<Segment<'a>> {
        self.split_with_opt(input, skip_special).unwrap_or_else(|| {
            if input.is_empty() {
                Vec::new()
            } else {
                vec![Segment::Text(input)]
            }
        })
    }

    /// [`Self::split_with`], but `None` when nothing matched — the whole input is
    /// one text segment — so the common case (plain text) allocates nothing.
    pub fn split_with_opt<'a>(
        &self,
        input: &'a str,
        skip_special: bool,
    ) -> Option<Vec<Segment<'a>>> {
        // When there are few distinct start bytes, use SIMD memchr to skip
        // positions that cannot start any added token. This avoids scanning
        // the full input through the Aho-Corasick automaton.
        if !self.prefilter && self.teddy.is_none() {
            return Some(self.split_full_scan(input, skip_special));
        }
        let n = input.len();
        let threads = crate::fanout::threads();
        let hits = if n >= PARALLEL_SPLIT_MIN && threads > 1 {
            // Large input: verify the candidates of byte ranges in parallel (each
            // match check reads only a short window of the text); only ordering
            // them into segments is sequential.
            let ranges = threads.min(n / (PARALLEL_SPLIT_MIN / 2));
            let per = n.div_ceil(ranges);
            let parts = crate::fanout::map(ranges, |i| {
                let (lo, hi) = (i * per, ((i + 1) * per).min(n));
                self.hits(input, self.candidates(input.as_bytes(), lo, hi))
            });
            parts.concat()
        } else if self.ac_scan {
            // Leftmost-longest non-overlapping matches: the resolution
            // `segments_from_hits` applies (a skipped special still consumes its
            // span, as a match here does).
            self.ac
                .find_iter(input)
                .map(|m| (m.start(), m.end(), self.ac_ids[m.pattern().as_usize()]))
                .collect()
        } else if let Some(t) = &self.teddy {
            self.hits(input, Self::teddy_candidates(t, input.as_bytes(), 0, n))
        } else {
            self.hits(input, self.candidates(input.as_bytes(), 0, n))
        };
        if hits.is_empty() {
            return None;
        }
        Some(self.segments_from_hits(input, &hits, skip_special))
    }

    /// The Teddy prefilter's candidates in `lo..hi` (see [`Self::candidates`]), as
    /// a concrete iterator.
    fn teddy_candidates<'h>(
        t: &'h aho_corasick::packed::Searcher,
        b: &'h [u8],
        lo: usize,
        hi: usize,
    ) -> impl Iterator<Item = usize> + 'h {
        // Every start of a prefix in `lo..hi` (a match may run past `hi`),
        // overlapping ones included: resume one byte past each.
        let hay = &b[..(hi + 3).min(b.len())];
        let mut from = lo;
        std::iter::from_fn(move || {
            let m = t.find_in(hay, aho_corasick::Span::from(from..hay.len()))?;
            (m.start() < hi).then(|| {
                from = m.start() + 1;
                m.start()
            })
        })
    }

    /// Positions in `lo..hi` where an added token may start, per the prefilter
    /// (every occurrence, overlapping ones included: a token may start inside a
    /// self-overlapping prefix like `<<`), ascending. Needs [`Self::prefilter`].
    fn candidates<'h>(
        &'h self,
        b: &'h [u8],
        lo: usize,
        hi: usize,
    ) -> Box<dyn Iterator<Item = usize> + 'h> {
        if let Some(t) = &self.teddy {
            return Box::new(Self::teddy_candidates(t, b, lo, hi));
        }
        let mut streams: Vec<Box<dyn Iterator<Item = usize> + 'h>> = Vec::new();
        for f in &self.finders {
            // Matches starting before `hi` may end past it.
            let end = (hi + f.needle().len() - 1).min(b.len());
            let hay = &b[..end];
            let mut from = lo;
            streams.push(Box::new(std::iter::from_fn(move || {
                let p = from + f.find(hay.get(from..)?)?;
                (p < hi).then(|| {
                    from = p + 1;
                    p
                })
            })));
        }
        let sb = &self.start_bytes;
        let hay = &b[lo..hi];
        match sb.len() {
            0 => {}
            1 => streams.push(Box::new(
                memchr::memchr_iter(sb[0], hay).map(move |p| lo + p),
            )),
            2 => streams.push(Box::new(
                memchr::memchr2_iter(sb[0], sb[1], hay).map(move |p| lo + p),
            )),
            _ => streams.push(Box::new(
                memchr::memchr3_iter(sb[0], sb[1], sb[2], hay).map(move |p| lo + p),
            )),
        }
        if streams.len() == 1 {
            return streams.pop().unwrap();
        }
        // Merge the ascending streams (their positions differ: each stream's
        // candidates start with its own first bytes).
        let mut heads: Vec<(usize, Box<dyn Iterator<Item = usize> + 'h>)> = streams
            .into_iter()
            .filter_map(|mut it| it.next().map(|p| (p, it)))
            .collect();
        Box::new(std::iter::from_fn(move || {
            let (k, _) = heads.iter().enumerate().min_by_key(|(_, (p, _))| *p)?;
            let p = heads[k].0;
            match heads[k].1.next() {
                Some(q) => heads[k].0 = q,
                None => drop(heads.swap_remove(k)),
            }
            Some(p)
        }))
    }

    /// Whether the 4 bytes at `b[pos..]` (zero past the end) begin with a quad.
    #[inline(always)]
    fn quad_ok(&self, b: &[u8], pos: usize) -> bool {
        let mut w = [0u8; 4];
        let k = (b.len() - pos).min(4);
        w[..k].copy_from_slice(&b[pos..pos + k]);
        let w = u32::from_le_bytes(w);
        self.quad.iter().any(|&(v, m)| w & m == v)
    }

    /// The added-token matches starting at `candidates` (ascending): each as
    /// `(start, end, id)`. Every candidate is checked on its own — including ones
    /// inside an earlier match, which [`Self::segments_from_hits`] then skips —
    /// so disjoint ranges of candidates can be checked independently.
    fn hits(
        &self,
        input: &str,
        candidates: impl Iterator<Item = usize>,
    ) -> Vec<(usize, usize, u32)> {
        let mut hits = Vec::new();
        let b = input.as_bytes();
        for pos in candidates {
            let may_start = if pos + 1 < b.len() {
                let k = (b[pos] as usize) << 8 | b[pos + 1] as usize;
                self.pair_ok[k >> 6] >> (k & 63) & 1 != 0
            } else {
                self.single_ok[b[pos] as usize]
            };
            if !may_start {
                continue;
            }
            // Past the end the bytes read 0, so a prefix longer than what is left
            // cannot match — nor could its token.
            if !self.quad.is_empty() && !self.quad_ok(b, pos) {
                continue;
            }
            // The longest token starting exactly here.
            let end = (pos + self.max_token_len).min(b.len());
            let at = aho_corasick::Input::new(b)
                .span(pos..end)
                .anchored(aho_corasick::Anchored::Yes);
            if let Some(m) = self.ac.find(at) {
                hits.push((pos, m.end(), self.ac_ids[m.pattern().as_usize()]));
            }
        }
        hits
    }

    /// Resolve [`Self::hits`] into segments, left to right: a hit inside an
    /// earlier match (emitted or skipped) is not a match.
    fn segments_from_hits<'a>(
        &self,
        input: &'a str,
        hits: &[(usize, usize, u32)],
        skip_special: bool,
    ) -> Vec<Segment<'a>> {
        let mut segments = Vec::new();
        let mut prev_end = 0;
        // End of the last match, emitted or not. A skipped special token still
        // consumes its span, so a candidate inside it cannot start a second
        // match — the full-scan path's automaton advances the same way.
        let mut scan_from = 0;
        for &(pos, end, id) in hits {
            if pos < scan_from {
                continue;
            }
            if skip_special && self.is_special(id) {
                scan_from = end;
                continue;
            }
            let (start, end) = self.strip_bounds(input, id, pos, end, prev_end);
            if start > prev_end {
                segments.push(Segment::Text(&input[prev_end..start]));
            }
            segments.push(Segment::Token(id));
            prev_end = end;
            scan_from = end;
        }

        if prev_end < input.len() {
            segments.push(Segment::Text(&input[prev_end..]));
        }
        if segments.is_empty() && !input.is_empty() {
            segments.push(Segment::Text(input));
        }

        segments
    }

    /// Expand a match span to absorb adjacent Unicode whitespace per the
    /// token's `lstrip`/`rstrip` flags, matching HuggingFace `AddedToken`
    /// behavior. Absorbed whitespace is excluded from the surrounding text.
    ///
    /// `lstrip` extends the start left over whitespace, bounded by `floor`
    /// (the end of the previous segment) so it never reclaims already-emitted
    /// text. `rstrip` extends the end right over whitespace. When two strip
    /// tokens share a whitespace run, the left token's `rstrip` consumes it
    /// first (via the advanced `floor`), so the right token's `lstrip` finds
    /// none — mirroring HF's left-to-right resolution.
    fn strip_bounds(
        &self,
        input: &str,
        id: u32,
        mut start: usize,
        mut end: usize,
        floor: usize,
    ) -> (usize, usize) {
        if self.lstrip[id as usize] {
            for (rel_i, c) in input[floor..start].char_indices().rev() {
                if c.is_whitespace() {
                    start = floor + rel_i;
                } else {
                    break;
                }
            }
        }
        if self.rstrip[id as usize] {
            for c in input[end..].chars() {
                if c.is_whitespace() {
                    end += c.len_utf8();
                } else {
                    break;
                }
            }
        }
        (start, end)
    }

    /// Full-scan fallback for >3 distinct start bytes.
    fn split_full_scan<'a>(&self, input: &'a str, skip_special: bool) -> Vec<Segment<'a>> {
        let mut segments = Vec::new();
        let mut prev_end = 0;

        for m in self.daac.leftmost_find_iter(input) {
            if skip_special && self.is_special(m.value()) {
                continue;
            }
            let (start, end) = self.strip_bounds(input, m.value(), m.start(), m.end(), prev_end);
            if start > prev_end {
                segments.push(Segment::Text(&input[prev_end..start]));
            }
            segments.push(Segment::Token(m.value()));
            prev_end = end;
        }

        if prev_end < input.len() {
            segments.push(Segment::Text(&input[prev_end..]));
        }

        segments
    }
}

impl fmt::Debug for AddedTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.token_lens.iter().filter(|&&len| len > 0).count();
        f.debug_struct("AddedTokens")
            .field("count", &count)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(id: u32, content: &str) -> AddedTokenConfig {
        AddedTokenConfig {
            id,
            content: content.to_string(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: false,
        }
    }

    /// Large inputs check candidates in parallel ranges; the result must equal
    /// the automaton's full scan, for every prefilter (shared prefix, 1–3 start
    /// bytes), self-overlapping tokens, strip flags and special skipping, with
    /// tokens landing on range boundaries.
    #[test]
    fn parallel_split_matches_full_scan() {
        let cfg = |id: u32, c: &str, lstrip: bool, rstrip: bool, special: bool| AddedTokenConfig {
            lstrip,
            rstrip,
            special,
            ..make_config(id, c)
        };
        let sets: Vec<Vec<AddedTokenConfig>> = vec![
            // Shared `<|` prefix (the `memmem` prefilter).
            vec![
                cfg(1, "<|a|>", false, false, true),
                cfg(2, "<|ab|>", true, true, false),
                cfg(3, "<||>", false, true, true),
            ],
            // One start byte, self-overlapping.
            vec![
                cfg(1, "<<", false, false, false),
                cfg(2, "<<<", true, false, true),
            ],
            // Two and three start bytes.
            vec![
                cfg(1, "[X]", false, false, true),
                cfg(2, "<y>", false, true, false),
            ],
            vec![
                cfg(1, "[X]", true, false, false),
                cfg(2, "<y>", false, false, true),
                cfg(3, "{z}", false, true, false),
            ],
            // Per-first-byte groups: a memchr for `<`, a finder for the lone
            // multibyte-led token (DeepSeek's `｜DSML｜` among `<…>` ones)...
            vec![
                cfg(1, "<｜a｜>", false, false, true),
                cfg(2, "<|ab|>", false, true, true),
                cfg(3, "</c>", true, false, false),
                cfg(4, "｜DSML｜", false, false, true),
            ],
            // ... memchrs for `<`, `[` and a finder for `/nothink` (GLM) ...
            vec![
                cfg(1, "<|a|>", false, false, true),
                cfg(2, "[MASK]", false, false, true),
                cfg(3, "[gMASK]", true, false, true),
                cfg(4, "/nothink", false, true, false),
            ],
            // ... three finders and no memchr.
            vec![
                cfg(1, "<|a|>", false, false, true),
                cfg(2, "<|ab|>", false, false, true),
                cfg(3, "｜DSML｜", true, true, true),
                cfg(4, "/nothink", false, false, false),
            ],
            // Many distinct 4-byte prefixes (the packed searcher's full load),
            // some of them tokens shorter than 4 bytes.
            (0..40u32)
                .map(|i| {
                    let c = (b'a' + (i % 26) as u8) as char;
                    let t = match i % 4 {
                        0 => format!("<|{c}{i}|>"),
                        1 => format!("[{c}{i}]"),
                        2 => format!("<{c}"),
                        _ => format!("/{c}x{i}"),
                    };
                    cfg(100 + i, &t, i % 3 == 0, i % 5 == 0, i % 2 == 0)
                })
                .collect(),
        ];
        let alphabet = [
            "<|a0|>",
            "[b1]",
            "<c",
            "/dx3",
            "<|e4|>",
            "[f5]",
            "<g",
            "/hx7",
            "<|",
            "[b",
            "/d",
            "a",
            "b",
            " ",
            "  ",
            "\n",
            "<",
            "<<",
            "<|",
            "|>",
            "[",
            "]",
            "X",
            "y",
            ">",
            "{",
            "}",
            "z",
            "é",
            "中",
            "<|a|>",
            "<|ab|>",
            "<||>",
            "<<<",
            "[X]",
            "<y>",
            "{z}",
            "｜",
            "，",
            "：",
            "｜DS",
            "｜DSML｜",
            "<｜a｜>",
            "</c>",
            "/",
            "/no",
            "/nothink",
            "[MASK]",
            "[gMASK]",
            "c>",
        ];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for configs in &sets {
            let at = AddedTokens::from_configs(configs).unwrap().unwrap();
            for round in 0..3 {
                let mut text = String::new();
                // A small input (the sequential scan) and two large ones.
                let target = if round == 0 {
                    5000
                } else {
                    PARALLEL_SPLIT_MIN * (1 + round) + 777
                };
                while text.len() < target {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    text.push_str(alphabet[(state % alphabet.len() as u64) as usize]);
                }
                for skip_special in [false, true] {
                    assert_eq!(
                        at.split_with(&text, skip_special),
                        at.split_full_scan(&text, skip_special),
                        "configs={configs:?} round={round} skip_special={skip_special}"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_configs() {
        let result = AddedTokens::from_configs(&[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn no_match() {
        let configs = vec![make_config(100, "<special>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("hello world");
        assert_eq!(segs, vec![Segment::Text("hello world")]);
    }

    #[test]
    fn single_match_at_start() {
        let configs = vec![make_config(100, "<s>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("<s>hello");
        assert_eq!(segs, vec![Segment::Token(100), Segment::Text("hello")]);
    }

    #[test]
    fn single_match_at_end() {
        let configs = vec![make_config(100, "</s>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("hello</s>");
        assert_eq!(segs, vec![Segment::Text("hello"), Segment::Token(100)]);
    }

    #[test]
    fn match_in_middle() {
        let configs = vec![make_config(42, "<sep>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("hello<sep>world");
        assert_eq!(
            segs,
            vec![
                Segment::Text("hello"),
                Segment::Token(42),
                Segment::Text("world"),
            ]
        );
    }

    #[test]
    fn multiple_matches() {
        let configs = vec![make_config(1, "<a>"), make_config(2, "<b>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("x<a>y<b>z");
        assert_eq!(
            segs,
            vec![
                Segment::Text("x"),
                Segment::Token(1),
                Segment::Text("y"),
                Segment::Token(2),
                Segment::Text("z"),
            ]
        );
    }

    #[test]
    fn adjacent_matches() {
        let configs = vec![make_config(1, "<a>"), make_config(2, "<b>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("<a><b>");
        assert_eq!(segs, vec![Segment::Token(1), Segment::Token(2)]);
    }

    #[test]
    fn longest_match_wins() {
        let configs = vec![make_config(1, "<file>"), make_config(2, "<filename>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("a<filename>b");
        assert_eq!(
            segs,
            vec![Segment::Text("a"), Segment::Token(2), Segment::Text("b"),]
        );
    }

    #[test]
    fn entire_input_is_added_token() {
        let configs = vec![make_config(99, "hello")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("hello");
        assert_eq!(segs, vec![Segment::Token(99)]);
    }

    #[test]
    fn empty_input() {
        let configs = vec![make_config(1, "<s>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("");
        assert!(segs.is_empty());
    }

    // ── token_to_id (content → id reverse lookup) ───────────────────────

    #[test]
    fn token_to_id_finds_added_token() {
        let configs = vec![make_config(42, "<special>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(at.token_to_id("<special>"), Some(42));
    }

    #[test]
    fn token_to_id_returns_none_for_unknown() {
        let configs = vec![make_config(1, "<known>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(at.token_to_id("<unknown>"), None);
    }

    #[test]
    fn token_to_id_and_id_to_token_are_inverses() {
        let configs = vec![
            make_config(10, "<bos>"),
            make_config(11, "<eos>"),
            make_config(12, "<pad>"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        for cfg in &configs {
            let id = at.token_to_id(&cfg.content).unwrap();
            assert_eq!(id, cfg.id);
            assert_eq!(at.id_to_token(id), Some(cfg.content.as_str()));
        }
    }

    // ── Unicode and multi-byte token content ────────────────────────────

    #[test]
    fn unicode_token_content() {
        let configs = vec![
            make_config(1, "▁"), // U+2581  (SentencePiece metaspace)
            make_config(2, "Ġ"), // U+0120  (GPT-2 space marker)
            make_config(3, "日本語"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(
            at.split("▁hello"),
            vec![Segment::Token(1), Segment::Text("hello")]
        );
        assert_eq!(
            at.split("Ġworld"),
            vec![Segment::Token(2), Segment::Text("world")]
        );
        assert_eq!(
            at.split("日本語text"),
            vec![Segment::Token(3), Segment::Text("text")]
        );
        assert_eq!(at.token_to_id("▁"), Some(1));
        assert_eq!(at.token_to_id("Ġ"), Some(2));
        assert_eq!(at.token_to_id("日本語"), Some(3));
    }

    #[test]
    fn emoji_token_content() {
        let configs = vec![make_config(7, "🌍")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(
            at.split("hello 🌍 world"),
            vec![
                Segment::Text("hello "),
                Segment::Token(7),
                Segment::Text(" world"),
            ]
        );
    }

    // ── is_special ──────────────────────────────────────────────────────

    #[test]
    fn is_special_only_for_marked_tokens() {
        let mut special = make_config(1, "<bos>");
        special.special = true;
        let non_special = make_config(2, "<extra>");
        let at = AddedTokens::from_configs(&[special, non_special])
            .unwrap()
            .unwrap();
        assert!(at.is_special(1));
        assert!(!at.is_special(2));
        assert!(!at.is_special(99)); // unknown id
    }

    #[test]
    fn iter_exposes_id_content_and_special_flag() {
        let mut special = make_config(1, "<bos>");
        special.special = true;
        let plain = make_config(2, "<extra>");
        let at = AddedTokens::from_configs(&[special, plain])
            .unwrap()
            .unwrap();

        let mut entries: Vec<_> = at.iter().collect();
        entries.sort_by_key(|entry| entry.id);

        assert_eq!(
            entries,
            vec![
                AddedTokenInfo {
                    id: 1,
                    content: "<bos>",
                    special: true,
                },
                AddedTokenInfo {
                    id: 2,
                    content: "<extra>",
                    special: false,
                },
            ]
        );
    }

    // ── len / is_empty ───────────────────────────────────────────────────

    #[test]
    fn len_returns_token_count() {
        let configs = vec![
            make_config(1, "<a>"),
            make_config(2, "<b>"),
            make_config(3, "<c>"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(at.len(), 3);
        assert!(!at.is_empty());
    }

    #[test]
    fn three_tokens_with_shared_start_byte() {
        // <, <s>, <sep> all start with '<'  — exercises the memchr prefilter
        // (≤3 distinct first bytes → SIMD path).
        let configs = vec![
            make_config(1, "<"),
            make_config(2, "<s>"),
            make_config(3, "<sep>"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        // Longest match: <sep> wins over <s> or <
        let segs = at.split("x<sep>y<s>z<");
        assert_eq!(
            segs,
            vec![
                Segment::Text("x"),
                Segment::Token(3),
                Segment::Text("y"),
                Segment::Token(2),
                Segment::Text("z"),
                Segment::Token(1),
            ]
        );
    }

    #[test]
    fn four_distinct_start_bytes() {
        // >3 distinct first bytes: no memchr prefilter (the quad scan instead,
        // where available).
        let configs = vec![
            make_config(1, "<bos>"),
            make_config(2, "[SEP]"),
            make_config(3, "{pad}"),
            make_config(4, "|mask|"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("<bos>[SEP]{pad}|mask|");
        assert_eq!(
            segs,
            vec![
                Segment::Token(1),
                Segment::Token(2),
                Segment::Token(3),
                Segment::Token(4),
            ]
        );
    }

    #[test]
    fn token_surrounded_by_text() {
        let configs = vec![make_config(5, "<mid>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("prefix <mid> suffix");
        assert_eq!(
            segs,
            vec![
                Segment::Text("prefix "),
                Segment::Token(5),
                Segment::Text(" suffix"),
            ]
        );
    }

    #[test]
    fn repeated_same_token() {
        let configs = vec![make_config(9, "<r>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("<r><r><r>");
        assert_eq!(
            segs,
            vec![Segment::Token(9), Segment::Token(9), Segment::Token(9)]
        );
    }

    // ── lstrip / rstrip whitespace absorption ───────────────────────────

    fn make_strip_config(id: u32, content: &str, lstrip: bool, rstrip: bool) -> AddedTokenConfig {
        AddedTokenConfig {
            id,
            content: content.to_string(),
            single_word: false,
            lstrip,
            rstrip,
            normalized: false,
            special: true,
        }
    }

    #[test]
    fn lstrip_absorbs_leading_whitespace() {
        let configs = vec![make_strip_config(1, "<s>", true, false)];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        // The spaces before <s> are absorbed into the token span; "ab" remains,
        // trailing "cd" is untouched.
        assert_eq!(
            at.split("ab   <s>cd"),
            vec![Segment::Text("ab"), Segment::Token(1), Segment::Text("cd")]
        );
    }

    #[test]
    fn rstrip_absorbs_trailing_whitespace() {
        let configs = vec![make_strip_config(1, "<s>", false, true)];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(
            at.split("ab<s>   cd"),
            vec![Segment::Text("ab"), Segment::Token(1), Segment::Text("cd")]
        );
    }

    #[test]
    fn strip_both_sides() {
        let configs = vec![make_strip_config(1, "<s>", true, true)];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(
            at.split("ab \t <s> \n cd"),
            vec![Segment::Text("ab"), Segment::Token(1), Segment::Text("cd")]
        );
    }

    #[test]
    fn no_strip_keeps_whitespace() {
        let configs = vec![make_strip_config(1, "<s>", false, false)];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(
            at.split("ab <s> cd"),
            vec![
                Segment::Text("ab "),
                Segment::Token(1),
                Segment::Text(" cd"),
            ]
        );
    }

    #[test]
    fn adjacent_strip_tokens_share_whitespace() {
        // The <|im_end|>\n<|im_start|> case from Phi-4: both tokens strip both
        // sides. The single \n between them must be absorbed exactly once and
        // produce no text token.
        let configs = vec![
            make_strip_config(1, "<|im_end|>", true, true),
            make_strip_config(2, "<|im_start|>", true, true),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(
            at.split("hi<|im_end|>\n<|im_start|>user"),
            vec![
                Segment::Text("hi"),
                Segment::Token(1),
                Segment::Token(2),
                Segment::Text("user"),
            ]
        );
    }

    #[test]
    fn lstrip_bounded_by_previous_token() {
        // A preceding token's span must not be reclaimed by the next token's
        // lstrip: there is no whitespace between the tokens here.
        let configs = vec![
            make_strip_config(1, "<a>", false, false),
            make_strip_config(2, "<b>", true, false),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(
            at.split("<a><b>"),
            vec![Segment::Token(1), Segment::Token(2)]
        );
    }

    // ── skip_special (HF `encode_special_tokens`) ───────────────────────

    #[test]
    fn skip_special_leaves_special_tokens_as_text() {
        let mut special = make_config(1, "<|user|>");
        special.special = true;
        let at = AddedTokens::from_configs(&[special]).unwrap().unwrap();

        assert_eq!(
            at.split_with("a<|user|>b", false),
            vec![Segment::Text("a"), Segment::Token(1), Segment::Text("b"),]
        );
        assert_eq!(
            at.split_with("a<|user|>b", true),
            vec![Segment::Text("a<|user|>b")]
        );
    }

    #[test]
    fn skip_special_keeps_non_special_tokens() {
        // The distinction HuggingFace draws: only entries flagged `special` are
        // skipped, so ordinary added vocabulary still tokenizes as itself.
        let mut special = make_config(1, "<|user|>");
        special.special = true;
        let plain = make_config(2, "<think>");
        let at = AddedTokens::from_configs(&[special, plain])
            .unwrap()
            .unwrap();

        assert_eq!(
            at.split_with("<|user|><think>", true),
            vec![Segment::Text("<|user|>"), Segment::Token(2),]
        );
    }

    #[test]
    fn skip_special_does_not_absorb_whitespace() {
        // A skipped token is text, so its strip flags must not eat the spaces
        // around it the way an emitted token would.
        let at = AddedTokens::from_configs(&[make_strip_config(1, "<s>", true, true)])
            .unwrap()
            .unwrap();

        assert_eq!(
            at.split_with("ab <s> cd", false),
            vec![Segment::Text("ab"), Segment::Token(1), Segment::Text("cd"),]
        );
        assert_eq!(
            at.split_with("ab <s> cd", true),
            vec![Segment::Text("ab <s> cd")]
        );
    }

    #[test]
    fn skip_special_consumes_the_matched_span() {
        // A candidate start byte *inside* a skipped token must not start a
        // second match: the full-scan automaton advances past the whole match,
        // and the prefiltered path has to agree.
        let mut outer = make_config(1, "<a<b>");
        outer.special = true;
        let inner = make_config(2, "<b>");
        let at = AddedTokens::from_configs(&[outer, inner]).unwrap().unwrap();

        assert_eq!(at.split_with("<a<b>", true), vec![Segment::Text("<a<b>")]);
    }

    #[test]
    fn skip_special_via_full_scan_path() {
        // >3 distinct start bytes forces the full-scan path.
        let mut special = make_config(1, "<bos>");
        special.special = true;
        let at = AddedTokens::from_configs(&[
            special,
            make_config(2, "[SEP]"),
            make_config(3, "{pad}"),
            make_config(4, "|mask|"),
        ])
        .unwrap()
        .unwrap();

        assert_eq!(
            at.split_with("<bos>[SEP]", true),
            vec![Segment::Text("<bos>"), Segment::Token(2),]
        );
    }

    #[test]
    fn configs_are_retained_for_rebuilding() {
        let configs = vec![
            make_strip_config(1, "<a>", true, false),
            make_config(2, "<b>"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(at.configs(), configs.as_slice());
    }

    #[test]
    fn strip_via_full_scan_path() {
        // >3 distinct start bytes forces the full-scan path; rstrip must still
        // absorb trailing whitespace there.
        let configs = vec![
            make_strip_config(1, "<bos>", false, true),
            make_config(2, "[SEP]"),
            make_config(3, "{pad}"),
            make_config(4, "|mask|"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(
            at.split("<bos>   x"),
            vec![Segment::Token(1), Segment::Text("x")]
        );
    }

    /// Tokens sharing a self-overlapping prefix: a match starting inside another
    /// occurrence of the prefix is still found.
    #[test]
    fn common_prefix_overlapping_occurrences() {
        let at = AddedTokens::from_configs(&[make_config(10, "<<a"), make_config(11, "<<b")])
            .unwrap()
            .unwrap();
        assert_eq!(
            at.split("x<<<a y<<<<b"),
            vec![
                Segment::Text("x<"),
                Segment::Token(10),
                Segment::Text(" y<<"),
                Segment::Token(11),
            ]
        );
    }
}
