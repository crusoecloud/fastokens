//! Hand-written pretokenizer scanner for the o200k / Kimi tiktoken pattern
//! family, fully replacing the regex engine on the hot path.
//!
//! The scanner reproduces the pattern's pretoken boundaries directly over the
//! UTF-8 text. ASCII (the common case for English, code, and structured text)
//! is classified with direct byte checks; non-ASCII codepoints are classified
//! with [`unicode_class`] tables built from the same `regex-syntax` data
//! `fancy-regex` uses, so results match the reference matcher (and tiktoken)
//! exactly — **there is no regex fallback**.
//!
//! The behaviour was derived from, and validated bit-for-bit against, the
//! reference regex / tiktoken over large multilingual corpora and an
//! adversarial suite.

use super::unicode_class::{Tables, tables};

/// Which recognized tiktoken pattern this scanner reproduces. The two differ in
/// whether Han runs are their own tokens (Kimi) and in the trailing class of a
/// punctuation run (o200k also absorbs `/`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanKind {
    O200k,
    Kimi,
}

/// Recognize a Split regex source as a scanner-supported pattern family.
pub(crate) fn recognize(source: &str) -> Option<ScanKind> {
    if source == crate::tiktoken::O200K_BASE_PATTERN {
        Some(ScanKind::O200k)
    } else if source == crate::tiktoken::KIMI_PATTERN {
        Some(ScanKind::Kimi)
    } else {
        None
    }
}

// ── Codepoint classifiers (ASCII fast path inline, tables for the rest) ───────
#[inline(always)]
fn is_letter(t: &Tables, c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        c.is_ascii_alphabetic()
    } else {
        t.is_letter(u)
    }
}
#[inline(always)]
fn is_number(t: &Tables, c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        c.is_ascii_digit()
    } else {
        t.is_number(u)
    }
}
#[inline(always)]
fn is_ws(t: &Tables, c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        matches!(u as u8, b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | b' ')
    } else {
        t.is_ws(u)
    }
}
#[inline(always)]
fn is_han(t: &Tables, c: char) -> bool {
    let u = c as u32;
    if u < 0x80 { false } else { t.is_han(u) }
}
/// Membership in the pattern's uppercase class `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]`
/// (Han excluded for Kimi, whose Han is consumed by the leading `[\p{Han}]+`).
#[inline(always)]
fn is_ugroup(t: &Tables, c: char, kimi: bool) -> bool {
    let u = c as u32;
    let m = if u < 0x80 {
        c.is_ascii_uppercase()
    } else {
        t.is_ugroup(u)
    };
    m && !(kimi && is_han(t, c))
}
/// Membership in the pattern's lowercase class `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`
/// (Han excluded for Kimi).
#[inline(always)]
fn is_lgroup(t: &Tables, c: char, kimi: bool) -> bool {
    let u = c as u32;
    let m = if u < 0x80 {
        c.is_ascii_lowercase()
    } else {
        t.is_lgroup(u)
    };
    m && !(kimi && is_han(t, c))
}
/// Membership in a letter run: in either case class (letter or mark, Han
/// excluded for Kimi).
#[inline(always)]
fn run_member(t: &Tables, c: char, kimi: bool) -> bool {
    is_ugroup(t, c, kimi) || is_lgroup(t, c, kimi)
}
/// The optional word-prefix class `[^\r\n\p{L}\p{N}]`.
#[inline(always)]
fn is_prefix(t: &Tables, c: char) -> bool {
    c != '\r' && c != '\n' && !is_letter(t, c) && !is_number(t, c)
}
/// The punctuation class `[^\s\p{L}\p{N}]`.
#[inline(always)]
fn is_punct(t: &Tables, c: char) -> bool {
    !is_ws(t, c) && !is_letter(t, c) && !is_number(t, c)
}

/// Length of a contraction suffix at `bs[0]` (which must be `'`), or 0.
/// Reproduces `(?i:'s|'t|'re|'ve|'m|'ll|'d)`.
#[inline]
fn contraction_len(bs: &[u8]) -> usize {
    if bs.len() < 2 || bs[1] >= 0x80 {
        return 0;
    }
    match bs[1] | 0x20 {
        b's' | b't' | b'm' | b'd' => 2,
        b'r' | b'v' if bs.len() >= 3 && (bs[2] | 0x20) == b'e' => 3,
        b'l' if bs.len() >= 3 && (bs[2] | 0x20) == b'l' => 3,
        _ => 0,
    }
}

/// Decode the codepoint at byte offset `i` (which must be a char boundary),
/// returning it and its UTF-8 length. ASCII is handled without decoding.
#[inline(always)]
fn char_at(text: &str, b: &[u8], i: usize) -> (char, usize) {
    let byte = b[i];
    if byte < 0x80 {
        (byte as char, 1)
    } else {
        let c = text[i..].chars().next().unwrap();
        (c, c.len_utf8())
    }
}

/// High bit of every byte in a `u64` lane.
const SWAR_HI: u64 = 0x8080_8080_8080_8080;

/// End (exclusive) of the maximal run of ASCII lowercase letters (`a`..=`z`)
/// starting at `pos`, scanning 8 bytes at a time. Stops at the first byte that
/// is not `a`..=`z` — including any non-ASCII byte (≥0x80), which the caller
/// treats as "the run may continue into a non-ASCII letter" and defers to the
/// scalar path. Purely `u64` arithmetic: identical and fast on every platform.
#[inline(always)]
fn ascii_lower_run_end(b: &[u8], mut pos: usize) -> usize {
    let n = b.len();
    while pos + 8 <= n {
        // SAFETY: pos + 8 <= n.
        let word = unsafe { (b.as_ptr().add(pos) as *const u64).read_unaligned() };
        if word & SWAR_HI != 0 {
            break; // non-ASCII byte present; resolve in the scalar tail
        }
        // High bit set in each lane that is NOT in 'a'..='z'.
        let ge_a = (word | SWAR_HI).wrapping_sub(0x6161_6161_6161_6161);
        let le_z = 0xFAFA_FAFA_FAFA_FAFA_u64.wrapping_sub(word);
        let non_lower = !(ge_a & le_z) & SWAR_HI;
        if non_lower != 0 {
            return pos + non_lower.to_le().trailing_zeros() as usize / 8;
        }
        pos += 8;
    }
    while pos < n {
        let x = unsafe { *b.get_unchecked(pos) };
        if x.wrapping_sub(b'a') < 26 {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// Bytes a punctuation pretoken may trail with after its `[^\s\p{L}\p{N}]+` core
/// — the pattern's `[\r\n/]*` (o200k) or `[\r\n]*` (Kimi) tail. These are exactly
/// the bytes that can follow a newline while staying inside one pretoken, so a
/// chunk split must not land inside such a run. Kept in sync with the trailing
/// loop in [`scan_core`].
const fn punct_trailing_bytes(kind: ScanKind) -> &'static [u8] {
    match kind {
        ScanKind::O200k => b"\r\n/",
        ScanKind::Kimi => b"\r\n",
    }
}

/// Split `text` into up to `n_chunks` `(start, end)` byte segments, each split
/// placed at a pretoken boundary so segments can be scanned/tokenized
/// independently.
///
/// The split goes right after the **last** `\r`/`\n` of the maximal whitespace
/// run around the nominal split point — the point where the pretoken containing
/// that newline ends. `\s*[\r\n]+` matches a whitespace run up to and including
/// its last newline, so the run may hold *interior* whitespace between newlines
/// (`\n  \n` is a single pretoken: `\s*` eats `\n  `, `[\r\n]+` the last `\n`);
/// a preceding `[^\s\p{L}\p{N}]+[\r\n]*` likewise ends at a newline. Splitting
/// merely after the *first* newline run — as this once did — can fall inside
/// such a pretoken and split it across two chunks, changing the tokenization.
///
/// There is a further subtlety for any kind whose punctuation pretoken trails
/// with a class beyond `[\r\n]` — o200k's `[\r\n/]*` (Kimi's is `[\r\n]*`): a
/// byte from that class right after the newline (o200k's `/`, or an alternating
/// run like `.\n/\n`) is still part of that *same* pretoken, so the newline sits
/// in its middle. Splitting after the last newline would cut it, so the boundary
/// is advanced past the whole trailing run — [`punct_trailing_bytes`] — to the
/// pretoken's true end. That run is the only place any alternative carries a
/// non-newline byte after a newline, so this is both necessary and sufficient
/// for every kind. (Where the run instead spans a `\s*[\r\n]+` token followed by
/// a separate punctuation token, advancing merely moves that complete token into
/// the earlier chunk — same tokenization either way. For Kimi the run is empty,
/// so the advance is a no-op.)
pub(crate) fn newline_chunk_bounds(
    text: &str,
    n_chunks: usize,
    kind: ScanKind,
) -> Vec<(usize, usize)> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    if n_chunks < 2 {
        return vec![(0, n)];
    }
    let t = tables();
    let nominal = n / n_chunks;
    let mut splits = vec![0usize];
    for i in 1..n_chunks {
        let from = i * nominal;
        let Some(rel) = memchr::memchr2(b'\n', b'\r', &bytes[from..]) else {
            break;
        };
        // Walk the maximal whitespace run (ASCII + Unicode) containing this
        // newline, tracking the last `\r`/`\n`; the pretoken ends right after it.
        let mut e = from + rel;
        let mut last_nl = e;
        while e < n {
            let (c, l) = char_at(text, bytes, e);
            if !is_ws(t, c) {
                break;
            }
            if bytes[e] == b'\n' || bytes[e] == b'\r' {
                last_nl = e;
            }
            e += l;
        }
        let mut boundary = last_nl + 1;
        // Advance past any punctuation-trailing run continuing the pretoken past
        // this newline (see the doc comment). A no-op for kinds whose trailing
        // class is only `[\r\n]` (e.g. Kimi).
        let trailing = punct_trailing_bytes(kind);
        while boundary < n && trailing.contains(&bytes[boundary]) {
            boundary += 1;
        }
        if boundary < n && boundary > *splits.last().unwrap() {
            splits.push(boundary);
        }
    }
    splits.push(n);
    splits.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Core single-pass scan: call `emit(start, end)` for each covering pretoken
/// byte range, in order. [`scan_seq`] collects these into a `Vec`; the fused
/// encode path BPEs each pretoken inline (no range list is materialized). Any
/// error returned by `emit` (e.g. an un-encodable byte during BPE) propagates.
pub(crate) fn scan_core<F>(kind: ScanKind, text: &str, mut emit: F) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    let t = tables();
    let b = text.as_bytes();
    let n = b.len();
    let kimi = kind == ScanKind::Kimi;
    let slash = kind == ScanKind::O200k;
    let mut i = 0usize;

    while i < n {
        let start = i;

        // ── Fast path: a plain ASCII word `[ ]?[A-Z]*[a-z]*` (≥1 letter) with
        // no contraction and no non-ASCII continuation — the dominant token in
        // natural-language text. Batch-emitting it with SWAR run scans skips the
        // general word machinery below. It fires only when the result is
        // provably identical to that machinery:
        //  - the letters start at `i` (no prefix) or after a single space prefix
        //    at `i` (a valid `[^\r\n\p{L}\p{N}]?`) followed by a letter;
        //  - the run is uppercase-run then lowercase-run — i.e. the `U+ L*` /
        //    `U* L+` form that is exactly one token (ASCII has no caseless
        //    letters, so no backtrack); a further uppercase after the lowercase
        //    run (`camelCase`) would split, so that defers to the scalar path;
        //  - it does not end on `'` (a possible contraction) or a non-ASCII byte
        //    (a possible non-ASCII letter/mark that the run would absorb).
        {
            let c0 = b[i];
            let lstart = if c0.wrapping_sub(b'a') < 26 {
                i
            } else if c0 == b' ' && i + 1 < n && b[i + 1].wrapping_sub(b'a') < 26 {
                i + 1
            } else {
                usize::MAX
            };
            if lstart != usize::MAX {
                let run_end = ascii_lower_run_end(b, lstart);
                if run_end == n || (b[run_end] < 0x80 && b[run_end] != b'\'') {
                    emit(start, run_end)?;
                    i = run_end;
                    continue;
                }
            }
        }

        let (c, clen) = char_at(text, b, i);

        // ── alt0: Han run (Kimi only) ──
        if kimi && is_han(t, c) {
            let mut e = i + clen;
            while e < n {
                let (cj, lj) = char_at(text, b, e);
                if is_han(t, cj) {
                    e += lj;
                } else {
                    break;
                }
            }
            emit(start, e)?;
            i = e;
            continue;
        }

        // ── Word: optional prefix + letter run + contraction ──
        // The letter run reproduces `U* L+` (alt1, tried first) else `U+ L*`
        // (alt2), where U = `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]` and
        // L = `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`. These classes OVERLAP (Lm/Lo/M are in
        // both), so the greedy `U*` gives back trailing overlap chars to satisfy
        // `L+`; we reproduce that with a single backtrack point.
        let run_start = if run_member(t, c, kimi) {
            Some(i)
        } else if is_prefix(t, c) && i + clen < n {
            let (c1, _) = char_at(text, b, i + clen);
            if run_member(t, c1, kimi) {
                Some(i + clen)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(run_start) = run_start {
            // Maximal U-group run from `run_start`, remembering the last position
            // within it that is also L-group (a candidate `L+` start).
            let mut u = run_start;
            let mut last_lg = usize::MAX;
            while u < n {
                let (cj, lj) = char_at(text, b, u);
                if !is_ugroup(t, cj, kimi) {
                    break;
                }
                if is_lgroup(t, cj, kimi) {
                    last_lg = u;
                }
                u += lj;
            }
            // Largest `L+` start ≤ u: prefer the char right after the U-run (a
            // pure `Ll`), else the last overlap char inside the U-run.
            let after_u_is_lgroup = u < n && is_lgroup(t, char_at(text, b, u).0, kimi);
            let lstart = if after_u_is_lgroup { u } else { last_lg };
            let mut e = if lstart != usize::MAX {
                // alt1: `L+` = maximal L-group run from `lstart`.
                let mut e = lstart;
                while e < n {
                    let (cj, lj) = char_at(text, b, e);
                    if !is_lgroup(t, cj, kimi) {
                        break;
                    }
                    e += lj;
                }
                e
            } else {
                // alt2: `U+ L*` with an empty `L*` (the U-run has no L-group char).
                u
            };
            if e < n && b[e] == b'\'' {
                e += contraction_len(&b[e..]);
            }
            emit(start, e)?;
            i = e;
            continue;
        }

        // ── Number: \p{N}{1,3} (no prefix) ──
        if is_number(t, c) {
            let mut cnt = 1usize;
            let mut e = i + clen;
            while cnt < 3 && e < n {
                let (cj, lj) = char_at(text, b, e);
                if is_number(t, cj) {
                    cnt += 1;
                    e += lj;
                } else {
                    break;
                }
            }
            emit(start, e)?;
            i = e;
            continue;
        }

        // ── Punctuation: ` ?[^\s\p{L}\p{N}]+[\r\n(/)]* ──
        {
            let (mut pstart, mut pc, mut pcl) = (i, c, clen);
            if c == ' ' && i + clen < n {
                let (c1, l1) = char_at(text, b, i + clen);
                if is_punct(t, c1) {
                    pstart = i + clen;
                    pc = c1;
                    pcl = l1;
                }
            }
            if is_punct(t, pc) {
                let mut e = pstart + pcl;
                while e < n {
                    let (cj, lj) = char_at(text, b, e);
                    if is_punct(t, cj) {
                        e += lj;
                    } else {
                        break;
                    }
                }
                // Trailing class `[\r\n/]*` (o200k) / `[\r\n]*` (Kimi). This byte
                // set is the canonical definition mirrored by
                // [`punct_trailing_bytes`], which `newline_chunk_bounds` uses to
                // avoid splitting a chunk inside this run.
                while e < n && (b[e] == b'\r' || b[e] == b'\n' || (slash && b[e] == b'/')) {
                    e += 1;
                }
                emit(start, e)?;
                i = e;
                continue;
            }
        }

        // ── Whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+ ──
        let mut e = i;
        let mut last_cp_start = i;
        while e < n {
            let (cj, lj) = char_at(text, b, e);
            if is_ws(t, cj) {
                last_cp_start = e;
                e += lj;
            } else {
                break;
            }
        }
        let we = e;
        // Last `\r`/`\n` (both ASCII) within the whitespace run.
        let mut last_nl = usize::MAX;
        let mut k = i;
        while k < we {
            if b[k] == b'\r' || b[k] == b'\n' {
                last_nl = k;
            }
            k += 1;
        }
        let end = if last_nl != usize::MAX {
            last_nl + 1 // \s*[\r\n]+ up to and including the last newline
        } else if we == n {
            we // trailing whitespace: \s+(?!\S) at EOF
        } else if last_cp_start > i {
            last_cp_start // \s+(?!\S): leave the last whitespace codepoint for the next token
        } else {
            we // single whitespace codepoint: \s+
        };
        emit(i, end)?;
        i = end;
    }

    Ok(())
}

/// Scan a segment into covering pretoken byte-ranges (collects [`scan_core`]).
/// Test helper; the encode path drives [`scan_core`] directly (no range list).
#[cfg(test)]
fn scan_seq(kind: ScanKind, text: &str) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(text.len() / 4 + 1);
    // `scan_core`'s emit is infallible here (we only collect ranges).
    let _ = scan_core(kind, text, |s, e| {
        out.push((s as u32, e as u32));
        Ok(())
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiktoken::KIMI_PATTERN;

    fn scan(kind: ScanKind, text: &str) -> Vec<String> {
        scan_seq(kind, text)
            .iter()
            .map(|&(s, e)| text[s as usize..e as usize].to_string())
            .collect()
    }

    /// Scanning a buffer split into [`newline_chunk_bounds`] segments must yield
    /// exactly the same pretokens as scanning it whole — even when a nominal
    /// split lands inside a `\s*[\r\n]+` pretoken that holds interior whitespace
    /// (`" \n  \n"` is one pretoken). Regression for a chunk boundary placed
    /// after the *first* newline run splitting such a pretoken across chunks.
    #[test]
    fn chunk_bounds_preserve_interior_newline_pretokens() {
        // Sized so the 2-way nominal split (byte 100) is the run's start.
        let text = format!("{} \n  \n{}", "a".repeat(100), "b".repeat(95));
        let whole = scan(ScanKind::Kimi, &text);
        assert!(
            whole.iter().any(|p| p == " \n  \n"),
            "run should be one pretoken: {whole:?}"
        );

        let bounds = newline_chunk_bounds(&text, 2, ScanKind::Kimi);
        assert!(bounds.len() >= 2, "expected a split: {bounds:?}");
        let chunked: Vec<String> = bounds
            .iter()
            .flat_map(|&(s, e)| scan(ScanKind::Kimi, &text[s..e]))
            .collect();
        assert_eq!(chunked, whole, "chunked scan diverged from whole scan");
    }

    /// o200k's punctuation pretoken trails with `[\r\n/]*`, so `.\n/` is a single
    /// pretoken with the newline in its *middle*. A chunk split placed right
    /// after that newline would cut the pretoken across chunks; the boundary must
    /// skip the trailing `[\r\n/]` run. (Kimi's trailing class is `[\r\n]*`, so it
    /// tokenizes `.\n/` as `.\n` + `/` and its boundary there is already correct.)
    /// Regression for issue #67: o200k multithreaded segmentation diverging from
    /// the single-threaded / HF result.
    #[test]
    fn o200k_chunk_bounds_preserve_slash_after_newline() {
        // Sized so the 2-way nominal split lands on the newline inside `.\n/`.
        let text = format!("{}.\n/{}", "a".repeat(100), "b".repeat(100));
        let whole = scan(ScanKind::O200k, &text);
        assert!(
            whole.iter().any(|p| p == ".\n/"),
            "`.\\n/` should be one o200k pretoken: {whole:?}"
        );

        let bounds = newline_chunk_bounds(&text, 2, ScanKind::O200k);
        assert!(bounds.len() >= 2, "expected a split: {bounds:?}");
        let chunked: Vec<String> = bounds
            .iter()
            .flat_map(|&(s, e)| scan(ScanKind::O200k, &text[s..e]))
            .collect();
        assert_eq!(
            chunked, whole,
            "chunked o200k scan diverged from whole scan"
        );
    }

    #[test]
    fn words_case_split() {
        assert_eq!(scan(ScanKind::O200k, "HTTPRequest"), vec!["HTTPRequest"]);
        assert_eq!(scan(ScanKind::O200k, "HelloWorld"), vec!["Hello", "World"]);
        assert_eq!(scan(ScanKind::O200k, "camelCase"), vec!["camel", "Case"]);
        assert_eq!(scan(ScanKind::O200k, "iOS"), vec!["i", "OS"]);
        assert_eq!(scan(ScanKind::O200k, "ALLCAPS"), vec!["ALLCAPS"]);
        assert_eq!(scan(ScanKind::O200k, "aBc"), vec!["a", "Bc"]);
    }

    #[test]
    fn contractions_and_prefix() {
        assert_eq!(scan(ScanKind::O200k, "don't"), vec!["don't"]);
        assert_eq!(scan(ScanKind::O200k, "I'll"), vec!["I'll"]);
        assert_eq!(scan(ScanKind::O200k, "O'Brien"), vec!["O", "'Brien"]);
        assert_eq!(
            scan(ScanKind::O200k, "hello world"),
            vec!["hello", " world"]
        );
        assert_eq!(scan(ScanKind::O200k, "a!b"), vec!["a", "!b"]);
    }

    #[test]
    fn numbers_punct_whitespace() {
        assert_eq!(scan(ScanKind::O200k, "1234567"), vec!["123", "456", "7"]);
        assert_eq!(scan(ScanKind::O200k, "3.14"), vec!["3", ".", "14"]);
        assert_eq!(scan(ScanKind::O200k, "!!!"), vec!["!!!"]);
        assert_eq!(scan(ScanKind::O200k, "a  b"), vec!["a", " ", " b"]);
        assert_eq!(scan(ScanKind::O200k, "trailing  "), vec!["trailing", "  "]);
        assert_eq!(
            scan(ScanKind::O200k, "foo\n\nbar"),
            vec!["foo", "\n\n", "bar"]
        );
        assert_eq!(scan(ScanKind::O200k, "  \n  x"), vec!["  \n", " ", " x"]);
    }

    #[test]
    fn unicode_and_han() {
        // Non-Han letters extend runs; accented word stays whole.
        assert_eq!(scan(ScanKind::Kimi, "café"), vec!["café"]);
        // Kimi: Han is its own run, split from surrounding scripts.
        assert_eq!(scan(ScanKind::Kimi, "你好world"), vec!["你好", "world"]);
        assert_eq!(
            scan(ScanKind::Kimi, "café中文test"),
            vec!["café", "中文", "test"]
        );
        assert_eq!(scan(ScanKind::Kimi, "中1文"), vec!["中", "1", "文"]);
        // Unicode whitespace (ideographic space) handled natively.
        assert_eq!(scan(ScanKind::Kimi, "a\u{3000}b"), vec!["a", "\u{3000}b"]);
        // o200k treats Han as ordinary letters (no Han alternative).
        assert_eq!(scan(ScanKind::O200k, "你好"), vec!["你好"]);
    }

    /// Scanning newline-delimited chunks independently (as the fused encode
    /// path does) must reproduce a whole-buffer scan exactly, for any chunk
    /// count. Guards the "newline runs are always pretoken boundaries" invariant.
    #[test]
    fn newline_chunking_matches_whole() {
        // Includes `" \n  \n"` and `" \n \t\n"`: `\s*[\r\n]+` pretokens with
        // interior whitespace between newlines, where a split after the first
        // newline run would fall inside the pretoken. Also includes `end.\n/usr`
        // and `x!\n/\n/y`: o200k punctuation pretokens whose `[\r\n/]*` trailing
        // run carries a newline in its middle, where a split after that newline
        // would fall inside the pretoken.
        let unit = "Hello world!\nCamelCase 中文 test\n\n  spaced  lines \n  \n\
                    café résumé 12345 don't \n \t\n更多文本\r\n\
                    end.\n/usr/bin\nx!\n/\n/y\n";
        let big = unit.repeat(400);
        for kind in [ScanKind::O200k, ScanKind::Kimi] {
            let whole = scan_seq(kind, &big);
            for n_chunks in [1usize, 2, 3, 7, 16, 64] {
                let mut combined = Vec::new();
                for (s, e) in newline_chunk_bounds(&big, n_chunks, kind) {
                    let base = s as u32;
                    for (a, b) in scan_seq(kind, &big[s..e]) {
                        combined.push((a + base, b + base));
                    }
                }
                assert_eq!(combined, whole, "kind={kind:?} n_chunks={n_chunks}");
            }
        }
    }

    #[test]
    fn recognizes_patterns() {
        assert_eq!(
            recognize(crate::tiktoken::O200K_BASE_PATTERN),
            Some(ScanKind::O200k)
        );
        assert_eq!(recognize(KIMI_PATTERN), Some(ScanKind::Kimi));
        assert_eq!(recognize("something else"), None);
    }

    /// The scanner must produce exactly the same covering pretoken ranges as the
    /// crate's own regex engine (PCRE2 for o200k, fancy-regex for Kimi's `&&`
    /// pattern), which is itself validated against tiktoken. Network-free.
    #[test]
    fn scanner_matches_regex_engine() {
        use crate::Split;
        use crate::pre_tokenized::PreTokenizedString;
        use serde_json::json;

        let corpus = [
            "",
            "Hello, world! HTTPRequest HelloWorld camelCase ALLCAPS iOS getHTTPResponse",
            "don't I'll O'Brien y'all wasn't 'tis can't won't",
            "1234567 3.14 1,000 42 007 mixed ABC123def a1b2c3",
            "  leading trailing  a  b  a   b\ttabs\there",
            "foo\n\nbar  \n  x\r\n\r\nwin end.  x.\n\n",
            "!!! ... @#$ a!b ( hello .. word e.g. U.S.A. snake_case kebab-case -42 ' -42",
            "café résumé naïve über straße señor niño",
            "你好世界 中文，世界 混合ABCと日本語 test中文HELLO café中文test 中1文 a中b",
            "こんにちは 안녕하세요 Привет мир Ελληνικά עברית العربية",
            "數據科學 fêteliefer.githubusercontent શહેરefeller эндey feature",
            "עצמ亚洲AVurant בדרך无码AVJobҮ下面tellremaining მჯდომ",
            "😀🚀 中文 test 한국어 test 中文 日本語 ①②③ Ｆｕｌｌ",
            "a\u{3000}b \u{00a0}nbsp \u{2028}line é\u{0301}\u{0302} ǅ titlecase",
            "The rain in Spain. ".to_string().repeat(50).leak(),
        ];

        for (kind, pat) in [
            (ScanKind::O200k, crate::tiktoken::O200K_BASE_PATTERN),
            (ScanKind::Kimi, KIMI_PATTERN),
        ] {
            let split = Split::from_config(&json!({ "Regex": pat }), "Isolated", false).unwrap();
            for s in &corpus {
                let mut pts = PreTokenizedString::from_text(s);
                split.pre_tokenize(&mut pts).unwrap();
                let regex_ranges: Vec<(u32, u32)> = pts
                    .splits()
                    .iter()
                    .filter(|sp| !sp.range.is_empty())
                    .map(|sp| (sp.range.start as u32, sp.range.end as u32))
                    .collect();

                let scan: Vec<(u32, u32)> = scan_seq(kind, s)
                    .into_iter()
                    .filter(|(a, b)| a != b)
                    .collect();

                assert_eq!(scan, regex_ranges, "kind={kind:?} input={s:?}");
            }
        }
    }
}
