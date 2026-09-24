//! Hand-written pretokenizer scanner for the cl100k / o200k / Kimi tiktoken
//! pattern family, fully replacing the regex engine on the hot path.
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

/// Which recognized tiktoken pattern this scanner reproduces. They differ in the
/// word run (cl100k is a plain `\p{L}+` with the contraction as its own leading
/// alternative; o200k / Kimi split runs by letter case and attach the
/// contraction as a word suffix), in whether Han runs are their own tokens
/// (Kimi), and in the trailing class of a punctuation run (o200k also absorbs
/// `/`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanKind {
    Cl100k,
    /// Qwen2/Qwen3: cl100k with single-digit numbers (`\p{N}` for `\p{N}{1,3}`).
    Qwen,
    /// Qwen3.5/3.8: [`ScanKind::Qwen`] with `\p{M}` counted as a letter (in the
    /// word run, out of the punctuation class).
    Qwen35,
    O200k,
    /// Mistral tekken: o200k with no contraction suffix and single-digit numbers.
    Tekken,
    Kimi,
    /// GPT-2's ByteLevel regex (`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
    /// ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`): case-sensitive contractions, and a
    /// literal space as the only prefix. See [`scan_core_gpt2`].
    Gpt2,
    /// DeepSeek's three-`Split` sequence (digits, then CJK runs, then a GPT-like
    /// pattern), reproduced in one pass. See [`recognize_deepseek`].
    DeepSeek,
}

/// Recognize a single Split regex source as a scanner-supported pattern family.
/// (DeepSeek is a *sequence* of Splits, recognized by [`recognize_deepseek`].)
pub(crate) fn recognize(source: &str) -> Option<ScanKind> {
    if source == crate::tiktoken::CL100K_BASE_PATTERN {
        Some(ScanKind::Cl100k)
    } else if source == crate::tiktoken::QWEN2_PATTERN {
        Some(ScanKind::Qwen)
    } else if source == crate::tiktoken::QWEN35_PATTERN {
        Some(ScanKind::Qwen35)
    } else if source == crate::tiktoken::O200K_BASE_PATTERN {
        Some(ScanKind::O200k)
    } else if source == crate::tiktoken::TEKKEN_PATTERN {
        Some(ScanKind::Tekken)
    } else if source == crate::tiktoken::KIMI_PATTERN {
        Some(ScanKind::Kimi)
    } else {
        None
    }
}

/// Recognize DeepSeek's three consecutive `Isolated` Split patterns (in order:
/// digits, CJK runs, the GPT-like pattern). The caller checks the surrounding
/// shape (`Sequence([Split, Split, Split, ByteLevel(bulk)])`, all `Isolated`,
/// none inverted); this checks the three regex sources are byte-identical to the
/// ones DeepSeek ships, so any drift falls back to the regex engine.
pub(crate) fn recognize_deepseek(p1: &str, p2: &str, p3: &str) -> bool {
    p1 == crate::tiktoken::DEEPSEEK_SPLIT1_PATTERN
        && p2 == crate::tiktoken::DEEPSEEK_SPLIT2_PATTERN
        && p3 == crate::tiktoken::DEEPSEEK_SPLIT3_PATTERN
}

/// Membership in DeepSeek's CJK class `[一-龥぀-ゟ゠-ヿ]`
/// (CJK Unified Ideographs + Hiragana + Katakana). All non-ASCII, so it never
/// affects the ASCII fast paths.
#[inline(always)]
pub(crate) fn is_ds_cjk(c: char) -> bool {
    let u = c as u32;
    (0x4E00..=0x9FA5).contains(&u)
        || (0x3040..=0x309F).contains(&u)
        || (0x30A0..=0x30FF).contains(&u)
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
/// `\p{M}` (a mark: in both case classes but not a letter). Non-ASCII only.
#[inline(always)]
fn is_mark(t: &Tables, c: char) -> bool {
    let u = c as u32;
    u >= 0x80 && t.is_ugroup(u) && t.is_lgroup(u) && !t.is_letter(u)
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

/// Like [`ascii_lower_run_end`] but for ASCII letters of *either* case
/// (`A`..=`Z` | `a`..=`z`) — the run body of cl100k's `\p{L}+`, which (unlike
/// o200k) does not split on case. Folds each lane to lowercase (`| 0x20`) before
/// the `a`..=`z` range test; the fold is only applied after confirming the whole
/// word is ASCII, so it cannot corrupt a UTF-8 lead/continuation byte. Stops at
/// the first non-letter or any non-ASCII byte (deferred to the scalar path,
/// which may extend the run into a non-ASCII `\p{L}`).
#[inline(always)]
fn ascii_alpha_run_end(b: &[u8], mut pos: usize) -> usize {
    const SWAR_CASE: u64 = 0x2020_2020_2020_2020;
    let n = b.len();
    while pos + 8 <= n {
        // SAFETY: pos + 8 <= n.
        let word = unsafe { (b.as_ptr().add(pos) as *const u64).read_unaligned() };
        if word & SWAR_HI != 0 {
            break; // non-ASCII byte present; resolve in the scalar tail
        }
        let lower = word | SWAR_CASE;
        // High bit set in each lane whose folded byte is NOT in 'a'..='z'.
        let ge_a = (lower | SWAR_HI).wrapping_sub(0x6161_6161_6161_6161);
        let le_z = 0xFAFA_FAFA_FAFA_FAFA_u64.wrapping_sub(lower);
        let non_alpha = !(ge_a & le_z) & SWAR_HI;
        if non_alpha != 0 {
            return pos + non_alpha.to_le().trailing_zeros() as usize / 8;
        }
        pos += 8;
    }
    while pos < n {
        let x = unsafe { *b.get_unchecked(pos) };
        if (x | 0x20).wrapping_sub(b'a') < 26 {
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
        ScanKind::O200k | ScanKind::Tekken => b"\r\n/",
        ScanKind::Cl100k
        | ScanKind::Qwen
        | ScanKind::Qwen35
        | ScanKind::Kimi
        | ScanKind::DeepSeek => b"\r\n",
        // No punct tail; a newline only ends a whitespace pretoken.
        ScanKind::Gpt2 => b"",
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
/// byte from that class right after the newline (o200k's `/`) may belong to
/// either side. After a punctuation run (`end.\n/usr`, or alternating
/// `x!\n/\n/y`) the newline sits inside that run's pretoken; after anything else
/// (`b\n/b's`) the newlines are a `\s*[\r\n]+` pretoken and the `/` opens the
/// next one. [`punct_trailing_bytes`] is the only place any alternative carries a
/// non-newline byte after a newline, so a newline followed by one of those bytes
/// is simply not used as a cut point, and any other newline is.
///
/// A newline is used only within half a chunk of the nominal split point;
/// failing one, a [`space_boundary`] there, and failing that the next usable
/// newline however far — so text with few newlines still splits evenly.
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
    // The pretoken boundary a newline at `q` implies, if it is unambiguous.
    let newline_boundary = |q: usize| -> Option<usize> {
        if kind == ScanKind::Gpt2 {
            // GPT-2 has no `\s*[\r\n]+`: a whitespace run ends its token only
            // before its last char (`\s+(?!\S)`), but the run's *start* always
            // opens one (every non-whitespace token stops at whitespace, and the
            // grammar has no look-behind), so cut there.
            let mut q = q;
            while q > 0 {
                let Some(c) = text[..q].chars().next_back() else {
                    break;
                };
                if !is_ws(t, c) {
                    break;
                }
                q -= c.len_utf8();
            }
            return Some(q);
        }
        // Walk the maximal whitespace run (ASCII + Unicode) containing this
        // newline, tracking the last `\r`/`\n`; the pretoken ends right after it.
        let mut e = q;
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
        // A trailing-class byte right after the run (o200k's `/`) makes this
        // ambiguous: it continues the pretoken when the newlines are a
        // punctuation run's `[\r\n/]*` tail, but opens the next pretoken when
        // they are a `\s*[\r\n]+` of their own (`b\n/b's`). Skip such a newline.
        // Never the case for kinds whose trailing class is only `[\r\n]`.
        let boundary = last_nl + 1;
        (boundary >= n || !punct_trailing_bytes(kind).contains(&bytes[boundary]))
            .then_some(boundary)
    };
    // The first unambiguous newline boundary from `from` on (and before `to`).
    let newline_from = |from: usize, to: usize| -> Option<usize> {
        memchr::memchr2_iter(b'\n', b'\r', &bytes[from..to])
            .find_map(|r| newline_boundary(from + r))
    };
    let nominal = n / n_chunks;
    let mut splits = vec![0usize];
    for i in 1..n_chunks {
        let from = i * nominal;
        // Prefer a newline near the nominal point; text with few newlines (a
        // paragraph-long line, minified data) falls back to a space boundary
        // there, and only then to the next newline however far away.
        let window = (from + nominal / 2).min(n);
        let Some(boundary) = newline_from(from, window)
            .or_else(|| space_boundary(bytes, from, window))
            .or_else(|| newline_from(window, n))
        else {
            break;
        };
        if boundary > 0 && boundary < n && boundary > *splits.last().unwrap() {
            splits.push(boundary);
        }
    }
    splits.push(n);
    splits.windows(2).map(|w| (w[0], w[1])).collect()
}

/// The first `p` in `from..to` where an ASCII space follows a printable ASCII
/// non-space byte — a pretoken boundary in every supported grammar, after which
/// the scan depends on the rest of the text alone, so chunks may be cut there.
///
/// The pretoken holding the byte before `p` ends at `p`: letter, digit and
/// punctuation runs all stop at `\s`, no contraction or `[\r\n/]*` tail holds a
/// space, and a space opens a token only as its first char. The grammars have
/// no look-behind, so the pretoken starting at `p` is the same whether or not
/// the text before `p` is present. Requiring an ASCII non-space byte (not just
/// any non-whitespace char) keeps Unicode whitespace such as U+3000 from joining
/// the space into one whitespace run across the cut.
fn space_boundary(bytes: &[u8], from: usize, to: usize) -> Option<usize> {
    let from = from.max(1);
    if from >= to {
        return None;
    }
    memchr::memchr_iter(b' ', &bytes[from..to])
        .map(|r| from + r)
        .find(|&p| (0x21..=0x7E).contains(&bytes[p - 1]))
}

/// Core single-pass scan: call `emit(start, end)` for each covering pretoken
/// byte range, in order. [`scan_seq`] collects these into a `Vec`; the fused
/// encode path BPEs each pretoken inline (no range list is materialized). Any
/// error returned by `emit` (e.g. an un-encodable byte during BPE) propagates.
pub(crate) fn scan_core<F>(kind: ScanKind, text: &str, mut emit: F) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    // DeepSeek's grammar (three sequenced splits) differs enough from the single
    // tiktoken regex that it gets its own pass rather than branching this one.
    if kind == ScanKind::DeepSeek {
        return scan_core_deepseek(text, &mut emit);
    }
    if kind == ScanKind::Gpt2 {
        return scan_core_gpt2(text, &mut emit);
    }
    let t = tables();
    let b = text.as_bytes();
    let n = b.len();
    let kimi = kind == ScanKind::Kimi;
    // Qwen is cl100k but for the digit cap (`\p{N}` vs `\p{N}{1,3}`).
    let cl = matches!(kind, ScanKind::Cl100k | ScanKind::Qwen | ScanKind::Qwen35);
    let digit_cap = if matches!(kind, ScanKind::Qwen | ScanKind::Qwen35 | ScanKind::Tekken) {
        1
    } else {
        3
    };
    // Qwen3.5: a mark is a letter — in the word run, out of the punct class. (It is
    // also prefix-eligible, but a prefix mark and a run-initial mark open the same
    // token, so treating it as a letter throughout is exact.)
    let marks = kind == ScanKind::Qwen35;
    let letter = |c: char| is_letter(t, c) || (marks && is_mark(t, c));
    let punct = |c: char| is_punct(t, c) && !(marks && is_mark(t, c));
    let slash = matches!(kind, ScanKind::O200k | ScanKind::Tekken);
    // Tekken's letter tokens take no contraction suffix.
    let contraction_suffix = kind != ScanKind::Tekken;
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
        if cl {
            // cl100k fast path: ` ?[A-Za-z]+` — the word `[^\r\n\p{L}\p{N}]?\p{L}+`
            // specialized to a space prefix and an all-ASCII, case-insensitive
            // letter run (cl100k does not split on case). A trailing `'` does not
            // attach here (the contraction is a separate leading alternative), so
            // unlike the o200k path there is no `'` guard. A non-space prefix
            // (e.g. `(word`) or a non-ASCII continuation defers to the scalar path.
            let c0 = b[i];
            let is_alpha = |x: u8| (x | 0x20).wrapping_sub(b'a') < 26;
            let lstart = if is_alpha(c0) {
                i
            } else if c0 == b' ' && i + 1 < n && is_alpha(b[i + 1]) {
                i + 1
            } else {
                usize::MAX
            };
            if lstart != usize::MAX {
                let run_end = ascii_alpha_run_end(b, lstart);
                if run_end == n || b[run_end] < 0x80 {
                    emit(start, run_end)?;
                    i = run_end;
                    continue;
                }
            }
        } else {
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

        // ── cl100k word/contraction (distinct from the o200k/Kimi word below) ──
        // cl100k's letter alternatives are `(?i:'s|'t|'re|'ve|'m|'ll|'d)` (a
        // *leading* alternative, tried first) and `[^\r\n\p{L}\p{N}]?\p{L}+` (a
        // plain letter run, no case split, no attached contraction). Everything
        // else — numbers, punctuation, whitespace — is shared with the branches
        // further down.
        if cl {
            // Leading contraction alternative (highest priority at a `'`).
            if c == '\'' {
                let clen2 = contraction_len(&b[i..]);
                if clen2 > 0 {
                    emit(start, i + clen2)?;
                    i += clen2;
                    continue;
                }
            }
            // Word: optional single prefix `[^\r\n\p{L}\p{N}]` then `\p{L}+`.
            let run_start = if letter(c) {
                Some(i)
            } else if is_prefix(t, c) && i + clen < n && letter(char_at(text, b, i + clen).0) {
                Some(i + clen)
            } else {
                None
            };
            if let Some(run_start) = run_start {
                let mut e = run_start;
                while e < n {
                    let (cj, lj) = char_at(text, b, e);
                    if !letter(cj) {
                        break;
                    }
                    e += lj;
                }
                emit(start, e)?;
                i = e;
                continue;
            }
            // Not a contraction or word: fall through to number/punct/whitespace.
        }

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
        // Skipped for cl100k, whose word/contraction was handled above; its plain
        // `\p{L}+` run must not fall into this case-aware `U*L+`/`U+L*` machinery
        // (which also absorbs `\p{M}` marks that cl100k leaves to punctuation).
        let run_start = if !cl && run_member(t, c, kimi) {
            Some(i)
        } else if !cl && is_prefix(t, c) && i + clen < n {
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
            if contraction_suffix && e < n && b[e] == b'\'' {
                e += contraction_len(&b[e..]);
            }
            emit(start, e)?;
            i = e;
            continue;
        }

        // ── Number: \p{N}{1,3} (Qwen: \p{N}) (no prefix) ──
        if is_number(t, c) {
            let mut cnt = 1usize;
            let mut e = i + clen;
            while cnt < digit_cap && e < n {
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
                if punct(c1) {
                    pstart = i + clen;
                    pc = c1;
                    pcl = l1;
                }
            }
            if punct(pc) {
                let mut e = pstart + pcl;
                while e < n {
                    let (cj, lj) = char_at(text, b, e);
                    if punct(cj) {
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

/// Length of GPT-2's case-sensitive contraction `'s|'t|'re|'ve|'m|'ll|'d` at
/// `bs[0]` (which must be `'`), or 0.
#[inline]
pub(crate) fn gpt2_contraction_len(bs: &[u8]) -> usize {
    match bs {
        [b'\'', b's' | b't' | b'm' | b'd', ..] => 2,
        [b'\'', b'r' | b'v', b'e', ..] | [b'\'', b'l', b'l', ..] => 3,
        _ => 0,
    }
}

/// The GPT-2 pass: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|
/// \s+(?!\S)|\s+`. At each position, in precedence order: a contraction (only
/// where a token starts — inside a punct run or after a space the `'` is punct);
/// else an optional single *space* then a letter, number or punct run (whichever
/// class the next char is); else whitespace, which leaves its last char to the
/// next token when a non-whitespace char follows (`\s+(?!\S)`).
fn scan_core_gpt2<F>(text: &str, mut emit: F) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    let t = tables();
    let b = text.as_bytes();
    let n = b.len();
    let mut i = 0usize;
    while i < n {
        let (c, _) = char_at(text, b, i);
        if c == '\'' {
            let k = gpt2_contraction_len(&b[i..]);
            if k > 0 {
                emit(i, i + k)?;
                i += k;
                continue;
            }
        }
        // The run's first char: `c` itself, or the char after a space prefix.
        let (body, c0) = if c == ' ' && i + 1 < n {
            let (c1, _) = char_at(text, b, i + 1);
            if is_ws(t, c1) { (i, c) } else { (i + 1, c1) }
        } else {
            (i, c)
        };
        let class = |x: char| {
            if is_letter(t, x) {
                0
            } else if is_number(t, x) {
                1
            } else if is_ws(t, x) {
                3
            } else {
                2
            }
        };
        let k0 = class(c0);
        if k0 != 3 {
            let mut e = body;
            while e < n {
                let (cj, lj) = char_at(text, b, e);
                if class(cj) != k0 {
                    break;
                }
                e += lj;
            }
            emit(i, e)?;
            i = e;
            continue;
        }
        // Whitespace run.
        let mut e = i;
        let mut last = i;
        while e < n {
            let (cj, lj) = char_at(text, b, e);
            if !is_ws(t, cj) {
                break;
            }
            last = e;
            e += lj;
        }
        let end = if e < n && last > i { last } else { e };
        emit(i, end)?;
        i = end;
    }
    Ok(())
}

/// The DeepSeek pass: reproduces `Sequence([Split(\p{N}{1,3}), Split(CJK),
/// Split(gpt-like)])` (all `Isolated`) in a single left-to-right scan.
///
/// The governing invariant is: **no pretoken crosses a CJK-range boundary.**
/// Split 2 isolates maximal runs of DeepSeek's CJK class (a codepoint *range*,
/// [`is_ds_cjk`]), so a transition between a CJK-range char and a non-range char
/// is always a cut. Split 1 caps digit runs at 3, and split 3 (a GPT-like
/// pattern) then applies *within* each piece. In one pass this becomes, at each
/// position, in precedence order:
/// - 1: digits `\p{N}{1,3}` (split 1; digits are never in the CJK range);
/// - 3a: `[<ascii punct>][A-Za-z]+` — a punct char glued to an ASCII word (ASCII,
///   never CJK);
/// - 3b: `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+` — a letter/mark run;
/// - 3c: ` ?[\p{P}\p{S}]+[\r\n]*` — a punctuation/symbol run (its optional space
///   and trailing `[\r\n]` are non-CJK, so they only attach to a non-CJK run);
/// - 3d–f: `\s*[\r\n]+ | \s+(?!\S) | \s+` — whitespace (never in the CJK range);
/// - gap: any leftover char (rare: `\p{C}` minus whitespace).
///
/// Every run (3b, 3c, gap) additionally stops at a CJK-range boundary, which is
/// what makes CJK letters group together (`中文`) yet split off both adjacent
/// non-CJK letters (`中x` → `中`,`x`) and in-range non-letters (`ヿ゠` → `ヿ`,`゠`,
/// since `゠` U+30A0 is punctuation). Verified byte-for-byte against the three
/// sequenced `Split`s in `deepseek_scanner_matches_regex_sequence`.
fn scan_core_deepseek<F>(text: &str, mut emit: F) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<(), String>,
{
    let t = tables();
    let b = text.as_bytes();
    let n = b.len();
    // `[^\r\n\p{L}\p{P}\p{S}]`: the optional single prefix of split-3 alt b.
    let is_ds_prefix = |c: char| c != '\r' && c != '\n' && !is_letter(t, c) && !t.is_psym(c as u32);
    // `[\p{L}\p{M}]` (CJK included; runs are kept on one side of the CJK boundary
    // explicitly, below).
    let run_body = |c: char| run_member(t, c, false);
    let mut i = 0usize;

    while i < n {
        let start = i;

        // Fast path: ` ?[A-Za-z]+` — split-3 alt b specialized to an all-ASCII
        // letter run (case-insensitive; DeepSeek does not split on case). ASCII is
        // never in the CJK range, so no boundary check is needed. A non-space/
        // non-letter start, or a non-ASCII continuation that might extend the run
        // (into a `\p{L}`/`\p{M}` or a CJK char), defers to the scalar path.
        {
            let c0 = b[i];
            let is_alpha = |x: u8| (x | 0x20).wrapping_sub(b'a') < 26;
            let lstart = if is_alpha(c0) {
                i
            } else if c0 == b' ' && i + 1 < n && is_alpha(b[i + 1]) {
                i + 1
            } else {
                usize::MAX
            };
            if lstart != usize::MAX {
                let run_end = ascii_alpha_run_end(b, lstart);
                if run_end == n || b[run_end] < 0x80 {
                    emit(start, run_end)?;
                    i = run_end;
                    continue;
                }
            }
        }

        let (c, clen) = char_at(text, b, i);

        // 1. Digits: `\p{N}{1,3}` (split 1). Digits are never CJK-range.
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

        // 3a. `[<ascii punct>][A-Za-z]+`: an ASCII-punctuation char glued to an
        //     ASCII letter run (ASCII only, so no CJK boundary is involved).
        if (c as u32) < 0x80 && (c as u8).is_ascii_punctuation() && i + clen < n {
            let nb = b[i + clen];
            if (nb | 0x20).wrapping_sub(b'a') < 26 {
                let e = ascii_alpha_run_end(b, i + clen);
                emit(start, e)?;
                i = e;
                continue;
            }
        }

        // 3b. `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`, all on one CJK side. The
        //     optional prefix must share the run's side (else split 2 cut between).
        let run_start = if run_body(c) {
            Some(i)
        } else if is_ds_prefix(c) && i + clen < n {
            let (nc, _) = char_at(text, b, i + clen);
            (run_body(nc) && is_ds_cjk(c) == is_ds_cjk(nc)).then_some(i + clen)
        } else {
            None
        };
        if let Some(run_start) = run_start {
            let side = is_ds_cjk(char_at(text, b, run_start).0);
            let mut e = run_start;
            while e < n {
                let (cj, lj) = char_at(text, b, e);
                if !run_body(cj) || is_ds_cjk(cj) != side {
                    break;
                }
                e += lj;
            }
            emit(start, e)?;
            i = e;
            continue;
        }

        // 3c. ` ?[\p{P}\p{S}]+[\r\n]*`, all on one CJK side. The optional leading
        //     space and trailing `[\r\n]` are non-CJK, so they attach only when the
        //     run itself is non-CJK.
        {
            let (mut pstart, mut pc, mut pcl) = (i, c, clen);
            if c == ' ' && i + clen < n {
                let (c1, l1) = char_at(text, b, i + clen);
                if t.is_psym(c1 as u32) && !is_ds_cjk(c1) {
                    pstart = i + clen;
                    pc = c1;
                    pcl = l1;
                }
            }
            if t.is_psym(pc as u32) {
                let side = is_ds_cjk(pc);
                let mut e = pstart + pcl;
                while e < n {
                    let (cj, lj) = char_at(text, b, e);
                    if !t.is_psym(cj as u32) || is_ds_cjk(cj) != side {
                        break;
                    }
                    e += lj;
                }
                if !side {
                    while e < n && (b[e] == b'\r' || b[e] == b'\n') {
                        e += 1;
                    }
                }
                emit(start, e)?;
                i = e;
                continue;
            }
        }

        // 3d–f. Whitespace: `\s*[\r\n]+ | \s+(?!\S) | \s+` (mirrors scan_core;
        //       whitespace is never CJK-range).
        if is_ws(t, c) {
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
            let mut last_nl = usize::MAX;
            let mut k = i;
            while k < we {
                if b[k] == b'\r' || b[k] == b'\n' {
                    last_nl = k;
                }
                k += 1;
            }
            // `\s+(?!\S)` leaves the last whitespace codepoint for the *following*
            // token to take as its prefix — but only when that token is in the
            // same split-3 piece. Split 1 (digits) and split 2 (CJK) cut their
            // matches into separate pieces first, so a whitespace run immediately
            // before a digit or CJK char is a complete piece and `(?!\S)` keeps it
            // whole. (This is the one place DeepSeek's sequenced splits differ from
            // the other kinds' single regex, where a digit is just another
            // alternative in the same pass.)
            let next_starts_new_piece = we < n && {
                let (nc, _) = char_at(text, b, we);
                is_number(t, nc) || is_ds_cjk(nc)
            };
            let end = if last_nl != usize::MAX {
                last_nl + 1
            } else if we == n || next_starts_new_piece {
                we
            } else if last_cp_start > i {
                last_cp_start
            } else {
                we
            };
            emit(i, end)?;
            i = end;
            continue;
        }

        // Gap: a maximal run of characters matching no split-3 alternative — rare,
        // `\p{C}` minus whitespace (controls, format chars like ZWJ, private use).
        // An `Isolated` Split keeps such an unmatched span as one pretoken, stopping
        // at a CJK boundary and before any char that starts a real token (including
        // a same-side gap char that prefixes a following `[\p{L}\p{M}]+`).
        let side = is_ds_cjk(c);
        let mut e = i + clen;
        while e < n {
            let (cj, lj) = char_at(text, b, e);
            if is_number(t, cj)
                || run_body(cj)
                || t.is_psym(cj as u32)
                || is_ws(t, cj)
                || is_ds_cjk(cj) != side
            {
                break;
            }
            let after = e + lj;
            if after < n {
                let (ac, _) = char_at(text, b, after);
                if run_body(ac) && is_ds_cjk(ac) == is_ds_cjk(cj) {
                    break; // cj is the optional prefix of the next same-side letter run
                }
            }
            e += lj;
        }
        emit(i, e)?;
        i = e;
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
    ///
    /// The opposite case must hold too: after a letter, `\n` is a pretoken of its
    /// own and the `/` opens the next one (`/b's`), so advancing past the `/`
    /// there cut that pretoken. A newline followed by `/` is ambiguous, and is
    /// no longer used as a cut point at all.
    #[test]
    fn o200k_chunk_bounds_preserve_slash_after_newline() {
        for (text, pretoken) in [
            // Sized so the 2-way nominal split lands on the newline.
            (
                format!("{}.\n/{}", "a".repeat(100), "b".repeat(100)),
                ".\n/",
            ),
            (
                format!("{}b\n/b's {}", "a ".repeat(50), "c".repeat(100)),
                "/b's",
            ),
        ] {
            let whole = scan(ScanKind::O200k, &text);
            assert!(
                whole.iter().any(|p| p == pretoken),
                "{pretoken:?} should be one o200k pretoken: {whole:?}"
            );
            for n_chunks in [2, 3, 5] {
                let bounds = newline_chunk_bounds(&text, n_chunks, ScanKind::O200k);
                let chunked: Vec<String> = bounds
                    .iter()
                    .flat_map(|&(s, e)| scan(ScanKind::O200k, &text[s..e]))
                    .collect();
                assert_eq!(chunked, whole, "chunked o200k scan diverged: {bounds:?}");
            }
        }
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
                    end.\n/usr/bin\nx!\n/\n/y\n\n\n   foo\t\n bar  \n\u{00a0}baz\n 'sx\n";
        let big = unit.repeat(400);
        for kind in [
            ScanKind::O200k,
            ScanKind::Kimi,
            ScanKind::Cl100k,
            ScanKind::Qwen,
            ScanKind::Qwen35,
            ScanKind::Tekken,
            ScanKind::DeepSeek,
            ScanKind::Gpt2,
        ] {
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

    /// Text with few or no newlines is still chunked, at space boundaries (see
    /// [`space_boundary`]), and scanning those chunks independently reproduces
    /// a whole-buffer scan exactly for every grammar.
    #[test]
    fn space_chunking_matches_whole() {
        // No newlines at all, and every kind of char next to a space: letters in
        // both cases and scripts, digits, punctuation (incl. `/` and `'`),
        // contractions, CJK and ideographic space (U+3000), NBSP, marks, runs of
        // spaces and tabs.
        let unit = "Hello world! CamelCase HTTPRequest 中文 test  spaced   out \t tab \
                    café résumé naïve 12345 1,000 don't I'LL we're /usr/bin a/b x! /y \
                    ！？ 全角　空格 \u{3000} ideographic\u{3000}space \u{00a0}nbsp \
                    a\u{0301} e\u{0301}x (paren) [bracket] {brace} \"quote\" 'single' \
                    ` backtick ~tilde ^caret _under .dot ,comma ;semi :colon @at #hash ";
        let big = unit.repeat(300);
        assert!(!big.contains('\n'));
        for kind in [
            ScanKind::O200k,
            ScanKind::Kimi,
            ScanKind::Cl100k,
            ScanKind::Qwen,
            ScanKind::Qwen35,
            ScanKind::Tekken,
            ScanKind::DeepSeek,
            ScanKind::Gpt2,
        ] {
            let whole = scan_seq(kind, &big);
            for n_chunks in [2usize, 3, 7, 16, 64, 500] {
                let bounds = newline_chunk_bounds(&big, n_chunks, kind);
                assert!(
                    n_chunks < 16 || bounds.len() >= n_chunks / 2,
                    "kind={kind:?}"
                );
                let mut combined = Vec::new();
                for (s, e) in bounds {
                    let base = s as u32;
                    for (a, b) in scan_seq(kind, &big[s..e]) {
                        combined.push((a + base, b + base));
                    }
                }
                assert_eq!(combined, whole, "kind={kind:?} n_chunks={n_chunks}");
            }
        }
    }

    /// Randomized [`space_chunking_matches_whole`]: random strings over the
    /// chars that sit next to boundaries in some grammar, cut into many chunks.
    #[test]
    fn chunking_fuzz_matches_whole() {
        let alphabet = [
            " ", " ", " ", "  ", "\t", "a", "b", "Z", "Qu", "1", "7", ".", ",", "/", "'", "'s",
            "!", "(", "-", "_", "中", "文", "。", "！", "\u{3000}", "\u{00a0}", "é", "\u{0301}",
            "ä", "Ω", "\u{200d}", "😀", "\n", "\r\n", "\u{1f}", "\u{7f}", "0", "٣",
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..40 {
            let len = 2000 + (rnd() % 6000) as usize;
            let newline_rare = round % 2 == 0;
            let mut text = String::new();
            while text.len() < len {
                let mut piece = alphabet[(rnd() % alphabet.len() as u64) as usize];
                if newline_rare && piece.contains('\n') && rnd() % 16 != 0 {
                    piece = " ";
                }
                text.push_str(piece);
            }
            for kind in [
                ScanKind::O200k,
                ScanKind::Kimi,
                ScanKind::Cl100k,
                ScanKind::Qwen,
                ScanKind::Qwen35,
                ScanKind::Tekken,
                ScanKind::DeepSeek,
                ScanKind::Gpt2,
            ] {
                let whole = scan_seq(kind, &text);
                for n_chunks in [2usize, 5, 13, 40, 200] {
                    let mut combined = Vec::new();
                    for (s, e) in newline_chunk_bounds(&text, n_chunks, kind) {
                        let base = s as u32;
                        for (a, b) in scan_seq(kind, &text[s..e]) {
                            combined.push((a + base, b + base));
                        }
                    }
                    let diverged = combined.iter().zip(&whole).position(|(a, b)| a != b);
                    assert!(
                        combined == whole,
                        "kind={kind:?} n_chunks={n_chunks} round={round}: chunked scan diverged \
                         at pretoken {diverged:?}"
                    );
                }
            }
        }
    }

    /// The DeepSeek single-pass scanner must reproduce, byte-for-byte, the result
    /// of applying DeepSeek's three `Isolated` `Split`s in sequence via the crate's
    /// own regex engine — over digits, CJK (in and out of the specific ranges),
    /// punctuation-glued words, symbols, and control/format chars.
    #[test]
    fn deepseek_scanner_matches_regex_sequence() {
        use crate::Split;
        use crate::pre_tokenized::PreTokenizedString;
        use serde_json::json;

        let mk = |p: &str| Split::from_config(&json!({ "Regex": p }), "Isolated", false).unwrap();
        let s1 = mk(crate::tiktoken::DEEPSEEK_SPLIT1_PATTERN);
        let s2 = mk(crate::tiktoken::DEEPSEEK_SPLIT2_PATTERN);
        let s3 = mk(crate::tiktoken::DEEPSEEK_SPLIT3_PATTERN);

        let corpus = [
            "",
            "Hello, world! HTTPRequest HelloWorld camelCase don't I'll y'all",
            "1234567 3.14 1,000 42 007 mixed ABC123def a1b2c3 100000000",
            "_foo .NET /usr/bin/env snake_case kebab-case a!b (hello) [tag] {x}",
            "  leading trailing  a  b  a   b\ttabs\there end.\n\nfoo\r\nbar",
            "你好世界 中文，世界 混合ABCと日本語 test中文HELLO 中1文 a中b カタカナ ひらがな",
            "汉字\u{20000}扩展 あいうえお アイウエオ 龥一 ヿ゠", // CJK ext-B (outside range) vs in-range
            "café résumé naïve über straße señor 😀🚀 ①②③ ＦＵＬＬ",
            "emoji👍\u{200d}👍 zwj a\u{200d}b \u{0000}ctrl \u{0007}bell \u{feff}bom",
            "e\u{0301}\u{0302} mark .\n/\n. punct+sym <=> |~ $100 #tag @user",
            // Whitespace runs immediately before a digit / CJK: split 1 / split 2
            // cut those first, so `\s+(?!\S)` keeps the whole run (regression for
            // ".indd   2\n201").
            ".indd   2\n201  3   你好    世界\t42 x  y   z",
            "a   1 b  2  \n  3\ttab   中 end   ",
            "The rain in Spain. ".to_string().repeat(30).leak(),
        ];

        for s in corpus {
            let mut pts = PreTokenizedString::from_text(s);
            s1.pre_tokenize(&mut pts).unwrap();
            s2.pre_tokenize(&mut pts).unwrap();
            s3.pre_tokenize(&mut pts).unwrap();
            let regex_ranges: Vec<(u32, u32)> = pts
                .splits()
                .iter()
                .filter(|sp| !sp.range.is_empty())
                .map(|sp| (sp.range.start as u32, sp.range.end as u32))
                .collect();
            let scan: Vec<(u32, u32)> = scan_seq(ScanKind::DeepSeek, s)
                .into_iter()
                .filter(|(a, b)| a != b)
                .collect();
            assert_eq!(scan, regex_ranges, "input={s:?}");
        }
    }

    /// Diagnostic (manual): compare the DeepSeek scanner to the 3-`Split` regex
    /// sequence over real documents. Set `DS_DIAG_FILE` to a LongBench-style
    /// `data.json`. Prints the first diverging window. Skipped when unset.
    #[test]
    fn deepseek_real_data_diag() {
        let Ok(path) = std::env::var("DS_DIAG_FILE") else {
            return;
        };
        use crate::Split;
        use crate::pre_tokenized::PreTokenizedString;
        use serde_json::json;
        let mk = |p: &str| Split::from_config(&json!({ "Regex": p }), "Isolated", false).unwrap();
        let s1 = mk(crate::tiktoken::DEEPSEEK_SPLIT1_PATTERN);
        let s2 = mk(crate::tiktoken::DEEPSEEK_SPLIT2_PATTERN);
        let s3 = mk(crate::tiktoken::DEEPSEEK_SPLIT3_PATTERN);
        let text = std::fs::read_to_string(&path).unwrap();
        let data: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
        let samples: Vec<String> = data
            .iter()
            .filter_map(|it| it.get("context").and_then(|v| v.as_str()).map(String::from))
            .take(50)
            .collect();
        for (si, s) in samples.iter().enumerate() {
            let mut pts = PreTokenizedString::from_text(s);
            s1.pre_tokenize(&mut pts).unwrap();
            s2.pre_tokenize(&mut pts).unwrap();
            s3.pre_tokenize(&mut pts).unwrap();
            let rx: Vec<(u32, u32)> = pts
                .splits()
                .iter()
                .filter(|sp| !sp.range.is_empty())
                .map(|sp| (sp.range.start as u32, sp.range.end as u32))
                .collect();
            let sc: Vec<(u32, u32)> = scan_seq(ScanKind::DeepSeek, s)
                .into_iter()
                .filter(|(a, b)| a != b)
                .collect();
            if sc != rx {
                let k = sc.iter().zip(&rx).position(|(a, b)| a != b).unwrap_or(0);
                let lo = k.saturating_sub(3);
                eprintln!("sample {si}: first diff at pretoken {k}");
                eprintln!("  scan: {:?}", &sc[lo..(k + 4).min(sc.len())]);
                eprintln!("  rxeq: {:?}", &rx[lo..(k + 4).min(rx.len())]);
                let ctx_start = sc[lo].0 as usize;
                let ctx_end = (sc.get(k + 4).map(|r| r.1).unwrap_or(sc[k].1)) as usize;
                eprintln!("  text: {:?}", &s[ctx_start..ctx_end.min(s.len())]);
                panic!("scanner diverged from regex sequence on sample {si}");
            }
        }
        eprintln!("no divergence over {} samples", samples.len());
    }

    #[test]
    fn recognizes_patterns() {
        assert_eq!(
            recognize(crate::tiktoken::CL100K_BASE_PATTERN),
            Some(ScanKind::Cl100k)
        );
        assert_eq!(
            recognize(crate::tiktoken::O200K_BASE_PATTERN),
            Some(ScanKind::O200k)
        );
        assert_eq!(recognize(KIMI_PATTERN), Some(ScanKind::Kimi));
        assert_eq!(recognize("something else"), None);
        // DeepSeek is a sequence, recognized by its three sources together.
        assert!(recognize_deepseek(
            crate::tiktoken::DEEPSEEK_SPLIT1_PATTERN,
            crate::tiktoken::DEEPSEEK_SPLIT2_PATTERN,
            crate::tiktoken::DEEPSEEK_SPLIT3_PATTERN,
        ));
        assert!(!recognize_deepseek("a", "b", "c"));
    }

    /// cl100k differs from o200k/Kimi: `\p{L}+` runs are not split on case, and
    /// the contraction is a standalone leading alternative rather than a word
    /// suffix. These spellings are the behaviours those two facts imply.
    #[test]
    fn cl100k_words_and_contractions() {
        use ScanKind::Cl100k as C;
        // No case split: mixed-case letters stay one token.
        assert_eq!(scan(C, "HelloWorld"), vec!["HelloWorld"]);
        assert_eq!(scan(C, "camelCase"), vec!["camelCase"]);
        assert_eq!(scan(C, "iOS"), vec!["iOS"]);
        // Contraction is its own token, split off the preceding word.
        assert_eq!(scan(C, "don't"), vec!["don", "'t"]);
        assert_eq!(scan(C, "I'll"), vec!["I", "'ll"]);
        assert_eq!(scan(C, "they're"), vec!["they", "'re"]);
        // A leading `'` not forming a contraction is a word prefix.
        assert_eq!(scan(C, "'x"), vec!["'x"]);
        // Space-prefixed word.
        assert_eq!(scan(C, "hello world"), vec!["hello", " world"]);
        // A combining mark is not `\p{L}`, so it leaves the word (→ punctuation).
        assert_eq!(scan(C, "cafe\u{0301}"), vec!["cafe", "\u{0301}"]);
        // Numbers still cap at 3.
        assert_eq!(scan(C, "1234567"), vec!["123", "456", "7"]);
    }

    /// The scanner gate treats `Removed`+`invert` (Phi-4's cl100k spelling) as
    /// equivalent to `Isolated` because the recognized patterns are total. Prove
    /// it: both spellings must yield the identical splits from the regex engine.
    #[test]
    fn removed_inverted_equals_isolated_for_covering() {
        use crate::Split;
        use crate::pre_tokenized::PreTokenizedString;
        use serde_json::json;

        let pat = crate::tiktoken::CL100K_BASE_PATTERN;
        let iso = Split::from_config(&json!({ "Regex": pat }), "Isolated", false).unwrap();
        let rem = Split::from_config(&json!({ "Regex": pat }), "Removed", true).unwrap();
        assert_eq!(iso.scan_kind(), Some(ScanKind::Cl100k));
        assert_eq!(rem.scan_kind(), Some(ScanKind::Cl100k));

        let corpus = [
            "Hello, world! don't I'll 1234 café\u{0301}\n\n  x\r\ny'all -42",
            "(word) [tag] a!b snake_case kebab-case  \n  trailing  ",
        ];
        for s in corpus {
            let ranges = |split: &Split| -> Vec<(usize, usize)> {
                let mut pts = PreTokenizedString::from_text(s);
                split.pre_tokenize(&mut pts).unwrap();
                pts.splits()
                    .iter()
                    .filter(|sp| !sp.range.is_empty())
                    .map(|sp| (sp.range.start, sp.range.end))
                    .collect()
            };
            assert_eq!(ranges(&iso), ranges(&rem), "input={s:?}");
        }
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
            (ScanKind::Cl100k, crate::tiktoken::CL100K_BASE_PATTERN),
            (ScanKind::Qwen, crate::tiktoken::QWEN2_PATTERN),
            (ScanKind::Qwen35, crate::tiktoken::QWEN35_PATTERN),
            (ScanKind::O200k, crate::tiktoken::O200K_BASE_PATTERN),
            (ScanKind::Tekken, crate::tiktoken::TEKKEN_PATTERN),
            (ScanKind::Kimi, KIMI_PATTERN),
            (
                ScanKind::Gpt2,
                crate::pre_tokenizers::byte_level::GPT2_PATTERN,
            ),
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

    /// Random char soup over every class the grammars distinguish (multibyte
    /// digits, whitespace, marks, caseless/titlecase letters, symbols, contraction
    /// letters next to apostrophes) against the crate's regex engine.
    ///
    /// `ſ` (U+017F) is left out on purpose: the regex engines case-fold it to `s`,
    /// so `'ſ` is a `(?i:'s)` contraction to them, while the scanner — like
    /// `tokenizers` v1's bitcannon, whose ids we match — only takes an ASCII letter
    /// after the apostrophe (see [`contraction_len`]).
    #[test]
    fn scanner_matches_regex_soup() {
        use crate::Split;
        use crate::pre_tokenized::PreTokenizedString;
        use serde_json::json;

        let pool: &[&str] = &[
            "a",
            "Z",
            "s",
            "t",
            "e",
            "l",
            "L",
            "'",
            "'",
            "0",
            "7",
            " ",
            " ",
            "\t",
            "\n",
            "\r",
            "\x0b",
            "!",
            ".",
            "/",
            "(",
            "-",
            "_",
            "é",
            "É",
            "ß",
            "ǅ",
            "中",
            "文",
            "ー",
            "ʰ",
            "\u{0301}",
            "\u{094d}",
            "हि",
            "م",
            "١",
            "٣",
            "１",
            "２",
            "²",
            "½",
            "Ⅻ",
            "\u{0085}",
            "\u{00a0}",
            "\u{2000}",
            "\u{2028}",
            "\u{3000}",
            "’",
            "“",
            "—",
            "…",
            "，",
            "😀",
            "\u{1f3fd}",
            "\u{200d}",
            "€",
            "\u{0001}",
            "\u{007f}",
            "\u{feff}",
            "𝐀",
            "𝟎",
        ];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (kind, pat) in [
            (ScanKind::Cl100k, crate::tiktoken::CL100K_BASE_PATTERN),
            (ScanKind::Qwen, crate::tiktoken::QWEN2_PATTERN),
            (ScanKind::Qwen35, crate::tiktoken::QWEN35_PATTERN),
            (ScanKind::O200k, crate::tiktoken::O200K_BASE_PATTERN),
            (ScanKind::Tekken, crate::tiktoken::TEKKEN_PATTERN),
            (ScanKind::Kimi, KIMI_PATTERN),
            (
                ScanKind::Gpt2,
                crate::pre_tokenizers::byte_level::GPT2_PATTERN,
            ),
        ] {
            let split = Split::from_config(&json!({ "Regex": pat }), "Isolated", false).unwrap();
            for round in 0..3000 {
                let len = 1 + (next() % 60) as usize;
                let mut s = String::new();
                for _ in 0..len {
                    s.push_str(pool[(next() % pool.len() as u64) as usize]);
                }
                let mut pts = PreTokenizedString::from_text(&s);
                split.pre_tokenize(&mut pts).unwrap();
                let regex_ranges: Vec<(u32, u32)> = pts
                    .splits()
                    .iter()
                    .filter(|sp| !sp.range.is_empty())
                    .map(|sp| (sp.range.start as u32, sp.range.end as u32))
                    .collect();
                let scan: Vec<(u32, u32)> = scan_seq(kind, &s)
                    .into_iter()
                    .filter(|(a, b)| a != b)
                    .collect();
                assert_eq!(
                    scan, regex_ranges,
                    "round {round} kind={kind:?} input={s:?}"
                );
            }
        }
    }

    /// DeepSeek-focused random soup against the three sequenced regex `Split`s and
    /// the bit-parallel scanner (`scan_simd::scan_fast`): CJK-range letters, punct,
    /// marks and unassigned codepoints on both sides of the range edges, ASCII punct
    /// glued to letters, gaps, and whitespace before digits / CJK.
    #[test]
    fn deepseek_soup_matches_regex_and_bulk() {
        use crate::Split;
        use crate::pre_tokenized::PreTokenizedString;
        use serde_json::json;

        let mk = |p: &str| Split::from_config(&json!({ "Regex": p }), "Isolated", false).unwrap();
        let s1 = mk(crate::tiktoken::DEEPSEEK_SPLIT1_PATTERN);
        let s2 = mk(crate::tiktoken::DEEPSEEK_SPLIT2_PATTERN);
        let s3 = mk(crate::tiktoken::DEEPSEEK_SPLIT3_PATTERN);
        let pool: &[&str] = &[
            "a",
            "Z",
            "q",
            "'",
            ".",
            "_",
            "(",
            "!",
            "/",
            "$",
            "0",
            "7",
            " ",
            " ",
            " ",
            "\t",
            "\n",
            "\r",
            "\x0b",
            "\x01",
            "\x7f",
            "é",
            "ß",
            "\u{0301}",
            "ʰ",
            "中",
            "龥",
            "龦",
            "一",
            "あ",
            "ア",
            "ー",
            "ヿ",
            "゠",
            "・",
            "\u{3040}",
            "\u{3097}",
            "\u{3099}",
            "゛",
            "々",
            "〇",
            "。",
            "「",
            "\u{20000}",
            "１",
            "²",
            "½",
            "\u{00a0}",
            "\u{3000}",
            "\u{2028}",
            "’",
            "“",
            "—",
            "€",
            "😀",
            "\u{200d}",
            "\u{feff}",
            "\u{e000}",
            "\u{0085}",
            "𝐀",
        ];
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let fast = |t: &str| {
            let mut v = Vec::new();
            crate::pre_tokenizers::scan_simd::scan_fast(ScanKind::DeepSeek, t, |a, b| {
                v.push((a as u32, b as u32));
                Ok(())
            })
            .unwrap();
            v
        };
        for round in 0..6000 {
            let len = 1 + (next() % 50) as usize;
            let mut s = String::new();
            for _ in 0..len {
                s.push_str(pool[(next() % pool.len() as u64) as usize]);
            }
            let mut pts = PreTokenizedString::from_text(&s);
            s1.pre_tokenize(&mut pts).unwrap();
            s2.pre_tokenize(&mut pts).unwrap();
            s3.pre_tokenize(&mut pts).unwrap();
            let rx: Vec<(u32, u32)> = pts
                .splits()
                .iter()
                .filter(|sp| !sp.range.is_empty())
                .map(|sp| (sp.range.start as u32, sp.range.end as u32))
                .collect();
            let scan: Vec<(u32, u32)> = scan_seq(ScanKind::DeepSeek, &s)
                .into_iter()
                .filter(|(a, b)| a != b)
                .collect();
            assert_eq!(scan, rx, "scalar vs regex, round {round} input={s:?}");
            assert_eq!(fast(&s), scan, "bulk vs scalar, round {round} input={s:?}");
        }
    }

    /// Kimi over Han-heavy soup — BMP ideographs, Han-script `\p{Lm}` (々) and
    /// `\p{Nl}` (〇), a radical (`\p{So}`), an Extension-B ideograph, next to kana
    /// and Hangul (no stand-in: scalar islands), CJK punctuation, spaces, digits,
    /// apostrophes: the regex, the scalar scanner, the bulk callback path and the
    /// whole-text bitmap all agree.
    #[test]
    fn kimi_han_soup_matches_regex_and_bulk() {
        use crate::Split;
        use crate::pre_tokenized::PreTokenizedString;
        use serde_json::json;

        let split = Split::from_config(&json!({ "Regex": KIMI_PATTERN }), "Isolated", false).unwrap();
        let pool: &[&str] = &[
            "中", "文", "的", "件", "一", "龥", "々", "〇", "⺀", "\u{20000}", "あ", "ア", "ー", "한",
            "。", "，", "「", "」", "、", "：", " ", " ", "\n", "\t", "a", "B", "z", "'", "s", "1", "2",
            "!", ".", "é", "１", "\u{3000}", "\u{0301}",
        ];
        let mut state = 0xa076_1d64_78bd_642fu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..6000 {
            let len = 1 + (next() % 60) as usize;
            let dense = next() % 3; // 0: any, 1-2: mostly Han and CJK punctuation
            let mut s = String::new();
            for _ in 0..len {
                let r = next();
                let i = if dense > 0 && r % 4 != 0 { (r >> 8) % 10 } else { (r >> 8) % pool.len() as u64 };
                s.push_str(pool[i as usize]);
            }
            let mut pts = PreTokenizedString::from_text(&s);
            split.pre_tokenize(&mut pts).unwrap();
            let rx: Vec<(u32, u32)> = pts
                .splits()
                .iter()
                .filter(|sp| !sp.range.is_empty())
                .map(|sp| (sp.range.start as u32, sp.range.end as u32))
                .collect();
            let scalar: Vec<(u32, u32)> =
                scan_seq(ScanKind::Kimi, &s).into_iter().filter(|(a, b)| a != b).collect();
            assert_eq!(scalar, rx, "scalar vs regex, round {round} {s:?}");
            let mut fast = Vec::new();
            crate::pre_tokenizers::scan_simd::scan_fast(ScanKind::Kimi, &s, |a, b| {
                fast.push((a as u32, b as u32));
                Ok(())
            })
            .unwrap();
            assert_eq!(fast, scalar, "bulk vs scalar, round {round} {s:?}");
            if let Some(bs) = crate::pre_tokenizers::scan_simd::bulk_starts(ScanKind::Kimi, &s) {
                let got: Vec<(u32, u32)> = bs.spans(s.len()).map(|(a, b)| (a as u32, b as u32)).collect();
                assert_eq!(got, scalar, "bitmap vs scalar, round {round} {s:?}");
            }
        }
    }
}
