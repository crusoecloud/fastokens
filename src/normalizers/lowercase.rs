use std::borrow::Cow;

/// `Lowercase` normalizer: maps every character to its Unicode lowercase form.
///
/// Mirrors HuggingFace `tokenizers.normalizers.Lowercase`, which lowercases one
/// character at a time (`NormalizedString::lowercase` pushes the characters of
/// `char::to_lowercase` for each character in turn). That is deliberately *not*
/// `str::to_lowercase`: the std method additionally applies the Greek
/// final-sigma rule, rendering `ὈΔΥΣΣΕΎΣ` as `ὀδυσσεύς` where HuggingFace
/// produces `ὀδυσσεύσ`. U+03A3 is the only character the two disagree on, and
/// matching HuggingFace is what this crate is for.
#[derive(Debug)]
pub struct Lowercase;

impl Lowercase {
    /// Lowercase `input`.
    ///
    /// Returns `Cow::Borrowed` when every character is already lowercase, so
    /// lowercase ASCII, digits, punctuation and uncased scripts never allocate.
    pub fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let Some(start) = first_change(input) else {
            return Cow::Borrowed(input);
        };
        // A hint, not a bound: lowercasing preserves length for ASCII and keeps
        // or shrinks most of Unicode, but a few code points grow (`İ` U+0130 ->
        // `i` + U+0307, `Ⱥ` U+023A -> `ⱥ` U+2C65), costing at most one realloc.
        let mut out = String::with_capacity(input.len());
        out.push_str(&input[..start]);
        lower_into(&input[start..], &mut out);
        Cow::Owned(out)
    }
}

/// High bit of every byte in a `u64` lane.
const SWAR_HI: u64 = 0x8080_8080_8080_8080;

/// High bit of every lane of `word` that holds `A`..=`Z`.
///
/// `word` must be all-ASCII (`word & SWAR_HI == 0`). `(b | 0x80) - b'A'` leaves
/// the lane's high bit set iff `b >= b'A'`, and `0xDA - b` (`b'Z' + 0x80`) sets
/// it iff `b <= b'Z'`; with every byte below `0x80` neither subtraction can
/// borrow into the next lane.
#[inline(always)]
fn ascii_upper_mask(word: u64) -> u64 {
    let ge_a = (word | SWAR_HI).wrapping_sub(0x4141_4141_4141_4141);
    let le_z = 0xDADA_DADA_DADA_DADA_u64.wrapping_sub(word);
    ge_a & le_z & SWAR_HI
}

/// Whether `char::to_lowercase` rewrites `c`.
///
/// The mapping is the identity exactly when it yields a single character equal
/// to `c`; a different character, or the 1->2 expansion of `İ` U+0130, is a
/// change. `c.is_uppercase()` is not a substitute: it is false for titlecase
/// letters such as `ǅ` U+01C5, which do lowercase (to `ǆ` U+01C6).
#[inline]
fn needs_lowering(c: char) -> bool {
    let mut lower = c.to_lowercase();
    lower.next() != Some(c) || lower.next().is_some()
}

/// Byte offset of the first character `char::to_lowercase` would rewrite.
fn first_change(input: &str) -> Option<usize> {
    let bytes = input.as_bytes();
    let mut pos = 0;
    loop {
        // Skip runs of already-lowercase ASCII eight bytes at a time.
        while let Some(chunk) = bytes[pos..].first_chunk::<8>() {
            let word = u64::from_le_bytes(*chunk);
            if word & SWAR_HI != 0 {
                break; // non-ASCII in this window: resolve it character by character
            }
            let upper = ascii_upper_mask(word);
            if upper != 0 {
                // Lane `i` carries its flag at bit `8 * i + 7`.
                return Some(pos + upper.trailing_zeros() as usize / 8);
            }
            pos += 8;
        }
        let c = input[pos..].chars().next()?;
        if needs_lowering(c) {
            return Some(pos);
        }
        pos += c.len_utf8();
    }
}

/// Append the per-character lowercase form of `input` to `out`.
fn lower_into(input: &str, out: &mut String) {
    let bytes = input.as_bytes();
    let mut pos = 0;
    loop {
        // SAFETY: the inner loop only appends the lanes of an all-ASCII word
        // with bit 0x20 possibly set, so every byte written stays below 0x80
        // and `out` remains valid UTF-8.
        let raw = unsafe { out.as_mut_vec() };
        while let Some(chunk) = bytes[pos..].first_chunk::<8>() {
            let word = u64::from_le_bytes(*chunk);
            if word & SWAR_HI != 0 {
                break;
            }
            // `0x80 >> 2 == 0x20` is exactly the bit that turns `A`..=`Z` into
            // `a`..=`z`, so this lowercases eight bytes without a branch.
            let lowered = word | (ascii_upper_mask(word) >> 2);
            raw.extend_from_slice(&lowered.to_le_bytes());
            pos += 8;
        }
        let Some(c) = input[pos..].chars().next() else {
            return;
        };
        out.extend(c.to_lowercase());
        pos += c.len_utf8();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn already_lowercase_ascii_is_borrowed() {
        let out = Lowercase.normalize("hello, world! 42 -- long enough for swar");
        assert_eq!(out, "hello, world! 42 -- long enough for swar");
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    #[test]
    fn empty_is_borrowed() {
        let out = Lowercase.normalize("");
        assert_eq!(out, "");
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    #[test]
    fn lowercases_ascii() {
        let out = Lowercase.normalize("Hello, WORLD! 42");
        assert_eq!(out, "hello, world! 42");
        assert!(matches!(out, Cow::Owned(_)));
    }

    /// A capital placed at every byte offset in turn, to pin the SWAR lane
    /// indexing, the window boundary at 8 bytes and the scalar tail.
    #[test]
    fn uppercase_at_every_offset() {
        let base = "the quick brown fox jumps ok"; // 28 bytes: 3 full words + a tail
        // `base` is ASCII, so byte offsets and character positions coincide.
        let with_at = |i: usize, ch: char| -> String {
            base.char_indices()
                .map(|(j, c)| if j == i { ch } else { c })
                .collect()
        };
        for i in 0..base.len() {
            assert_eq!(
                Lowercase.normalize(&with_at(i, 'Z')),
                with_at(i, 'z'),
                "capital at offset {i}"
            );
        }
    }

    #[test]
    fn titlecase_letters_are_lowered() {
        // U+01C5 is titlecase `ǅ` and U+01C4 its uppercase `Ǆ`; both lowercase
        // to U+01C6. `char::is_uppercase` is false for the titlecase form.
        let out = Lowercase.normalize("\u{01C5}\u{01C4}");
        assert_eq!(out, "\u{01C6}\u{01C6}");
    }

    #[test]
    fn dotted_capital_i_expands() {
        // U+0130 is the only character whose lowercase form is two characters.
        let out = Lowercase.normalize("\u{0130}stanbul");
        assert_eq!(out, "i\u{0307}stanbul");
    }

    /// HuggingFace lowercases per character, so a word-final sigma stays `σ`.
    /// `str::to_lowercase` would render it `ς`; the `assert_ne!` keeps a future
    /// refactor towards it from silently diverging from HuggingFace.
    #[test]
    fn final_sigma_is_not_contextual() {
        let input = "\u{1F48}\u{0394}\u{03A5}\u{03A3}\u{03A3}\u{0395}\u{038E}\u{03A3}";
        let out = Lowercase.normalize(input);
        assert_eq!(
            out,
            "\u{1F40}\u{03B4}\u{03C5}\u{03C3}\u{03C3}\u{03B5}\u{03CD}\u{03C3}"
        );
        assert_ne!(out, input.to_lowercase());
    }

    #[test]
    fn uncased_text_is_borrowed() {
        let out = Lowercase.normalize("\u{4F60}\u{597D} \u{5E9}\u{5DC}\u{5D5}\u{5DD} 🙂 caf\u{e9}");
        assert_eq!(
            out,
            "\u{4F60}\u{597D} \u{5E9}\u{5DC}\u{5D5}\u{5DD} 🙂 caf\u{e9}"
        );
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    #[test]
    fn mixed_ascii_and_non_ascii() {
        let out = Lowercase.normalize("caf\u{c9} AU LAIT \u{4F60} \u{c4}bc");
        assert_eq!(out, "caf\u{e9} au lait \u{4F60} \u{e4}bc");
    }
}
