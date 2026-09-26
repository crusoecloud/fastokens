use std::borrow::Cow;
use std::sync::LazyLock;

use icu_normalizer::{ComposingNormalizer, ComposingNormalizerBorrowed};
use unicode_normalization_alignments::UnicodeNormalization;

static NFC_NORMALIZER: LazyLock<ComposingNormalizerBorrowed<'static>> =
    LazyLock::new(ComposingNormalizer::new_nfc);

/// NFC (Canonical Decomposition, followed by Canonical Composition) normalizer.
///
/// Applies Unicode NFC normalization to the input text. If the input is already
/// in NFC form the original string is returned without allocation.
///
/// Matches HF `tokenizers`, which uses the Unicode 9.0 tables of
/// `unicode-normalization-alignments`. Characters assigned since then have
/// combining class 0 there, so e.g. HF leaves U+06E0 U+07FD alone, while current
/// tables (ICU4X) reorder it.
#[derive(Debug)]
pub struct Nfc;

impl Nfc {
    /// Normalize `input` to NFC form.
    ///
    /// Returns `Cow::Borrowed` when the input is already NFC, avoiding
    /// allocation. ICU4X finds the longest already-normalized prefix in a
    /// single pass; only the rest goes through the Unicode 9.0 normalizer. This
    /// is exact because normalization data only grows between versions: anything
    /// Unicode 9.0 would change, ICU4X changes too, at or before the same point.
    pub fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let (head, tail) = NFC_NORMALIZER.split_normalized(input);
        if tail.is_empty() {
            return Cow::Borrowed(head);
        }
        let mut out = String::with_capacity(input.len());
        out.push_str(head);
        out.extend(tail.nfc().map(|(c, _)| c));
        Cow::Owned(out)
    }

    /// Whether `input` is already NFC (so [`Self::normalize`] is the identity on
    /// it). ASCII always is; otherwise ICU4X's quick check, a single pass that
    /// never allocates.
    pub fn is_normalized(&self, input: &str) -> bool {
        input.is_ascii() || NFC_NORMALIZER.is_normalized(input)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    #[test]
    fn ascii_unchanged() {
        let out = Nfc.normalize("hello world");
        assert_eq!(out, "hello world");
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    #[test]
    fn already_composed() {
        // U+00E9 = LATIN SMALL LETTER E WITH ACUTE (precomposed)
        let out = Nfc.normalize("\u{e9}");
        assert_eq!(out, "\u{e9}");
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    #[test]
    fn decomposes_then_composes() {
        // NFD: 'e' + U+0301 (COMBINING ACUTE ACCENT) -> NFC: U+00E9
        let out = Nfc.normalize("e\u{0301}");
        assert_eq!(out, "\u{e9}");
        assert!(matches!(out, Cow::Owned(_)));
    }

    #[test]
    fn empty_string() {
        let out = Nfc.normalize("");
        assert_eq!(out, "");
        assert!(matches!(out, Cow::Borrowed(_)));
    }

    #[test]
    fn mixed_ascii_and_decomposed() {
        // "caf" + 'e' + combining acute -> "caf\u{e9}"
        let out = Nfc.normalize("cafe\u{0301}!");
        assert_eq!(out, "caf\u{e9}!");
    }

    #[test]
    fn keeps_order_of_marks_newer_than_unicode_9() {
        // U+07FD (ccc 220 since Unicode 11) is unassigned in Unicode 9.0, so HF
        // does not move it before U+06E0 (ccc 230).
        for input in [
            "\u{6e0}\u{7fd}",
            "\u{308}\u{7fd}",
            "\u{1aeb}\u{35b}",
            "\u{346}\u{8cf}",
            "a\u{7fd}\u{301}",
        ] {
            assert_eq!(Nfc.normalize(input), input, "{input:?}");
        }
        // Marks known to Unicode 9.0 are still reordered and composed.
        assert_eq!(Nfc.normalize("a\u{301}\u{323}"), "\u{1ea1}\u{301}");
        assert_eq!(
            Nfc.normalize("x\u{6e0}\u{7fd}e\u{301}"),
            "x\u{6e0}\u{7fd}\u{e9}"
        );
    }

    #[test]
    fn matches_unicode_9_normalization() {
        // Every code point alone, then after a starter and before a mark.
        let unicode9 = |s: &str| s.nfc().map(|(c, _)| c).collect::<String>();
        for c in (0..=0x10ffffu32).filter_map(char::from_u32) {
            for s in [
                c.to_string(),
                format!("a{c}"),
                format!("a{c}\u{301}\u{323}"),
            ] {
                assert_eq!(Nfc.normalize(&s), unicode9(&s), "{s:?}");
            }
        }
        // Pairs of combining marks from both sides of the version split.
        let marks: Vec<char> = (0x300..0x370)
            .chain(0x591..0x5c8)
            .chain(0x610..0x61b)
            .chain(0x64b..0x660)
            .chain(0x6d6..0x6ee)
            .chain(0x7eb..0x7fe)
            .chain(0x898..0x900)
            .chain(0x1ab0..0x1b00)
            .chain(0x1dc0..0x1e00)
            .chain(0x20d0..0x20f1)
            .filter_map(char::from_u32)
            .collect();
        for &m1 in &marks {
            for &m2 in &marks {
                let s = format!("e{m1}{m2}");
                assert_eq!(Nfc.normalize(&s), unicode9(&s), "{s:?}");
            }
        }
    }

    #[test]
    fn hangul_composition() {
        // Hangul Jamo: U+1100 U+1161 U+11A8 -> composed U+AC01
        let out = Nfc.normalize("\u{1100}\u{1161}\u{11A8}");
        assert_eq!(out, "\u{AC01}");
    }
}
