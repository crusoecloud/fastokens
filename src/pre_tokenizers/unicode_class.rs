//! Unicode codepoint classification tables for the pretokenizer scanner.
//!
//! The tables are built from [`regex_syntax`] — the same crate `fancy-regex`
//! (and therefore tiktoken's reference matcher) is built on — so a codepoint's
//! membership here is identical to `\p{…}` under that engine, with no
//! Unicode-version drift. This is what lets the scanner classify non-ASCII
//! codepoints natively instead of falling back to a regex.

use std::sync::OnceLock;

use regex_syntax::hir::{Class, HirKind};

/// A dense bitset over all Unicode scalar values (`0..=0x10FFFF`).
struct ClassSet {
    bits: Box<[u64]>,
}

impl ClassSet {
    /// Build the set of codepoints matched by a single-class regex `pattern`
    /// (e.g. `r"\p{Han}"` or `r"[\p{Lu}\p{Lt}]"`).
    fn from_pattern(pattern: &str) -> Self {
        let hir = regex_syntax::parse(pattern)
            .unwrap_or_else(|e| panic!("invalid class pattern {pattern:?}: {e}"));
        let mut bits = vec![0u64; 0x110000_usize.div_ceil(64)].into_boxed_slice();
        match hir.kind() {
            HirKind::Class(Class::Unicode(cls)) => {
                for range in cls.iter() {
                    for cp in (range.start() as u32)..=(range.end() as u32) {
                        bits[(cp >> 6) as usize] |= 1u64 << (cp & 63);
                    }
                }
            }
            other => panic!("pattern {pattern:?} is not a unicode class: {other:?}"),
        }
        Self { bits }
    }

    #[inline(always)]
    fn contains(&self, cp: u32) -> bool {
        (self.bits[(cp >> 6) as usize] >> (cp & 63)) & 1 != 0
    }
}

/// The classification tables the scanner needs.
pub struct Tables {
    letter: ClassSet, // \p{L}
    number: ClassSet, // \p{N}
    ws: ClassSet,     // \s
    han: ClassSet,    // \p{Han}
    ugroup: ClassSet, // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]  (the pattern's "uppercase" class)
    lgroup: ClassSet, // [\p{Ll}\p{Lm}\p{Lo}\p{M}]        (the pattern's "lowercase" class)
    psym: ClassSet,   // [\p{P}\p{S}]  (DeepSeek's punctuation-or-symbol run class)
}

impl Tables {
    #[inline(always)]
    pub fn is_letter(&self, cp: u32) -> bool {
        self.letter.contains(cp)
    }
    #[inline(always)]
    pub fn is_number(&self, cp: u32) -> bool {
        self.number.contains(cp)
    }
    #[inline(always)]
    pub fn is_ws(&self, cp: u32) -> bool {
        self.ws.contains(cp)
    }
    #[inline(always)]
    pub fn is_han(&self, cp: u32) -> bool {
        self.han.contains(cp)
    }
    /// Membership in `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]` (the `U*`/`U+` class).
    #[inline(always)]
    pub fn is_ugroup(&self, cp: u32) -> bool {
        self.ugroup.contains(cp)
    }
    /// Membership in `[\p{Ll}\p{Lm}\p{Lo}\p{M}]` (the `L+`/`L*` class).
    #[inline(always)]
    pub fn is_lgroup(&self, cp: u32) -> bool {
        self.lgroup.contains(cp)
    }
    /// Membership in `[\p{P}\p{S}]` (DeepSeek's punctuation-or-symbol class).
    #[inline(always)]
    pub fn is_psym(&self, cp: u32) -> bool {
        self.psym.contains(cp)
    }
}

/// The process-wide tables, built once on first use (~a few ms).
pub fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| Tables {
        letter: ClassSet::from_pattern(r"\p{L}"),
        number: ClassSet::from_pattern(r"\p{N}"),
        ws: ClassSet::from_pattern(r"\s"),
        han: ClassSet::from_pattern(r"\p{Han}"),
        ugroup: ClassSet::from_pattern(r"[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]"),
        lgroup: ClassSet::from_pattern(r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]"),
        psym: ClassSet::from_pattern(r"[\p{P}\p{S}]"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_agrees() {
        let t = tables();
        for b in b'A'..=b'Z' {
            assert!(t.is_letter(b as u32) && t.is_ugroup(b as u32) && !t.is_lgroup(b as u32));
        }
        for b in b'a'..=b'z' {
            assert!(t.is_letter(b as u32) && t.is_lgroup(b as u32) && !t.is_ugroup(b as u32));
        }
        for b in b'0'..=b'9' {
            assert!(t.is_number(b as u32) && !t.is_letter(b as u32));
        }
        for &b in &[b'\t', b'\n', b'\r', b' ', 0x0b, 0x0c] {
            assert!(t.is_ws(b as u32));
        }
    }

    #[test]
    fn unicode_samples() {
        let t = tables();
        assert!(t.is_han('中' as u32) && t.is_letter('中' as u32));
        assert!(!t.is_han('a' as u32) && !t.is_han('あ' as u32));
        assert!(t.is_letter('é' as u32) && t.is_lgroup('é' as u32) && !t.is_ugroup('é' as u32));
        assert!(t.is_ugroup('É' as u32) && !t.is_lgroup('É' as u32));
        // CJK / Hebrew (Lo) are in BOTH groups — the source of the case-split subtlety.
        assert!(t.is_ugroup('中' as u32) && t.is_lgroup('中' as u32));
        assert!(t.is_ugroup('\u{05e2}' as u32) && t.is_lgroup('\u{05e2}' as u32)); // ע
        // combining acute (M) is in both case groups
        assert!(t.is_ugroup('\u{0301}' as u32) && t.is_lgroup('\u{0301}' as u32));
        assert!(t.is_ws('\u{3000}' as u32) && t.is_ws('\u{00a0}' as u32)); // ideographic space, nbsp
        assert!(t.is_number('٣' as u32)); // arabic-indic digit three
        assert!(t.is_ugroup('ǅ' as u32) && !t.is_lgroup('ǅ' as u32)); // titlecase letter (Lt)
    }
}
