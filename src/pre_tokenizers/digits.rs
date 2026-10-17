//! HuggingFace's `Digits` pre-tokenizer: splits numbers from the surrounding
//! text.
//!
//! A "digit" is any `char::is_numeric` codepoint (general category `Nd`, `Nl` or
//! `No`), exactly the predicate HF `tokenizers` splits on. With
//! `individual_digits` every such char is its own split (`"a123"` → `a`,`1`,`2`,
//! `3`); without it, each maximal run of them is (`"a123"` → `a`,`123`).
//!
//! Note this is *not* the same set as a regex `\p{N}`: the regex engines' Unicode
//! tables can lag the standard library's, so a `Split` on `\p{N}` is not an exact
//! stand-in for this step.

use super::Error;
use crate::pre_tokenized::{PreTokenizedString, Split as PtSplit};

/// A compiled `Digits` pre-tokenizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Digits {
    individual_digits: bool,
}

impl Digits {
    pub fn new(individual_digits: bool) -> Self {
        Self { individual_digits }
    }

    /// Whether each digit is split off on its own (vs. runs of digits).
    pub fn individual_digits(&self) -> bool {
        self.individual_digits
    }

    /// Refine the splits of `pts` in place. Added-token splits pass through.
    pub fn pre_tokenize(&self, pts: &mut PreTokenizedString) -> Result<(), Error> {
        let mut new_splits = Vec::with_capacity(pts.splits().len() + 8);
        for split in pts.splits() {
            if split.token_id.is_some() {
                new_splits.push(split.clone());
                continue;
            }
            let base = split.range.start;
            self.split_ranges(pts.split_text(split), |s, e| {
                new_splits.push(PtSplit {
                    range: (base + s)..(base + e),
                    token_id: None,
                });
                Ok::<(), std::convert::Infallible>(())
            })
            .unwrap_or_else(|never| match never {});
        }
        pts.refine_splits(new_splits);
        Ok(())
    }

    /// Call `emit(start, end)` for each non-empty piece of `text`, in order, with
    /// byte offsets relative to `text`. The pieces cover `text` exactly.
    #[inline]
    pub(crate) fn split_ranges<E, F>(&self, text: &str, mut emit: F) -> Result<(), E>
    where
        F: FnMut(usize, usize) -> Result<(), E>,
    {
        let b = text.as_bytes();
        // Common case: no ASCII digit and no non-ASCII byte, so nothing to split.
        if !b.iter().any(|&c| c.is_ascii_digit() || c >= 0x80) {
            if !b.is_empty() {
                emit(0, b.len())?;
            }
            return Ok(());
        }
        // `start` opens the current piece; `in_num` says whether it is a run of
        // digits (only tracked for the contiguous mode).
        let mut start = 0;
        let mut in_num = false;
        for (i, c) in text.char_indices() {
            if c.is_numeric() {
                if self.individual_digits {
                    if start < i {
                        emit(start, i)?;
                    }
                    start = i + c.len_utf8();
                    emit(i, start)?;
                } else if !in_num {
                    if start < i {
                        emit(start, i)?;
                    }
                    start = i;
                    in_num = true;
                }
            } else if in_num {
                emit(start, i)?;
                start = i;
                in_num = false;
            }
        }
        if start < b.len() {
            emit(start, b.len())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(individual: bool, text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        Digits::new(individual)
            .split_ranges(text, |s, e| {
                out.push(&text[s..e]);
                Ok::<(), ()>(())
            })
            .unwrap();
        out
    }

    #[test]
    fn individual_digits() {
        assert_eq!(pieces(true, "a123b"), ["a", "1", "2", "3", "b"]);
        assert_eq!(pieces(true, "42"), ["4", "2"]);
        assert_eq!(pieces(true, "x"), ["x"]);
        assert_eq!(pieces(true, ""), Vec::<&str>::new());
        // Non-ASCII numerics: Arabic-Indic, fullwidth, circled, Roman numeral.
        assert_eq!(pieces(true, "٣４①Ⅻ"), ["٣", "４", "①", "Ⅻ"]);
        assert_eq!(pieces(true, "价格12元"), ["价格", "1", "2", "元"]);
    }

    #[test]
    fn contiguous_digits() {
        assert_eq!(pieces(false, "a123b45"), ["a", "123", "b", "45"]);
        assert_eq!(pieces(false, "12 34"), ["12", " ", "34"]);
        assert_eq!(pieces(false, "٣４①x"), ["٣４①", "x"]);
        assert_eq!(pieces(false, "abc"), ["abc"]);
    }

    #[test]
    fn added_token_splits_pass_through() {
        let mut pts = PreTokenizedString::new(
            "a1<s>23".to_string(),
            vec![
                PtSplit {
                    range: 0..2,
                    token_id: None,
                },
                PtSplit {
                    range: 2..5,
                    token_id: Some(7),
                },
                PtSplit {
                    range: 5..7,
                    token_id: None,
                },
            ],
        );
        Digits::new(true).pre_tokenize(&mut pts).unwrap();
        let got: Vec<(&str, Option<u32>)> = pts
            .splits()
            .iter()
            .map(|s| (pts.split_text(s), s.token_id))
            .collect();
        assert_eq!(
            got,
            [
                ("a", None),
                ("1", None),
                ("<s>", Some(7)),
                ("2", None),
                ("3", None)
            ]
        );
    }
}
