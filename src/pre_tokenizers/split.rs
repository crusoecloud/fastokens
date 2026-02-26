use fancy_regex::Regex;
use serde::Deserialize;

use crate::pre_tokenized::{PreTokenizedString, Split as PtSplit};

use super::Error;

/// A pattern for a Split pre-tokenizer: either a literal string or a regex.
#[derive(Clone, Debug, Deserialize)]
pub enum Pattern {
    /// Literal string match (will be regex-escaped).
    String(String),
    /// Regular expression (used as-is).
    Regex(String),
}

impl Pattern {
    /// Compile into a [`fancy_regex::Regex`].
    fn compile(&self) -> Result<Regex, fancy_regex::Error> {
        match self {
            Self::String(s) => Ok(Regex::new(&fancy_regex::escape(s))?),
            Self::Regex(r) => Ok(Regex::new(r)?),
        }
    }
}

/// How matched delimiters are handled in the output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, strum::EnumString)]
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
    regex: Regex,
    behavior: SplitBehavior,
    invert: bool,
}

impl TryFrom<SplitRaw> for Split {
    type Error = fancy_regex::Error;

    fn try_from(raw: SplitRaw) -> Result<Self, fancy_regex::Error> {
        let regex = raw.pattern.compile()?;
        Ok(Self {
            regex,
            behavior: raw.behavior,
            invert: raw.invert,
        })
    }
}

impl Split {
    /// Build a [`Split`] from raw JSON fields.
    ///
    /// `pattern` must be a JSON object with either a `"String"` key (literal,
    /// will be regex-escaped) or a `"Regex"` key (used as-is).
    #[cfg(test)]
    fn from_config(
        pattern: serde_json::Value,
        behavior: &str,
        invert: bool,
    ) -> Result<Self, serde_json::Error> {
        serde_json::from_value(serde_json::json!({
            "pattern": pattern,
            "behavior": behavior,
            "invert": invert,
        }))
    }

    /// Refine the splits of a [`PreTokenizedString`] in place.
    ///
    /// Since Split only re-slices text (no content transformation), this is
    /// zero-copy: the buffer stays unchanged and only the split ranges are
    /// replaced.
    pub fn pre_tokenize(&self, pts: &mut PreTokenizedString) -> Result<(), Error> {
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
        let mut segments = Vec::new();
        let mut prev_end = 0;

        for m in self.regex.find_iter(input) {
            let m = m?;
            // Skip zero-width matches (e.g. from lookaheads).
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    // ── Behavior tests ──────────────────────────────────

    #[test]
    fn split_removed() {
        let s = Split::from_config(json!({"String": "-"}), "Removed", false).unwrap();
        assert_eq!(
            s.split("the-final--countdown").unwrap(),
            vec!["the", "final", "countdown"],
        );
    }

    #[test]
    fn split_isolated() {
        let s = Split::from_config(json!({"String": "-"}), "Isolated", false).unwrap();
        assert_eq!(
            s.split("the-final--countdown").unwrap(),
            vec!["the", "-", "final", "-", "-", "countdown"],
        );
    }

    #[test]
    fn split_merged_with_previous() {
        let s = Split::from_config(json!({"String": "-"}), "MergedWithPrevious", false).unwrap();
        assert_eq!(
            s.split("the-final--countdown").unwrap(),
            vec!["the-", "final-", "-", "countdown"],
        );
    }

    #[test]
    fn split_merged_with_next() {
        let s = Split::from_config(json!({"String": "-"}), "MergedWithNext", false).unwrap();
        assert_eq!(
            s.split("the-final--countdown").unwrap(),
            vec!["the", "-final", "-", "-countdown"],
        );
    }

    #[test]
    fn split_contiguous() {
        let s = Split::from_config(json!({"String": "-"}), "Contiguous", false).unwrap();
        assert_eq!(
            s.split("the-final--countdown").unwrap(),
            vec!["the", "-", "final", "--", "countdown"],
        );
    }

    // ── Invert tests ────────────────────────────────────

    #[test]
    fn split_invert_removed() {
        let s = Split::from_config(json!({"Regex": "\\d+"}), "Removed", true).unwrap();
        assert_eq!(s.split("abc123def456").unwrap(), vec!["123", "456"]);
    }

    #[test]
    fn split_invert_isolated() {
        let s = Split::from_config(json!({"Regex": "\\d+"}), "Isolated", true).unwrap();
        assert_eq!(
            s.split("abc123def456").unwrap(),
            vec!["abc", "123", "def", "456"],
        );
    }

    // ── Edge cases ──────────────────────────────────────

    #[test]
    fn split_empty_input() {
        let s = Split::from_config(json!({"String": "-"}), "Isolated", false).unwrap();
        assert!(s.split("").unwrap().is_empty());
    }

    #[test]
    fn split_no_matches() {
        let s = Split::from_config(json!({"String": "-"}), "Isolated", false).unwrap();
        assert_eq!(s.split("hello world").unwrap(), vec!["hello world"]);
    }

    #[test]
    fn split_all_delimiters() {
        let s = Split::from_config(json!({"String": "-"}), "Removed", false).unwrap();
        assert!(s.split("---").unwrap().is_empty());
    }

    #[test]
    fn split_delimiter_at_start() {
        let s = Split::from_config(json!({"String": "-"}), "MergedWithPrevious", false).unwrap();
        assert_eq!(s.split("-hello").unwrap(), vec!["-", "hello"]);
    }

    #[test]
    fn split_delimiter_at_end() {
        let s = Split::from_config(json!({"String": "-"}), "MergedWithNext", false).unwrap();
        assert_eq!(s.split("hello-").unwrap(), vec!["hello", "-"]);
    }

    #[test]
    fn split_default_behavior() {
        let s = Split::from_config(json!({"String": " "}), "Isolated", false).unwrap();
        assert_eq!(s.split("a b c").unwrap(), vec!["a", " ", "b", " ", "c"]);
    }

    #[test]
    fn split_string_pattern_not_treated_as_regex() {
        let s = Split::from_config(json!({"String": "[a]"}), "Isolated", false).unwrap();
        // "[a]" is literal, not a character class
        assert_eq!(s.split("a[a]b").unwrap(), vec!["a", "[a]", "b"]);
    }

    #[test]
    fn split_regex_whitespace() {
        let s = Split::from_config(json!({"Regex": "\\s+"}), "Isolated", false).unwrap();
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
        Split::from_config(json!({"Foo": "bar"}), "Isolated", false).unwrap_err();
    }

    #[test]
    fn error_bad_regex() {
        Split::from_config(json!({"Regex": "(unclosed"}), "Isolated", false).unwrap_err();
    }

    #[test]
    fn error_unknown_behavior() {
        Split::from_config(json!({"String": "-"}), "Foobar", false).unwrap_err();
    }
}
