use std::collections::{HashMap, HashSet};
use std::fmt;

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};

use crate::{json_structs::AddedTokenConfig, models::BuildError};

/// A compiled set of added tokens that can be matched against input text.
///
/// The HuggingFace `tokenizer.json` format includes an `added_tokens` array of
/// literal patterns that are matched *before* the normal tokenization pipeline.
/// Matched spans are assigned their token IDs directly; unmatched spans pass
/// through normalization, pre-tokenization and the model as usual.
pub struct AddedTokens {
    daac: DoubleArrayAhoCorasick<u32>,
    /// Mapping from token ID to token content string.
    id_to_content: HashMap<u32, String>,
    /// Set of token IDs marked as special (e.g. BOS/EOS).
    special_ids: HashSet<u32>,
}

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

impl AddedTokens {
    /// Build from the `added_tokens` array in `tokenizer.json`.
    ///
    /// Returns `None` if there are no added tokens.
    pub fn from_configs(configs: &[AddedTokenConfig]) -> Result<Option<Self>, BuildError> {
        if configs.is_empty() {
            return Ok(None);
        }

        let max_id = configs.iter().map(|c| c.id).max().unwrap_or(0);
        let mut token_lens = vec![0usize; (max_id + 1) as usize];

        let mut id_to_content = HashMap::with_capacity(configs.len());
        let mut special_ids = HashSet::new();

        let patterns: Vec<(&str, u32)> = configs
            .iter()
            .map(|c| {
                token_lens[c.id as usize] = c.content.len();
                id_to_content.insert(c.id, c.content.clone());
                if c.special {
                    special_ids.insert(c.id);
                }
                (c.content.as_str(), c.id)
            })
            .collect();

        let daac = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(daachorse::MatchKind::LeftmostLongest)
            .build_with_values(patterns)
            .map_err(|e| BuildError(format!("error building added-tokens DAAC: {e}")))?;

        Ok(Some(Self {
            daac,
            id_to_content,
            special_ids,
        }))
    }

    /// Look up the string content of an added token by ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_content.get(&id).map(String::as_str)
    }

    /// Check if a token ID is a special added token.
    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    /// Return the number of added tokens.
    pub fn len(&self) -> usize {
        self.id_to_content.len()
    }

    /// Split `input` into segments: spans matching added tokens and spans of
    /// regular text.
    ///
    /// Added tokens are matched leftmost-longest. Non-overlapping matches are
    /// emitted as [`Segment::Token`]; the gaps between them as
    /// [`Segment::Text`].
    pub fn split<'a>(&self, input: &'a str) -> Vec<Segment<'a>> {
        let mut segments = Vec::new();
        let mut prev_end = 0;

        for m in self.daac.leftmost_find_iter(input) {
            if m.start() > prev_end {
                segments.push(Segment::Text(&input[prev_end..m.start()]));
            }
            segments.push(Segment::Token(m.value()));
            prev_end = m.end();
        }

        if prev_end < input.len() {
            segments.push(Segment::Text(&input[prev_end..]));
        }

        segments
    }
}

impl fmt::Debug for AddedTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AddedTokens").finish_non_exhaustive()
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
}
