use std::{
    collections::{BTreeMap, HashMap},
    fmt,
};

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};
use quick_cache::sync::Cache;
use serde::Deserialize;
use serde_json::Value;

use super::{BuildError, TokenizeError};

type TokenId = u32;
type MergeMap = HashMap<(u32, u32), (u32, u32)>;
type Vocab = HashMap<String, u32>;

const INVALID_TOKEN: u32 = u32::MAX;
const DEFAULT_CACHE_CAPACITY: usize = 200_000;

/// Raw deserialization helper for [`Bpe`].
#[derive(Deserialize)]
struct RawBpe {
    #[serde(default)]
    vocab: Vocab,
    /// Each element is either `"a b"` (string) or `["a","b"]` (two-element
    /// array) depending on the tokenizer.
    #[serde(default)]
    merges: Vec<Value>,
    #[allow(dead_code)]
    dropout: Option<f64>,
    #[allow(dead_code)]
    unk_token: Option<String>,
    #[allow(dead_code)]
    continuing_subword_prefix: Option<String>,
    #[allow(dead_code)]
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    fuse_unk: bool,
    #[serde(default)]
    #[allow(dead_code)]
    byte_fallback: bool,
}

/// A compiled BPE tokenization model.
///
/// Constructed from a vocabulary and merge list (typically via
/// [`ModelConfig::Bpe`]). Uses a forward DP algorithm backed by an Aho-Corasick
/// automaton for efficient tokenization.
///
/// [`ModelConfig::Bpe`]: crate::ModelConfig::Bpe
#[derive(Deserialize)]
#[serde(try_from = "RawBpe")]
pub struct Bpe {
    daac: DoubleArrayAhoCorasick<TokenId>,
    merge_map: MergeMap,
    unmerge_map: Vec<(TokenId, TokenId)>,
    next_prefix_map: Vec<Option<TokenId>>,
    /// Pre-computed token lengths for O(1) lookup instead of HashMap access.
    token_lens: Vec<usize>,
    cache: Option<Cache<String, Vec<u32>>>,
    cache_capacity: usize,
}

impl TryFrom<RawBpe> for Bpe {
    type Error = BuildError;

    fn try_from(raw: RawBpe) -> Result<Self, BuildError> {
        let merge_map = parse_merges(&raw.vocab, &raw.merges)?;
        Self::new(&raw.vocab, merge_map, DEFAULT_CACHE_CAPACITY)
    }
}

/// Result of computing a BPE encoding decomposition.
enum Decomposition {
    /// Successfully reduced to two tokens: the last merge step.
    Pair(TokenId, TokenId),
    /// Text contains characters not in the base vocabulary (special tokens like
    /// `<unk>`). Not a BPE orphan.
    CharsNotInVocab,
    /// All characters are in vocab but BPE gets stuck at >2 tokens. This token
    /// is never produced by BPE and should be excluded from matching.
    Stuck,
}

/// Compute the BPE encoding decomposition for a token by simulating bottom-up
/// BPE on its text. Returns `Pair(left, right)` such that the last merge step
/// in BPE(text) is `left + right → token`.
fn encoding_decomposition(text: &str, vocab: &Vocab, merge_map: &MergeMap) -> Decomposition {
    // Start with single characters
    let mut tokens: Vec<TokenId> = Vec::new();
    for ch in text.chars() {
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        match vocab.get(s) {
            Some(&tid) => tokens.push(tid),
            None => return Decomposition::CharsNotInVocab,
        }
    }

    if tokens.len() < 2 {
        return Decomposition::CharsNotInVocab;
    }

    // Greedily apply the lowest-rank merge until 2 tokens remain.
    while tokens.len() > 2 {
        let mut best_rank = u32::MAX;
        let mut best_pos = usize::MAX;
        let mut best_new = 0;
        for i in 0..tokens.len() - 1 {
            let pair = (tokens[i], tokens[i + 1]);
            if let Some(&(rank, new_id)) = merge_map.get(&pair)
                && rank < best_rank
            {
                best_rank = rank;
                best_pos = i;
                best_new = new_id;
            }
        }
        if best_pos == usize::MAX {
            return Decomposition::Stuck;
        }
        tokens[best_pos] = best_new;
        tokens.remove(best_pos + 1);
    }

    Decomposition::Pair(tokens[0], tokens[1])
}

fn parse_merges(vocab: &Vocab, merges: &[Value]) -> Result<MergeMap, BuildError> {
    let mut merge_map = MergeMap::new();
    for (rank, entry) in merges.iter().enumerate() {
        let (left, right) = parse_merge_entry(entry)?;
        let &left_id = vocab
            .get(left)
            .ok_or_else(|| BuildError(format!("merge token not in vocab: {left:?}")))?;
        let &right_id = vocab
            .get(right)
            .ok_or_else(|| BuildError(format!("merge token not in vocab: {right:?}")))?;
        let merged = format!("{left}{right}");
        let &merged_id = vocab
            .get(&merged)
            .ok_or_else(|| BuildError(format!("merged token not in vocab: {merged:?}")))?;
        merge_map.insert((left_id, right_id), (rank as u32, merged_id));
    }
    Ok(merge_map)
}

fn parse_merge_entry(entry: &Value) -> Result<(&str, &str), BuildError> {
    match entry {
        Value::String(s) => {
            let (left, right) = s
                .split_once(' ')
                .ok_or_else(|| BuildError(format!("invalid merge entry (no space): {s:?}")))?;
            Ok((left, right))
        }
        Value::Array(arr) if arr.len() == 2 => {
            let left = arr[0]
                .as_str()
                .ok_or_else(|| BuildError(format!("merge element not a string: {:?}", arr[0])))?;
            let right = arr[1]
                .as_str()
                .ok_or_else(|| BuildError(format!("merge element not a string: {:?}", arr[1])))?;
            Ok((left, right))
        }
        _ => Err(BuildError(format!(
            "unrecognized merge entry format: {entry:?}"
        ))),
    }
}

impl Bpe {
    /// Sets up a BPE encoder.
    pub fn new(
        vocab: &Vocab,
        merge_map: MergeMap,
        cache_capacity: usize,
    ) -> Result<Self, BuildError> {
        if vocab.is_empty() {
            return Err(BuildError(
                "cannot build Bpe with empty vocabulary".to_string(),
            ));
        }

        let vocab_r: BTreeMap<u32, &str> = vocab.iter().map(|(s, &id)| (id, s.as_str())).collect();

        // Safe: vocab is not empty
        let max_token = vocab_r.keys().max().copied().unwrap();

        // Compute the unmerge_map using BPE encoding decompositions. For each
        // non-base token, we simulate BPE on its text to find the last merge
        // step. This gives the correct decomposition for
        // is_compatible_token_pair, even when the merge list contains multiple
        // binary splits per token.
        //
        // Also track "orphan" tokens: multi-char tokens whose characters are
        // all in the vocab but BPE gets stuck at
        // >2 tokens. These are never produced by BPE and must be excluded from
        // DAAC matching.
        let mut unmerge_map = (0..=max_token).map(|t| (t, t)).collect::<Vec<_>>();
        let mut is_orphan = vec![false; (max_token + 1) as usize];
        for (&tid, text) in &vocab_r {
            if text.chars().count() < 2 {
                continue; // base token
            }
            match encoding_decomposition(text, vocab, &merge_map) {
                Decomposition::Pair(left, right) => {
                    unmerge_map[tid as usize] = (left, right);
                }
                Decomposition::Stuck => {
                    is_orphan[tid as usize] = true;
                }
                Decomposition::CharsNotInVocab => {}
            }
        }

        let daac = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(daachorse::MatchKind::LeftmostLongest)
            .build_with_values(vocab_r.iter().filter_map(|(&token, pattern)| {
                (!is_orphan[token as usize]).then_some((pattern, token))
            }))
            .map_err(|e| BuildError(format!("error building DAAC: {e}")))?;

        let token_lens = (0..=max_token)
            .map(|t| {
                Ok(vocab_r
                    .get(&t)
                    .ok_or_else(|| {
                        BuildError(format!("non-contiguous tokens - token {t} is missing"))
                    })?
                    .len())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let next_prefix_map = (0..=max_token)
            .map(|token| {
                // The token_lens construction already checked that these all
                // exist, and would have returned an error if it didn't.
                let token_str = vocab_r
                    .get(&token)
                    .expect("non-contiguous tokens despite having already checked this");
                let (last_char_start, _) = token_str.char_indices().next_back()?;
                if last_char_start == 0 {
                    return None;
                }
                Some(
                    daac.leftmost_find_iter(&token_str[..last_char_start])
                        .next()?
                        .value(),
                )
            })
            .collect();

        Ok(Self {
            daac,
            merge_map,
            unmerge_map,
            next_prefix_map,
            token_lens,
            cache: (cache_capacity > 0).then(|| Cache::new(cache_capacity)),
            cache_capacity,
        })
    }

    fn cache(&self) -> Option<&Cache<String, Vec<u32>>> {
        self.cache.as_ref()
    }

    /// Checks if `(t1, t2)` could have been output in that order as part of a
    /// valid encoding.
    ///
    /// # Panics
    ///
    /// Panics if `t1` or `t2` are not valid tokens in the vocabulary.
    pub fn is_compatible_token_pair(&self, mut t1: TokenId, mut t2: TokenId) -> bool {
        if t1 == INVALID_TOKEN {
            return false;
        }

        let mut limit = u32::MAX;
        loop {
            if let Some((_, t)) = self.merge_map.get(&(t1, t2))
                && *t < limit
            {
                return false;
            }

            if t1 > t2 {
                limit = t1;
                t1 = self.unmerge_map[t1 as usize].1;
                if t1 == limit {
                    limit = t2 + 1;
                    t2 = self.unmerge_map[t2 as usize].0;
                    if t2 + 1 == limit {
                        return true;
                    }
                }
            } else {
                limit = t2 + 1;
                t2 = self.unmerge_map[t2 as usize].0;
                if t2 + 1 == limit {
                    limit = t1;
                    t1 = self.unmerge_map[t1 as usize].1;
                    if t1 == limit {
                        return true;
                    }
                }
            }
        }
    }

    fn next_match(&self, input: &str) -> Option<TokenId> {
        let m = self.daac.leftmost_find_iter(input).next()?;
        // Verify the match starts at the beginning of the slice. If the
        // vocabulary doesn't cover the first byte, the DAAC may return a match
        // starting later — which would give a wrong endpoint calculation.
        (m.start() == 0).then_some(m.value())
    }

    /// Tokenizes using a forward DP that tracks the last token of each prefix's
    /// BPE encoding.
    ///
    /// At each reachable byte position, all matching tokens (longest to
    /// shortest) are tried. A token is accepted if it is compatible with the
    /// previous token at that position (i.e. the two would not merge under BPE
    /// rules). The first token to reach a given endpoint wins — by BPE
    /// uniqueness, all valid paths agree on the last token at every position.
    pub fn tokenize(&self, input: &str) -> Result<Vec<TokenId>, TokenizeError> {
        if input.is_empty() {
            return Ok(vec![]);
        }

        let cache = self.cache();

        if let Some(cache) = cache
            && let Some(hit) = cache.get(input)
        {
            return Ok(hit);
        }

        let n = input.len();
        // last_token[i] = last token of the BPE encoding of input[0..i].
        // INVALID_TOKEN means position i is unreachable.
        let mut last_token = vec![INVALID_TOKEN; n + 1];

        // Position 0: no predecessor, so every matching token is compatible.
        let longest = self
            .next_match(input)
            .ok_or_else(|| TokenizeError("no match at position 0".to_string()))?;
        let mut token = longest;
        loop {
            let end = self.token_lens[token as usize];
            if last_token[end] == INVALID_TOKEN {
                last_token[end] = token;
            }
            match self.next_prefix_map.get(token as usize).copied().flatten() {
                Some(shorter) => token = shorter,
                None => break,
            }
        }

        // Extend from each reachable position.
        for pos in 1..n {
            let prev = last_token[pos];
            if prev == INVALID_TOKEN {
                continue;
            }
            let Some(longest) = self.next_match(&input[pos..]) else {
                continue;
            };
            let mut token = longest;
            loop {
                let end = pos + self.token_lens[token as usize];
                if last_token[end] == INVALID_TOKEN && self.is_compatible_token_pair(prev, token) {
                    last_token[end] = token;
                }
                match self.next_prefix_map.get(token as usize).copied().flatten() {
                    Some(shorter) => token = shorter,
                    None => break,
                }
            }
        }

        if last_token[n] == INVALID_TOKEN {
            return Err(TokenizeError(format!(
                "failed to tokenize: end of input not \
                 reachable (input length {n})"
            )));
        }

        // Reconstruct by following the last_token chain backwards.
        let mut out = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let token = last_token[pos];
            out.push(token);
            pos -= self.token_lens[token as usize];
        }
        out.reverse();

        if let Some(cache) = cache {
            cache.insert(input.to_string(), out.clone());
        }

        Ok(out)
    }
}

impl Clone for Bpe {
    fn clone(&self) -> Self {
        Self {
            daac: self.daac.clone(),
            merge_map: self.merge_map.clone(),
            unmerge_map: self.unmerge_map.clone(),
            next_prefix_map: self.next_prefix_map.clone(),
            token_lens: self.token_lens.clone(),
            // Copy existence/capacity but not cache contents.
            cache: self
                .cache
                .is_some()
                .then(|| Cache::new(self.cache_capacity)),
            cache_capacity: self.cache_capacity,
        }
    }
}

impl fmt::Debug for Bpe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bpe")
            .field("vocab_size", &self.token_lens.len())
            .field("merges", &self.merge_map.len())
            .field("cache_capacity", &self.cache_capacity)
            .finish()
    }
}

impl PartialEq for Bpe {
    fn eq(&self, other: &Self) -> bool {
        self.daac == other.daac
            && self.merge_map == other.merge_map
            && self.unmerge_map == other.unmerge_map
            && self.next_prefix_map == other.next_prefix_map
            && self.token_lens == other.token_lens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json_structs::ModelConfig;

    /// Build a small test vocabulary and merge list.
    ///
    /// Base tokens: a=0, b=1, c=2, d=3
    /// Merges (in order):
    ///   "a b"  → ab=4
    ///   "c d"  → cd=5
    ///   "ab cd" → abcd=6
    fn test_bpe() -> Bpe {
        let vocab: Vocab = [
            ("a", 0),
            ("b", 1),
            ("c", 2),
            ("d", 3),
            ("ab", 4),
            ("cd", 5),
            ("abcd", 6),
        ]
        .into_iter()
        .map(|(s, id)| (s.to_string(), id))
        .collect();

        let merges: Vec<Value> = vec![
            Value::String("a b".into()),
            Value::String("c d".into()),
            Value::String("ab cd".into()),
        ];

        let merge_map = parse_merges(&vocab, &merges).unwrap();
        Bpe::new(&vocab, merge_map, 0).unwrap()
    }

    #[test]
    fn empty_input() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn single_char() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("a").unwrap(), vec![0]);
        assert_eq!(bpe.tokenize("d").unwrap(), vec![3]);
    }

    #[test]
    fn simple_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("ab").unwrap(), vec![4]);
        assert_eq!(bpe.tokenize("cd").unwrap(), vec![5]);
    }

    #[test]
    fn chained_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("abcd").unwrap(), vec![6]);
    }

    #[test]
    fn partial_merge() {
        let bpe = test_bpe();
        // "abc" → "ab" + "c" (no merge for (ab, c))
        assert_eq!(bpe.tokenize("abc").unwrap(), vec![4, 2]);
    }

    #[test]
    fn repeated_merge() {
        let bpe = test_bpe();
        assert_eq!(bpe.tokenize("abab").unwrap(), vec![4, 4]);
    }

    #[test]
    fn deserialize_from_json() {
        let json = serde_json::json!({
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": ["a b"]
        });
        let config: ModelConfig = serde_json::from_value(json).unwrap();
        assert!(matches!(config, ModelConfig::Bpe(_)));
    }

    #[test]
    fn deserialize_array_merges() {
        let json = serde_json::json!({
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": [["a", "b"]]
        });
        let config: ModelConfig = serde_json::from_value(json).unwrap();
        let ModelConfig::Bpe(bpe) = config else {
            panic!("expected Bpe variant");
        };
        assert_eq!(bpe.tokenize("ab").unwrap(), vec![2]);
    }

    #[test]
    fn cache_returns_same_result() {
        let vocab: Vocab = [("a", 0), ("b", 1), ("ab", 2)]
            .into_iter()
            .map(|(s, id)| (s.to_string(), id))
            .collect();
        let merges = vec![Value::String("a b".into())];
        let merge_map = parse_merges(&vocab, &merges).unwrap();
        let bpe = Bpe::new(&vocab, merge_map, 100).unwrap();

        let first = bpe.tokenize("ab").unwrap();
        let second = bpe.tokenize("ab").unwrap();
        assert_eq!(first, second);
        assert_eq!(first, vec![2]);
    }
}
