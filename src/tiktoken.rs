//! Support for the `tiktoken` model format (OpenAI's `.tiktoken` / `tiktoken.model`
//! files, e.g. `cl100k_base`, `o200k_base`, or the `tiktoken.model` shipped by
//! models such as Kimi-K2).
//!
//! A tiktoken model file is a list of `base64(token_bytes) rank` lines, e.g.
//!
//! ```text
//! IQ== 0
//! Ig== 1
//! ...
//! ```
//!
//! This file carries **only** the mergeable ranks (a byte-level BPE vocabulary
//! where the rank doubles as both the token id and the merge priority). It does
//! *not* contain the pre-tokenization regex (`pat_str`) or the special tokens —
//! those live in companion code (tiktoken's registry, `tokenization_*.py`, …),
//! so they must be supplied separately via [`TiktokenConfig`].
//!
//! The ranks are converted into the same internal representation used for
//! HuggingFace byte-level BPE tokenizers: each token's bytes are mapped through
//! the GPT-2 byte-to-unicode table and the merge list is regenerated from the
//! ranks. The resulting [`Tokenizer`](crate::Tokenizer) runs on the existing
//! fused byte-level BPE path.

use std::collections::HashMap;

use crate::Error;

/// The pre-tokenization regex used by OpenAI's `cl100k_base` encoding
/// (GPT-3.5 / GPT-4).
///
/// This is the canonical (non-possessive) form, which is equivalent to the
/// pattern tiktoken ships but compiles on the regex engines used internally.
pub const CL100K_BASE_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// Qwen2/Qwen3's pre-tokenization regex: [`CL100K_BASE_PATTERN`] with single-digit
/// numbers (`\p{N}` instead of `\p{N}{1,3}`).
pub const QWEN2_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// Qwen3.5 / Qwen3.8's pre-tokenization regex: [`QWEN2_PATTERN`] with marks in
/// the letter run (`[\p{L}\p{M}]+`) and out of the punctuation class.
pub const QWEN35_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// Mistral's "tekken" pre-tokenization regex (Mistral-Nemo, Nemotron):
/// [`O200K_BASE_PATTERN`] without the contraction suffix and with single-digit
/// numbers (`\p{N}` instead of `\p{N}{1,3}`).
pub const TEKKEN_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*",
    r"|\p{N}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// The pre-tokenization regex used by OpenAI's `o200k_base` encoding
/// (GPT-4o and later).
pub const O200K_BASE_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// The pre-tokenization regex used by Moonshot's Kimi models (K2 and later),
/// verbatim from the `pat_str` in the `tokenization_kimi.py` they ship.
///
/// It differs from [`O200K_BASE_PATTERN`] in two ways: a leading `[\p{Han}]+`
/// alternative makes Han runs their own pretokens (with Han then excluded from
/// the letter classes via `&&[^\p{Han}]`), and the trailing class of a
/// punctuation run is `[\r\n]*` rather than o200k's `[\r\n/]*`, so o200k also
/// absorbs a `/` there. Note `/` itself is matched by `[^\s\p{L}\p{N}]+` under
/// both patterns; only the trailing position differs.
///
/// This must stay **byte-identical** to the model's own `pat_str`:
/// [`crate::pre_tokenizers::scan`] recognizes the scanner fast path by exact
/// string comparison, so any drift silently falls back to the regex engine.
pub const KIMI_PATTERN: &str = concat!(
    r"[\p{Han}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// Number of consecutive ids Kimi reserves for special tokens, starting at the
/// end of the mergeable ranks (`num_reserved_special_tokens` in
/// `tokenization_kimi.py`).
pub const KIMI_RESERVED_SPECIAL_TOKENS: u32 = 256;

/// DeepSeek (V3 / V3.2) pre-tokenization is a *sequence* of three `Isolated`
/// `Split`s applied in order, then `ByteLevel`. Unlike the single-regex tiktoken
/// families, no one pattern describes it; the scanner
/// ([`crate::pre_tokenizers::scan::ScanKind::DeepSeek`]) reproduces the combined
/// effect of the three in one pass. These three constants are byte-identical to
/// the `Regex` sources in DeepSeek's `tokenizer.json`, matched exactly by
/// [`crate::pre_tokenizers::scan::recognize_deepseek`]; any drift silently falls
/// back to the regex engine.
///
/// Split 1: isolate runs of 1–3 digits.
pub const DEEPSEEK_SPLIT1_PATTERN: &str = r"\p{N}{1,3}";
/// Split 2: isolate runs of CJK Unified Ideographs / Hiragana / Katakana. Note
/// this is a *specific* codepoint range, narrower than `\p{Han}`.
pub const DEEPSEEK_SPLIT2_PATTERN: &str = "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+";
/// Split 3: a GPT-like word/number/punct/whitespace pattern. `[\r\n]` here are
/// literal CR/LF characters in the source (not backslash escapes), matching the
/// model file. Applied *after* splits 1 and 2, so its `[\p{L}\p{M}]+` runs never
/// cross a digit or CJK boundary those splits already cut.
pub const DEEPSEEK_SPLIT3_PATTERN: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

/// A recognized family of tiktoken-based model repository.
///
/// A `tiktoken.model` file carries only mergeable ranks, so loading one from a
/// model repository requires knowing which pre-tokenization pattern and
/// special-token layout it belongs to. That cannot be read from the ranks file;
/// it is inferred from `tokenizer_config.json`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TiktokenFamily {
    /// Moonshot Kimi (K2 and later). See [`TiktokenConfig::kimi`].
    Kimi,
}

impl TiktokenFamily {
    /// Infer the family from a `tokenizer_config.json`.
    ///
    /// `tokenizer_class` alone is not sufficient: Kimi declares the generic
    /// `"TikTokenTokenizer"`, which another vendor could reuse with a different
    /// pattern. The discriminator is the module in `auto_map.AutoTokenizer`,
    /// which for Kimi names its own `tokenization_kimi` (as in
    /// `"tokenization_kimi.TikTokenTokenizer"`).
    #[must_use]
    pub fn detect(tokenizer_class: Option<&str>, auto_map_tokenizer: Option<&str>) -> Option<Self> {
        let module = auto_map_tokenizer?.split('.').next()?;
        match (module, tokenizer_class) {
            ("tokenization_kimi", Some("TikTokenTokenizer")) => Some(Self::Kimi),
            _ => None,
        }
    }
}

/// Everything needed to turn a set of mergeable ranks into a working
/// tokenizer: the pre-tokenization regex and the special tokens.
///
/// Use [`TiktokenConfig::cl100k_base`] / [`TiktokenConfig::o200k_base`] for the
/// standard OpenAI encodings, [`TiktokenConfig::kimi`] for Kimi models, or
/// [`TiktokenConfig::new`] to supply a custom pattern and special-token table
/// (e.g. when loading a model's own `tiktoken.model`).
#[derive(Clone, Debug)]
pub struct TiktokenConfig {
    /// The pre-tokenization regex (`pat_str`). Applied with `Isolated`
    /// behavior, matching tiktoken's `regex.findall` over the input.
    pub pattern: String,
    /// Special tokens as `(content, id)` pairs. These are matched literally
    /// before the BPE model (like HuggingFace added tokens) and are marked
    /// special, so they can be skipped on decode.
    pub special_tokens: Vec<(String, u32)>,
}

impl TiktokenConfig {
    /// Build a config from a pattern and an explicit list of special tokens.
    pub fn new(pattern: impl Into<String>, special_tokens: Vec<(String, u32)>) -> Self {
        Self {
            pattern: pattern.into(),
            special_tokens,
        }
    }

    /// Config for OpenAI's `cl100k_base` encoding (GPT-3.5 / GPT-4).
    pub fn cl100k_base() -> Self {
        Self::new(
            CL100K_BASE_PATTERN,
            vec![
                ("<|endoftext|>".into(), 100257),
                ("<|fim_prefix|>".into(), 100258),
                ("<|fim_middle|>".into(), 100259),
                ("<|fim_suffix|>".into(), 100260),
                ("<|endofprompt|>".into(), 100276),
            ],
        )
    }

    /// Config for OpenAI's `o200k_base` encoding (GPT-4o and later).
    pub fn o200k_base() -> Self {
        Self::new(
            O200K_BASE_PATTERN,
            vec![
                ("<|endoftext|>".into(), 199999),
                ("<|endofprompt|>".into(), 200018),
            ],
        )
    }

    /// Config for Moonshot's Kimi models (K2 and later).
    ///
    /// Unlike the OpenAI encodings, Kimi's special-token table is not a fixed
    /// list: it is derived from the vocabulary size. Kimi reserves
    /// [`KIMI_RESERVED_SPECIAL_TOKENS`] ids immediately after the mergeable
    /// ranks, names the ones its `tokenizer_config.json` declares, and fills the
    /// remainder with `<|reserved_token_{id}|>` placeholders. This reproduces
    /// `tokenization_kimi.py` exactly:
    ///
    /// ```text
    /// special_tokens_mapping.get(i, f"<|reserved_token_{i}|>"): i
    /// for i in range(num_base_tokens, num_base_tokens + num_reserved_special_tokens)
    /// ```
    ///
    /// `num_ranks` is the number of mergeable ranks (i.e. the first reserved id).
    /// `named` supplies the declared `(id, content)` pairs, normally taken from
    /// `tokenizer_config.json`'s `added_tokens_decoder`.
    ///
    /// Entries in `named` whose id falls outside the reserved window are
    /// **ignored**, matching the reference implementation — an id below
    /// `num_ranks` denotes a regular BPE token, and registering it as a special
    /// token would change tokenization.
    pub fn kimi(num_ranks: u32, named: impl IntoIterator<Item = (u32, String)>) -> Self {
        let end = num_ranks.saturating_add(KIMI_RESERVED_SPECIAL_TOKENS);
        let mut names: HashMap<u32, String> = named
            .into_iter()
            .filter(|(id, _)| (num_ranks..end).contains(id))
            .collect();
        let mut special_tokens = Vec::with_capacity(KIMI_RESERVED_SPECIAL_TOKENS as usize);
        for id in num_ranks..end {
            let content = names
                .remove(&id)
                .unwrap_or_else(|| format!("<|reserved_token_{id}|>"));
            special_tokens.push((content, id));
        }
        Self::new(KIMI_PATTERN, special_tokens)
    }

    /// Resolve a preset config by name (`"cl100k_base"` or `"o200k_base"`).
    ///
    /// Kimi is deliberately absent: its special-token table depends on the
    /// vocabulary size and the model's declared token names, so it cannot be
    /// produced from a name alone. Use [`TiktokenConfig::kimi`] instead.
    pub fn from_preset(name: &str) -> Option<Self> {
        match name {
            "cl100k_base" | "cl100k" => Some(Self::cl100k_base()),
            "o200k_base" | "o200k" => Some(Self::o200k_base()),
            _ => None,
        }
    }
}

/// Parse the contents of a tiktoken model file into `(token_bytes, rank)`
/// entries.
///
/// Each non-empty line must be `<base64> <rank>`; blank lines are ignored.
/// This mirrors tiktoken's own `load_tiktoken_bpe`.
pub fn parse_tiktoken_model(contents: &str) -> Result<Vec<(Vec<u8>, u32)>, Error> {
    let mut out = Vec::new();
    for (i, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split_whitespace();
        let token_b64 = fields
            .next()
            .ok_or_else(|| Error::Tiktoken(format!("line {}: empty entry", i + 1)))?;
        let rank_str = fields
            .next()
            .ok_or_else(|| Error::Tiktoken(format!("line {}: missing rank", i + 1)))?;
        let bytes = base64_decode(token_b64)
            .map_err(|e| Error::Tiktoken(format!("line {}: {e}", i + 1)))?;
        let rank: u32 = rank_str
            .parse()
            .map_err(|_| Error::Tiktoken(format!("line {}: invalid rank {rank_str:?}", i + 1)))?;
        out.push((bytes, rank));
    }
    if out.is_empty() {
        return Err(Error::Tiktoken("no token entries found".into()));
    }
    Ok(out)
}

/// Decode a standard-alphabet base64 string (padding optional).
fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    #[inline]
    fn sextet(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }

    let bytes = s.as_bytes();
    // Ignore any trailing '=' padding.
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1] == b'=' {
        end -= 1;
    }

    let mut out = Vec::with_capacity(end * 3 / 4 + 1);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for &c in &bytes[..end] {
        let v = sextet(c).ok_or_else(|| format!("invalid base64 character {:?}", c as char))?;
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Kimi pattern / preset ───────────────────────────────────────────────

    /// [`KIMI_PATTERN`] must stay byte-identical to the `pat_str` in the
    /// `tokenization_kimi.py` Moonshot ships, because
    /// [`crate::pre_tokenizers::scan::recognize`] matches it by exact string
    /// comparison — any drift silently disables the scanner fast path instead of
    /// failing. This literal is an independent transcription of that `pat_str`
    /// (from `moonshotai/Kimi-K2.6`), so an accidental edit to the `concat!`
    /// above fails here rather than degrading performance in silence.
    #[test]
    fn kimi_pattern_matches_the_model_pat_str() {
        const KIMI_PAT_STR_FROM_MODEL: &str = r#"[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;
        assert_eq!(KIMI_PATTERN, KIMI_PAT_STR_FROM_MODEL);
    }

    #[test]
    fn kimi_config_fills_the_reserved_window() {
        let config = TiktokenConfig::kimi(163_584, []);
        assert_eq!(config.pattern, KIMI_PATTERN);
        assert_eq!(config.special_tokens.len(), 256);
        // Contiguous ids across the whole reserved window, all placeholders.
        assert_eq!(
            config.special_tokens.first().unwrap(),
            &("<|reserved_token_163584|>".to_string(), 163_584)
        );
        assert_eq!(
            config.special_tokens.last().unwrap(),
            &("<|reserved_token_163839|>".to_string(), 163_839)
        );
        let ids: Vec<u32> = config.special_tokens.iter().map(|&(_, id)| id).collect();
        assert_eq!(ids, (163_584..163_840).collect::<Vec<u32>>());
    }

    /// Declared names override the placeholder at their id; every other slot
    /// keeps its placeholder. Mirrors Kimi-K2.6's real `added_tokens_decoder`.
    #[test]
    fn kimi_config_applies_declared_names() {
        let config = TiktokenConfig::kimi(
            163_584,
            [
                (163_584, "[BOS]".to_string()),
                (163_586, "<|im_end|>".to_string()),
                (163_839, "[PAD]".to_string()),
            ],
        );
        assert_eq!(config.special_tokens.len(), 256);
        let by_id: HashMap<u32, &str> = config
            .special_tokens
            .iter()
            .map(|(content, id)| (*id, content.as_str()))
            .collect();
        assert_eq!(by_id[&163_584], "[BOS]");
        assert_eq!(by_id[&163_586], "<|im_end|>");
        assert_eq!(by_id[&163_839], "[PAD]");
        // Undeclared slots keep placeholders.
        assert_eq!(by_id[&163_585], "<|reserved_token_163585|>");
        assert_eq!(by_id[&163_700], "<|reserved_token_163700|>");
    }

    /// Ids outside the reserved window are ignored: below `num_ranks` they name
    /// regular BPE tokens, and promoting one to a special token would change
    /// tokenization. Matches `tokenization_kimi.py`, which only ever looks up
    /// ids inside the window.
    #[test]
    fn kimi_config_ignores_ids_outside_the_reserved_window() {
        let config = TiktokenConfig::kimi(
            163_584,
            [
                (5, "regular-bpe-token".to_string()),
                (163_583, "last-rank".to_string()),
                (163_840, "past-the-window".to_string()),
                (999_999, "way-out".to_string()),
            ],
        );
        assert_eq!(config.special_tokens.len(), 256);
        assert!(
            config
                .special_tokens
                .iter()
                .all(|(content, _)| content.starts_with("<|reserved_token_")),
            "out-of-window names must not leak into the table"
        );
    }

    #[test]
    fn kimi_is_absent_from_from_preset() {
        // Cannot be derived from a name alone — needs the vocab size.
        assert!(TiktokenConfig::from_preset("kimi").is_none());
    }

    #[test]
    fn base64_roundtrip() {
        // "!" -> IQ==, "\"" -> Ig==, "#" -> Iw==
        assert_eq!(base64_decode("IQ==").unwrap(), b"!");
        assert_eq!(base64_decode("Ig==").unwrap(), b"\"");
        assert_eq!(base64_decode("Iw==").unwrap(), b"#");
        // Multi-byte with no padding.
        assert_eq!(base64_decode("aGVsbG8").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
    }

    #[test]
    fn base64_rejects_garbage() {
        assert!(base64_decode("not base64!!").is_err());
    }

    #[test]
    fn parse_simple_model() {
        let contents = "IQ== 0\nIg== 1\n\nIw== 2\n";
        let ranks = parse_tiktoken_model(contents).unwrap();
        assert_eq!(
            ranks,
            vec![(b"!".to_vec(), 0), (b"\"".to_vec(), 1), (b"#".to_vec(), 2)]
        );
    }

    #[test]
    fn parse_rejects_empty() {
        assert!(parse_tiktoken_model("\n\n").is_err());
    }

    #[test]
    fn parse_rejects_missing_rank() {
        assert!(parse_tiktoken_model("IQ==\n").is_err());
    }

    #[test]
    fn presets_resolve() {
        assert!(TiktokenConfig::from_preset("cl100k_base").is_some());
        assert!(TiktokenConfig::from_preset("o200k_base").is_some());
        assert!(TiktokenConfig::from_preset("nope").is_none());
    }

    /// End-to-end over a tiny hand-crafted byte-level BPE: exercises merge
    /// generation, the fused Split+ByteLevel path, special-token splitting,
    /// and ByteLevel decode round-trip — with no external fixtures.
    #[test]
    fn end_to_end_tiny_bpe() {
        use crate::Tokenizer;

        // Single bytes 0..=3, then merges "ab"=4 and "abc"=5 (ranks double as
        // ids and merge priorities, just like a real tiktoken model).
        let ranks: Vec<(Vec<u8>, u32)> = vec![
            (b"a".to_vec(), 0),
            (b"b".to_vec(), 1),
            (b"c".to_vec(), 2),
            (b" ".to_vec(), 3),
            (b"ab".to_vec(), 4),
            (b"abc".to_vec(), 5),
        ];
        let config = TiktokenConfig::new(r"\S+| +", vec![("<|end|>".to_string(), 6)]);
        let tok = Tokenizer::from_tiktoken_ranks(&ranks, config).unwrap();

        // Greedy lowest-rank merges: "abc" -> [abc], "ab" -> [ab], repeats.
        assert_eq!(tok.encode("abc").unwrap(), vec![5]);
        assert_eq!(tok.encode("ab").unwrap(), vec![4]);
        assert_eq!(tok.encode("abab").unwrap(), vec![4, 4]);
        // Pre-tokenizer isolates the whitespace run into its own piece.
        assert_eq!(tok.encode("ab c").unwrap(), vec![4, 3, 2]);

        // Special token is split out and emitted directly.
        let ids = tok.encode("ab<|end|>c").unwrap();
        assert_eq!(ids, vec![4, 6, 2]);
        assert_eq!(tok.token_to_id("<|end|>"), Some(6));
        assert_eq!(tok.id_to_token(6), Some("<|end|>"));
        assert!(tok.is_special_token(6));

        // Decode round-trips; skip_special_tokens drops the special.
        assert_eq!(tok.decode(&ids, false).unwrap(), "ab<|end|>c");
        assert_eq!(tok.decode(&ids, true).unwrap(), "abc");
    }

    /// The opt-in prefix cache returns results bit-identical to the uncached
    /// path — for exact repeats, for inputs sharing a byte prefix with a cached
    /// one (the shared-system-prompt case), and across an LRU of prefixes —
    /// exercising both the miss (store) and hit (reuse) paths on the scanner.
    #[test]
    fn prefix_cache_matches_uncached() {
        use crate::Tokenizer;

        // A byte-level vocab (all 256 single bytes) under the o200k pattern, so
        // the scanner fast path — where the prefix cache lives — engages.
        let ranks: Vec<(Vec<u8>, u32)> = (0..256u32).map(|i| (vec![i as u8], i)).collect();
        let cfg = TiktokenConfig::new(O200K_BASE_PATTERN, vec![]);

        // Two distinct shared prefixes, each well over the 8 KiB / 256-token
        // reuse thresholds, with lines that begin at hard boundaries.
        let pa = "the quick brown fox jumps over the lazy dog 12 times\n".repeat(250);
        let pb = "SECTION header: values (a, b, c) and 1, 2, 3 -- see notes.\n".repeat(250);
        let tails = [
            "",
            "x",
            " trailing",
            "\n\n  \nmore text here\n",
            "!!!weird***",
            "\t\tindented line\n",
            "'contraction test's fine",
        ];

        // Reference encodings computed with the cache OFF.
        let tok = Tokenizer::from_tiktoken_ranks(&ranks, cfg.clone()).unwrap();
        let mut inputs = Vec::new();
        for p in [&pa, &pb] {
            for t in tails {
                inputs.push(format!("{p}{t}"));
            }
        }
        let expected: Vec<Vec<u32>> = inputs.iter().map(|s| tok.encode(s).unwrap()).collect();

        // With the cache ON, every input must match — first pass primes the
        // cache (misses), second pass reuses the shared prefixes (hits).
        let mut cached = Tokenizer::from_tiktoken_ranks(&ranks, cfg).unwrap();
        cached.enable_input_cache(4);
        for _ in 0..2 {
            for (s, exp) in inputs.iter().zip(&expected) {
                assert_eq!(&cached.encode(s).unwrap(), exp);
            }
        }
    }

    /// The `from_tiktoken_str` path parses the base64 model format and builds a
    /// working tokenizer (bytes with non-ASCII values round-trip too).
    #[test]
    fn from_str_roundtrip_with_space_byte() {
        use crate::Tokenizer;

        // "YQ==" = "a", "IA==" = " ", "YSA=" = "a " (merge).
        let contents = "YQ== 0\nIA== 1\nYSA= 2\n";
        let config = TiktokenConfig::new(r"\S+| +|\S", vec![]);
        let tok = Tokenizer::from_tiktoken_str(contents, config).unwrap();
        // The space byte is stored byte-level ("Ġ") yet decodes back to " ".
        assert_eq!(tok.decode(&[1], false).unwrap(), " ");
        assert_eq!(tok.encode("a").unwrap(), vec![0]);
    }
}
