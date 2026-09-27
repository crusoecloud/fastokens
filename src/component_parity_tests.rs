//! Network-free parity tests: each pipeline component is built from the same
//! JSON config in both fastokens and HuggingFace `tokenizers` and compared on
//! fixed edge cases plus seeded fuzz. Also covers cross-call state (the
//! thread-local split cache) and stack safety at the component level.

use serde_json::{Value, json};
use tokenizers::{Decoder as _, Normalizer as _, OffsetReferential, OffsetType, PreTokenizer as _};

use crate::decoders::Decoder;
use crate::test_support::{
    MARKS, Rng, WHITESPACE, adversarial_text, assert_no_failures, prose, random_scalar,
};
use crate::tiktoken::QWEN2_PATTERN;
use crate::{
    CL100K_BASE_PATTERN, DecoderConfig, KIMI_PATTERN, Normalizer, NormalizerConfig,
    O200K_BASE_PATTERN, PreTokenizedString, PreTokenizer, PreTokenizerConfig, Tokenizer,
};

// ── Harness ─────────────────────────────────────────────────────────────────

fn ours_normalize(config: &Value, input: &str) -> String {
    let cfg: NormalizerConfig = serde_json::from_value(config.clone()).unwrap();
    Normalizer::from_config(cfg)
        .unwrap()
        .normalize(input)
        .into_owned()
}

fn hf_normalize(config: &Value, input: &str) -> String {
    let n: tokenizers::NormalizerWrapper = serde_json::from_value(config.clone()).unwrap();
    let mut s = tokenizers::NormalizedString::from(input);
    n.normalize(&mut s).unwrap();
    s.get().to_string()
}

fn ours_pre_tokenizer(config: &Value) -> PreTokenizer {
    let cfg: PreTokenizerConfig = serde_json::from_value(config.clone()).unwrap();
    PreTokenizer::from_config(cfg).unwrap()
}

fn ours_pieces(pt: &PreTokenizer, input: &str) -> Result<Vec<String>, String> {
    let mut pts = PreTokenizedString::from_text(input);
    pt.pre_tokenize(&mut pts).map_err(|e| e.to_string())?;
    Ok(pts
        .splits()
        .iter()
        .filter(|s| !s.range.is_empty())
        .map(|s| pts.split_text(s).to_string())
        .collect())
}

fn hf_pieces(pt: &tokenizers::PreTokenizerWrapper, input: &str) -> Vec<String> {
    let mut pts = tokenizers::PreTokenizedString::from(input);
    pt.pre_tokenize(&mut pts).unwrap();
    pts.get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(s, _, _)| s.to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Compare a pre-tokenizer config on every input; returns failure lines.
fn compare_pre_tokenizer(config: &Value, inputs: &[String]) -> Vec<String> {
    let ours = ours_pre_tokenizer(config);
    let hf: tokenizers::PreTokenizerWrapper = serde_json::from_value(config.clone()).unwrap();
    let mut failures = Vec::new();
    for input in inputs {
        let want = hf_pieces(&hf, input);
        match ours_pieces(&ours, input) {
            Ok(got) if got == want => {}
            Ok(got) => failures.push(format!(
                "{config}\n  input: {input:?}\n  ours: {got:?}\n  hf:   {want:?}"
            )),
            Err(e) => failures.push(format!("{config}\n  input: {input:?}\n  error: {e}")),
        }
    }
    failures
}

fn fuzz_inputs(seed: u64, n: usize, pieces: usize) -> Vec<String> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let k = 1 + rng.below(pieces);
            adversarial_text(&mut rng, k)
        })
        .collect()
}

// ── Normalizers ─────────────────────────────────────────────────────────────

/// Two combining marks, one from a recent Unicode version. HF leaves them in
/// input order; a normalizer with newer combining-class data reorders them,
/// which changes the bytes and so the token ids.
const MARK_PAIRS: &[&str] = &[
    "\u{6e0}\u{7fd}",
    "\u{341}\u{7fd}",
    "\u{30a}\u{7fd}",
    "\u{368}\u{7fd}",
    "\u{308}\u{7fd}",
    "\u{1aeb}\u{35b}",
    "\u{1aeb}\u{34e}",
    "\u{346}\u{8cf}",
];

#[test]
fn nfc_matches_huggingface_on_combining_marks() {
    let config = json!({"type": "NFC"});
    let mut inputs: Vec<String> = MARK_PAIRS.iter().map(|s| s.to_string()).collect();
    // Ordinary NFC work that must keep matching.
    inputs.extend(
        [
            "e\u{301}",
            "a\u{308}\u{301}",
            "\u{1100}\u{1161}\u{11a8}",
            "\u{212b}",
            "q\u{307}\u{323}",
            "\u{5d1}\u{5bc}\u{5b0}",
            "\u{645}\u{64f}\u{62d}\u{64e}\u{645}\u{651}\u{64e}\u{62f}",
        ]
        .map(String::from),
    );
    // Random mark stacks, with and without a base letter.
    let mut rng = Rng::new(0x5eed_0002);
    for _ in 0..3000 {
        let mut s = String::new();
        if rng.chance(1, 2) {
            s.push(rng.pick(&['a', 'e', 'o', 'n', 'ا', 'ב', 'क', 'ᄀ']));
        }
        for _ in 0..2 + rng.below(3) {
            s.push(rng.pick(MARKS));
        }
        inputs.push(s);
    }

    let failures: Vec<String> = inputs
        .iter()
        .filter_map(|input| {
            let (ours, hf) = (ours_normalize(&config, input), hf_normalize(&config, input));
            (ours != hf).then(|| {
                format!(
                    "  {input:?}: ours {:?} hf {:?}",
                    ours.chars().map(|c| c as u32).collect::<Vec<_>>(),
                    hf.chars().map(|c| c as u32).collect::<Vec<_>>()
                )
            })
        })
        .collect();
    assert_no_failures("NFC vs HF", &failures);
}

#[test]
fn nfc_matches_huggingface_on_random_text() {
    let config = json!({"type": "NFC"});
    let mut rng = Rng::new(0x5eed_0003);
    let failures: Vec<String> = (0..3000)
        .map(|_| {
            (0..1 + rng.below(12))
                .map(|_| random_scalar(&mut rng))
                .collect::<String>()
        })
        .filter_map(|input| {
            let (ours, hf) = (
                ours_normalize(&config, &input),
                hf_normalize(&config, &input),
            );
            (ours != hf).then(|| format!("  {input:?}: ours {ours:?} hf {hf:?}"))
        })
        .collect();
    assert_no_failures("NFC vs HF (random scalars)", &failures);
}

#[test]
fn replace_prepend_and_sequence_normalizers_match_huggingface() {
    let configs = [
        json!({"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"}),
        json!({"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "}),
        json!({"type": "Replace", "pattern": {"Regex": "(\\n)+"}, "content": "$1"}),
        json!({"type": "Replace", "pattern": {"String": "  "}, "content": ""}),
        json!({"type": "Prepend", "prepend": "\u{2581}"}),
        json!({"type": "Sequence", "normalizers": [
            {"type": "Prepend", "prepend": "\u{2581}"},
            {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
        ]}),
        json!({"type": "Sequence", "normalizers": [
            {"type": "Replace", "pattern": {"Regex": "[\\r\\n]+"}, "content": "\n"},
            {"type": "Replace", "pattern": {"String": "\n"}, "content": " \u{2581}"},
        ]}),
    ];
    let inputs = fuzz_inputs(0x5eed_0004, 400, 20);
    let mut failures = Vec::new();
    for config in &configs {
        for input in inputs
            .iter()
            .map(String::as_str)
            .chain(["", " ", "  ", "\n"])
        {
            let (ours, hf) = (ours_normalize(config, input), hf_normalize(config, input));
            if ours != hf {
                failures.push(format!("{config}\n  {input:?}: ours {ours:?} hf {hf:?}"));
            }
        }
    }
    assert_no_failures("normalizers vs HF", &failures);
}

// ── Pre-tokenizers ──────────────────────────────────────────────────────────

/// Every Split behavior, with and without `invert`, over patterns that match
/// runs, single chars, lookaheads and literal strings.
#[test]
fn split_behaviors_match_huggingface() {
    let patterns = [
        json!({"Regex": "\\s+"}),
        json!({"Regex": "\\d"}),
        json!({"Regex": "[aeiou]+"}),
        json!({"Regex": "\\s+(?!\\S)"}),
        json!({"String": " "}),
        json!({"String": "\u{2581}"}),
        json!({"String": "ab"}),
    ];
    let behaviors = [
        "Removed",
        "Isolated",
        "MergedWithPrevious",
        "MergedWithNext",
        "Contiguous",
    ];
    let mut inputs = fuzz_inputs(0x5eed_0005, 150, 16);
    inputs.extend(
        [
            "",
            " ",
            "  a  ",
            "ab ab",
            "abab",
            "1 2 3",
            "aeiou",
            "x",
            "\u{2581}\u{2581}x\u{2581}",
            "  \n  x  ",
            "a1b22c333",
        ]
        .map(String::from),
    );
    let mut failures = Vec::new();
    for pattern in &patterns {
        for behavior in behaviors {
            for invert in [false, true] {
                let config = json!({
                    "type": "Split", "pattern": pattern, "behavior": behavior, "invert": invert,
                });
                failures.extend(compare_pre_tokenizer(&config, &inputs));
            }
        }
    }
    assert_no_failures("Split behaviors vs HF", &failures);
}

/// The tokenizer-family regexes, on adversarial input and on inputs large
/// enough (≥ 16 KiB) for the parallel PCRE2 / fancy-regex matchers.
#[test]
fn split_tokenizer_patterns_match_huggingface() {
    let mut inputs = fuzz_inputs(0x5eed_0006, 300, 40);
    let mut rng = Rng::new(0x5eed_0007);
    for _ in 0..6 {
        let mut big = String::new();
        while big.len() < 48 * 1024 {
            big.push_str(&adversarial_text(&mut rng, 50));
            // Long whitespace runs so some straddle a parallel chunk boundary.
            if rng.chance(1, 3) {
                big.push_str(&rng.pick(WHITESPACE).repeat(200 + rng.below(3000)));
            }
        }
        inputs.push(big);
    }
    let mut failures = Vec::new();
    for pattern in [
        QWEN2_PATTERN,
        CL100K_BASE_PATTERN,
        O200K_BASE_PATTERN,
        KIMI_PATTERN,
    ] {
        let config = json!({
            "type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false,
        });
        failures.extend(compare_pre_tokenizer(&config, &inputs));
    }
    assert_no_failures("tokenizer patterns vs HF", &failures);
}

/// Patterns that can match the empty string must not derail the matcher on
/// multi-byte text (a zero-width match must advance by a whole character).
#[test]
fn split_empty_matching_patterns_match_huggingface() {
    let inputs: Vec<String> = ["é", "aé b", "中文 x", "😀😀", "x\u{301}y", "  é  "]
        .map(String::from)
        .into_iter()
        .chain(fuzz_inputs(0x5eed_0008, 100, 12))
        .collect();
    let mut failures = Vec::new();
    for pattern in ["x*", "\\s*", "(?=a)", "a?"] {
        let config = json!({
            "type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false,
        });
        failures.extend(compare_pre_tokenizer(&config, &inputs));
    }
    assert_no_failures("empty-matching patterns vs HF", &failures);
}

/// Sequences of splits run the generic multi-split path (not the single-split
/// PCRE2 fast path), and ByteLevel with a regex / prefix space runs the
/// non-bulk ByteLevel path.
#[test]
fn pre_tokenizer_sequences_and_byte_level_match_huggingface() {
    let configs = [
        json!({"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "\\d{1,3}"}, "behavior": "Isolated", "invert": false},
            {"type": "Split", "pattern": {"Regex": QWEN2_PATTERN}, "behavior": "Isolated", "invert": false},
        ]}),
        json!({"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithNext", "invert": false},
            {"type": "Split", "pattern": {"Regex": "\\p{P}+"}, "behavior": "Isolated", "invert": false},
        ]}),
        json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}),
        json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}),
        json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": false}),
        json!({"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN2_PATTERN}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        ]}),
    ];
    let inputs = fuzz_inputs(0x5eed_0009, 300, 30);
    let mut failures = Vec::new();
    for config in &configs {
        // ByteLevel rewrites content, so compare the normalized pieces.
        let ours = ours_pre_tokenizer(config);
        let hf: tokenizers::PreTokenizerWrapper = serde_json::from_value(config.clone()).unwrap();
        for input in &inputs {
            let mut pts = tokenizers::PreTokenizedString::from(input.as_str());
            hf.pre_tokenize(&mut pts).unwrap();
            let want: Vec<String> = pts
                .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
                .into_iter()
                .map(|(s, _, _)| s.to_string())
                .filter(|s| !s.is_empty())
                .collect();
            match ours_pieces(&ours, input) {
                Ok(got) if got == want => {}
                Ok(got) => failures.push(format!(
                    "{config}\n  input: {input:?}\n  ours: {got:?}\n  hf:   {want:?}"
                )),
                Err(e) => failures.push(format!("{config}\n  input: {input:?}\n  error: {e}")),
            }
        }
    }
    assert_no_failures("pre-tokenizer sequences / ByteLevel vs HF", &failures);
}

// ── Split incremental cache ─────────────────────────────────────────────────

fn isolated_split(pattern: &str) -> PreTokenizer {
    ours_pre_tokenizer(&json!({
        "type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false,
    }))
}

/// Pre-tokenizing B right after A (same `Split`, same thread) must give what a
/// fresh `Split` gives for B. Fuzzes the split cache with ≥ 4 KiB shared
/// prefixes (prose, whitespace runs or adversarial text, ending in whitespace
/// half the time) and short random tails, plus some long tails so the
/// post-restart suffix takes the parallel matcher.
#[test]
fn split_cache_reuse_matches_fresh_split_fuzz() {
    let mut rng = Rng::new(0x5eed_0010);
    let mut failures = Vec::new();
    for pattern in [QWEN2_PATTERN, CL100K_BASE_PATTERN] {
        // The cache is one slot keyed by instance, so running `reference`
        // between `cached` calls both resets it and computes B from scratch.
        let (cached, reference) = (isolated_split(pattern), isolated_split(pattern));
        let fresh_for = |input: &str| ours_pieces(&reference, input).unwrap();
        for _ in 0..120 {
            let mut prefix = match rng.below(3) {
                0 => prose(&mut rng, 900),
                1 => rng.pick(WHITESPACE).repeat(4200 + rng.below(2000)),
                _ => {
                    let mut s = String::new();
                    while s.len() < 4200 {
                        s.push_str(&adversarial_text(&mut rng, 40));
                    }
                    s
                }
            };
            if rng.chance(1, 2) {
                prefix.push_str(&rng.pick(WHITESPACE).repeat(1 + rng.below(6)));
            }
            let tail = |rng: &mut Rng| {
                if rng.chance(1, 10) {
                    prose(rng, 4000)
                } else {
                    {
                        let k = 1 + rng.below(6);
                        adversarial_text(rng, k)
                    }
                }
            };
            let (a, b) = (
                format!("{prefix}{}", tail(&mut rng)),
                format!("{prefix}{}", tail(&mut rng)),
            );
            ours_pieces(&cached, &a).unwrap();
            let got = ours_pieces(&cached, &b).unwrap();
            let want = fresh_for(&b);
            if got != want {
                let at = got.iter().zip(&want).position(|(x, y)| x != y).unwrap_or(0);
                failures.push(format!(
                    "  pattern {pattern:.20}… A tail {:?} B tail {:?}: first diff at piece {at}: \
                     ours {:?} fresh {:?}",
                    &a[prefix.len()..].chars().take(40).collect::<String>(),
                    &b[prefix.len()..].chars().take(40).collect::<String>(),
                    &got[at..(at + 3).min(got.len())],
                    &want[at..(at + 3).min(want.len())],
                ));
            }
        }
    }
    assert_no_failures("split cache vs fresh (fuzz)", &failures);
}

/// The cache is keyed by `Split` instance: interleaving two Splits with
/// different patterns on the same inputs must not leak matches across them.
#[test]
fn split_cache_is_not_shared_between_split_instances() {
    let prefix = "word  ".repeat(1000);
    let inputs = [
        format!("{prefix}abc"),
        format!("{prefix}  \n"),
        format!("{prefix}1234"),
    ];
    let (qwen, digits) = (isolated_split(QWEN2_PATTERN), isolated_split("\\d+|\\s+"));
    for _ in 0..3 {
        for input in &inputs {
            assert_eq!(
                ours_pieces(&qwen, input).unwrap(),
                ours_pieces(&isolated_split(QWEN2_PATTERN), input).unwrap()
            );
            assert_eq!(
                ours_pieces(&digits, input).unwrap(),
                ours_pieces(&isolated_split("\\d+|\\s+"), input).unwrap()
            );
        }
    }
}

// ── Stack safety ────────────────────────────────────────────────────────────

/// Pre-tokenize long single-character whitespace runs (and runs interrupted by
/// a word) with every regex engine on a thread with a small stack.
#[test]
fn long_whitespace_runs_do_not_overflow_a_small_stack() {
    let mut inputs = Vec::new();
    for ws in [
        " ", "\t", "\n", "\r\n", "\u{3000}", "\u{a0}", "\u{2028}", " \n",
    ] {
        inputs.push(ws.repeat(64 * 1024 / ws.len()));
        inputs.push(format!("{}word{}", ws.repeat(8192), ws.repeat(8192)));
    }
    let configs = [
        // PCRE2 (and its parallel matcher) via the fast path.
        json!({"type": "Split", "pattern": {"Regex": QWEN2_PATTERN}, "behavior": "Isolated", "invert": false}),
        // fancy-regex: Kimi's class intersection is not PCRE2-compatible.
        json!({"type": "Split", "pattern": {"Regex": KIMI_PATTERN}, "behavior": "Isolated", "invert": false}),
        // Generic path + GPT-2 regex on fancy-regex.
        json!({"type": "Split", "pattern": {"Regex": QWEN2_PATTERN}, "behavior": "MergedWithNext", "invert": false}),
        json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}),
    ];
    crate::test_support::run_in_child(
        "component_parity_tests::long_whitespace_runs_do_not_overflow_a_small_stack",
        move || {
            std::thread::Builder::new()
                .stack_size(256 * 1024)
                .spawn(move || {
                    for config in &configs {
                        let pt = ours_pre_tokenizer(config);
                        for input in &inputs {
                            let pieces = ours_pieces(&pt, input).unwrap();
                            assert!(!pieces.is_empty());
                        }
                    }
                })
                .unwrap()
                .join()
                .unwrap();
        },
    );
}

// ── Added tokens ────────────────────────────────────────────────────────────

/// Characters the synthetic vocabulary knows (one id per char, no merges, so
/// ids reveal exactly which spans became added tokens).
const ALPHABET: &str = "abcx <>|/\t\n\u{3000}\u{a0}é中";

fn char_vocab_tokenizer_json(added: &[Value], normalizer: Value) -> String {
    let mut chars: Vec<char> = ALPHABET.chars().collect();
    chars.sort_unstable();
    chars.dedup();
    let vocab: serde_json::Map<String, Value> = chars
        .iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), Value::from(i)))
        .collect();
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": normalizer,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    })
    .to_string()
}

fn added(id: u32, content: &str, special: bool, lstrip: bool, rstrip: bool) -> Value {
    json!({
        "id": id, "content": content, "single_word": false, "lstrip": lstrip,
        "rstrip": rstrip, "normalized": false, "special": special,
    })
}

/// Random text over the alphabet with added-token contents spliced in.
fn added_token_inputs(seed: u64, contents: &[&str]) -> Vec<String> {
    let alphabet: Vec<char> = ALPHABET.chars().collect();
    let mut rng = Rng::new(seed);
    let mut inputs: Vec<String> = contents.iter().map(|c| c.to_string()).collect();
    for _ in 0..600 {
        let mut s = String::new();
        for _ in 0..1 + rng.below(12) {
            if rng.chance(1, 3) {
                s.push_str(rng.pick(contents));
            } else {
                for _ in 0..1 + rng.below(3) {
                    s.push(rng.pick(&alphabet));
                }
            }
        }
        inputs.push(s);
    }
    inputs
}

fn compare_added_tokens(json: &str, inputs: &[String]) -> Vec<String> {
    use std::str::FromStr;
    let ours = Tokenizer::from_json(serde_json::from_str(json).unwrap()).unwrap();
    let mut hf = tokenizers::Tokenizer::from_str(json).unwrap();
    let mut failures = Vec::new();
    for skip_special in [false, true] {
        hf.set_encode_special_tokens(skip_special);
        let policy = if skip_special {
            crate::AddedTokenPolicy::SkipSpecial
        } else {
            crate::AddedTokenPolicy::All
        };
        for input in inputs {
            let want = hf.encode(input.as_str(), false).unwrap().get_ids().to_vec();
            let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ours.encode_with_policy(input, false, policy)
            }));
            match got {
                Ok(Ok(got)) if got == want => {}
                Ok(Ok(got)) => failures.push(format!(
                    "  skip_special={skip_special} {input:?}: ours {got:?} hf {want:?}"
                )),
                Ok(Err(e)) => {
                    failures.push(format!("  skip_special={skip_special} {input:?}: {e}"))
                }
                Err(_) => failures.push(format!("  skip_special={skip_special} {input:?}: PANIC")),
            }
        }
    }
    failures
}

/// Overlapping and prefix-sharing tokens, lstrip/rstrip over ASCII and Unicode
/// whitespace, special and non-special entries. Three distinct start bytes,
/// so matching takes the memchr-prefiltered path.
#[test]
fn added_tokens_match_huggingface_prefilter_path() {
    let base = ALPHABET
        .chars()
        .collect::<std::collections::BTreeSet<_>>()
        .len() as u32;
    let tokens = [
        added(base, "<|a|>", true, true, false),
        added(base + 1, "<|a|>b", false, false, false),
        added(base + 2, "<x>", true, false, true),
        added(base + 3, "</x>", true, true, true),
        added(base + 4, "|/|", false, true, true),
    ];
    let json = char_vocab_tokenizer_json(&tokens, Value::Null);
    let inputs = added_token_inputs(0x5eed_0011, &["<|a|>", "<|a|>b", "<x>", "</x>", "|/|"]);
    assert_no_failures(
        "added tokens (prefilter) vs HF",
        &compare_added_tokens(&json, &inputs),
    );
}

/// Same, with ≥ 4 distinct start bytes so matching takes the full-scan path,
/// and with contents that contain whitespace or multi-byte characters.
#[test]
fn added_tokens_match_huggingface_full_scan_path() {
    let base = ALPHABET
        .chars()
        .collect::<std::collections::BTreeSet<_>>()
        .len() as u32;
    let tokens = [
        added(base, "<|a|>", true, true, true),
        added(base + 1, "é x", false, false, false),
        added(base + 2, "\tb", true, true, false),
        added(base + 3, "ccc", false, false, true),
        added(base + 4, "中中", true, true, true),
        added(base + 5, "a\u{3000}", false, false, false),
    ];
    let json = char_vocab_tokenizer_json(&tokens, Value::Null);
    let inputs = added_token_inputs(
        0x5eed_0012,
        &["<|a|>", "é x", "\tb", "ccc", "中中", "a\u{3000}"],
    );
    assert_no_failures(
        "added tokens (full scan) vs HF",
        &compare_added_tokens(&json, &inputs),
    );
}

/// Added tokens with a normalizer that rewrites the surrounding text
/// (Gemma-style space → ▁): text between tokens is normalized per segment.
#[test]
fn added_tokens_with_a_rewriting_normalizer_match_huggingface() {
    let base = ALPHABET
        .chars()
        .collect::<std::collections::BTreeSet<_>>()
        .len() as u32;
    let tokens = [
        added(base, "<x>", true, false, false),
        added(base + 1, "</x>", true, false, false),
        added(base + 2, "\u{2581}", false, false, false),
    ];
    let normalizer = json!({"type": "Replace", "pattern": {"String": " "}, "content": "é"});
    let json = char_vocab_tokenizer_json(&tokens, normalizer);
    let inputs = added_token_inputs(0x5eed_0013, &["<x>", "</x>", " <x> ", "\n</x>\n"]);
    assert_no_failures(
        "added tokens + normalizer vs HF",
        &compare_added_tokens(&json, &inputs),
    );
}

/// `single_word` tokens only match when not adjacent to word characters.
#[test]
fn single_word_added_tokens_match_huggingface() {
    let base = ALPHABET
        .chars()
        .collect::<std::collections::BTreeSet<_>>()
        .len() as u32;
    let tokens = [json!({
        "id": base, "content": "abc", "single_word": true, "lstrip": false,
        "rstrip": false, "normalized": false, "special": false,
    })];
    let json = char_vocab_tokenizer_json(&tokens, Value::Null);
    let inputs = [
        "abc", "abcabc", "xabc", "abcx", "x abc x", "<abc>", "abc\n", "éabc", "中abc",
    ]
    .map(String::from);
    assert_no_failures(
        "single_word added tokens vs HF",
        &compare_added_tokens(&json, &inputs),
    );
}

/// `normalized: true` added tokens are matched against the normalized text:
/// with a normalizer that rewrites spaces, the token's (normalized) content
/// must still be found.
#[test]
fn normalized_added_tokens_match_huggingface() {
    let base = ALPHABET
        .chars()
        .collect::<std::collections::BTreeSet<_>>()
        .len() as u32;
    let tokens = [
        json!({
            "id": base, "content": "éx", "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": true, "special": false,
        }),
        json!({
            "id": base + 1, "content": "a b", "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": true, "special": false,
        }),
        added(base + 2, "<x>", true, false, false),
    ];
    let normalizer = json!({"type": "Replace", "pattern": {"String": " "}, "content": "é"});
    let json = char_vocab_tokenizer_json(&tokens, normalizer);
    let inputs = [" x", "éx", "a x b", "a b", "aéb", "<x> x", "c  x"].map(String::from);
    assert_no_failures(
        "normalized added tokens vs HF",
        &compare_added_tokens(&json, &inputs),
    );
}

// ── Decoders ────────────────────────────────────────────────────────────────

#[test]
fn byte_level_decoder_matches_huggingface() {
    compare_decoders(&[json!({
        "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true,
    })]);
}

#[test]
fn sentencepiece_style_decoders_match_huggingface() {
    compare_decoders(&[
        // Gemma 4.
        json!({"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ]}),
        // Llama 2 / SentencePiece.
        json!({"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ]}),
        json!({"type": "ByteFallback"}),
    ]);
}

fn compare_decoders(configs: &[Value]) {
    const POOL: &[&str] = &[
        "<0x41>",
        "<0xE2>",
        "<0x82>",
        "<0xAC>",
        "<0xF0>",
        "<0x9F>",
        "<0x98>",
        "<0x80>",
        "<0xC3>",
        "<0xA9>",
        "<0x0A>",
        "<0x20>",
        "<0xff>",
        "<0x4>",
        "<0xZZ>",
        "<0x411>",
        "\u{2581}hello",
        "\u{2581}",
        "\u{2581}\u{2581}",
        "Ġworld",
        "Ċ",
        "ĠĠ",
        "Ã©",
        "å",
        "ł",
        "a",
        "x",
        "xx",
        "",
        " ",
        "<unk>",
        "<|im_end|>",
        // Mixed: HF keeps a token's raw UTF-8 when any char is off-table.
        "Ġhello world",
        "Ġ\u{2581}x",
        "ĊĠ中",
    ];
    let mut rng = Rng::new(0x5eed_0014);
    let mut failures = Vec::new();
    for config in configs {
        let dcfg: DecoderConfig = serde_json::from_value(config.clone()).unwrap();
        let ours = Decoder::from_config(dcfg).unwrap();
        let hf: tokenizers::DecoderWrapper = serde_json::from_value(config.clone()).unwrap();
        for _ in 0..800 {
            let tokens: Vec<String> = (0..rng.below(10))
                .map(|_| rng.pick(POOL).to_string())
                .collect();
            let want = hf.decode(tokens.clone()).unwrap();
            match ours.decode(tokens.clone()) {
                Ok(got) if got == want => {}
                Ok(got) => {
                    failures.push(format!("{config}\n  {tokens:?}: ours {got:?} hf {want:?}"))
                }
                Err(e) => failures.push(format!("{config}\n  {tokens:?}: {e}")),
            }
        }
    }
    assert_no_failures("decoders vs HF", &failures);
}

// ── BPE model edge cases ────────────────────────────────────────────────────

fn tiny_bpe_json(vocab: Value, merges: Value, extra: Value) -> String {
    let mut model = json!({"type": "BPE", "vocab": vocab, "merges": merges});
    for (k, v) in extra.as_object().unwrap() {
        model[k] = v.clone();
    }
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
        "model": model,
    })
    .to_string()
}

fn compare_encodes(json: &str, inputs: &[&str]) -> Vec<String> {
    use std::str::FromStr;
    let ours = Tokenizer::from_json(serde_json::from_str(json).unwrap()).unwrap();
    let hf = tokenizers::Tokenizer::from_str(json).unwrap();
    inputs
        .iter()
        .filter_map(|input| {
            let want = hf.encode(*input, false).unwrap().get_ids().to_vec();
            match ours.encode(input) {
                Ok(got) if got == want => None,
                Ok(got) => Some(format!("  {input:?}: ours {got:?} hf {want:?}")),
                Err(e) => Some(format!("  {input:?}: error {e} (hf {want:?})")),
            }
        })
        .collect()
}

/// A vocab entry that the merges can never produce ("abc" when only `a b` and
/// `b c` exist) must not be emitted by the whole-token shortcut — unless
/// `ignore_merges` is set, in which case HF emits it directly.
#[test]
fn bpe_tokens_unreachable_through_merges_match_huggingface() {
    let vocab = json!({"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4, "abc": 5});
    let merges = json!(["a b", "b c"]);
    let inputs = ["abc", "abcabc", "ab", "bc", "cab", "abcab"];
    for ignore_merges in [false, true] {
        let json = tiny_bpe_json(
            vocab.clone(),
            merges.clone(),
            json!({"ignore_merges": ignore_merges}),
        );
        assert_no_failures(
            &format!("orphan vocab tokens, ignore_merges={ignore_merges}"),
            &compare_encodes(&json, &inputs),
        );
    }
}

/// A character outside the vocabulary: with an `unk_token` HF emits its id
/// (one per unknown char, or one per run with `fuse_unk`); with byte fallback
/// it emits byte tokens. fastokens must agree, and never panic.
#[test]
fn bpe_out_of_vocabulary_characters_match_huggingface() {
    let mut vocab = json!({"<unk>": 0, "a": 1, "b": 2, "ab": 3});
    let inputs = ["aXb", "中", "ab中ab", "X", "aXYb", "XY"];
    let mut failures = Vec::new();
    for fuse_unk in [false, true] {
        let json = tiny_bpe_json(
            vocab.clone(),
            json!(["a b"]),
            json!({"unk_token": "<unk>", "fuse_unk": fuse_unk}),
        );
        failures.extend(
            compare_encodes(&json, &inputs)
                .into_iter()
                .map(|f| format!("fuse_unk={fuse_unk}{f}")),
        );
    }
    assert_no_failures("unk_token", &failures);

    for b in 0u32..256 {
        vocab[format!("<0x{b:02X}>")] = Value::from(4 + b);
    }
    let json = tiny_bpe_json(vocab, json!(["a b"]), json!({"byte_fallback": true}));
    assert_no_failures(
        "byte_fallback",
        &compare_encodes(&json, &["aXb", "中", "ab😀ab", "é"]),
    );

    // No unk token and no byte fallback: must return (not panic).
    let json = tiny_bpe_json(json!({"a": 0, "b": 1}), json!([]), json!({}));
    let ours = Tokenizer::from_json(serde_json::from_str(&json).unwrap()).unwrap();
    let _ = ours.encode("aXb");
}

/// BPE models with `end_of_word_suffix` (e.g. CLIP's `</w>`) or
/// `continuing_subword_prefix` (`##`) build their initial symbols and merges
/// with those markers; fastokens must load them and match HF.
#[test]
fn bpe_word_suffix_and_subword_prefix_match_huggingface() {
    use std::str::FromStr;
    let split_on_space = json!({"type": "Split", "pattern": {"String": " "}, "behavior": "Removed", "invert": false});
    let cases = [
        (
            "end_of_word_suffix",
            json!({"a": 0, "b": 1, "c": 2, "a</w>": 3, "b</w>": 4, "c</w>": 5, "ab": 6, "abc</w>": 7, "ab</w>": 8}),
            json!(["a b", "ab c</w>", "a b</w>"]),
            json!({"end_of_word_suffix": "</w>"}),
        ),
        (
            "continuing_subword_prefix",
            json!({"a": 0, "b": 1, "c": 2, "##a": 3, "##b": 4, "##c": 5, "ab": 6, "abc": 7}),
            json!(["a ##b", "ab ##c"]),
            json!({"continuing_subword_prefix": "##"}),
        ),
    ];
    let inputs = ["abc", "ab", "cab", "abc ab c", "a b c", "ba"];
    let mut failures = Vec::new();
    for (what, vocab, merges, extra) in cases {
        let mut json: Value = serde_json::from_str(&tiny_bpe_json(vocab, merges, extra)).unwrap();
        json["pre_tokenizer"] = split_on_space.clone();
        let hf = tokenizers::Tokenizer::from_str(&json.to_string()).unwrap();
        let ours = match Tokenizer::from_json(json) {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!("  {what}: fastokens failed to load: {e}"));
                continue;
            }
        };
        for input in inputs {
            let want = hf.encode(input, false).unwrap().get_ids().to_vec();
            match ours.encode(input) {
                Ok(got) if got == want => {}
                Ok(got) => failures.push(format!("  {what} {input:?}: ours {got:?} hf {want:?}")),
                Err(e) => failures.push(format!("  {what} {input:?}: error {e} (hf {want:?})")),
            }
        }
    }
    assert_no_failures("BPE suffix / prefix options", &failures);
}

/// GPT-2's own ByteLevel pre-tokenizer goes to the hand-written scanner, which
/// merges raw bytes and caches whole pretokens. A model with a word suffix or
/// dropout must not take it: neither the suffix nor skipped merges exist there.
#[test]
fn scanner_path_is_not_used_for_word_affixes_or_dropout() {
    use crate::pre_tokenizers::BYTE_TO_CHAR;
    use std::str::FromStr;
    let byte_level = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true});
    let model = |suffix: bool, merges: &[&str], extra: Value| {
        let mut vocab = serde_json::Map::new();
        for c in BYTE_TO_CHAR {
            let n = vocab.len();
            vocab.insert(c.to_string(), Value::from(n));
            if suffix {
                vocab.insert(format!("{c}</w>"), Value::from(n + 1));
            }
        }
        for m in merges {
            let (l, r) = m.split_once(' ').unwrap();
            let n = vocab.len();
            vocab.insert(format!("{l}{r}"), Value::from(n));
        }
        let mut model = json!({"type": "BPE", "vocab": vocab, "merges": merges});
        for (k, v) in extra.as_object().unwrap() {
            model[k] = v.clone();
        }
        model
    };
    let cases = [
        (
            "end_of_word_suffix",
            model(
                true,
                &["h e", "he l", "hel l", "hell o</w>", "Ġ w", "Ġw o", "Ġwo r"],
                json!({"end_of_word_suffix": "</w>"}),
            ),
        ),
        (
            "dropout 1.0",
            model(
                false,
                &["h e", "he l", "hel l", "hell o", "Ġ w", "Ġw o"],
                json!({"dropout": 1.0}),
            ),
        ),
    ];
    let inputs = ["hello", "hello world", "hell", "héllo wor", "I'LL hello"];
    let mut failures = Vec::new();
    for (what, model) in cases {
        let json = json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "post_processor": null, "decoder": null,
            "pre_tokenizer": byte_level.clone(), "model": model,
        });
        let hf = tokenizers::Tokenizer::from_str(&json.to_string()).unwrap();
        let ours = Tokenizer::from_json(json).unwrap();
        for input in inputs {
            let want = hf.encode(input, false).unwrap().get_ids().to_vec();
            match ours.encode(input) {
                Ok(got) if got == want => {}
                Ok(got) => failures.push(format!("  {what} {input:?}: ours {got:?} hf {want:?}")),
                Err(e) => failures.push(format!("  {what} {input:?}: error {e} (hf {want:?})")),
            }
        }
    }
    assert_no_failures("scanner path with affixes / dropout", &failures);
}

// ── Scanner fast path ───────────────────────────────────────────────────────

fn scan_ranges(kind: crate::pre_tokenizers::scan::ScanKind, text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    crate::pre_tokenizers::scan::scan_core(kind, text, |s, e| {
        if s < e {
            out.push((s, e));
        }
        Ok(())
    })
    .unwrap();
    out
}

/// The hand-written o200k / Kimi scanner replaces the regex engine outright,
/// so it must reproduce the regex's pretokens on adversarial input too — in
/// particular mark sequences, Unicode whitespace and newline runs, where its
/// Unicode tables and hand-coded backtracking could drift from the regex.
#[test]
fn scanner_matches_regex_engine_fuzz() {
    use crate::pre_tokenizers::scan::ScanKind;
    let inputs = fuzz_inputs(0x5eed_0015, 3000, 30);
    let mut failures = Vec::new();
    for (kind, pattern) in [
        (ScanKind::O200k, O200K_BASE_PATTERN),
        (ScanKind::Kimi, KIMI_PATTERN),
    ] {
        let split = isolated_split(pattern);
        for input in &inputs {
            let mut pts = PreTokenizedString::from_text(input);
            split.pre_tokenize(&mut pts).unwrap();
            let want: Vec<(usize, usize)> = pts
                .splits()
                .iter()
                .filter(|s| !s.range.is_empty())
                .map(|s| (s.range.start, s.range.end))
                .collect();
            let got = scan_ranges(kind, input);
            if got != want {
                let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
                let show = |r: &[(usize, usize)]| -> Vec<&str> {
                    r[at..(at + 3).min(r.len())]
                        .iter()
                        .map(|&(s, e)| &input[s..e])
                        .collect()
                };
                failures.push(format!(
                    "  {kind:?} {input:?}: ours {:?} regex {:?}",
                    show(&got),
                    show(&want)
                ));
            }
        }
    }
    assert_no_failures("scanner vs regex engine", &failures);
}

/// Scanning newline-bounded chunks independently (the parallel fused path)
/// must equal scanning the whole buffer, for random buffers and chunk counts.
#[test]
fn newline_chunk_bounds_preserve_pretokens_fuzz() {
    use crate::pre_tokenizers::scan::{ScanKind, newline_chunk_bounds};
    let mut rng = Rng::new(0x5eed_0016);
    let mut failures = Vec::new();
    for _ in 0..60 {
        let mut text = String::new();
        while text.len() < 4096 {
            let k = 1 + rng.below(20);
            text.push_str(&adversarial_text(&mut rng, k));
            if rng.chance(1, 3) {
                text.push_str(rng.pick(&[
                    "\n",
                    ".\n/",
                    "!\n/\n/",
                    " \n \t\n",
                    "\r\n",
                    "\n\u{3000}\n",
                ]));
            }
        }
        for kind in [ScanKind::O200k, ScanKind::Kimi] {
            let whole = scan_ranges(kind, &text);
            for n_chunks in [2, 3, 5, 17, 64] {
                let chunked: Vec<(usize, usize)> = newline_chunk_bounds(&text, n_chunks, kind)
                    .into_iter()
                    .flat_map(|(s, e)| {
                        scan_ranges(kind, &text[s..e])
                            .into_iter()
                            .map(move |(a, b)| (a + s, b + s))
                    })
                    .collect();
                if chunked != whole {
                    let at = chunked
                        .iter()
                        .zip(&whole)
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    let show = |r: &[(usize, usize)]| -> Vec<String> {
                        r[at..(at + 2).min(r.len())]
                            .iter()
                            .map(|&(s, e)| text[s..e].to_string())
                            .collect()
                    };
                    failures.push(format!(
                        "  {kind:?} n_chunks={n_chunks}: chunked {:?} whole {:?}",
                        show(&chunked),
                        show(&whole)
                    ));
                }
            }
        }
    }
    assert_no_failures("newline chunk bounds", &failures);
}

// ── Focused regressions ─────────────────────────────────────────────────────

/// An `rstrip` token absorbs whitespace that the next added-token match
/// starts in. As in HF, the next token still matches, and its `lstrip` stops
/// at the end of the previous token's span.
#[test]
fn added_token_rstrip_into_next_match_does_not_panic() {
    let base = ALPHABET
        .chars()
        .collect::<std::collections::BTreeSet<_>>()
        .len() as u32;
    let tokens = [
        added(base, "<|a|>", true, false, true),
        added(base + 1, "\tb", true, true, false),
        added(base + 2, "ccc", false, false, false),
        added(base + 3, "中中", false, false, false),
    ];
    let json = char_vocab_tokenizer_json(&tokens, Value::Null);
    let inputs = ["<|a|>\tb", "<|a|> \tb", "x<|a|>\t\tb"].map(String::from);
    assert_no_failures(
        "rstrip into next match",
        &compare_added_tokens(&json, &inputs),
    );
}

/// o200k's word alternative takes an optional non-letter prefix, so `/usr`
/// after a newline is one pretoken; a parallel chunk boundary placed after the
/// newline must not be advanced past the `/` into the middle of it.
#[test]
fn o200k_chunk_bounds_keep_slash_prefixed_word_whole() {
    use crate::pre_tokenizers::scan::{ScanKind, newline_chunk_bounds};
    // 100 bytes, the newline, then 84: the 2-way nominal split (byte 92)
    // precedes the newline, so the boundary lands right after it.
    let text = format!("{}\n/usr{}", "a ".repeat(50), " b".repeat(40));
    let whole = scan_ranges(ScanKind::O200k, &text);
    assert!(
        whole.iter().any(|&(s, e)| &text[s..e] == "/usr"),
        "`/usr` should be one pretoken"
    );
    let chunked: Vec<(usize, usize)> = newline_chunk_bounds(&text, 2, ScanKind::O200k)
        .into_iter()
        .flat_map(|(s, e)| {
            scan_ranges(ScanKind::O200k, &text[s..e])
                .into_iter()
                .map(move |(a, b)| (a + s, b + s))
        })
        .collect();
    assert_eq!(chunked, whole);
}

/// The GPT-2 regex behind `ByteLevel(use_regex=true)` is case-sensitive
/// upstream: `'LL` is not a contraction.
#[test]
fn byte_level_regex_contractions_are_case_sensitive_like_huggingface() {
    let config = json!({
        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true,
    });
    let ours = ours_pre_tokenizer(&config);
    let hf: tokenizers::PreTokenizerWrapper = serde_json::from_value(config).unwrap();
    for input in ["I'LL", "WE'RE HERE", "it'S", "I'll"] {
        let mut pts = tokenizers::PreTokenizedString::from(input);
        hf.pre_tokenize(&mut pts).unwrap();
        let want: Vec<String> = pts
            .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s.to_string())
            .collect();
        assert_eq!(ours_pieces(&ours, input).unwrap(), want, "{input:?}");
    }
}

/// Added tokens whose content has characters outside the GPT-2 byte alphabet
/// (a literal space or newline) decode through `ByteLevel` as themselves.
#[test]
fn byte_level_decoder_passes_through_non_alphabet_chars_like_huggingface() {
    use std::str::FromStr;
    let mut vocab = serde_json::Map::new();
    for (i, c) in crate::pre_tokenizers::BYTE_TO_CHAR.iter().enumerate() {
        vocab.insert(c.to_string(), Value::from(i));
    }
    let json = json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [
            added(256, "hello world", false, false, false),
            added(257, "\n\n", false, false, false),
            added(258, "<|x|>", true, false, false),
            // Mixed: HF keeps the whole token's raw UTF-8 (so `Ġ` stays `Ġ`).
            added(259, "Ġx y", false, false, false),
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    })
    .to_string();
    let ours = Tokenizer::from_json(serde_json::from_str(&json).unwrap()).unwrap();
    let hf = tokenizers::Tokenizer::from_str(&json).unwrap();
    for ids in [
        vec![256],
        vec![257],
        vec![259],
        vec![72, 256, 257, 258, 259, 105],
    ] {
        for skip in [false, true] {
            assert_eq!(
                ours.decode(&ids, skip).unwrap(),
                hf.decode(&ids, skip).unwrap(),
                "{ids:?} skip={skip}"
            );
        }
    }
}

/// `encode` of ≥ 64 KiB inputs from a worker of another rayon pool (e.g. in a
/// caller's `par_iter`) must not nest BPE pool installs without bound; see
/// `pre_tokenized::bpe_threads`.
#[test]
fn encode_from_a_foreign_rayon_pool_does_not_nest_without_bound() {
    use rayon::prelude::*;
    crate::test_support::run_in_child(
        "component_parity_tests::encode_from_a_foreign_rayon_pool_does_not_nest_without_bound",
        || {
            // Byte-level vocab + o200k pattern: the scanner path, whose
            // parallel threshold is 64 KiB.
            let ranks: Vec<(Vec<u8>, u32)> = (0..256u32).map(|i| (vec![i as u8], i)).collect();
            let tok = Tokenizer::from_tiktoken_ranks(
                &ranks,
                crate::TiktokenConfig::new(O200K_BASE_PATTERN, vec![]),
            )
            .unwrap();
            let base = "the quick brown fox jumps over the lazy dog\n".repeat(1600);
            let inputs: Vec<String> = (0..400).map(|i| format!("{i}{base}")).collect();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .stack_size(256 * 1024)
                .build()
                .unwrap();
            let lens: Vec<usize> = pool.install(|| {
                inputs
                    .par_iter()
                    .map(|s| tok.encode(s).unwrap().len())
                    .collect()
            });
            assert!(lens.iter().all(|&n| n > 0));
        },
    );
}

// ── Rarely used options and unsupported components ──────────────────────────

/// `dropout: 1.0` drops every merge in HF, deterministically.
#[test]
fn bpe_dropout_one_disables_merges_like_huggingface() {
    let vocab = json!({"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4});
    let json = tiny_bpe_json(vocab, json!(["a b", "ab c"]), json!({"dropout": 1.0}));
    assert_no_failures(
        "dropout 1.0",
        &compare_encodes(&json, &["ab", "abc", "cab", "abcab"]),
    );
}

/// `truncation` / `padding` blocks in `tokenizer.json` are applied by HF's
/// `encode`.
#[test]
fn tokenizer_json_truncation_and_padding_match_huggingface() {
    use std::str::FromStr;
    let mut base: Value = serde_json::from_str(&tiny_bpe_json(
        json!({"<pad>": 0, "a": 1, "b": 2, "c": 3, "ab": 4, "<s>": 5, "</s>": 6}),
        json!(["a b"]),
        json!({}),
    ))
    .unwrap();
    // A BOS/EOS template: with `add_special_tokens`, truncation must leave room
    // for both, and they must survive it.
    let special = |id: u32, content: &str| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false,
               "rstrip": false, "normalized": false, "special": true})
    };
    base["added_tokens"] = json!([special(5, "<s>"), special(6, "</s>")]);
    base["post_processor"] = json!({
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<s>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "</s>", "type_id": 0}},
        ],
        "pair": [
            {"Sequence": {"id": "A", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
        ],
        "special_tokens": {
            "<s>": {"id": "<s>", "ids": [5], "tokens": ["<s>"]},
            "</s>": {"id": "</s>", "ids": [6], "tokens": ["</s>"]},
        },
    });
    let truncation = |direction: &str, max_length: usize| {
        json!({"direction": direction, "max_length": max_length,
               "strategy": "LongestFirst", "stride": 0})
    };
    let padding = |strategy: Value, direction: &str| {
        json!({"strategy": strategy, "direction": direction, "pad_to_multiple_of": null,
               "pad_id": 0, "pad_type_id": 0, "pad_token": "<pad>"})
    };
    let variants = [
        ("truncation right", truncation("Right", 4), Value::Null),
        ("truncation left", truncation("Left", 4), Value::Null),
        (
            "padding fixed",
            Value::Null,
            padding(json!({"Fixed": 8}), "Right"),
        ),
        (
            "padding longest, left",
            Value::Null,
            padding(json!("BatchLongest"), "Left"),
        ),
        (
            "truncation and padding",
            truncation("Right", 5),
            padding(json!({"Fixed": 6}), "Right"),
        ),
    ];
    let inputs = ["ab", "abcabcabc", "c", "cccccccccc"];
    let mut failures = Vec::new();
    for (what, truncation, padding) in variants {
        let mut json = base.clone();
        json["truncation"] = truncation;
        json["padding"] = padding;
        let hf = tokenizers::Tokenizer::from_str(&json.to_string()).unwrap();
        let ours = Tokenizer::from_json(json).unwrap();
        for add_special in [false, true] {
            for input in inputs {
                let want = hf.encode(input, add_special).unwrap().get_ids().to_vec();
                let got = ours.encode_with_special_tokens(input, add_special).unwrap();
                if got != want {
                    failures.push(format!(
                        "  {what} add_special={add_special} {input:?}: ours {got:?} hf {want:?}"
                    ));
                }
            }
            // Batch padding is decided over the whole batch.
            let want: Vec<Vec<u32>> = hf
                .encode_batch(inputs.to_vec(), add_special)
                .unwrap()
                .iter()
                .map(|e| e.get_ids().to_vec())
                .collect();
            let got = ours.encode_batch(&inputs, add_special).unwrap();
            if got != want {
                failures.push(format!(
                    "  {what} add_special={add_special} batch: ours {got:?} hf {want:?}"
                ));
            }
        }
    }

    // Below the two added tokens, HF's unchecked subtraction wraps and nothing
    // is cut. A debug build of HF panics there instead, so compare with HF
    // without truncation.
    let mut json = base.clone();
    json["truncation"] = truncation("Right", 1);
    let ours = Tokenizer::from_json(json).unwrap();
    let hf = tokenizers::Tokenizer::from_str(&base.to_string()).unwrap();
    for input in inputs {
        let want = hf.encode(input, true).unwrap().get_ids().to_vec();
        let got = ours.encode_with_special_tokens(input, true).unwrap();
        if got != want {
            failures.push(format!(
                "  truncation below specials {input:?}: ours {got:?} hf {want:?}"
            ));
        }
    }
    assert_no_failures("tokenizer.json truncation / padding", &failures);
}

/// Component types fastokens does not implement must be rejected at load
/// time with an error naming them — never loaded and silently skipped.
#[test]
fn unsupported_components_are_rejected_with_a_clear_error() {
    let base: Value = serde_json::from_str(&tiny_bpe_json(
        json!({"[UNK]": 0, "a": 1, "b": 2}),
        json!([]),
        json!({}),
    ))
    .unwrap();
    let cases = [
        (
            "model",
            "WordPiece",
            json!({"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "a": 1}}),
        ),
        (
            "model",
            "WordLevel",
            json!({"type": "WordLevel", "unk_token": "[UNK]", "vocab": {"[UNK]": 0, "a": 1}}),
        ),
        (
            "model",
            "Unigram",
            json!({"type": "Unigram", "unk_id": 0, "vocab": [["[UNK]", 0.0], ["a", -1.0]], "byte_fallback": false}),
        ),
        ("normalizer", "NFD", json!({"type": "NFD"})),
        ("normalizer", "NFKC", json!({"type": "NFKC"})),
        ("normalizer", "NFKD", json!({"type": "NFKD"})),
        (
            "normalizer",
            "StripAccents",
            json!({"type": "StripAccents"}),
        ),
        (
            "normalizer",
            "Strip",
            json!({"type": "Strip", "strip_left": true, "strip_right": true}),
        ),
        (
            "normalizer",
            "BertNormalizer",
            json!({"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true}),
        ),
        ("normalizer", "ByteLevel", json!({"type": "ByteLevel"})),
        ("pre_tokenizer", "Whitespace", json!({"type": "Whitespace"})),
        (
            "pre_tokenizer",
            "WhitespaceSplit",
            json!({"type": "WhitespaceSplit"}),
        ),
        (
            "pre_tokenizer",
            "Punctuation",
            json!({"type": "Punctuation", "behavior": "Isolated"}),
        ),
        (
            "pre_tokenizer",
            "Metaspace",
            json!({"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": true}),
        ),
        (
            "pre_tokenizer",
            "Digits",
            json!({"type": "Digits", "individual_digits": true}),
        ),
        (
            "pre_tokenizer",
            "BertPreTokenizer",
            json!({"type": "BertPreTokenizer"}),
        ),
        (
            "pre_tokenizer",
            "UnicodeScripts",
            json!({"type": "UnicodeScripts"}),
        ),
        (
            "decoder",
            "WordPiece",
            json!({"type": "WordPiece", "prefix": "##", "cleanup": true}),
        ),
        (
            "decoder",
            "Metaspace",
            json!({"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": true}),
        ),
        (
            "decoder",
            "BPEDecoder",
            json!({"type": "BPEDecoder", "suffix": "</w>"}),
        ),
        (
            "decoder",
            "CTC",
            json!({"type": "CTC", "pad_token": "<pad>", "word_delimiter_token": "|", "cleanup": true}),
        ),
        (
            "post_processor",
            "BertProcessing",
            json!({"type": "BertProcessing", "sep": ["[SEP]", 102], "cls": ["[CLS]", 101]}),
        ),
        (
            "post_processor",
            "RobertaProcessing",
            json!({"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0], "trim_offsets": true, "add_prefix_space": true}),
        ),
    ];
    let mut failures = Vec::new();
    for (field, name, config) in cases {
        let mut json = base.clone();
        json[field] = config;
        match Tokenizer::from_json(json) {
            Ok(_) => failures.push(format!("  {field} {name}: loaded (silently unsupported)")),
            Err(e) => {
                let msg = e.to_string();
                if !msg.contains("unsupported") || !msg.contains(name) {
                    failures.push(format!("  {field} {name}: unclear error: {msg}"));
                }
            }
        }
    }
    assert_no_failures("unsupported components", &failures);
}
