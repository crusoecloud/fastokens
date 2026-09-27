//! End-to-end parity with HuggingFace `tokenizers` on real models, for the
//! flows the fixed-corpus tests in `lib.rs` cannot reach:
//!
//! - cross-call state: different inputs sharing a long prefix on one thread
//!   (split cache), several tokenizers sharing a thread (cache keys), the
//!   opt-in prefix cache;
//! - randomized adversarial input, long whitespace runs on small stacks, and
//!   inputs large enough for every parallel path and cache growth;
//! - the metaspace / byte-fallback pipeline (Gemma 4).
//!
//! Needs the HF Hub or a warm HF cache.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::test_support::{
    MARKS, Rng, SPLIT_CACHE_PAIRS, WHITESPACE, adversarial_text, assert_no_failures, load_hf,
    load_ours, prose, random_scalar,
};
use crate::{DecodeStream, NewToken, Tokenizer};

const QWEN3: &str = "Qwen/Qwen3-0.6B";
const GPT_OSS: &str = "openai/gpt-oss-120b";
const KIMI: &str = "hoangquan456/Kimi-K2.5";
const GEMMA4: &str = "google/gemma-4-31B-it";
const MINIMAX: &str = "MiniMaxAI/MiniMax-M2.1";

/// Both implementations of one model.
struct Pair {
    name: &'static str,
    ours: Tokenizer,
    hf: tokenizers::Tokenizer,
}

/// The pair for `name`, loaded once per test process and shared by tests.
/// Sharing is safe: every cache involved is thread-local or keyed by input.
fn shared_pair(name: &'static str) -> &'static Pair {
    static PAIRS: Mutex<Option<HashMap<&'static str, &'static Pair>>> = Mutex::new(None);
    let mut pairs = PAIRS.lock().unwrap_or_else(|e| e.into_inner());
    pairs
        .get_or_insert_with(HashMap::new)
        .entry(name)
        .or_insert_with(|| {
            Box::leak(Box::new(Pair {
                name,
                ours: load_ours(name),
                hf: load_hf(name),
            }))
        })
}

impl Pair {
    fn hf_ids(&self, input: &str, add_special_tokens: bool) -> Vec<u32> {
        self.hf
            .encode(input, add_special_tokens)
            .unwrap_or_else(|e| panic!("{}: HF encode failed: {e}", self.name))
            .get_ids()
            .to_vec()
    }

    /// Describe how `got` differs from HF's ids for `input`, or `None`.
    fn diff(&self, input: &str, got: &[u32], add_special_tokens: bool) -> Option<String> {
        let want = self.hf_ids(input, add_special_tokens);
        (got != want).then(|| describe_diff(self.name, input, got, &want))
    }

    /// Encode with fastokens (on the calling thread) and compare with HF.
    fn check(&self, input: &str, add_special_tokens: bool) -> Option<String> {
        match self
            .ours
            .encode_with_special_tokens(input, add_special_tokens)
        {
            Ok(got) => self.diff(input, &got, add_special_tokens),
            Err(e) => Some(format!(
                "{}: {:?}: encode error {e}",
                self.name,
                preview(input)
            )),
        }
    }
}

fn preview(s: &str) -> String {
    if s.chars().count() <= 60 {
        return s.to_string();
    }
    let head: String = s.chars().take(30).collect();
    let tail: String = s
        .chars()
        .rev()
        .take(30)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}…[{} bytes]…{tail}", s.len())
}

fn describe_diff(model: &str, input: &str, got: &[u32], want: &[u32]) -> String {
    let at = got
        .iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .unwrap_or(got.len().min(want.len()));
    let from = at.saturating_sub(3);
    format!(
        "{model}: {:?}: {} vs {} ids, first diff at {at}\n    ours: {:?}\n    hf:   {:?}",
        preview(input),
        got.len(),
        want.len(),
        &got[from..(at + 4).min(got.len())],
        &want[from..(at + 4).min(want.len())],
    )
}

// ── Split cache across different inputs ─────────────────────────────────────

fn shared_prefix_pair_failures(pair: &Pair, pairs: &[(String, String)]) -> Vec<String> {
    let mut failures = Vec::new();
    for (a, b) in pairs {
        // A primes the thread-local caches; B must not depend on it.
        let _ = pair.ours.encode(a);
        if let Some(f) = pair.check(b, false) {
            failures.push(format!("after A={:?}:\n  {f}", preview(a)));
        }
    }
    // The production shape: many requests through the rayon pool, whose
    // worker threads keep their caches across items.
    let batch: Vec<&str> = pairs
        .iter()
        .flat_map(|(a, b)| [a.as_str(), b.as_str()])
        .collect();
    let got = pair.ours.encode_batch(&batch, false).unwrap();
    for (input, ids) in batch.iter().zip(&got) {
        if let Some(f) = pair.diff(input, ids, false) {
            failures.push(format!("encode_batch:\n  {f}"));
        }
    }
    failures
}

#[test]
fn qwen3_split_cache_known_pairs() {
    let pair = shared_pair(QWEN3);
    let prefix = "word ".repeat(1000);
    let pairs: Vec<(String, String)> = SPLIT_CACHE_PAIRS
        .iter()
        .map(|(a, b)| (format!("{prefix}{a}"), format!("{prefix}{b}")))
        .collect();
    assert_no_failures(
        "Qwen3 B-after-A (known pairs)",
        &shared_prefix_pair_failures(pair, &pairs),
    );
}

/// Shared prefixes ≥ 4 KiB (the split cache's reuse threshold) from prose,
/// single-character whitespace runs and adversarial text, ending in
/// whitespace half the time; short random tails.
fn shared_prefix_pairs(seed: u64, n: usize) -> Vec<(String, String)> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
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
            let (ka, kb) = (1 + rng.below(6), 1 + rng.below(6));
            let a = format!("{prefix}{}", adversarial_text(&mut rng, ka));
            let b = format!("{prefix}{}", adversarial_text(&mut rng, kb));
            (a, b)
        })
        .collect()
}

#[test]
fn shared_prefix_pairs_match_huggingface() {
    let pairs = shared_prefix_pairs(0x5eed_1001, 150);
    let mut failures = Vec::new();
    for model in [QWEN3, GPT_OSS, KIMI, GEMMA4] {
        failures.extend(shared_prefix_pair_failures(shared_pair(model), &pairs));
    }
    assert_no_failures("B-after-A shared-prefix fuzz", &failures);
}

// ── Combining marks ─────────────────────────────────────────────────────────

#[test]
fn qwen3_combining_mark_pairs_match_huggingface() {
    let pair = shared_pair(QWEN3);
    let inputs = [
        "\u{6e0}\u{7fd}",
        "\u{341}\u{7fd}",
        "\u{30a}\u{7fd}",
        "\u{368}\u{7fd}",
        "\u{308}\u{7fd}",
        "\u{1aeb}\u{35b}",
        "\u{1aeb}\u{34e}",
        "\u{346}\u{8cf}",
        // The same pairs inside words and after other text.
        "word\u{6e0}\u{7fd} x",
        "a\u{346}\u{8cf}b",
    ];
    let failures: Vec<String> = inputs.iter().filter_map(|s| pair.check(s, false)).collect();
    assert_no_failures("Qwen3 combining-mark pairs", &failures);
}

// ── Long whitespace runs on small stacks ────────────────────────────────────

fn long_whitespace_inputs() -> Vec<String> {
    let mut inputs = Vec::new();
    for ws in [" ", "\t", "\n", "\r\n", "\u{3000}", "\u{a0}", " \n", "\n "] {
        // Below and above the 16 KiB parallel-regex threshold.
        inputs.push(ws.repeat(8192 / ws.len()));
        inputs.push(ws.repeat(64 * 1024 / ws.len()));
        inputs.push(format!("{}word{}", ws.repeat(4096), ws.repeat(4096)));
    }
    inputs
}

#[test]
fn long_whitespace_runs_match_huggingface_on_a_small_stack() {
    crate::test_support::run_in_child(
        "model_parity_tests::long_whitespace_runs_match_huggingface_on_a_small_stack",
        long_whitespace_runs_body,
    );
}

fn long_whitespace_runs_body() {
    let inputs = long_whitespace_inputs();
    let mut failures = Vec::new();
    for model in [QWEN3, GPT_OSS, KIMI, GEMMA4] {
        let pair = shared_pair(model);
        let want: Vec<Vec<u32>> = inputs.iter().map(|s| pair.hf_ids(s, false)).collect();
        // Small stack; an overflow aborts the child process and fails the test.
        let got: Vec<Vec<u32>> = std::thread::scope(|scope| {
            std::thread::Builder::new()
                .stack_size(512 * 1024)
                .spawn_scoped(scope, || {
                    inputs
                        .iter()
                        .map(|s| pair.ours.encode(s).unwrap())
                        .collect()
                })
                .unwrap()
                .join()
                .unwrap()
        });
        let batched = pair.ours.encode_batch(&inputs, false).unwrap();
        for ((input, want), (got, batched)) in
            inputs.iter().zip(&want).zip(got.iter().zip(&batched))
        {
            if got != want {
                failures.push(describe_diff(model, input, got, want));
            }
            if batched != want {
                failures.push(format!(
                    "encode_batch: {}",
                    describe_diff(model, input, batched, want)
                ));
            }
        }
    }
    assert_no_failures("long whitespace runs", &failures);
}

// ── Randomized parity ───────────────────────────────────────────────────────

fn fuzz_failures(pair: &Pair, seed: u64, n: usize) -> Vec<String> {
    let mut rng = Rng::new(seed);
    let mut failures = Vec::new();
    for i in 0..n {
        let input = match i % 4 {
            0 => (0..1 + rng.below(16))
                .map(|_| random_scalar(&mut rng))
                .collect(),
            1 => {
                // Marks stacked on (or without) a base character.
                let mut s = String::new();
                for _ in 0..1 + rng.below(4) {
                    if rng.chance(1, 2) {
                        s.push(random_scalar(&mut rng));
                    }
                    for _ in 0..1 + rng.below(3) {
                        s.push(rng.pick(MARKS));
                    }
                }
                s
            }
            _ => {
                let k = 1 + rng.below(30);
                adversarial_text(&mut rng, k)
            }
        };
        if let Some(f) = pair.check(&input, i % 2 == 0) {
            failures.push(f);
        }
    }
    failures
}

/// Decoding random id sequences (biased toward specials and the low ids where
/// byte / byte-fallback tokens live) matches HF, with both skip flags.
fn decode_failures(pair: &Pair, seed: u64, n: usize) -> Vec<String> {
    let vocab = pair.hf.get_vocab_size(true) as u32;
    let specials: Vec<u32> = pair
        .hf
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, t)| t.special)
        .map(|(id, _)| id)
        .collect();
    let mut rng = Rng::new(seed);
    let mut failures = Vec::new();
    for _ in 0..n {
        let ids: Vec<u32> = (0..1 + rng.below(12))
            .map(|_| match rng.below(4) {
                0 if !specials.is_empty() => rng.pick(&specials),
                1 => rng.below(600) as u32,
                _ => rng.below(vocab as usize) as u32,
            })
            .collect();
        for skip in [false, true] {
            let want = pair.hf.decode(&ids, skip).unwrap();
            match pair.ours.decode(&ids, skip) {
                Ok(got) if got == want => {}
                Ok(got) => failures.push(format!(
                    "{}: decode({ids:?}, skip={skip}): ours {got:?} hf {want:?}",
                    pair.name
                )),
                Err(e) => failures.push(format!("{}: decode({ids:?}): {e}", pair.name)),
            }
        }
    }
    failures
}

#[test]
fn random_inputs_match_huggingface() {
    let mut failures = Vec::new();
    for (i, model) in [QWEN3, GPT_OSS, KIMI, GEMMA4].into_iter().enumerate() {
        let pair = shared_pair(model);
        failures.extend(fuzz_failures(pair, 0x5eed_2000 + i as u64, 2000));
    }
    assert_no_failures("random-input encode parity", &failures);
}

#[test]
fn random_id_sequences_decode_like_huggingface() {
    let mut failures = Vec::new();
    for (i, model) in [QWEN3, GPT_OSS, KIMI, GEMMA4].into_iter().enumerate() {
        let pair = shared_pair(model);
        failures.extend(decode_failures(pair, 0x5eed_2100 + i as u64, 1500));
    }
    assert_no_failures("random-id decode parity", &failures);
}

// ── Large inputs: parallel paths and cache growth ───────────────────────────

/// ~200 KiB with tens of thousands of distinct short pretokens, so one encode
/// crosses every parallel threshold (16 KiB regex, 64 KiB scan / BPE) and
/// grows the pretoken cache past its initial size.
fn large_input(seed: u64, newlines: bool) -> String {
    let mut rng = Rng::new(seed);
    let mut s = String::new();
    while s.len() < 200 * 1024 {
        match rng.below(6) {
            0 => {
                let k = 1 + rng.below(20);
                s.push_str(&adversarial_text(&mut rng, k));
            }
            1 => {
                // A unique pseudo-word, to defeat the caches.
                s.push(' ');
                for _ in 0..3 + rng.below(12) {
                    s.push((b'a' + rng.below(26) as u8) as char);
                }
            }
            2 => s.push_str(&format!(" {}", rng.below(100_000))),
            _ => s.push_str(&prose(&mut rng, 20)),
        }
        if newlines && rng.chance(1, 8) {
            s.push_str(rng.pick(&["\n", "\n\n", " \n", "\r\n", ".\n/", "\n  \n"]));
        }
    }
    if newlines {
        s
    } else {
        s.replace(['\n', '\r'], " ")
    }
}

#[test]
fn large_inputs_match_huggingface_twice() {
    let inputs = [
        large_input(0x5eed_3001, true),
        large_input(0x5eed_3002, false),
        // One enormous pretoken, and 15/16-byte pretokens at the buffer end
        // (pretoken-cache key packing boundaries).
        format!("{} tail", "abcdefghij".repeat(7000)),
        format!("{} abcdefghijklmn", prose(&mut Rng::new(1), 30)),
        format!("{} abcdefghijklmno", prose(&mut Rng::new(2), 30)),
    ];
    let mut failures = Vec::new();
    for model in [QWEN3, GPT_OSS, KIMI, GEMMA4, MINIMAX] {
        let pair = shared_pair(model);
        for input in &inputs {
            let want = pair.hf_ids(input, false);
            // Cold, then warm: the second pass reads back cached pretokens.
            for pass in ["cold", "warm"] {
                let got = pair.ours.encode(input).unwrap();
                if got != want {
                    failures.push(format!(
                        "{pass}: {}",
                        describe_diff(model, input, &got, &want)
                    ));
                }
            }
        }
    }
    assert_no_failures("large inputs", &failures);
}

// ── Several tokenizers sharing one thread ───────────────────────────────────

/// The thread-local split/BPE caches are keyed by instance; alternating
/// tokenizers over the same inputs on one thread must not leak results.
#[test]
fn tokenizers_sharing_a_thread_match_huggingface() {
    let pairs = [
        shared_pair(QWEN3),
        shared_pair(GPT_OSS),
        shared_pair(GEMMA4),
    ];
    let prefix = prose(&mut Rng::new(3), 1200);
    let mut inputs = vec![
        "Hello world, this is a test.".to_string(),
        format!("{prefix} first tail"),
        format!("{prefix}  \n second tail"),
        large_input(0x5eed_4001, true),
    ];
    let mut rng = Rng::new(0x5eed_4002);
    inputs.extend((0..40).map(|_| {
        let k = 1 + rng.below(20);
        adversarial_text(&mut rng, k)
    }));
    let mut failures = Vec::new();
    for _round in 0..2 {
        for input in &inputs {
            for pair in &pairs {
                failures.extend(pair.check(input, false));
            }
        }
    }
    assert_no_failures("tokenizers sharing a thread", &failures);
}

// ── Opt-in prefix cache ─────────────────────────────────────────────────────

/// The prefix cache on a real vocabulary (o200k scanner path), where a reuse
/// cut at the wrong place changes merges. Covers divergence inside
/// whitespace and newline runs, `\r\n`, o200k's `[\r\n/]*` punctuation tail,
/// non-ASCII prefixes, LRU thrash at capacity 1, exact repeats,
/// `add_special_tokens`, > 64 KiB inputs and concurrent `encode_batch`.
#[test]
fn prefix_cache_matches_huggingface() {
    let pair = shared_pair(GPT_OSS);
    let mut rng = Rng::new(0x5eed_5001);
    let prefixes = [
        prose(&mut rng, 2500),
        format!("{}\n", "网友们说 données café — ".repeat(600)),
        format!("{}\n", "line with code();\r\n".repeat(700)),
        large_input(0x5eed_5002, true),
    ];
    let tails = [
        "",
        "x",
        "\n",
        "\nx",
        "   \n",
        "  \n  x",
        "\r\nnext",
        "!\n/usr/bin",
        ".\n/\n/y",
        "\n\n\nabc",
        "'s tail",
        " 中文",
        "\u{3000}x",
    ];
    let mut inputs = Vec::new();
    for p in &prefixes {
        for t in tails {
            inputs.push(format!("{p}{t}"));
        }
    }
    let mut failures = Vec::new();
    for capacity in [1, 4] {
        let mut cached = load_ours(GPT_OSS);
        cached.enable_input_cache(capacity);
        for round in 0..2 {
            for (i, input) in inputs.iter().enumerate() {
                let special = (i + round) % 3 == 0;
                match cached.encode_with_special_tokens(input, special) {
                    Ok(got) => failures.extend(
                        pair.diff(input, &got, special)
                            .map(|f| format!("capacity {capacity} round {round}: {f}")),
                    ),
                    Err(e) => failures.push(format!("capacity {capacity}: error {e}")),
                }
            }
        }
        let got = cached.encode_batch(&inputs, false).unwrap();
        for (input, ids) in inputs.iter().zip(&got) {
            failures.extend(
                pair.diff(input, ids, false)
                    .map(|f| format!("capacity {capacity} encode_batch: {f}")),
            );
        }
    }
    assert_no_failures("prefix cache", &failures);
}

/// `add_tokens` must invalidate cached encodings: a later input sharing a
/// cached prefix has to recognize the new token.
#[test]
fn prefix_cache_is_invalidated_by_add_tokens() {
    let mut ours = load_ours(GPT_OSS);
    let mut hf = load_hf(GPT_OSS);
    ours.enable_input_cache(4);
    let prefix = prose(&mut Rng::new(4), 3000);
    let input = format!("{prefix}\nzzqq tail");
    ours.encode(&input).unwrap();
    ours.encode(&input).unwrap();

    ours.add_tokens(&[NewToken::new("zzqq")]).unwrap();
    hf.add_tokens(&[tokenizers::AddedToken::from("zzqq", false)]);
    let want = hf.encode(input.as_str(), false).unwrap().get_ids().to_vec();
    assert_eq!(ours.encode(&input).unwrap(), want);
    assert_eq!(
        ours.encode(&format!("{prefix}\nzzqq other")).unwrap(),
        hf.encode(format!("{prefix}\nzzqq other"), false)
            .unwrap()
            .get_ids()
    );
}

// ── Gemma 4: metaspace + byte fallback ──────────────────────────────────────

fn gemma() -> &'static Pair {
    shared_pair(GEMMA4)
}

const GEMMA_EXTRA: &[&str] = &[
    // Chat template with spaces and newlines around the control tokens.
    "<bos><start_of_turn>user\nHello there<end_of_turn>\n<start_of_turn>model\n",
    "Hello <end_of_turn> world",
    " <start_of_turn> ",
    "text<end_of_turn>\n\n<start_of_turn>model\nsure",
    // Characters the vocab only covers through byte fallback.
    "𪛖",
    "𐍈 ꙮ \u{e0041}",
    "a𪛖b 𪛖𪛖 x",
    // The metaspace itself in the input.
    "\u{2581}already\u{2581}escaped",
    "   leading and trailing   ",
];

#[test]
fn gemma4_matches_huggingface() {
    let pair = gemma();
    let mut failures = Vec::new();
    for input in crate::tests::CORPUS.iter().chain(GEMMA_EXTRA) {
        for special in [false, true] {
            failures.extend(pair.check(input, special));
        }
        let hf_ids = pair.hf_ids(input, false);
        let (want, got) = (
            pair.hf.decode(&hf_ids, false).unwrap(),
            pair.ours.decode(&hf_ids, false),
        );
        if got.as_deref().ok() != Some(want.as_str()) {
            failures.push(format!("decode {input:?}: ours {got:?} hf {want:?}"));
        }
    }
    assert_eq!(
        pair.ours.vocab_size(),
        pair.hf.get_vocab_size(true),
        "vocab_size"
    );
    assert_no_failures("Gemma 4 vs HF", &failures);
}

/// Vocab-aware bigram splitting claims to be output-preserving; turning it
/// off must not change a single id.
#[test]
fn gemma4_bigram_splitting_is_output_preserving() {
    let with = &gemma().ours;
    let mut without = load_ours(GEMMA4);
    assert!(
        with.needs_vocab_splitting,
        "Gemma 4 should use bigram splitting"
    );
    without.needs_vocab_splitting = false;
    let mut rng = Rng::new(0x5eed_6001);
    let mut inputs: Vec<String> = crate::tests::CORPUS.iter().map(|s| s.to_string()).collect();
    inputs.extend((0..500).map(|_| {
        let k = 1 + rng.below(30);
        adversarial_text(&mut rng, k)
    }));
    inputs.push(large_input(0x5eed_6002, true));
    let failures: Vec<String> = inputs
        .iter()
        .filter_map(|input| {
            let (a, b) = (with.encode(input).unwrap(), without.encode(input).unwrap());
            (a != b).then(|| describe_diff("gemma4 split vs unsplit", input, &a, &b))
        })
        .collect();
    assert_no_failures("bigram splitting", &failures);
}

/// Streaming a character that the vocab only encodes as several byte-fallback
/// tokens must reassemble it — no U+FFFD chunk, and the full text back.
#[test]
fn gemma4_decode_stream_reassembles_byte_fallback_chars() {
    let tok = &gemma().ours;
    for text in ["𪛖", "x 𪛖 y", "𐍈ꙮ\u{e0041}", "Hello 𪛖 world 😀 done"] {
        let ids = tok.encode(text).unwrap();
        let mut stream = DecodeStream::new(Vec::new(), false);
        let mut out = String::new();
        for id in ids {
            if let Some(chunk) = stream.step(tok, vec![id]).unwrap() {
                assert!(!chunk.contains('\u{fffd}'), "{text:?}: chunk {chunk:?}");
                out.push_str(&chunk);
            }
        }
        assert_eq!(out, text);
    }
}
