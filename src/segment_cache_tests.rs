//! The segment cache against the same pipeline without it, on synthetic
//! tokenizers covering each encode path: the scanner families (fused and as a
//! plain regex), byte-level `Split` with and without a prefix space, the
//! `Prepend`, `NFC` and `Lowercase` normalizers, and a sequence of splits.
//! Hub models are in `model_parity_tests`.

use serde_json::{Value, json};

use crate::{
    AddedTokenPolicy, EncodeSegment, NewToken, Tokenizer,
    test_support::{Rng, chat_requests, prose},
    tiktoken::{self, TiktokenConfig},
};

const SCAN_SPECIALS: [&str; 4] = ["<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"];

/// A tiktoken-style tokenizer for `pattern`: every byte, some merges so
/// multi-byte tokens occur, and [`SCAN_SPECIALS`] as special tokens.
fn scan_test_tokenizer(pattern: &str) -> Tokenizer {
    let mut ranks: Vec<(Vec<u8>, u32)> = (0..=255u8).map(|b| (vec![b], u32::from(b))).collect();
    // Each merge must split into exactly two earlier tokens.
    for merged in [
        " t", "th", "he", "in", "er", "an", " a", "re", "on", "  ", "\n\n", "e ", " the", "ing",
        "  \n", "}\n", "He", "ll", "Hell", "Hello", "'s", " d", " don",
    ] {
        let rank = ranks.len() as u32;
        ranks.push((merged.as_bytes().to_vec(), rank));
    }
    let specials = SCAN_SPECIALS
        .iter()
        .enumerate()
        .map(|(i, s)| (s.to_string(), (ranks.len() + i) as u32))
        .collect();
    Tokenizer::from_tiktoken_ranks(&ranks, TiktokenConfig::new(pattern, specials)).unwrap()
}

/// A byte-level BPE `tokenizer.json` (every byte symbol plus a few merges)
/// with the given normalizer and pre-tokenizer, a special added token, a
/// non-special one that strips surrounding whitespace, and a special one
/// matched on the normalized text (HF's two-stage matching).
fn segment_test_json(normalizer: Value, pre_tokenizer: Value) -> Value {
    let mut alphabet: Vec<char> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    alphabet.sort_unstable();
    let mut vocab: serde_json::Map<String, Value> = alphabet
        .iter()
        .enumerate()
        .map(|(id, c)| (c.to_string(), json!(id)))
        .collect();
    let merges = ["Ġ t", "h e", "Ġt he", "i n", "in g", "Ċ Ċ"];
    for merge in merges {
        let id = vocab.len();
        vocab.insert(merge.replace(' ', ""), json!(id));
    }
    let n = vocab.len();
    let added = |id: usize, content: &str, strip: bool, normalized: bool, special: bool| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": strip,
               "rstrip": strip, "normalized": normalized, "special": special})
    };
    json!({
        "version": "1.0",
        "added_tokens": [
            added(n, "<|a|>", false, false, true),
            added(n + 1, "<|b|>", true, false, false),
            added(n + 2, "<|c|>", false, true, true),
        ],
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": null,
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    })
}

fn segment_test_tokenizer(normalizer: Value, pre_tokenizer: Value) -> Tokenizer {
    Tokenizer::from_json(segment_test_json(normalizer, pre_tokenizer)).unwrap()
}

type Build = Box<dyn Fn() -> Tokenizer + Sync>;

/// One tokenizer per encode path, each built fresh by its closure.
fn configs() -> Vec<(&'static str, Build)> {
    let byte_level = |regex: bool, prefix: bool| json!({"type": "ByteLevel", "add_prefix_space": prefix, "trim_offsets": true, "use_regex": regex});
    let split = |regex: &str| json!({"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated", "invert": false});
    vec![
        (
            "o200k scanner",
            Box::new(|| scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN)),
        ),
        (
            "kimi scanner",
            Box::new(|| scan_test_tokenizer(tiktoken::KIMI_PATTERN)),
        ),
        (
            "o200k regex",
            Box::new(|| scan_test_tokenizer(&format!("(?:{})", tiktoken::O200K_BASE_PATTERN))),
        ),
        (
            "o200k split + byte-level, NFC",
            Box::new(move || {
                segment_test_tokenizer(
                    json!({"type": "NFC"}),
                    json!({"type": "Sequence", "pretokenizers": [
                        split(tiktoken::O200K_BASE_PATTERN), byte_level(false, false)
                    ]}),
                )
            }),
        ),
        (
            "byte-level regex + prefix space, NFC",
            Box::new(move || {
                segment_test_tokenizer(json!({"type": "NFC"}), byte_level(true, true))
            }),
        ),
        (
            "lowercase, byte-level regex",
            Box::new(move || {
                segment_test_tokenizer(json!({"type": "Lowercase"}), byte_level(true, false))
            }),
        ),
        (
            "prepend normalizer, bulk byte-level",
            Box::new(move || {
                segment_test_tokenizer(
                    json!({"type": "Prepend", "prepend": ">"}),
                    byte_level(false, false),
                )
            }),
        ),
        (
            "sequence of splits",
            Box::new(move || {
                segment_test_tokenizer(
                    Value::Null,
                    json!({"type": "Sequence", "pretokenizers": [
                        split(r"\p{N}{1,3}"), split(r"\s+|[^\s\p{N}]+"), byte_level(false, false)
                    ]}),
                )
            }),
        ),
    ]
}

/// The added tokens a request stream for `tok` is wrapped in. `<|C|>` is
/// `<|c|>` once lowercased: the Lowercase config matches it as that normalized
/// token, the others encode it as text.
fn specials_of(tok: &Tokenizer) -> Vec<&'static str> {
    if tok.token_to_id("<|a|>").is_some() {
        vec!["<|a|>", "<|b|>", "<|c|>", "<|C|>"]
    } else {
        SCAN_SPECIALS.to_vec()
    }
}

/// Request streams with history rewrites, request by request and as a batch,
/// cold and warm, with a roomy budget and one of a few hundred KiB, which
/// still takes the longest segments (a segment costing more than 1/64 of the
/// budget is not cached) but holds only part of a stream.
#[test]
fn segment_cache_matches_uncached_encoding() {
    for (name, build) in &configs() {
        let plain = build();
        let requests = chat_requests(7, &specials_of(&plain), 40);
        let expected: Vec<Vec<u32>> = requests.iter().map(|r| plain.encode(r).unwrap()).collect();

        for budget in [64 << 20, 768 << 10] {
            // Cold: every segment of the longest request misses at once.
            let mut cold = build();
            cold.enable_segment_cache(budget);
            let last = requests.len() - 1;
            assert_eq!(
                cold.encode(&requests[last]).unwrap(),
                expected[last],
                "{name}: cold"
            );

            let mut cached = build();
            cached.enable_segment_cache(budget);
            for (i, (request, want)) in requests.iter().zip(&expected).enumerate() {
                assert_eq!(
                    &cached.encode(request).unwrap(),
                    want,
                    "{name}: request {i}, budget {budget}"
                );
            }
            // Again, now mostly from the cache, and concurrently.
            assert_eq!(
                cached.encode_batch(&requests, false).unwrap(),
                expected,
                "{name}: batch"
            );
            let stats = cached.segment_cache_stats().unwrap();
            assert!(
                stats.entries > 0 && stats.hits > stats.misses,
                "{name}: {stats:?}"
            );
            assert!(stats.bytes <= budget, "{name}: {stats:?}");
        }
    }
}

/// Every added-token policy, and `encode_segments` with a trusted template
/// around an untrusted body that contains the same text as a trusted segment:
/// the policy decides where the segment boundaries are, the cache keys by
/// the normalized text between them.
#[test]
fn segment_cache_matches_under_every_policy() {
    for (name, build) in &configs() {
        let plain = build();
        let mut cached = build();
        cached.enable_segment_cache(1 << 20);
        let specials = specials_of(&plain);
        for (i, request) in chat_requests(11, &specials, 20).iter().enumerate() {
            for policy in [AddedTokenPolicy::All, AddedTokenPolicy::SkipSpecial] {
                let special = i % 2 == 0;
                assert_eq!(
                    cached.encode_with_policy(request, special, policy).unwrap(),
                    plain.encode_with_policy(request, special, policy).unwrap(),
                    "{name}: request {i}, {policy:?}"
                );
            }
            let segments = [
                EncodeSegment::special(specials[0]),
                EncodeSegment::ordinary(request),
                EncodeSegment::special(request),
                EncodeSegment::special(specials[1]),
            ];
            assert_eq!(
                cached.encode_segments(&segments).unwrap(),
                plain.encode_segments(&segments).unwrap(),
                "{name}: request {i}, encode_segments"
            );
        }
    }
}

/// `encode_segments`: an untrusted segment is cached whole, so a message body
/// re-sent as its own segment is a hit the second time; the ids match the same
/// call without the cache, and `encode` of the flattened text.
#[test]
fn segment_cache_covers_untrusted_segments_of_encode_segments() {
    for (name, build) in &configs() {
        let plain = build();
        let mut cached = build();
        cached.enable_segment_cache(1 << 20);
        let specials = specials_of(&plain);
        let body = prose(&mut Rng::new(3), 200);
        let segments = [
            EncodeSegment::special(specials[0]),
            EncodeSegment::ordinary(&body),
            EncodeSegment::special(specials[1]),
        ];
        for round in 0..2 {
            assert_eq!(
                cached.encode_segments(&segments).unwrap(),
                plain.encode_segments(&segments).unwrap(),
                "{name}: round {round}"
            );
        }
        let stats = cached.segment_cache_stats().unwrap();
        assert_eq!((stats.hits, stats.misses), (1, 1), "{name}: {stats:?}");
    }
}

/// `encode_segments`: an untrusted segment too long to ever be cached is not
/// looked up, and still encodes as without the cache.
#[test]
fn segment_cache_skips_untrusted_segments_too_long_to_cache() {
    for (name, build) in &configs() {
        let plain = build();
        let mut cached = build();
        // An entry may cost at most 1/64 of the budget: 1 KiB here.
        cached.enable_segment_cache(64 << 10);
        let specials = specials_of(&plain);
        let short = prose(&mut Rng::new(5), 15);
        let long = prose(&mut Rng::new(6), 2000);
        assert!(long.len() > 1024, "{name}");
        let segments = [
            EncodeSegment::special(specials[0]),
            EncodeSegment::ordinary(&short),
            EncodeSegment::special(specials[1]),
            EncodeSegment::ordinary(&long),
        ];
        for round in 0..2 {
            assert_eq!(
                cached.encode_segments(&segments).unwrap(),
                plain.encode_segments(&segments).unwrap(),
                "{name}: round {round}"
            );
        }
        let stats = cached.segment_cache_stats().unwrap();
        assert_eq!((stats.hits, stats.misses), (1, 1), "{name}: {stats:?}");
        assert!(stats.miss_bytes < 1024, "{name}: {stats:?}");
    }
}

/// Four sessions replayed twice from four threads sharing one cache too small
/// for all of them, so entries keep being evicted and promoted, on a scanner
/// path and on the general path.
#[test]
fn segment_cache_is_shared_safely_across_threads() {
    let configs = configs();
    for (name, build) in [&configs[0], &configs[4]] {
        let plain = build();
        let specials = specials_of(&plain);
        let streams: Vec<Vec<String>> = (0..4)
            .map(|s| chat_requests(100 + s, &specials, 40))
            .collect();
        let expected: Vec<Vec<Vec<u32>>> = streams
            .iter()
            .map(|s| s.iter().map(|r| plain.encode(r).unwrap()).collect())
            .collect();
        let mut cached = build();
        cached.enable_segment_cache(768 << 10);
        let cached = &cached;
        std::thread::scope(|scope| {
            for (stream, want) in streams.iter().zip(&expected) {
                scope.spawn(move || {
                    for round in 0..2 {
                        for (i, (request, want)) in stream.iter().zip(want).enumerate() {
                            assert_eq!(
                                &cached.encode(request).unwrap(),
                                want,
                                "{name}: round {round}, request {i}"
                            );
                        }
                    }
                });
            }
        });
        let stats = cached.segment_cache_stats().unwrap();
        assert!(
            stats.entries > 0 && stats.hits > stats.misses,
            "{name}: {stats:?}"
        );
        assert!(stats.bytes <= 768 << 10, "{name}: {stats:?}");
    }
}

#[test]
fn segment_cache_counts_lookups_and_ignores_plain_inputs() {
    let mut tok = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
    tok.enable_segment_cache(1 << 20);
    let plain = prose(&mut Rng::new(1), 80);
    let plain_ids = tok.encode(&plain).unwrap();
    assert_eq!(
        tok.segment_cache_stats().unwrap(),
        crate::SegmentCacheStats {
            max_bytes: 1 << 20,
            ..Default::default()
        }
    );

    let a = format!("<|im_start|>{plain}<|im_end|>");
    tok.encode(&a).unwrap();
    let stats = tok.segment_cache_stats().unwrap();
    assert_eq!((stats.hits, stats.misses), (0, 1));
    assert_eq!(stats.miss_bytes, plain.len() as u64);
    assert_eq!(stats.entries, 1);
    assert!(stats.bytes >= plain.len() + plain_ids.len() * 4);

    // The same segment after other tokens, twice in one input: two hits.
    let b = format!("<tool_call>{plain}</tool_call>{plain}<|im_end|>");
    tok.encode(&b).unwrap();
    let stats = tok.segment_cache_stats().unwrap();
    assert_eq!((stats.hits, stats.misses), (2, 1));
    assert_eq!(stats.hit_bytes, 2 * plain.len() as u64);
    assert_eq!(stats.entries, 1);

    // The same on the general path (a tokenizer with a normalizer, an input
    // that is not its own normalized form): a new segment repeated within one
    // input is a miss, then a hit.
    let mut tok = segment_test_tokenizer(
        json!({"type": "NFC"}),
        json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}),
    );
    tok.enable_segment_cache(1 << 20);
    tok.encode("<|a|>new e\u{301}<|a|>new e\u{301}<|a|>")
        .unwrap();
    let stats = tok.segment_cache_stats().unwrap();
    assert_eq!((stats.hits, stats.misses), (1, 1));
    assert_eq!(stats.hit_bytes, stats.miss_bytes);
    assert_eq!(stats.entries, 1);

    // A zero budget turns the cache off.
    tok.enable_segment_cache(0);
    assert!(tok.segment_cache_stats().is_none());
}

/// `add_tokens` clears the cache: a segment that contained the new token's
/// text was cached as one piece and is now split by it, so its entry can no
/// longer be looked up anyway.
#[test]
fn segment_cache_stays_correct_when_tokens_are_added() {
    let build = || {
        segment_test_tokenizer(
            Value::Null,
            json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}),
        )
    };
    let mut plain = build();
    let mut cached = build();
    cached.enable_segment_cache(1 << 20);
    let input = "<|a|>say hi to <|d|> now<|a|>say hi to <|d|> now";
    assert_eq!(cached.encode(input).unwrap(), plain.encode(input).unwrap());
    for tok in [&mut plain, &mut cached] {
        tok.add_tokens(&[NewToken::special("<|d|>")]).unwrap();
    }
    assert_eq!(cached.segment_cache_stats().unwrap().entries, 0);
    // "say hi to <|d|> now" was one segment; it is now three.
    assert_eq!(cached.encode(input).unwrap(), plain.encode(input).unwrap());
    assert_eq!(cached.segment_cache_stats().unwrap().entries, 2);
}

/// BPE dropout makes an encoding random by design, so there is nothing to
/// reuse: the cache stays off for such a model.
#[test]
fn segment_cache_stays_off_with_bpe_dropout() {
    let mut config = segment_test_json(
        Value::Null,
        json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}),
    );
    config["model"]["dropout"] = json!(0.5);
    let mut tok = Tokenizer::from_json(config).unwrap();
    tok.enable_segment_cache(1 << 20);
    assert!(tok.segment_cache_stats().is_none());
    tok.encode("<|a|>the thing<|a|>the thing<|a|>").unwrap();
    assert!(tok.segment_cache_stats().is_none());
}
