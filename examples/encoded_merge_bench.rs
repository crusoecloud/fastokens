//! Benchmark the generic encoded-BPE cache-miss merger.
//!
//! Each input is a fresh ASCII pretoken-shaped string, so it is already in the
//! representation consumed by `Bpe::tokenize` and cannot hit either BPE cache.
//! The generic non-fused tokenizer calls this model entry for each encoded
//! split; using it directly keeps pre-tokenization outside the timed operation.
//! The leading `z` also keeps the whole input from matching a pair token,
//! forcing the encoded merge path. Use `--symbols` to select the exact
//! initial-symbol bucket, including the 32/33 crossover.

use std::{collections::HashSet, env, hint::black_box, time::Instant};

use fastokens::models::bpe::Bpe;
use serde_json::{Map, Value, json};

const DEFAULT_ITERATIONS: usize = 8_192;
const WARMUP: usize = 1_024;
const ALPHABET: &[u8] = b"abcdefghijklmnop";

fn fixture() -> Bpe {
    let mut vocab = Map::new();
    for (id, &byte) in ALPHABET.iter().enumerate() {
        vocab.insert((byte as char).to_string(), Value::from(id as u32));
    }
    vocab.insert("z".into(), Value::from(ALPHABET.len() as u32));

    // Every pair of body symbols is mergeable. The leading `z` is deliberately
    // not part of this table, so it prevents the whole input from being a
    // vocabulary match while leaving the measured body merge-heavy.
    let mut merges = Vec::with_capacity(ALPHABET.len() * ALPHABET.len());
    for &left in ALPHABET {
        for &right in ALPHABET {
            let merged = format!("{}{}", left as char, right as char);
            let id = vocab.len() as u32;
            vocab.insert(merged, Value::from(id));
            merges.push(Value::String(format!("{} {}", left as char, right as char)));
        }
    }

    serde_json::from_value(json!({"vocab": vocab, "merges": merges}))
        .expect("benchmark BPE fixture must deserialize")
}

fn inputs(symbols: usize, count: usize, state: &mut u64) -> Vec<String> {
    assert!(
        symbols >= 2,
        "a measured input must not be a one-token match"
    );
    let mut seen = HashSet::with_capacity(count);
    let mut result = Vec::with_capacity(count);
    while result.len() < count {
        let mut value = *state;
        let mut input = String::with_capacity(symbols);
        input.push('z');
        for _ in 1..symbols {
            // A deterministic stream gives every invocation the same workload,
            // while the set makes each call a cache miss in the BPE caches.
            value ^= value << 13;
            value ^= value >> 7;
            value ^= value << 17;
            input.push(ALPHABET[value as usize & (ALPHABET.len() - 1)] as char);
        }
        *state = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
        if seen.insert(input.clone()) {
            result.push(input);
        }
    }
    result
}

fn main() {
    let mut symbols = None;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--symbols" => {
                symbols = Some(
                    args.next()
                        .expect("--symbols needs a value")
                        .parse()
                        .expect("invalid symbol count"),
                )
            }
            "--iterations" => {
                iterations = args
                    .next()
                    .expect("--iterations needs a value")
                    .parse()
                    .expect("invalid iteration count")
            }
            other => panic!("unknown argument {other:?}"),
        }
    }
    let symbols = symbols.expect("usage: encoded_merge_bench --symbols N [--iterations N]");
    assert!(
        symbols <= 64,
        "the benchmark only covers the short branch and its guard"
    );
    assert!(iterations > 0, "iterations must be positive");

    let bpe = fixture();
    let mut state = 0x243f_6a88_85a3_08d3u64 ^ symbols as u64;
    let all_inputs = inputs(symbols, WARMUP + iterations, &mut state);
    let (warmup, measured) = all_inputs.split_at(WARMUP);

    for input in warmup {
        black_box(bpe.tokenize(input).expect("benchmark input must tokenize"));
    }

    let start = Instant::now();
    let mut checksum = 0u64;
    for input in measured {
        let ids = bpe.tokenize(input).expect("benchmark input must tokenize");
        for id in ids {
            checksum = checksum.rotate_left(7) ^ u64::from(id);
        }
    }
    let elapsed = start.elapsed();
    black_box(checksum);

    let ns_per_op = elapsed.as_secs_f64() * 1e9 / iterations as f64;
    eprintln!(
        "encoded generic cache misses: symbols={symbols}, iterations={iterations}, checksum={checksum}"
    );
    println!(r#"{{"metric":"ns/op","value":{ns_per_op:.3}}}"#);
}
