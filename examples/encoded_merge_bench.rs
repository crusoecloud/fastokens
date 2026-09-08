//! Benchmark the generic encoded-BPE cache-miss merger.
//!
//! Each input is a fresh ASCII pretoken-shaped string, so it is already in the
//! representation consumed by `Bpe::tokenize` and cannot hit either BPE cache.
//! The leading `c` also keeps the whole input from matching the `ab` vocabulary
//! token, forcing the encoded merge path. Use `--symbols` to select the exact
//! initial-symbol bucket, including the 32/33 crossover.

use std::{collections::HashSet, env, hint::black_box, time::Instant};

use fastokens::models::bpe::Bpe;
use serde_json::json;

const DEFAULT_ITERATIONS: usize = 8_192;
const WARMUP: usize = 1_024;
const ALPHABET: &[u8] = b"abcdefghijklmnop";

fn fixture() -> Bpe {
    serde_json::from_value(json!({
        "vocab": {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
            "i": 8,
            "j": 9,
            "k": 10,
            "l": 11,
            "m": 12,
            "n": 13,
            "o": 14,
            "p": 15,
            "ab": 16
        },
        "merges": ["a b"]
    }))
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
        input.push('c');
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
    let warmup = inputs(symbols, WARMUP, &mut state);
    let measured = inputs(symbols, iterations, &mut state);

    for input in &warmup {
        black_box(bpe.tokenize(input).expect("benchmark input must tokenize"));
    }

    let start = Instant::now();
    let mut checksum = 0u64;
    for input in &measured {
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
