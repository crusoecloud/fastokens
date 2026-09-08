//! Benchmark the generic encoded-BPE cache-miss merger.
//!
//! Each input is a fresh encoded pretoken-shaped string, so it is already in the
//! representation consumed by `Bpe::tokenize` and cannot hit either BPE cache.
//! The generic non-fused tokenizer calls this model entry for each encoded
//! split; using it directly keeps pre-tokenization outside the timed operation.
//! The leading `z` also keeps the whole input from matching a pair token,
//! forcing the encoded merge path. Use `--symbols` to select the exact
//! initial-symbol bucket, including the 32/33 crossover. Pass `--cpu-time` to
//! also report process CPU time for the measured loop on Unix.

use std::{
    env,
    hint::black_box,
    mem::MaybeUninit,
    time::{Duration, Instant},
};

use fastokens::models::bpe::Bpe;
use serde_json::{Map, Value, json};

const DEFAULT_ITERATIONS: usize = 8_192;
const WARMUP: usize = 1_024;
const ALPHABET: &[u8] = b"abcdefghijklmnop";
const ALIAS_BASE: u32 = 0x1000;
const ALIAS_COUNT: usize = 16_384;

fn alias_char(index: usize) -> char {
    char::from_u32(ALIAS_BASE + index as u32).expect("benchmark alias must be a Unicode scalar")
}

#[cfg(unix)]
fn process_cpu_time() -> Duration {
    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    assert_eq!(result, 0, "getrusage failed with status {result}");

    let usage = unsafe { usage.assume_init() };
    let user = timeval_duration(usage.ru_utime);
    let system = timeval_duration(usage.ru_stime);
    user.checked_add(system)
        .expect("process CPU time overflowed")
}

#[cfg(unix)]
fn timeval_duration(timeval: libc::timeval) -> Duration {
    let seconds = u64::try_from(timeval.tv_sec).expect("negative CPU time seconds");
    let micros = u64::try_from(timeval.tv_usec).expect("negative CPU time microseconds");
    Duration::from_secs(seconds)
        .checked_add(Duration::from_micros(micros))
        .expect("CPU time overflowed")
}

#[cfg(not(unix))]
fn process_cpu_time() -> Duration {
    panic!("--cpu-time requires a Unix process CPU clock")
}

fn fixture() -> Bpe {
    let mut vocab = Map::new();
    for (id, &byte) in ALPHABET.iter().enumerate() {
        vocab.insert((byte as char).to_string(), Value::from((id + 1) as u32));
    }
    vocab.insert("z".into(), Value::from(0u32));

    // Every pair of body symbols is mergeable. The leading `z` is not part of
    // this table, so it prevents the whole input from matching a vocabulary
    // token while leaving the measured body merge-heavy.
    let mut merges = Vec::with_capacity(ALPHABET.len() * ALPHABET.len());
    for &left in ALPHABET {
        for &right in ALPHABET {
            let merged = format!("{}{}", left as char, right as char);
            let id = vocab.len() as u32;
            vocab.insert(merged, Value::from(id));
            merges.push(Value::String(format!("{} {}", left as char, right as char)));
        }
    }

    // The aliases give the cache-miss corpus a large finite key space without
    // changing the 16-symbol merge topology. Each alias maps to one of the
    // dense body IDs, but no alias is itself a representative vocabulary token,
    // so a generated input cannot take the whole-token fast path.
    for index in 0..ALIAS_COUNT {
        let id = 1 + (index % ALPHABET.len()) as u32;
        vocab.insert(alias_char(index).to_string(), Value::from(id));
    }

    serde_json::from_value(json!({"vocab": vocab, "merges": merges}))
        .expect("benchmark BPE fixture must deserialize")
}

fn inputs(symbols: usize, count: usize) -> Vec<String> {
    assert!(
        symbols >= 2,
        "a measured input must not be a one-token match"
    );
    let body_len = symbols - 1;
    let capacity = (0..body_len)
        .try_fold(1usize, |capacity, _| capacity.checked_mul(ALIAS_COUNT))
        .unwrap_or(usize::MAX);
    assert!(
        count <= capacity,
        "requested {count} inputs but --symbols {symbols} has capacity {capacity}"
    );

    // Decode each ordinal in base ALIAS_COUNT. The per-position affine map is
    // bijective because ALIAS_COUNT is a power of two and 5 is odd, so this
    // produces exactly `count` distinct strings without retrying a HashSet.
    let mut result = Vec::with_capacity(count);
    for ordinal in 0..count {
        let mut value = ordinal;
        let mut input = String::with_capacity(1 + body_len * 3);
        input.push('z');
        for position in 0..body_len {
            let digit = value % ALIAS_COUNT;
            value /= ALIAS_COUNT;
            let alias = digit
                .wrapping_mul(5)
                .wrapping_add(position.wrapping_mul(257))
                & (ALIAS_COUNT - 1);
            input.push(alias_char(alias));
        }
        result.push(input);
    }
    result
}

fn main() {
    let mut symbols = None;
    let mut iterations = DEFAULT_ITERATIONS;
    let mut measure_cpu_time = false;
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
            "--cpu-time" => measure_cpu_time = true,
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
    let total_inputs = WARMUP
        .checked_add(iterations)
        .expect("warmup plus iterations overflowed");
    let all_inputs = inputs(symbols, total_inputs);
    let (warmup, measured) = all_inputs.split_at(WARMUP);

    for input in warmup {
        black_box(bpe.tokenize(input).expect("benchmark input must tokenize"));
    }

    let cpu_start = measure_cpu_time.then(process_cpu_time);
    let start = Instant::now();
    let mut checksum = 0u64;
    for input in measured {
        let ids = bpe.tokenize(input).expect("benchmark input must tokenize");
        for id in ids {
            checksum = checksum.rotate_left(7) ^ u64::from(id);
        }
    }
    let elapsed = start.elapsed();
    let cpu_elapsed = cpu_start.map(|start| {
        process_cpu_time()
            .checked_sub(start)
            .expect("process CPU clock moved backwards")
    });
    black_box(checksum);

    let ns_per_op = elapsed.as_secs_f64() * 1e9 / iterations as f64;
    eprintln!(
        "encoded generic cache misses: symbols={symbols}, iterations={iterations}, checksum={checksum}"
    );
    println!(r#"{{"metric":"ns/op","value":{ns_per_op:.3}}}"#);
    if let Some(cpu_elapsed) = cpu_elapsed {
        let cpu_ns_per_op = cpu_elapsed.as_secs_f64() * 1e9 / iterations as f64;
        println!(r#"{{"metric":"cpu-ns/op","value":{cpu_ns_per_op:.3}}}"#);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn corpus_is_finite_and_unique_at_small_and_boundary_sizes() {
        for symbols in [2, 3, 4, 16, 32, 33] {
            let inputs = inputs(symbols, 256);
            assert_eq!(inputs.len(), 256);
            assert!(inputs.iter().all(|input| input.chars().count() == symbols));
            let unique: HashSet<_> = inputs.iter().collect();
            assert_eq!(unique.len(), inputs.len());
        }
    }

    #[test]
    #[should_panic(expected = "has capacity 16384")]
    fn corpus_rejects_requests_larger_than_the_smallest_domain() {
        let _ = inputs(2, ALIAS_COUNT + 1);
    }
}
