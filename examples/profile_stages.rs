use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::json;

fn main() -> Result<()> {
    let model_name = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16";

    let tokenizer =
        fastokens::Tokenizer::from_model(model_name).context("failed to load tokenizer")?;

    let text = std::fs::read_to_string("outputs.txt").context("failed to read outputs.txt")?;
    let words: Vec<&str> = text.split_whitespace().collect();
    let prefix_len = (32_000.0_f64 / 70_000.0 * 70_000.0).round() as usize;
    let (prefix_words, rest) = words.split_at(prefix_len);
    let prefix = prefix_words.join(" ");
    let chunk_words = &rest[..3000];
    let input = format!("{} {}", prefix, chunk_words.join(" "));
    println!("Input length: {} bytes", input.len());

    let n = 10;

    // Profile full encode (with warmup)
    let _ = tokenizer.encode(&input);
    let t = Instant::now();
    for _ in 0..n {
        let ids = tokenizer.encode(&input).unwrap();
        std::hint::black_box(ids);
    }
    println!(
        "Full encode:    {:.2} ms avg ({n} iters)",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );

    // Build the initial pre-tokenized string
    use fastokens::pre_tokenized::{PreTokenizedString, Split as PtSplit};
    let normalizer = tokenizer.normalizer();
    let normalized = match normalizer {
        Some(n) => n.normalize(&input),
        None => std::borrow::Cow::Borrowed(input.as_str()),
    };
    let mut buffer = String::with_capacity(input.len());
    buffer.push_str(&normalized);
    let splits = vec![PtSplit {
        range: 0..buffer.len(),
        token_id: None,
    }];
    let base_pts = PreTokenizedString::new(buffer, splits);

    // Create Split pre-tokenizer directly
    let split_pattern = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let split =
        fastokens::Split::from_config(&json!({"Regex": split_pattern}), "Isolated", false).unwrap();

    // Create ByteLevel pre-tokenizer directly
    let byte_level = fastokens::ByteLevel::from_config(false, true, false).unwrap();

    // Profile Split only
    {
        let mut pts = base_pts.clone();
        split.pre_tokenize(&mut pts).unwrap();
    }
    let t = Instant::now();
    for _ in 0..n {
        let mut pts = base_pts.clone();
        split.pre_tokenize(&mut pts).unwrap();
    }
    println!(
        "Split only:     {:.2} ms avg",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );

    // Profile ByteLevel only (after Split)
    let mut after_split = base_pts.clone();
    split.pre_tokenize(&mut after_split).unwrap();
    let num_splits = after_split.splits().len();
    println!("Splits after Split: {num_splits}");
    {
        let mut pts = after_split.clone();
        byte_level.pre_tokenize(&mut pts).unwrap();
    }
    let t = Instant::now();
    for _ in 0..n {
        let mut pts = after_split.clone();
        byte_level.pre_tokenize(&mut pts).unwrap();
    }
    println!(
        "ByteLevel only: {:.2} ms avg",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );

    // Profile BPE tokenize (non-fused, via full pre-tokenizer)
    let model = tokenizer.model();
    let pre_tok = tokenizer.pre_tokenizer().unwrap();
    let mut after_pretok = base_pts.clone();
    pre_tok.pre_tokenize(&mut after_pretok).unwrap();
    println!("Splits after pre-tok: {}", after_pretok.splits().len());

    // Count unique splits
    {
        let mut unique: std::collections::HashSet<&str> = std::collections::HashSet::new();
        let mut single_token = 0usize;
        for s in after_pretok.splits() {
            let t = after_pretok.split_text(s);
            unique.insert(t);
            if t.len() <= 4 {
                single_token += 1;
            }
        }
        println!(
            "Unique splits: {} (short <=4 bytes: {})",
            unique.len(),
            single_token
        );
    }

    {
        let _ = after_pretok.tokenize(|t, out| model.tokenize_into(t, out));
    }
    let t = Instant::now();
    for _ in 0..n {
        let ids = after_pretok
            .tokenize(|t, out| model.tokenize_into(t, out))
            .unwrap();
        std::hint::black_box(ids);
    }
    println!(
        "BPE tokenize:   {:.2} ms avg",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );

    // Profile fused BPE tokenize (Split-only + fused BPE)
    {
        let mut pts = base_pts.clone();
        split.pre_tokenize(&mut pts).unwrap();
        let _ = pts.tokenize(|t, out| model.tokenize_into_fused(t, out));
    }
    let t = Instant::now();
    for _ in 0..n {
        let mut pts = base_pts.clone();
        split.pre_tokenize(&mut pts).unwrap();
        let ids = pts
            .tokenize(|t, out| model.tokenize_into_fused(t, out))
            .unwrap();
        std::hint::black_box(ids);
    }
    println!(
        "Fused tok:      {:.2} ms avg (Split + fused BPE)",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );

    // Profile BPE tokenize sequential (no rayon)
    {
        let _ = after_pretok.tokenize_sequential_pub(|t, out| model.tokenize_into(t, out));
    }
    let t = Instant::now();
    for _ in 0..n {
        let ids = after_pretok
            .tokenize_sequential_pub(|t, out| model.tokenize_into(t, out))
            .unwrap();
        std::hint::black_box(ids);
    }
    println!(
        "BPE seq:        {:.2} ms avg (sequential, no rayon)",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );

    // Profile iteration overhead only (no BPE, just iterate splits)
    let t = Instant::now();
    for _ in 0..n {
        let mut count = 0u64;
        for s in after_pretok.splits() {
            count += after_pretok.split_text(s).len() as u64;
        }
        std::hint::black_box(count);
    }
    println!(
        "Iter only:      {:.2} ms avg (iterate splits, no BPE)",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );

    // Profile build_pre_tokenized (added tokens + NFC + buffer copy)
    {
        let _ = tokenizer.build_pre_tokenized(&input);
    }
    let t = Instant::now();
    for _ in 0..n {
        let pts = tokenizer.build_pre_tokenized(&input);
        std::hint::black_box(pts);
    }
    println!(
        "build_pre_tok:  {:.2} ms avg",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );

    // Profile NFC normalization alone
    let normalizer = tokenizer.normalizer();
    {
        if let Some(n) = normalizer {
            let _ = n.normalize(&input);
        }
    }
    let t = Instant::now();
    for _ in 0..n {
        if let Some(n) = normalizer {
            let r = n.normalize(&input);
            std::hint::black_box(r);
        }
    }
    println!(
        "NFC normalize:  {:.2} ms avg",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );

    // Profile just buffer copy (to measure memcpy cost)
    let t = Instant::now();
    for _ in 0..n {
        let s = input.to_string();
        std::hint::black_box(s);
    }
    println!(
        "String copy:    {:.2} ms avg",
        t.elapsed().as_secs_f64() * 1000.0 / n as f64
    );

    // Profile contains("<|") prefilter
    let t = Instant::now();
    for _ in 0..100 {
        let r = input.contains("<|");
        std::hint::black_box(r);
    }
    println!(
        "contains <|:    {:.2} ms avg (100 iters)",
        t.elapsed().as_secs_f64() * 1000.0 / 100.0
    );

    // Profile NFC on the actual input with 100 iters for accuracy
    let t = Instant::now();
    for _ in 0..100 {
        if let Some(n) = normalizer {
            let r = n.normalize(&input);
            std::hint::black_box(r);
        }
    }
    println!(
        "NFC (100 iter): {:.2} ms avg",
        t.elapsed().as_secs_f64() * 1000.0 / 100.0
    );

    // Profile from_text (string allocation + copy)
    let t = Instant::now();
    for _ in 0..100 {
        let pts = PreTokenizedString::from_text(&input);
        std::hint::black_box(pts);
    }
    println!(
        "from_text:      {:.2} ms avg (100 iters)",
        t.elapsed().as_secs_f64() * 1000.0 / 100.0
    );

    // Profile memchr for '<'
    let t = Instant::now();
    for _ in 0..100 {
        let r = input.as_bytes().iter().position(|&b| b == b'<');
        std::hint::black_box(r);
    }
    println!(
        "find '<':       {:.2} ms avg (100 iters)",
        t.elapsed().as_secs_f64() * 1000.0 / 100.0
    );

    println!("\nDone.");
    Ok(())
}
