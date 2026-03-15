use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;
use std::{io, time::Instant};

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::api::sync::Api;
use indicatif::{ProgressBar, ProgressStyle};

/// Tokenizer benchmark tool
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// HuggingFace Hub model name (e.g. deepseek-ai/DeepSeek-V3)
    model: String,

    /// Dataset to use (default: "zai-org/LongBench-v2")
    #[arg(long, default_value = "zai-org/LongBench-v2")]
    dataset: String,

    /// Maximum number of samples to process
    #[arg(short = 'n', long)]
    max_samples: Option<usize>,

    /// Output CSV file path for per-input benchmark results
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Print per-sample results instead of a progress bar
    #[arg(short, long)]
    verbose: bool,

    /// Batch size for batched encoding benchmark. When set, groups samples
    /// into batches of this size and benchmarks encode_batch.
    #[arg(short = 'b', long)]
    batch_size: Option<usize>,
}

/// Return the JSON filename for a known HuggingFace Hub dataset.
fn dataset_json_file(dataset: &str) -> Result<&'static str> {
    match dataset {
        "RyokoAI/ShareGPT52K" => Ok("sg_90k_part1.json"),
        "zai-org/LongBench-v2" => Ok("data.json"),
        _ => anyhow::bail!("unknown dataset: {dataset:?}"),
    }
}

/// Extract a text sample from a single JSON item, based on the dataset.
fn extract_text(dataset: &str, item: &serde_json::Value) -> Result<Option<String>> {
    match dataset {
        "RyokoAI/ShareGPT52K" => {
            let Some(messages) = item.get("conversations").and_then(|v| v.as_array()) else {
                return Ok(None);
            };
            let parts: Vec<String> = messages
                .iter()
                .filter_map(|msg| {
                    let role = msg
                        .get("from")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let value = msg.get("value").and_then(|v| v.as_str())?;
                    if value.is_empty() {
                        return None;
                    }
                    Some(format!("[{role}]: {value}"))
                })
                .collect();
            if parts.is_empty() {
                return Ok(None);
            }
            Ok(Some(parts.join("\n\n")))
        }
        "zai-org/LongBench-v2" => {
            let Some(context) = item.get("context").and_then(|v| v.as_str()) else {
                return Ok(None);
            };
            if context.is_empty() {
                return Ok(None);
            }
            Ok(Some(context.to_string()))
        }
        _ => anyhow::bail!("unknown dataset: {dataset:?}"),
    }
}

/// Load text samples from a HuggingFace Hub dataset.
fn load_dataset(dataset: &str, max_items: Option<usize>, verbose: bool) -> Result<Vec<String>> {
    let json_file = dataset_json_file(dataset)?;

    if verbose {
        eprintln!("Downloading {dataset} from HuggingFace Hub...");
    }
    let api = Api::new().context("failed to create HuggingFace Hub API")?;
    let repo = api.dataset(dataset.to_string());
    let json_path = repo.get(json_file).context("failed to download dataset")?;
    if verbose {
        eprintln!("Downloaded to: {}", json_path.display());
    }

    let text = std::fs::read_to_string(&json_path)
        .with_context(|| format!("failed to read {}", json_path.display()))?;
    let data: Vec<serde_json::Value> =
        serde_json::from_str(&text).context("failed to parse dataset JSON")?;

    let limit = max_items.unwrap_or(usize::MAX);
    let samples: Vec<String> = data
        .iter()
        .take(limit)
        .map(|item| extract_text(dataset, item))
        .filter_map(Result::transpose)
        .collect::<Result<_>>()?;

    Ok(samples)
}

fn print_summary(
    label: &str,
    n: usize,
    total_chars: u64,
    total_tokens: u64,
    total_hf: Duration,
    total_ft: Duration,
) {
    let nf = n as f64;
    let hf_ms = total_hf.as_secs_f64() * 1000.0;
    let ft_ms = total_ft.as_secs_f64() * 1000.0;
    let speedup = hf_ms / ft_ms;

    println!();
    println!("═══════════════════════════════════════════");
    println!("  {label} ({n} samples)");
    println!("═══════════════════════════════════════════");
    println!("  Total chars:    {total_chars}");
    println!("  Total tokens:   {total_tokens}");
    println!("───────────────────────────────────────────");
    println!("  HF total:       {hf_ms:>10.2} ms");
    println!("  fastokens total:{ft_ms:>10.2} ms");
    println!("  Speedup:        {speedup:>10.2}x");
    println!("───────────────────────────────────────────");
    println!("  HF avg/sample:  {:>10.3} ms", hf_ms / nf);
    println!("  ft avg/sample:  {:>10.3} ms", ft_ms / nf);
    println!(
        "  HF throughput:  {:>10.2} MB/s",
        total_chars as f64 / total_hf.as_secs_f64() / 1_000_000.0
    );
    println!(
        "  ft throughput:  {:>10.2} MB/s",
        total_chars as f64 / total_ft.as_secs_f64() / 1_000_000.0
    );
    println!("═══════════════════════════════════════════");
}

fn bench_sequential(
    chunks: &[String],
    hf_tokenizer: &tokenizers::Tokenizer,
    tokenizer: &fastokens::Tokenizer,
    verbose: bool,
    mut csv_writer: Option<&mut io::BufWriter<std::fs::File>>,
) -> Result<()> {
    let pb = if verbose {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(chunks.len() as u64);
        pb.set_style(
            ProgressStyle::with_template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({eta})")
                .expect("valid template")
                .progress_chars("=> "),
        );
        pb
    };

    let mut total_hf = Duration::ZERO;
    let mut total_ft = Duration::ZERO;
    let mut total_tokens: u64 = 0;
    let mut total_chars: u64 = 0;

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_len = chunk.len();

        let t0 = Instant::now();
        let enc_hf = hf_tokenizer
            .encode_fast(chunk.as_str(), true)
            .map_err(|e| anyhow::anyhow!(e))
            .context("HF tokenizer encode failed")?;
        let enc_hf = enc_hf.get_ids();
        let t1 = Instant::now();
        let enc = tokenizer
            .encode_with_special_tokens(chunk, true)
            .context("fastokens encode failed")?;
        let t2 = Instant::now();

        if enc_hf != enc {
            panic!(
                "Output mismatch for input {i} ({} differences):\n\
                 Input: {:?}\n hf[:100]: {:?}\n ft[:100]: {:?}",
                std::iter::zip(enc_hf.iter().copied(), enc.iter().copied())
                    .filter(|(a, b)| a != b)
                    .count(),
                chunk,
                &enc_hf[..enc_hf.len().min(100)],
                &enc[..enc.len().min(100)],
            );
        }

        let dt_hf = t1 - t0;
        let dt = t2 - t1;

        total_hf += dt_hf;
        total_ft += dt;
        total_tokens += enc.len() as u64;
        total_chars += chunk_len as u64;

        if verbose {
            println!(
                "[{}/{}] {} chars, {} tokens | hf: {:.3} ms, ft: {:.3} ms ({:.1}x)",
                i + 1,
                chunks.len(),
                chunk_len,
                enc.len(),
                dt_hf.as_secs_f64() * 1000.0,
                dt.as_secs_f64() * 1000.0,
                dt_hf.as_secs_f64() / dt.as_secs_f64(),
            );
        } else {
            pb.inc(1);
        }

        if let Some(w) = csv_writer.as_mut() {
            writeln!(
                w,
                "{},{},{},{},{}",
                i,
                chunk_len,
                enc.len(),
                dt_hf.as_secs_f64() * 1000.0,
                dt.as_secs_f64() * 1000.0,
            )
            .context("failed to write CSV row")?;
        }
    }

    pb.finish();
    print_summary(
        "Benchmark Summary",
        chunks.len(),
        total_chars,
        total_tokens,
        total_hf,
        total_ft,
    );
    Ok(())
}

fn bench_batched(
    chunks: &[String],
    hf_tokenizer: &tokenizers::Tokenizer,
    tokenizer: &fastokens::Tokenizer,
    batch_size: usize,
    verbose: bool,
    mut csv_writer: Option<&mut io::BufWriter<std::fs::File>>,
) -> Result<()> {
    use tokenizers::EncodeInput;

    let num_batches = (chunks.len() + batch_size - 1) / batch_size;

    let pb = if verbose {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(num_batches as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] [{bar:40}] {pos}/{len} batches ({eta})",
            )
            .expect("valid template")
            .progress_chars("=> "),
        );
        pb
    };

    let mut total_hf = Duration::ZERO;
    let mut total_ft = Duration::ZERO;
    let mut total_tokens: u64 = 0;
    let mut total_chars: u64 = 0;

    for (batch_idx, batch) in chunks.chunks(batch_size).enumerate() {
        let batch_chars: usize = batch.iter().map(|s| s.len()).sum();

        let hf_inputs: Vec<EncodeInput> = batch
            .iter()
            .map(|s| EncodeInput::Single(s.as_str().into()))
            .collect();

        let t0 = Instant::now();
        let hf_results = hf_tokenizer
            .encode_batch(hf_inputs, true)
            .map_err(|e| anyhow::anyhow!(e))
            .context("HF encode_batch failed")?;
        let t1 = Instant::now();

        let ft_results = tokenizer
            .encode_batch(batch, true)
            .context("fastokens encode_batch failed")?;
        let t2 = Instant::now();

        // Verify correctness per sample within the batch.
        for (j, (hf_enc, ft_enc)) in hf_results.iter().zip(ft_results.iter()).enumerate() {
            let hf_ids = hf_enc.get_ids();
            if hf_ids != ft_enc.as_slice() {
                let global_idx = batch_idx * batch_size + j;
                panic!(
                    "Output mismatch for input {global_idx} ({} differences):\n\
                     Input: {:?}\n hf[:100]: {:?}\n ft[:100]: {:?}",
                    std::iter::zip(hf_ids.iter().copied(), ft_enc.iter().copied())
                        .filter(|(a, b)| a != b)
                        .count(),
                    &batch[j],
                    &hf_ids[..hf_ids.len().min(100)],
                    &ft_enc[..ft_enc.len().min(100)],
                );
            }
        }

        let dt_hf = t1 - t0;
        let dt_ft = t2 - t1;

        total_hf += dt_hf;
        total_ft += dt_ft;
        let batch_tokens: u64 = ft_results.iter().map(|v| v.len() as u64).sum();
        total_tokens += batch_tokens;
        total_chars += batch_chars as u64;

        if verbose {
            println!(
                "[batch {}/{}] {} samples, {} chars, {} tokens | hf: {:.3} ms, ft: {:.3} ms ({:.1}x)",
                batch_idx + 1,
                num_batches,
                batch.len(),
                batch_chars,
                batch_tokens,
                dt_hf.as_secs_f64() * 1000.0,
                dt_ft.as_secs_f64() * 1000.0,
                dt_hf.as_secs_f64() / dt_ft.as_secs_f64(),
            );
        } else {
            pb.inc(1);
        }

        if let Some(w) = csv_writer.as_mut() {
            writeln!(
                w,
                "{},{},{},{},{}",
                batch_idx,
                batch_chars,
                batch_tokens,
                dt_hf.as_secs_f64() * 1000.0,
                dt_ft.as_secs_f64() * 1000.0,
            )
            .context("failed to write CSV row")?;
        }
    }

    pb.finish();
    print_summary(
        &format!("Batched Benchmark Summary (batch_size={batch_size})"),
        chunks.len(),
        total_chars,
        total_tokens,
        total_hf,
        total_ft,
    );
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let samples = load_dataset(&args.dataset, args.max_samples, args.verbose)?;
    let chunks: Vec<String> = samples;

    if args.verbose {
        eprintln!("Loaded {} samples", chunks.len());
        eprintln!("Fetching tokenizer for {}...", args.model);
    }
    let hf_tokenizer = tokenizers::Tokenizer::from_pretrained(&args.model, None)
        .map_err(|e| anyhow::anyhow!(e))
        .context("failed to load HF tokenizer")?;
    let tokenizer = fastokens::Tokenizer::from_model(&args.model)
        .context("failed to load fastokens tokenizer")?;

    let mut csv_writer = args
        .output
        .as_ref()
        .map(|path| {
            let mut w = io::BufWriter::new(
                std::fs::File::create(path)
                    .with_context(|| format!("failed to create {}", path.display()))?,
            );
            writeln!(
                w,
                "input_index,input_char_len,output_token_len,\
                 hf_duration_ms,fastokens_duration_ms"
            )
            .context("failed to write CSV header")?;
            Ok::<_, anyhow::Error>(w)
        })
        .transpose()?;

    // Warmup: encode one sample to populate caches.
    if let Some(chunk) = chunks.first() {
        let _ = tokenizer.encode_with_special_tokens(chunk, true);
    }

    if let Some(batch_size) = args.batch_size {
        bench_batched(
            &chunks,
            &hf_tokenizer,
            &tokenizer,
            batch_size,
            args.verbose,
            csv_writer.as_mut(),
        )?;
    } else {
        bench_sequential(
            &chunks,
            &hf_tokenizer,
            &tokenizer,
            args.verbose,
            csv_writer.as_mut(),
        )?;
    }

    Ok(())
}
