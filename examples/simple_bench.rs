use std::io::Write;
use std::path::PathBuf;
use std::{io, path::Path, time::Instant};

use anyhow::{Context, Result};
use clap::Parser;
use hf_hub::api::sync::Api;
use indicatif::{ProgressBar, ProgressStyle};

const PREFIX_FACTOR: f64 = 32_000.0 / 70_000.0;
const TARGET_PROMPT_LEN: usize = 70_000;

const OUTPUTS: &str = "outputs.txt";
const CHUNK_SIZE: usize = 3_000;

/// Tokenizer benchmark tool
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Opts {
    /// HuggingFace Hub model name (e.g. deepseek-ai/DeepSeek-V3)
    model: String,

    /// Dataset to use (e.g. "sharegpt"); omit for default file mode
    #[arg(long)]
    dataset: Option<String>,

    /// Maximum number of conversations to process (dataset mode only)
    #[arg(long)]
    max_conversations: Option<usize>,

    /// Output CSV file path for per-input benchmark results
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn split_file_into_chunks<P: AsRef<Path>>(file_path: P, chunk_size: usize) -> Result<Vec<String>> {
    let file_path = file_path.as_ref();
    let text = std::fs::read_to_string(file_path)
        .with_context(|| format!("failed to read {}", file_path.display()))?;
    let words: Vec<&str> = text.split_whitespace().collect();

    let shared_prefix_len = (PREFIX_FACTOR * TARGET_PROMPT_LEN as f64).round() as usize;
    let (shared_prefix, rest) = words.split_at(shared_prefix_len);
    let shared_prefix = shared_prefix.join(" ");

    Ok(rest
        .chunks_exact(chunk_size)
        .map(|chunk| format!("{} {}", shared_prefix, chunk.join(" ")))
        .collect())
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
fn load_dataset(dataset: &str, max_items: Option<usize>) -> Result<Vec<String>> {
    let json_file = dataset_json_file(dataset)?;

    println!("Downloading {dataset} from HuggingFace Hub...");
    let api = Api::new().context("failed to create HuggingFace Hub API")?;
    let repo = api.dataset(dataset.to_string());
    let json_path = repo.get(json_file).context("failed to download dataset")?;
    println!("Downloaded to: {}", json_path.display());

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

fn main() -> Result<()> {
    let opts = Opts::parse();

    let chunks: Vec<String> = if let Some(dataset) = &opts.dataset {
        let samples = load_dataset(dataset, opts.max_conversations)?;
        println!("Loaded {} samples", samples.len());
        samples
    } else {
        split_file_into_chunks(OUTPUTS, CHUNK_SIZE)?
    };

    println!("Fetching tokenizer for {}...", opts.model);
    let hf_tokenizer = tokenizers::Tokenizer::from_pretrained(&opts.model, None)
        .map_err(|e| anyhow::anyhow!(e))
        .context("failed to load HF tokenizer")?;
    let tokenizer = fastokens::Tokenizer::from_model(&opts.model)
        .context("failed to load fastokens tokenizer")?;

    let mut csv_writer = opts
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

    println!("Running simple benchmark...");

    let pb = csv_writer.as_ref().map(|_| {
        let pb = ProgressBar::new(chunks.len() as u64);
        pb.set_style(
            ProgressStyle::with_template("[{elapsed_precise}] [{bar:40}] {pos}/{len} ({eta})")
                .expect("valid template")
                .progress_chars("=> "),
        );
        pb
    });

    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_len = chunk.len();

        let t0 = Instant::now();
        let enc_hf = hf_tokenizer
            .encode_fast(chunk.as_str(), true)
            .map_err(|e| anyhow::anyhow!(e))
            .context("HF tokenizer encode failed")?;
        let enc_hf = enc_hf.get_ids();
        let t1 = Instant::now();
        let enc = tokenizer.encode(chunk).context("fastokens encode failed")?;
        let t2 = Instant::now();

        if enc_hf != enc {
            panic!(
                "Output mismatch for input {i} ({} differences):\n\
                 Input: {:?}\n hf[:100]: {:?}\n ft[:100]: {:?}",
                std::iter::zip(enc_hf.iter().copied(), enc.iter().copied())
                    .filter(|(a, b)| a != b)
                    .count()
                    + enc_hf.len().abs_diff(enc.len()),
                chunk,
                &enc_hf[..enc_hf.len().min(100)],
                &enc[..enc.len().min(100)],
            );
        }

        let dt_hf = t1 - t0;
        let dt = t2 - t1;

        if let Some(pb) = &pb {
            pb.inc(1);
        } else {
            println!("Input length: {}, Output: {} tokens", chunk_len, enc.len());
            println!("  Duration: hf: {dt_hf:?}, our: {dt:?}");
            println!();
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

    if let Some(pb) = &pb {
        pb.finish();
    }

    Ok(())
}
