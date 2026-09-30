//! A `bench/rust` worker for gigatoken, speaking its protocol: invoked as
//! `gigatoken-worker --worker gt MODEL --long-n N ...`, prints the same JSON
//! (the same data slices, digests and timing rules as bench/rust's workers);
//! `gigatoken-worker bulk --worker gt MODEL FORM MODE ...` likewise for the
//! multi-GB benchmark (bench/rust/src/bulk.rs).
//!
//! gigatoken runs through what its Python API calls: `encode` →
//! `Tokenizer::encode_with_added_tokens_flat` on one tokenizer (its pretoken
//! cache persists across calls), `encode_batch_list` → the pooled parallel
//! batch encoder (`encode_docs_ragged`), at its default thread count.
//!
//! gigatoken builds only on nightly Rust (`rust-toolchain.toml` selects it) and
//! is not on crates.io, so this is its own crate over its source release:
//!
//!     ./fetch.sh && cargo build --release
//!     cd ../rust && cargo run --release -- --libs hf,gt,ft \
//!         --gt-worker ../gigatoken-worker/target/release/gigatoken-worker

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use gigatoken_rs::load_tokenizer::hf::{HfTokenizer, load_hf_slice};
use gigatoken_rs::load_tokenizer::tiktoken::load_tiktoken;
use gigatoken_rs::pretokenize::PretokenizerType;
use gigatoken_rs::{Tokenizer, WorkerPool, encode_docs_ragged};
use serde_json::{Value, json};

struct Args {
    long_n: usize,
    long_warm: usize,
    chat_n: usize,
    batch_n: usize,
    chat_warm: usize,
    model: String,
}

fn parse_args() -> Result<Args> {
    let mut a = Args { long_n: 200, long_warm: 40, chat_n: 5000, batch_n: 20000, chat_warm: 2000, model: String::new() };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().context("missing value");
        match k.as_str() {
            "--worker" => {
                let lib = val()?;
                if lib != "gt" {
                    bail!("this worker runs gigatoken (gt), not {lib}");
                }
                a.model = val()?;
            }
            "--long-n" => a.long_n = val()?.parse()?,
            "--long-warm" => a.long_warm = val()?.parse()?,
            "--chat-n" => a.chat_n = val()?.parse()?,
            "--batch-n" => a.batch_n = val()?.parse()?,
            "--chat-warm" => a.chat_warm = val()?.parse()?,
            other => bail!("unknown argument {other:?}"),
        }
    }
    Ok(a)
}

struct Data {
    long: Vec<String>,
    long_warm: Vec<String>,
    chat: Vec<String>,
    batch: Vec<String>,
    chat_warm: Vec<String>,
}

/// The same slices as bench/rust's (and bench/serving.py's) `load_data`.
fn load_data(a: &Args) -> Result<Data> {
    let api = hf_hub::api::sync::Api::new()?;
    let read = |repo: &str, file: &str| -> Result<Vec<Value>> {
        let path = api.dataset(repo.to_string()).get(file)?;
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    };
    let long_docs: Vec<String> = read("zai-org/LongBench-v2", "data.json")?
        .iter()
        .filter_map(|d| d["context"].as_str().filter(|s| !s.is_empty()).map(str::to_owned))
        .collect();
    let chats: Vec<String> = read("RyokoAI/ShareGPT52K", "sg_90k_part1.json")?
        .iter()
        .filter_map(|item| {
            let msgs = item["conversations"].as_array()?;
            let parts: Vec<&str> =
                msgs.iter().filter_map(|m| m["value"].as_str().filter(|s| !s.is_empty())).collect();
            (!parts.is_empty()).then(|| parts.join("\n\n"))
        })
        .collect();
    let take = |v: &[String], lo: usize, n: usize| v[lo.min(v.len())..(lo + n).min(v.len())].to_vec();
    Ok(Data {
        long: take(&long_docs, 0, a.long_n),
        long_warm: take(&long_docs, a.long_n, a.long_warm),
        chat: take(&chats, 0, a.chat_n),
        batch: take(&chats, a.chat_n, a.batch_n),
        chat_warm: take(&chats, a.chat_n + a.batch_n, a.chat_warm),
    })
}

/// FNV-1a over a document's ids (as bench/rust).
fn digest(ids: impl IntoIterator<Item = u32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for id in ids {
        for b in id.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// As gigatoken's Python `Tokenizer(...)` loads these models: Kimi from its
/// tiktoken vocabulary with the config's special tokens, the others from
/// tokenizer.json.
fn load(repo: &str) -> Result<Tokenizer> {
    let api = hf_hub::api::sync::Api::new()?.model(repo.to_string());
    if repo == "moonshotai/Kimi-K3" {
        let config: Value = serde_json::from_str(&std::fs::read_to_string(api.get("tokenizer_config.json")?)?)?;
        let specials = config["added_tokens_decoder"]
            .as_object()
            .context("added_tokens_decoder")?
            .iter()
            .map(|(id, t)| Ok((t["content"].as_str().context("content")?.to_string(), id.parse()?)))
            .collect::<Result<Vec<(String, u32)>>>()?;
        return load_tiktoken(api.get("tiktoken.model")?, PretokenizerType::Kimi, specials)
            .map_err(|e| anyhow::anyhow!("{e:?}"));
    }
    match load_hf_slice(&std::fs::read(api.get("tokenizer.json")?)?).map_err(|e| anyhow::anyhow!("{e:?}"))? {
        HfTokenizer::Bpe(t) => Ok(t),
        HfTokenizer::SentencePiece(_) => bail!("{repo}: SentencePiece models are not benchmarked here"),
    }
}

fn repo(model: &str) -> Result<&'static str> {
    Ok(match model {
        "GLM-5.3" => "zai-org/GLM-5.3",
        "Kimi-K3" => "moonshotai/Kimi-K3",
        "DeepSeek-V4.1-Flash" => "deepseek-ai/DeepSeek-V4.1-Flash",
        m => bail!("unknown model {m}"),
    })
}

/// A bulk form's documents: the text and its n+1 byte offsets (as bench/rust's `bulk`).
struct Corpus {
    text: String,
    offs: Vec<u64>,
}

impl Corpus {
    fn load(dir: &Path, form: &str, warm: bool) -> Result<Self> {
        let (text, idx) = match warm {
            true => ("warm.txt".to_string(), format!("{form}.warm.idx")),
            false => ("pool.txt".to_string(), format!("{form}.idx")),
        };
        let text = String::from_utf8(std::fs::read(dir.join(text))?).context("corpus is not UTF-8")?;
        let offs =
            std::fs::read(dir.join(idx))?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(Corpus { text, offs })
    }

    fn docs(&self) -> Vec<&[u8]> {
        self.offs.windows(2).map(|w| &self.text.as_bytes()[w[0] as usize..w[1] as usize]).collect()
    }

    fn batch_bounds(&self, budget: f64) -> Vec<usize> {
        let (mut bounds, mut start) = (vec![0], self.offs[0]);
        for i in 1..self.offs.len() - 1 {
            if (self.offs[i] - start) as f64 >= budget {
                bounds.push(i);
                start = self.offs[i];
            }
        }
        bounds.push(self.offs.len() - 1);
        bounds
    }
}

/// The bulk worker: `bulk --worker gt MODEL FORM MODE --dir D --batch-mb B --digests-out P`.
fn bulk(argv: Vec<String>) -> Result<()> {
    let (mut w, mut dir, mut batch_mb, mut out) = (Vec::new(), PathBuf::new(), 16.0f64, PathBuf::new());
    let mut it = argv.into_iter();
    while let Some(k) = it.next() {
        let mut val = || it.next().context("missing value");
        match k.as_str() {
            "--worker" => w = vec![val()?, val()?, val()?, val()?],
            "--dir" => dir = val()?.into(),
            "--batch-mb" => batch_mb = val()?.parse()?,
            "--digests-out" => out = val()?.into(),
            other => bail!("unknown argument {other:?}"),
        }
    }
    let [lib, model, form, mode] = &w[..] else { bail!("bulk needs --worker gt MODEL FORM MODE") };
    if lib != "gt" {
        bail!("this worker runs gigatoken (gt), not {lib}");
    }
    let mut tok = load(repo(model)?)?;
    let workers = WorkerPool::new();
    let corpus = Corpus::load(&dir, form, false)?;
    let warm = Corpus::load(&dir, form, true)?;
    let mut run = |c: &Corpus, keep: bool| {
        let docs = c.docs();
        let (mut dt, mut tokens, mut digs) = (0.0, 0usize, Vec::new());
        if mode == "encode" {
            let mut slots: Vec<Vec<u32>> = vec![Vec::new(); 64];
            for group in docs.chunks(64) {
                let t0 = Instant::now();
                for (d, slot) in group.iter().zip(&mut slots) {
                    // A fresh vector per call, as the Python `encode` returns one.
                    let mut ids = Vec::new();
                    tok.encode_with_added_tokens_flat(d, &mut ids);
                    *slot = ids;
                }
                dt += t0.elapsed().as_secs_f64();
                if keep {
                    for ids in &slots[..group.len()] {
                        digs.push(digest(ids.iter().copied()));
                        tokens += ids.len();
                    }
                }
            }
        } else {
            for b in c.batch_bounds(batch_mb * 1e6).windows(2) {
                let t0 = Instant::now();
                let (flat, lens) = encode_docs_ragged(&workers, &tok, &docs[b[0]..b[1]]);
                dt += t0.elapsed().as_secs_f64();
                if keep {
                    let mut at = 0usize;
                    for &l in &lens {
                        digs.push(digest(flat[at..at + l as usize].iter().copied()));
                        at += l as usize;
                    }
                    tokens += flat.len();
                }
            }
        }
        (dt, tokens, digs)
    };
    run(&warm, false);
    let (dt, tokens, digs) = run(&corpus, true);
    std::fs::write(&out, digs.iter().flat_map(|d| d.to_le_bytes()).collect::<Vec<u8>>())?;
    let bytes = corpus.offs.last().unwrap() - corpus.offs[0];
    println!(
        "{}",
        json!({
            "version": "gigatoken 0.10.0", "n": digs.len(), "bytes": bytes, "tokens": tokens,
            "seconds": dt, "mb_s": bytes as f64 / dt / 1e6,
        })
    );
    Ok(())
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("bulk") {
        return bulk(std::env::args().skip(2).collect());
    }
    let a = parse_args()?;
    let repo = repo(&a.model)?;
    let data = load_data(&a)?;
    let mut tok = load(repo)?;
    let workers = WorkerPool::new();
    let batch = |tok: &Tokenizer, texts: &[String]| {
        let docs: Vec<&[u8]> = texts.iter().map(|s| s.as_bytes()).collect();
        encode_docs_ragged(&workers, tok, &docs)
    };
    let mut ids = Vec::new();
    for t in data.long_warm.iter().chain(&data.chat_warm) {
        ids.clear();
        tok.encode_with_added_tokens_flat(t.as_bytes(), &mut ids);
    }
    batch(&tok, &data.chat_warm[..data.chat_warm.len().min(256)]);

    let mut per_request = |texts: &[String]| {
        let (mut lat, mut digs) = (Vec::with_capacity(texts.len()), Vec::with_capacity(texts.len()));
        for t in texts {
            let t0 = Instant::now();
            // A fresh vector per call, as the Python `encode` returns one.
            let mut ids = Vec::new();
            tok.encode_with_added_tokens_flat(t.as_bytes(), &mut ids);
            lat.push(t0.elapsed().as_secs_f64());
            digs.push(digest(ids));
        }
        let bytes: usize = texts.iter().map(String::len).sum();
        let total: f64 = lat.iter().sum();
        lat.sort_by(f64::total_cmp);
        json!({
            "n": texts.len(), "bytes": bytes, "mb_s": bytes as f64 / total / 1e6,
            "p50_ms": lat[lat.len() / 2] * 1e3, "digests": digs,
        })
    };
    let long = per_request(&data.long);
    let chat = per_request(&data.chat);
    let mut dt = 0.0;
    let mut digs = Vec::with_capacity(data.batch.len());
    for b in data.batch.chunks(256) {
        let t0 = Instant::now();
        let (flat, lens) = batch(&tok, b);
        dt += t0.elapsed().as_secs_f64();
        let mut at = 0usize;
        for &l in &lens {
            digs.push(digest(flat[at..at + l as usize].iter().copied()));
            at += l as usize;
        }
    }
    let bytes: usize = data.batch.iter().map(String::len).sum();
    println!(
        "{}",
        json!({
            "version": "gigatoken 0.10.0", "long": long, "chat": chat,
            "batch": {"n": data.batch.len(), "bytes": bytes, "mb_s": bytes as f64 / dt / 1e6, "digests": digs},
        })
    );
    Ok(())
}
