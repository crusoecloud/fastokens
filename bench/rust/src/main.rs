//! Rust-level counterpart of `bench/serving.py`: fastokens vs HuggingFace
//! `tokenizers` 1.0 (the `tk-encode` engine behind the 1.0 Python package) and
//! its latest 0.x release, with no Python in the way — each library's own native
//! API.
//!
//! Same scenarios, data slices and rules as the Python benchmark:
//!
//!   long   one encode per LongBench-v2 context (long-context prompts)
//!   chat   one encode per ShareGPT conversation (chat-sized prompts)
//!   batch  batch encode over ShareGPT conversations, 256 per call
//!
//! Libraries (`--libs`, default `hf,ft`; the first is the reference for parity
//! and speedups), each through its fastest native path at its default threads:
//!
//!   hf   tokenizers 1.0: `PipelineTokenizer::encode_into` (single inputs; no
//!        `Encoding` built) and `PipelineTokenizer::encode` (batches, rayon)
//!   hf0  tokenizers 0.x: `Tokenizer::encode_fast` / `encode_batch_fast` (no
//!        offsets)
//!   ft   fastokens: `Tokenizer::encode` / `encode_batch`
//!   gt   an external worker binary (`--gt-worker PATH`) speaking this worker
//!        protocol — gigatoken builds only on nightly Rust, so it cannot be a
//!        dependency here
//!
//! Each (library, model) runs in a fresh child process; a worker warms up on
//! documents disjoint from the timed ones and encodes every timed document once.
//! Every document's ids are hashed and compared.
//!
//! Kimi ships only a `tiktoken.model`; its HF tokenizer is the conversion
//! `bench/kimi_hf.py` writes (run `bench/serving.py` or that module once). Where
//! it disagrees with fastokens, `bench/serving.py` settles it against tiktoken.
//!
//!     cargo run --release -- [--models GLM-5.3,...] [--libs hf,hf0,ft] [--repeat 3]
//!
//! `cargo run --release -- bulk ...` is the multi-GB benchmark instead (see
//! `bulk.rs`).

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tokenizers::pipeline::EncodeOptions;

mod bulk;

const MODELS: [(&str, &str); 3] = [
    ("GLM-5.3", "zai-org/GLM-5.3"),
    ("Kimi-K3", "moonshotai/Kimi-K3"),
    ("DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4.1-Flash"),
];

#[derive(Clone)]
struct Args {
    models: Vec<String>,
    long_n: usize,
    long_warm: usize,
    chat_n: usize,
    batch_n: usize,
    chat_warm: usize,
    repeat: usize,
    libs: Vec<String>,
    gt_worker: Option<PathBuf>,
    worker: Option<(String, String)>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        models: MODELS.iter().map(|m| m.0.to_string()).collect(),
        long_n: 200,
        long_warm: 40,
        chat_n: 5000,
        batch_n: 20000,
        chat_warm: 2000,
        repeat: 1,
        libs: vec!["hf".into(), "ft".into()],
        gt_worker: None,
        worker: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().context("missing value");
        match k.as_str() {
            "--models" => a.models = val()?.split(',').map(str::to_string).collect(),
            "--long-n" => a.long_n = val()?.parse()?,
            "--long-warm" => a.long_warm = val()?.parse()?,
            "--chat-n" => a.chat_n = val()?.parse()?,
            "--batch-n" => a.batch_n = val()?.parse()?,
            "--chat-warm" => a.chat_warm = val()?.parse()?,
            "--repeat" => a.repeat = val()?.parse()?,
            "--libs" => a.libs = val()?.split(',').map(str::to_string).collect(),
            "--gt-worker" => a.gt_worker = Some(val()?.into()),
            "--worker" => a.worker = Some((val()?, val()?)),
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

/// The same slices as `bench/serving.py`'s `load_data`.
fn load_data(a: &Args) -> Result<Data> {
    let api = hf_hub::api::sync::Api::new()?;
    let read = |repo: &str, file: &str| -> Result<Vec<Value>> {
        let path = api.dataset(repo.to_string()).get(file)?;
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    };
    let long_docs: Vec<String> = read("zai-org/LongBench-v2", "data.json")?
        .iter()
        .filter_map(|d| {
            d["context"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
        })
        .collect();
    let chats: Vec<String> = read("RyokoAI/ShareGPT52K", "sg_90k_part1.json")?
        .iter()
        .filter_map(|item| {
            let msgs = item["conversations"].as_array()?;
            let parts: Vec<&str> = msgs
                .iter()
                .filter_map(|m| m["value"].as_str().filter(|s| !s.is_empty()))
                .collect();
            (!parts.is_empty()).then(|| parts.join("\n\n"))
        })
        .collect();
    let take =
        |v: &[String], lo: usize, n: usize| v[lo.min(v.len())..(lo + n).min(v.len())].to_vec();
    Ok(Data {
        long: take(&long_docs, 0, a.long_n),
        long_warm: take(&long_docs, a.long_n, a.long_warm),
        chat: take(&chats, 0, a.chat_n),
        batch: take(&chats, a.chat_n, a.batch_n),
        chat_warm: take(&chats, a.chat_n + a.batch_n, a.chat_warm),
    })
}

fn hf_json(repo: &str) -> Result<PathBuf> {
    if repo == "moonshotai/Kimi-K3" {
        let home = std::env::var("HOME").unwrap_or_default();
        let p =
            PathBuf::from(home).join(".cache/fastokens-bench/moonshotai--Kimi-K3/tokenizer.json");
        if !p.exists() {
            bail!(
                "{} missing: run `python bench/kimi_hf.py`'s conversion first (bench/serving.py does)",
                p.display()
            );
        }
        return Ok(p);
    }
    Ok(hf_hub::api::sync::Api::new()?
        .model(repo.to_string())
        .get("tokenizer.json")?)
}

/// FNV-1a over a document's ids: equal digests <=> (almost surely) equal ids.
fn digest(ids: impl IntoIterator<Item = u32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for id in ids {
        for b in id.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// The in-process libraries behind one interface.
enum Lib {
    Ft(Box<fastokens::Tokenizer>),
    Hf(Box<tokenizers::pipeline::PipelineTokenizer>),
    Hf0(Box<tokenizers_0::Tokenizer>),
}

impl Lib {
    fn load(lib: &str, repo: &str) -> Result<Self> {
        Ok(match lib {
            "ft" => Lib::Ft(Box::new(fastokens::Tokenizer::from_model(repo)?)),
            "hf" => {
                let json = tokenizers::canonicalize_file(hf_json(repo)?)?;
                Lib::Hf(Box::new(
                    tokenizers::from_json(&json).map_err(|e| anyhow::anyhow!("{e}"))?,
                ))
            }
            "hf0" => Lib::Hf0(Box::new(
                tokenizers_0::Tokenizer::from_file(hf_json(repo)?)
                    .map_err(|e| anyhow::anyhow!("{e}"))?,
            )),
            _ => bail!("unknown lib {lib}"),
        })
    }

    fn version(&self) -> &'static str {
        match self {
            Lib::Ft(_) => "fastokens (this tree)",
            Lib::Hf(_) => "tokenizers 1.0.0-rc.2",
            Lib::Hf0(_) => "tokenizers 0.23.2",
        }
    }

    /// Encode one input; only this is timed. `digest_last` then hashes the ids.
    fn encode(&self, text: &str, s: &mut Scratch) {
        match self {
            Lib::Ft(t) => s.ft = t.encode(text).expect("fastokens encode"),
            Lib::Hf(t) => {
                s.hf.clear();
                t.encode_into(text, &EncodeOptions::no_specials(), &mut s.hf)
                    .expect("tokenizers encode");
            }
            Lib::Hf0(t) => {
                s.hf0 = Some(t.encode_fast(text, false).expect("tokenizers 0.x encode"));
            }
        }
    }

    fn tokens_last(&self, s: &Scratch) -> usize {
        match self {
            Lib::Ft(_) => s.ft.len(),
            Lib::Hf(_) => s.hf.len(),
            Lib::Hf0(_) => s.hf0.as_ref().map_or(0, |e| e.get_ids().len()),
        }
    }

    fn digest_last(&self, s: &Scratch) -> u64 {
        match self {
            Lib::Ft(_) => digest(s.ft.iter().copied()),
            Lib::Hf(_) => digest(s.hf.iter().map(|&x| u32::from(x))),
            Lib::Hf0(_) => digest(s.hf0.as_ref().map_or(&[][..], |e| e.get_ids()).iter().copied()),
        }
    }

    /// Encode a batch; only this is timed. The results are hashed afterwards.
    fn encode_batch<S: AsRef<str> + Sync>(&self, texts: &[S]) -> Batch {
        match self {
            Lib::Ft(t) => Batch::Ft(
                t.encode_batch(texts, false)
                    .expect("fastokens encode_batch"),
            ),
            Lib::Hf(t) => {
                let refs: Vec<&str> = texts.iter().map(AsRef::as_ref).collect();
                Batch::Hf(
                    t.encode(refs.as_slice(), &EncodeOptions::no_specials())
                        .wait()
                        .expect("tokenizers encode batch"),
                )
            }
            Lib::Hf0(t) => {
                let refs: Vec<&str> = texts.iter().map(AsRef::as_ref).collect();
                Batch::Hf0(
                    t.encode_batch_fast(refs, false)
                        .expect("tokenizers 0.x encode batch"),
                )
            }
        }
    }
}

/// Reused per-input output buffers.
#[derive(Default)]
struct Scratch {
    ft: Vec<u32>,
    hf: Vec<tokenizers::pipeline::PipelineToken>,
    hf0: Option<tokenizers_0::Encoding>,
}

/// A batch's results, as each library returns them.
enum Batch {
    Ft(Vec<Vec<u32>>),
    Hf(Vec<tokenizers::pipeline::Encoding>),
    Hf0(Vec<tokenizers_0::Encoding>),
}

impl Batch {
    fn tokens(&self) -> usize {
        match self {
            Batch::Ft(v) => v.iter().map(Vec::len).sum(),
            Batch::Hf(v) => v.iter().map(|e| e.ids().len()).sum(),
            Batch::Hf0(v) => v.iter().map(|e| e.get_ids().len()).sum(),
        }
    }

    fn digests(&self) -> Vec<u64> {
        match self {
            Batch::Ft(v) => v.iter().map(|ids| digest(ids.iter().copied())).collect(),
            Batch::Hf(v) => v
                .iter()
                .map(|e| digest(e.ids().iter().map(|&x| u32::from(x))))
                .collect(),
            Batch::Hf0(v) => v
                .iter()
                .map(|e| digest(e.get_ids().iter().copied()))
                .collect(),
        }
    }
}

fn worker(lib: &str, model: &str, a: &Args) -> Result<Value> {
    let data = load_data(a)?;
    let repo = MODELS
        .iter()
        .find(|m| m.0 == model)
        .context("unknown model")?
        .1;
    let tok = Lib::load(lib, repo)?;
    let mut scratch = Scratch::default();
    for t in data.long_warm.iter().chain(&data.chat_warm) {
        tok.encode(t, &mut scratch);
    }
    tok.encode_batch(&data.chat_warm[..data.chat_warm.len().min(256)]);

    let mut per_request = |texts: &[String]| {
        let (mut lat, mut digs) = (
            Vec::with_capacity(texts.len()),
            Vec::with_capacity(texts.len()),
        );
        for t in texts {
            let t0 = Instant::now();
            tok.encode(t, &mut scratch);
            lat.push(t0.elapsed().as_secs_f64());
            digs.push(tok.digest_last(&scratch));
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
        let out = tok.encode_batch(b);
        dt += t0.elapsed().as_secs_f64();
        digs.extend(out.digests());
    }
    let bytes: usize = data.batch.iter().map(String::len).sum();
    Ok(json!({
        "version": tok.version(), "long": long, "chat": chat,
        "batch": {"n": data.batch.len(), "bytes": bytes, "mb_s": bytes as f64 / dt / 1e6, "digests": digs},
    }))
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Display name of a library, from its first worker's reported version.
fn label(lib: &str, version: &str) -> String {
    match lib {
        "ft" => "fastokens".to_string(),
        _ => version.replace("tokenizers ", "HF "),
    }
}

fn print_table(hdr: &[String], rows: &[Vec<String>]) {
    let w: Vec<usize> = (0..hdr.len())
        .map(|c| rows.iter().map(|r| r[c].len()).chain([hdr[c].len()]).max().unwrap())
        .collect();
    let line = |r: &[String]| {
        r.iter()
            .zip(&w)
            .map(|(x, n)| format!("{x:<n$}"))
            .collect::<Vec<_>>()
            .join("  ")
    };
    println!("{}", line(hdr));
    println!("{}", w.iter().map(|n| "-".repeat(*n)).collect::<Vec<_>>().join("  "));
    for r in rows {
        println!("{}", line(r));
    }
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("bulk") {
        return bulk::main(std::env::args().skip(2).collect());
    }
    let a = parse_args()?;
    if let Some((lib, model)) = &a.worker {
        println!("{}", worker(lib, model, &a)?);
        return Ok(());
    }
    let exe = std::env::current_exe()?;
    let pass: Vec<String> = [
        ("--long-n", a.long_n),
        ("--long-warm", a.long_warm),
        ("--chat-n", a.chat_n),
        ("--batch-n", a.batch_n),
        ("--chat-warm", a.chat_warm),
    ]
    .iter()
    .flat_map(|(k, v)| [k.to_string(), v.to_string()])
    .collect();
    for lib in &a.libs {
        match lib.as_str() {
            "hf" | "hf0" | "ft" => {}
            "gt" if a.gt_worker.is_some() => {}
            "gt" => bail!("--libs gt needs --gt-worker PATH"),
            other => bail!("unknown library {other:?} (expected hf, hf0, ft, gt)"),
        }
    }
    let n = a.libs.len();
    let ft = a.libs.iter().position(|l| l == "ft");
    let others: Vec<usize> = (0..n).filter(|&k| Some(k) != ft).collect();
    let (mut rows, mut lat_rows, mut labels) = (Vec::new(), Vec::new(), Vec::new());
    for model in &a.models {
        let mut runs: Vec<Vec<Value>> = vec![Vec::new(); n];
        for _ in 0..a.repeat {
            for (k, lib) in a.libs.iter().enumerate() {
                let bin = match lib.as_str() {
                    "gt" => a.gt_worker.clone().unwrap(),
                    _ => exe.clone(),
                };
                let out = Command::new(&bin)
                    .args(["--worker", lib, model])
                    .args(&pass)
                    .output()?;
                if !out.status.success() {
                    bail!(
                        "worker {lib}/{model} failed: {}",
                        String::from_utf8_lossy(&out.stderr)
                    );
                }
                runs[k].push(serde_json::from_slice(&out.stdout)?);
                eprintln!("  done {model:<20} {lib}");
            }
        }
        if labels.is_empty() {
            labels = (0..n)
                .map(|k| label(&a.libs[k], runs[k][0]["version"].as_str().unwrap_or("?")))
                .collect();
        }
        for sc in ["long", "chat", "batch"] {
            let med = |k: usize, key: &str| {
                median(runs[k].iter().filter_map(|r| r[sc][key].as_f64()).collect())
            };
            let reference = runs[0][0][sc]["digests"].as_array().unwrap();
            let mut status = Vec::new();
            for k in 1..n {
                let got = runs[k][0][sc]["digests"].as_array().unwrap();
                let diff = reference.iter().zip(got).filter(|(x, y)| x != y).count()
                    + reference.len().abs_diff(got.len());
                match (diff, model.as_str()) {
                    (0, _) => {}
                    (d, "Kimi-K3") => status.push(format!("{}: {d} differ*", labels[k])),
                    (d, _) => status.push(format!("{}: FAIL ({d})", labels[k])),
                }
            }
            let parity = if status.is_empty() { "PASS".to_string() } else { status.join(", ") };
            let r0 = &runs[0][0][sc];
            let mut row = vec![
                model.clone(),
                sc.to_string(),
                r0["n"].to_string(),
                format!("{} MB", r0["bytes"].as_u64().unwrap() / 1_000_000),
            ];
            row.extend((0..n).map(|k| format!("{:.1}", med(k, "mb_s"))));
            if let Some(f) = ft {
                row.extend(others.iter().map(|&k| format!("{:.2}x", med(f, "mb_s") / med(k, "mb_s"))));
            }
            row.push(parity);
            rows.push(row);
            if sc != "batch" {
                let mut r = vec![model.clone(), sc.to_string()];
                r.extend((0..n).map(|k| format!("{:.1}", med(k, "p50_ms") * 1e3)));
                lat_rows.push(r);
            }
        }
    }
    let mut hdr: Vec<String> = ["model", "scenario", "docs", "size"].map(String::from).to_vec();
    hdr.extend(labels.iter().map(|l| format!("{l} MB/s")));
    if ft.is_some() {
        hdr.extend(others.iter().map(|&k| format!("ft vs {}", labels[k])));
    }
    hdr.push("parity".into());
    print_table(&hdr, &rows);
    println!();
    let mut lhdr: Vec<String> = ["model", "scenario"].map(String::from).to_vec();
    lhdr.extend(labels.iter().map(|l| format!("{l} p50 µs")));
    print_table(&lhdr, &lat_rows);
    if a.models.iter().any(|m| m == "Kimi-K3") {
        println!(
            "  * Kimi-K3's HF tokenizer is a conversion of its tiktoken model; bench/serving.py checks such docs against tiktoken."
        );
    }
    Ok(())
}
