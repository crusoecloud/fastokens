//! `bulk`: `bench/bulk.py` at the Rust level. One multi-GB pool of text (built
//! by `python bench/bulk.py --prepare`), cut at the same bytes into small
//! (~100-token), medium (~2k-token) and large (~250k-token) documents; each form
//! encoded one call per document (`encode`) and in ~16 MB batches (`batch`),
//! through the same native calls as the serving benchmark.
//!
//!     cargo run --release -- bulk [--gb 3 | --dir PATH] [--libs hf,gt,ft] [--repeat 3]
//!         [--models GLM-5.3,...] [--forms small,medium,large] [--modes encode,batch]
//!         [--batch-mb 16] [--gt-worker ../gigatoken-worker/target/release/gigatoken-worker]
//!
//! Every (library, model, form, mode) runs in a fresh child process that warms
//! up on the corpus's disjoint 32 MB warm-up pool, then encodes every timed
//! document once. Only the encode calls are timed (per 64 documents in `encode`
//! mode). Each document's ids are hashed and every library and mode is checked
//! against the reference library's first mode; the hashes travel through a file.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::{Lib, MODELS, Scratch, label, median, print_table};

/// Documents per timed group in `encode` mode.
const GROUP: usize = 64;
/// At most this many disputed documents per (model, form) are settled by the judge.
const JUDGE_CHECKS: usize = 500;

/// A form's documents: the text and its n+1 byte offsets.
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
        let offs = std::fs::read(dir.join(idx))?
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Ok(Corpus { text, offs })
    }

    fn docs(&self) -> Vec<&str> {
        self.offs
            .windows(2)
            .map(|w| &self.text[w[0] as usize..w[1] as usize])
            .collect()
    }

    /// Document indices splitting the corpus into runs of >= `budget` bytes (as
    /// `bench/bulk.py`'s `batch_bounds`).
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

struct Args {
    dir: PathBuf,
    models: Vec<String>,
    libs: Vec<String>,
    forms: Vec<String>,
    modes: Vec<String>,
    batch_mb: f64,
    repeat: usize,
    gt_worker: Option<PathBuf>,
    worker: Option<[String; 4]>,
    digests_out: Option<PathBuf>,
}

fn parse_args(argv: Vec<String>) -> Result<Args> {
    let list = |s: String| s.split(',').map(str::to_string).collect::<Vec<_>>();
    let mut gb = 3.0f64;
    let mut dir = None;
    let mut a = Args {
        dir: PathBuf::new(),
        models: MODELS.iter().map(|m| m.0.to_string()).collect(),
        libs: vec!["hf".into(), "ft".into()],
        forms: list("small,medium,large".into()),
        modes: list("encode,batch".into()),
        batch_mb: 16.0,
        repeat: 1,
        gt_worker: None,
        worker: None,
        digests_out: None,
    };
    let mut it = argv.into_iter();
    while let Some(k) = it.next() {
        let mut val = || it.next().context("missing value");
        match k.as_str() {
            "--gb" => gb = val()?.parse()?,
            "--dir" => dir = Some(PathBuf::from(val()?)),
            "--models" => a.models = list(val()?),
            "--libs" => a.libs = list(val()?),
            "--forms" => a.forms = list(val()?),
            "--modes" => a.modes = list(val()?),
            "--batch-mb" => a.batch_mb = val()?.parse()?,
            "--repeat" => a.repeat = val()?.parse()?,
            "--gt-worker" => a.gt_worker = Some(val()?.into()),
            "--worker" => a.worker = Some([val()?, val()?, val()?, val()?]),
            "--digests-out" => a.digests_out = Some(val()?.into()),
            other => bail!("unknown argument {other:?}"),
        }
    }
    // As bench/bulk_corpus.py's corpus_dir (`{gb:g}` prints 3.0 as "3", like Rust's `{}`).
    a.dir = dir.unwrap_or_else(|| {
        let root = std::env::var("FASTOKENS_BENCH_CACHE").unwrap_or_else(|_| {
            format!("{}/.cache/fastokens-bench", std::env::var("HOME").unwrap_or_default())
        });
        PathBuf::from(root).join(format!("bulk-{gb}GB"))
    });
    Ok(a)
}

fn worker(a: &Args, [lib, model, form, mode]: &[String; 4]) -> Result<Value> {
    let repo = MODELS
        .iter()
        .find(|m| m.0 == model)
        .context("unknown model")?
        .1;
    let tok = Lib::load(lib, repo)?;
    let corpus = Corpus::load(&a.dir, form, false)?;
    let warm = Corpus::load(&a.dir, form, true)?;
    let budget = a.batch_mb * 1e6;
    let run = |c: &Corpus, keep: bool| {
        let docs = c.docs();
        let (mut dt, mut tokens, mut digs) = (0.0, 0usize, Vec::new());
        if mode == "encode" {
            let mut slots: Vec<Scratch> = (0..GROUP).map(|_| Scratch::default()).collect();
            for group in docs.chunks(GROUP) {
                let t0 = Instant::now();
                for (d, s) in group.iter().zip(&mut slots) {
                    tok.encode(d, s);
                }
                dt += t0.elapsed().as_secs_f64();
                if keep {
                    for s in &slots[..group.len()] {
                        digs.push(tok.digest_last(s));
                        tokens += tok.tokens_last(s);
                    }
                }
            }
        } else {
            for w in c.batch_bounds(budget).windows(2) {
                let t0 = Instant::now();
                let out = tok.encode_batch(&docs[w[0]..w[1]]);
                dt += t0.elapsed().as_secs_f64();
                if keep {
                    digs.extend(out.digests());
                    tokens += out.tokens();
                }
            }
        }
        (dt, tokens, digs)
    };
    run(&warm, false);
    let (dt, tokens, digs) = run(&corpus, true);
    let path = a.digests_out.as_ref().context("--digests-out")?;
    std::fs::write(path, digs.iter().flat_map(|d| d.to_le_bytes()).collect::<Vec<u8>>())?;
    let bytes = corpus.offs.last().unwrap() - corpus.offs[0];
    Ok(json!({
        "version": tok.version(), "n": digs.len(), "bytes": bytes, "tokens": tokens,
        "seconds": dt, "mb_s": bytes as f64 / dt / 1e6,
    }))
}

/// `n` with thousands separators.
fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn main(argv: Vec<String>) -> Result<()> {
    let a = parse_args(argv)?;
    if let Some(w) = &a.worker {
        println!("{}", worker(&a, w)?);
        return Ok(());
    }
    let meta: Value = serde_json::from_str(
        &std::fs::read_to_string(a.dir.join("meta.json")).with_context(|| {
            format!("{} has no corpus: run `python bench/bulk.py --prepare --gb N` first", a.dir.display())
        })?,
    )?;
    let total = meta["bytes"].as_f64().unwrap_or(1.0);
    let mix: Vec<String> = meta["composition"]
        .as_object()
        .map(|m| {
            // Largest share first, as bench/bulk.py prints it.
            let mut v: Vec<(&String, f64)> = m.iter().map(|(s, b)| (s, b.as_f64().unwrap_or(0.0))).collect();
            v.sort_by(|x, y| y.1.total_cmp(&x.1));
            v.iter().map(|(s, b)| format!("{s} {:.0}%", b / total * 100.0)).collect()
        })
        .unwrap_or_default();
    for lib in &a.libs {
        match lib.as_str() {
            "hf" | "hf0" | "ft" => {}
            "gt" if a.gt_worker.is_some() => {}
            "gt" => bail!("--libs gt needs --gt-worker PATH"),
            other => bail!("unknown library {other:?} (expected hf, hf0, ft, gt)"),
        }
    }
    let exe = std::env::current_exe()?;
    let n = a.libs.len();
    let ft = a.libs.iter().position(|l| l == "ft");
    let others: Vec<usize> = (0..n).filter(|&k| Some(k) != ft).collect();
    let (mut rows, mut labels, mut notes, mut kimi) = (Vec::new(), Vec::new(), Vec::new(), false);
    for model in &a.models {
        for form in &a.forms {
            // (lib, mode) -> first run's digests; runs[mode][lib] -> every run's JSON.
            let mut digs: Vec<((usize, usize), Vec<u64>)> = Vec::new();
            let mut runs: Vec<Vec<Vec<Value>>> = vec![vec![Vec::new(); n]; a.modes.len()];
            for (m, mode) in a.modes.iter().enumerate() {
                for rep in 0..a.repeat {
                    for (k, lib) in a.libs.iter().enumerate() {
                        let bin = if lib == "gt" { a.gt_worker.clone().unwrap() } else { exe.clone() };
                        let out_path = std::env::temp_dir()
                            .join(format!("fastokens-bulk-{}-{k}-{mode}.u64", std::process::id()));
                        let out = Command::new(&bin)
                            .args(["bulk", "--worker", lib, model, form, mode])
                            .arg("--dir")
                            .arg(&a.dir)
                            .args(["--batch-mb", &a.batch_mb.to_string()])
                            .arg("--digests-out")
                            .arg(&out_path)
                            .output()?;
                        if !out.status.success() {
                            bail!(
                                "worker {lib}/{model}/{form}/{mode} failed: {}",
                                String::from_utf8_lossy(&out.stderr)
                            );
                        }
                        let r: Value = serde_json::from_slice(&out.stdout)?;
                        eprintln!(
                            "  done {model:<20} {form:<6} {mode:<6} {lib}: {:.1} MB/s",
                            r["mb_s"].as_f64().unwrap_or(0.0)
                        );
                        if rep == 0 {
                            let d = std::fs::read(&out_path)?
                                .chunks_exact(8)
                                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                                .collect();
                            digs.push(((k, m), d));
                        }
                        std::fs::remove_file(&out_path)?;
                        runs[m][k].push(r);
                    }
                }
            }
            if labels.is_empty() {
                labels = (0..n)
                    .map(|k| label(&a.libs[k], runs[0][k][0]["version"].as_str().unwrap_or("?")))
                    .collect();
            }
            let reference = &digs.iter().find(|(key, _)| *key == (0, 0)).unwrap().1;
            let differing = |got: &Vec<u64>| -> Vec<usize> {
                let mut v: Vec<usize> = (0..reference.len().min(got.len()))
                    .filter(|&i| reference[i] != got[i])
                    .collect();
                v.extend(reference.len().min(got.len())..reference.len().max(got.len()));
                v
            };
            // Documents where libraries disagree are settled by tokenizers 0.x, HF's
            // long-standing implementation (Kimi's by tiktoken, in bench/bulk.py).
            let judge_label = "tokenizers 0.23.2";
            let mut truth = std::collections::HashMap::new();
            let disputed: std::collections::BTreeSet<usize> =
                digs.iter().flat_map(|(_, got)| differing(got)).filter(|&i| i < reference.len()).collect();
            if !disputed.is_empty() && model != "Kimi-K3" {
                let repo = MODELS.iter().find(|x| x.0 == model).context("unknown model")?.1;
                let judge = Lib::load("hf0", repo)?;
                let corpus = Corpus::load(&a.dir, form, false)?;
                let docs = corpus.docs();
                let mut s = Scratch::default();
                for &i in disputed.iter().take(JUDGE_CHECKS) {
                    judge.encode(docs[i], &mut s);
                    truth.insert(i, judge.digest_last(&s));
                }
                let agree = truth.iter().filter(|(i, t)| reference[**i] == **t).count();
                notes.push(format!(
                    "{model}/{form}: {} doc(s) disputed; {} matches {judge_label} on {agree}/{}",
                    disputed.len(),
                    labels[0],
                    truth.len()
                ));
            }
            for (m, mode) in a.modes.iter().enumerate() {
                let mut status = Vec::new();
                for ((k, dm), got) in &digs {
                    if *dm != m || (*k, *dm) == (0, 0) {
                        continue;
                    }
                    let diff = differing(got);
                    let settled = !truth.is_empty()
                        && diff.iter().all(|i| truth.get(i).is_some_and(|t| got.get(*i) == Some(t)));
                    match (diff.len(), model.as_str()) {
                        (0, _) => {}
                        (d, "Kimi-K3") => {
                            kimi = true;
                            status.push(format!("{} {mode}: {d} differ*", labels[*k]));
                        }
                        (d, _) if settled => status.push(format!("{} {mode}: {d} differ, match 0.x†", labels[*k])),
                        (d, _) => status.push(format!("{} {mode}: FAIL ({d})", labels[*k])),
                    }
                }
                let med = |k: usize| median(runs[m][k].iter().filter_map(|r| r["mb_s"].as_f64()).collect());
                let r0 = &runs[m][0][0];
                let docs = r0["n"].as_u64().unwrap_or(0);
                let mut row = vec![
                    model.clone(),
                    form.clone(),
                    mode.clone(),
                    thousands(docs),
                    format!("{:.2} GB", r0["bytes"].as_f64().unwrap_or(0.0) / 1e9),
                    thousands((r0["tokens"].as_f64().unwrap_or(0.0) / docs.max(1) as f64).round() as u64),
                ];
                row.extend((0..n).map(|k| format!("{:.1}", med(k))));
                if let Some(f) = ft {
                    row.extend(others.iter().map(|&k| format!("{:.2}x", med(f) / med(k))));
                }
                row.push(if status.is_empty() { "PASS".to_string() } else { status.join(", ") });
                rows.push(row);
            }
        }
    }
    let mut hdr: Vec<String> = ["model", "form", "mode", "docs", "size", "tok/doc"].map(String::from).to_vec();
    hdr.extend(labels.iter().map(|l| format!("{l} MB/s")));
    if ft.is_some() {
        hdr.extend(others.iter().map(|&k| format!("ft vs {}", labels[k])));
    }
    hdr.push("parity".into());
    println!(
        "corpus: {:.2} GB ({}); batch mode: ~{} MB per call",
        total / 1e9,
        mix.join(", "),
        a.batch_mb
    );
    print_table(&hdr, &rows);
    for n in &notes {
        println!("  † {n}");
    }
    if kimi {
        println!(
            "  * Kimi-K3's HF tokenizer is a conversion of its tiktoken model; bench/bulk.py checks such docs against tiktoken."
        );
    }
    Ok(())
}
