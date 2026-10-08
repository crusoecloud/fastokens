//! `gate`: the serving benchmark as a before/after performance gate. This binary
//! is the source side (the tree being changed); `--base-worker` is the same
//! harness built against the target branch's fastokens. `bench/perf-gate.sh`
//! builds both and runs this.
//!
//!     cargo run --release -- gate --base-worker PATH [--rounds 5]
//!         [--threshold 0.05] [--batch-threshold 0.10] [--models GLM-5.3,...]
//!         [--long-n 200] [--chat-n 5000] [--batch-n 20000]
//!         [--base-label main] [--head-label PR] [--summary FILE]
//!
//! Each round runs one fastokens worker (the serving benchmark's fresh-process
//! worker) per (side, model), alternating which side goes first, so both see the
//! same host drift. A side's score per (model, scenario) is its best round: the
//! least-contended window. A scenario regresses when the source's best is below
//! the target's by more than the threshold (`batch`, multi-threaded, gets its own
//! looser one). Models with a regression run as many rounds again before the
//! verdict, so one noisy window does not fail the gate.
//!
//! Token ids are compared too. Documents whose ids changed are a warning, not a
//! failure: parity with the reference tokenizers is the tests' job, and a
//! correctness fix changes ids on purpose.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::{MODELS, print_table};

const SCENARIOS: [&str; 3] = ["long", "chat", "batch"];
/// Worker sides, in `runs` order.
const TARGET: usize = 0;
const SOURCE: usize = 1;

struct Args {
    base_worker: PathBuf,
    models: Vec<String>,
    rounds: usize,
    threshold: f64,
    batch_threshold: f64,
    base_label: String,
    head_label: String,
    summary: Option<PathBuf>,
    /// Passed through to every worker.
    pass: Vec<String>,
}

fn parse_args(args: Vec<String>) -> Result<Args> {
    let mut base_worker = None;
    let mut a = Args {
        base_worker: PathBuf::new(),
        models: MODELS.iter().map(|m| m.0.to_string()).collect(),
        rounds: 5,
        threshold: 0.05,
        batch_threshold: 0.10,
        base_label: "target".into(),
        head_label: "source".into(),
        summary: None,
        pass: Vec::new(),
    };
    // The serving benchmark's slices (its defaults unless overridden).
    let mut sizes = [
        ("--long-n", 200usize),
        ("--long-warm", 40),
        ("--chat-n", 5000),
        ("--batch-n", 20000),
        ("--chat-warm", 2000),
    ];
    let mut it = args.into_iter();
    while let Some(k) = it.next() {
        let mut val = || it.next().context("missing value");
        match k.as_str() {
            "--base-worker" => base_worker = Some(PathBuf::from(val()?)),
            "--models" => a.models = val()?.split(',').map(str::to_string).collect(),
            "--rounds" => a.rounds = val()?.parse()?,
            "--threshold" => a.threshold = val()?.parse()?,
            "--batch-threshold" => a.batch_threshold = val()?.parse()?,
            "--base-label" => a.base_label = val()?,
            "--head-label" => a.head_label = val()?,
            "--summary" => a.summary = Some(val()?.into()),
            other => match sizes.iter_mut().find(|(name, _)| *name == other) {
                Some((_, v)) => *v = val()?.parse()?,
                None => bail!("unknown argument {other:?}"),
            },
        }
    }
    a.base_worker = base_worker.context("--base-worker PATH is required")?;
    if a.rounds == 0 {
        bail!("--rounds must be at least 1");
    }
    for m in &a.models {
        if !MODELS.iter().any(|x| x.0 == m) {
            bail!("unknown model {m:?}");
        }
    }
    a.pass = sizes
        .iter()
        .flat_map(|(k, v)| [k.to_string(), v.to_string()])
        .collect();
    Ok(a)
}

fn run_worker(exe: &PathBuf, model: &str, pass: &[String]) -> Result<Value> {
    let out = Command::new(exe)
        .args(["--worker", "ft", model])
        .args(pass)
        .output()
        .with_context(|| format!("running {}", exe.display()))?;
    if !out.status.success() {
        bail!(
            "worker {} / {model} failed: {}",
            exe.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(serde_json::from_slice(&out.stdout)?)
}

/// `runs[side]` for one model: every round's worker output.
type ModelRuns = [Vec<Value>; 2];

/// One round over `models`; even rounds run the target first, odd the source.
fn round(
    a: &Args,
    exes: &[PathBuf; 2],
    models: &[usize],
    r: usize,
    runs: &mut [ModelRuns],
) -> Result<()> {
    let order = if r.is_multiple_of(2) {
        [TARGET, SOURCE]
    } else {
        [SOURCE, TARGET]
    };
    for &m in models {
        for side in order {
            runs[m][side].push(run_worker(&exes[side], &a.models[m], &a.pass)?);
        }
        eprintln!("  round {} done {}", r + 1, a.models[m]);
    }
    Ok(())
}

fn best(runs: &[Value], sc: &str) -> f64 {
    runs.iter()
        .filter_map(|r| r[sc]["mb_s"].as_f64())
        .fold(0.0, f64::max)
}

/// Fractional change of the source's best over the target's.
fn change(runs: &ModelRuns, sc: &str) -> f64 {
    best(&runs[SOURCE], sc) / best(&runs[TARGET], sc) - 1.0
}

fn limit(a: &Args, sc: &str) -> f64 {
    if sc == "batch" {
        a.batch_threshold
    } else {
        a.threshold
    }
}

fn regressed(a: &Args, runs: &ModelRuns) -> bool {
    SCENARIOS.iter().any(|sc| change(runs, sc) < -limit(a, sc))
}

/// Documents whose ids differ between the two sides' first runs.
fn id_changes(runs: &ModelRuns, sc: &str) -> usize {
    let digests = |side: usize| {
        runs[side][0][sc]["digests"]
            .as_array()
            .cloned()
            .unwrap_or_default()
    };
    let (t, s) = (digests(TARGET), digests(SOURCE));
    t.iter().zip(&s).filter(|(x, y)| x != y).count() + t.len().abs_diff(s.len())
}

pub fn main(args: Vec<String>) -> Result<()> {
    let a = parse_args(args)?;
    let exes = [a.base_worker.clone(), std::env::current_exe()?];
    let all: Vec<usize> = (0..a.models.len()).collect();
    let mut runs: Vec<ModelRuns> = vec![Default::default(); a.models.len()];
    eprintln!(
        "perf gate: {} (source) vs {} (target), {} round(s)",
        a.head_label, a.base_label, a.rounds
    );
    for r in 0..a.rounds {
        round(&a, &exes, &all, r, &mut runs)?;
    }
    let flagged: Vec<usize> = all
        .iter()
        .copied()
        .filter(|&m| regressed(&a, &runs[m]))
        .collect();
    if !flagged.is_empty() {
        eprintln!(
            "perf gate: confirming {} model(s) with {} more round(s)",
            flagged.len(),
            a.rounds
        );
        for r in a.rounds..2 * a.rounds {
            round(&a, &exes, &flagged, r, &mut runs)?;
        }
    }

    let gha = std::env::var_os("GITHUB_ACTIONS").is_some();
    let (mut rows, mut failures, mut id_warnings) = (Vec::new(), 0usize, 0usize);
    let mut md = format!(
        "### Perf gate: `{}` vs `{}`\n\n| model | scenario | docs | size | target MB/s | source MB/s | change | ids changed | verdict |\n|---|---|--:|--:|--:|--:|--:|--:|---|\n",
        a.head_label, a.base_label
    );
    for (m, model) in a.models.iter().enumerate() {
        let r = &runs[m];
        for sc in SCENARIOS {
            let (t, s, c) = (best(&r[TARGET], sc), best(&r[SOURCE], sc), change(r, sc));
            let lim = limit(&a, sc);
            let fail = c < -lim;
            let ids = id_changes(r, sc);
            let first = &r[SOURCE][0][sc];
            let docs = first["n"].as_u64().unwrap_or(0);
            let size = format!("{} MB", first["bytes"].as_u64().unwrap_or(0) / 1_000_000);
            let verdict = if fail { "REGRESSION" } else { "ok" };
            failures += fail as usize;
            id_warnings += (ids > 0) as usize;
            if gha && fail {
                println!(
                    "::error title=Perf regression::{model} {sc}: {t:.1} -> {s:.1} MB/s ({:+.1}%, limit -{:.0}%)",
                    c * 100.0,
                    lim * 100.0
                );
            }
            if gha && ids > 0 {
                println!(
                    "::warning title=Token ids changed::{model} {sc}: {ids} of {docs} documents encode differently"
                );
            }
            rows.push(vec![
                model.clone(),
                sc.to_string(),
                docs.to_string(),
                size.clone(),
                format!("{t:.1}"),
                format!("{s:.1}"),
                format!("{:+.1}%", c * 100.0),
                ids.to_string(),
                verdict.to_string(),
            ]);
            let verdict_md = if fail { "**REGRESSION**" } else { "ok" };
            writeln!(
                md,
                "| {model} | {sc} | {docs} | {size} | {t:.1} | {s:.1} | {:+.1}% | {ids} | {verdict_md} |",
                c * 100.0
            )?;
        }
    }
    let note = format!(
        "Best of {} rounds per side (the target and source alternate going first); a model that regresses gets {} more before the verdict. \
         A scenario fails when the source is more than {:.0}% slower ({:.0}% for multi-threaded `batch`). \
         Changed token ids are reported, not failed.",
        a.rounds,
        a.rounds,
        a.threshold * 100.0,
        a.batch_threshold * 100.0
    );
    let hdr: Vec<String> = [
        "model",
        "scenario",
        "docs",
        "size",
        "target MB/s",
        "source MB/s",
        "change",
        "ids changed",
        "verdict",
    ]
    .map(String::from)
    .to_vec();
    println!();
    print_table(&hdr, &rows);
    println!("\n{note}");
    writeln!(md, "\n{note}")?;
    if let Some(path) = &a.summary {
        use std::io::Write as _;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?
            .write_all(md.as_bytes())?;
    }
    if id_warnings > 0 {
        eprintln!("perf gate: token ids changed in {id_warnings} (model, scenario) pair(s)");
    }
    if failures > 0 {
        bail!("perf gate failed: {failures} scenario(s) regressed");
    }
    eprintln!("perf gate passed");
    Ok(())
}
