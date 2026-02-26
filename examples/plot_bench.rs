use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use plotters::prelude::*;

/// Plot benchmark results from simple_bench CSV output.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Opts {
    /// Input CSV file (output from simple_bench)
    input: PathBuf,

    /// Output SVG file path
    #[arg(short, long)]
    output: PathBuf,
}

struct Record {
    output_token_len: f64,
    hf_duration_ms: f64,
    fastokens_duration_ms: f64,
}

fn load_csv(path: &Path) -> Result<Vec<Record>> {
    let mut reader = csv::ReaderBuilder::new()
        .trim(csv::Trim::All)
        .from_path(path)
        .with_context(|| format!("failed to open {}", path.display()))?;

    let mut records = Vec::new();
    for result in reader.records() {
        let row = result.context("failed to read CSV row")?;
        let output_token_len: f64 = row
            .get(2)
            .context("missing output_token_len column")?
            .parse()
            .context("invalid output_token_len")?;
        let hf_duration_ms: f64 = row
            .get(3)
            .context("missing hf_duration_ms column")?
            .parse()
            .context("invalid hf_duration_ms")?;
        let fastokens_duration_ms: f64 = row
            .get(4)
            .context("missing fastokens_duration_ms column")?
            .parse()
            .context("invalid fastokens_duration_ms")?;

        records.push(Record {
            output_token_len,
            hf_duration_ms,
            fastokens_duration_ms,
        });
    }

    Ok(records)
}

const HF_COLOR: RGBColor = RGBColor(99, 102, 241);
const FT_COLOR: RGBColor = RGBColor(244, 63, 94);

fn main() -> Result<()> {
    let opts = Opts::parse();
    let records = load_csv(&opts.input)?;

    if records.is_empty() {
        anyhow::bail!("no data rows found in CSV");
    }

    let max_tokens = records
        .iter()
        .map(|r| r.output_token_len)
        .fold(0.0_f64, f64::max);
    let max_duration = records
        .iter()
        .flat_map(|r| [r.hf_duration_ms, r.fastokens_duration_ms])
        .fold(0.0_f64, f64::max);

    // Add 5% padding so dots don't sit on the axes.
    let x_max = max_tokens * 1.05;
    let y_max = max_duration * 1.05;

    let root = SVGBackend::new(&opts.output, (900, 560)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Tokenizer Benchmark", ("sans-serif", 22).into_font())
        .margin(16)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..x_max, 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Output tokens")
        .y_desc("Duration (ms)")
        .axis_desc_style(("sans-serif", 15))
        .label_style(("sans-serif", 12))
        .light_line_style(RGBColor(230, 230, 230))
        .draw()?;

    // HF series
    chart
        .draw_series(PointSeries::of_element(
            records
                .iter()
                .map(|r| (r.output_token_len, r.hf_duration_ms)),
            4,
            HF_COLOR.filled(),
            &|c, s, st| Circle::new(c, s, st),
        ))?
        .label("HuggingFace tokenizers")
        .legend(|(x, y)| Circle::new((x + 8, y), 4, HF_COLOR.filled()));

    // fastokens series
    chart
        .draw_series(PointSeries::of_element(
            records
                .iter()
                .map(|r| (r.output_token_len, r.fastokens_duration_ms)),
            4,
            FT_COLOR.filled(),
            &|c, s, st| Circle::new(c, s, st),
        ))?
        .label("fastokens")
        .legend(|(x, y)| Circle::new((x + 8, y), 4, FT_COLOR.filled()));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .margin(12)
        .legend_area_size(18)
        .border_style(RGBColor(200, 200, 200))
        .background_style(WHITE.mix(0.9))
        .label_font(("sans-serif", 14))
        .draw()?;

    root.present()?;
    println!("Wrote {}", opts.output.display());
    Ok(())
}
