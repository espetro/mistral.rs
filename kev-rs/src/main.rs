//! kev-rs: Kev pointer-head serving on the mistral.rs hidden-states path.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Parser, Subcommand};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;

use kev_rs::api::{to_record, SystemOneRequest};
use kev_rs::checkpoint;
use kev_rs::encode::{self, Branch, Encoding, KevJson, MAX_BRANCH, MAX_STATE};
use kev_rs::engine::KevEngine;

#[derive(Parser)]
#[command(name = "kev-rs")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve /v1/systemone for an exported checkpoint.
    Serve {
        /// Directory written by scripts/export_checkpoint.py (model/, head.safetensors, kev.json),
        /// or a Hub repo holding one (`owner/repo[@revision]`, downloaded into the hf-hub cache).
        #[arg(long)]
        checkpoint: String,
        /// Bind address; use 0.0.0.0 inside containers.
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8009)]
        port: u16,
        #[arg(long)]
        paged: bool,
        /// Backbone precision; defaults to the checkpoint's exported dtype (kev.json).
        #[arg(long)]
        dtype: Option<String>,
        /// In-situ quantization: a bit width (2-8, e.g. `--isq 8`) or a type like q8_0, afq8.
        #[arg(long)]
        isq: Option<String>,
        /// The checkpoint name reported by /v1/models (e.g. jaredpalmer/kev-0.8b).
        #[arg(long, default_value = "kev-latest")]
        run: String,
        #[arg(long, default_value = "")]
        release_date: String,
    },
    /// Compare kev-rs probabilities against the torch reference JSON.
    Parity {
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        reference: PathBuf,
        #[arg(long)]
        paged: bool,
        #[arg(long)]
        dtype: Option<String>,
        #[arg(long)]
        isq: Option<String>,
    },
    /// Compare the record->ids encoder output against the reference JSON, token-exact.
    EncodeCheck {
        #[arg(long)]
        checkpoint: String,
        #[arg(long)]
        records: PathBuf,
        #[arg(long)]
        reference: PathBuf,
    },
}

#[derive(Deserialize)]
struct RefEntry {
    ids: Vec<u32>,
    state_len: usize,
    branches: Vec<RefBranch>,
    probs: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct RefBranch {
    ids: Vec<u32>,
    decide: usize,
    opts: Vec<usize>,
}

fn argmax(p: &[f32]) -> usize {
    p.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

async fn parity(
    checkpoint: &Path,
    reference: &Path,
    dtype: Option<&str>,
    isq: Option<&str>,
    paged: bool,
) -> Result<()> {
    let entries: Vec<RefEntry> = serde_json::from_str(&std::fs::read_to_string(reference)?)?;
    let engine = KevEngine::load(checkpoint, dtype, isq, paged, None).await?;
    let mut max_dp = 0f32;
    let mut sum_dp = 0f64;
    let mut n_cells = 0usize;
    let mut flips = 0usize;
    let mut n_q = 0usize;
    for (i, e) in entries.iter().enumerate() {
        let enc = Encoding {
            ids: e.ids.clone(),
            state_len: e.state_len,
            branches: e
                .branches
                .iter()
                .map(|b| Branch {
                    ids: b.ids.clone(),
                    decide: b.decide,
                    opts: b.opts.clone(),
                })
                .collect(),
        };
        let (got, m) = engine.probs(&enc).await?;
        for (j, (g, w)) in got.iter().zip(&e.probs).enumerate() {
            let cell_max = g
                .iter()
                .zip(w)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let flip = argmax(g) != argmax(w);
            flips += flip as usize;
            n_q += 1;
            n_cells += g.len();
            sum_dp += g
                .iter()
                .zip(w)
                .map(|(a, b)| (a - b).abs() as f64)
                .sum::<f64>();
            max_dp = max_dp.max(cell_max);
            println!(
                "record {i} q{j}: max|dp|={cell_max:.6} argmax_match={}",
                !flip
            );
        }
        println!(
            "record {i}: {} questions, {} tokens (latency {:.1} ms, prefix_hit={})",
            got.len(),
            m.tokens,
            m.latency_ms,
            m.prefix_cache_hit
        );
    }
    println!(
        "parity: {} records, {} questions, max|dp|={:.6}, mean|dp|={:.6}, flips={flips}/{n_q}",
        entries.len(),
        n_q,
        max_dp,
        if n_cells > 0 {
            sum_dp / n_cells as f64
        } else {
            0.0
        }
    );
    Ok(())
}

fn encode_check(checkpoint: &Path, records: &Path, reference: &Path) -> Result<()> {
    let kev: KevJson =
        serde_json::from_str(&std::fs::read_to_string(checkpoint.join("kev.json"))?)?;
    let tok = tokenizers::Tokenizer::from_file(checkpoint.join("model/tokenizer.json"))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let entries: Vec<RefEntry> = serde_json::from_str(&std::fs::read_to_string(reference)?)?;
    let ids = kev.special_ids();
    let mut raws: Vec<Value> = std::fs::read_to_string(records)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    // same ordering as reference_probs.py: most questions first
    raws.sort_by_key(|r| usize::MAX - r["questions"].as_object().map_or(0, |q| q.len()));
    let mut ok = 0usize;
    let mut total = 0usize;
    for (i, raw) in raws.iter().enumerate() {
        if i >= entries.len() {
            break;
        }
        // labelled record -> the serving request shape (api_request strips labels)
        let mut questions = IndexMap::new();
        for (qid, q) in raw["questions"].as_object().unwrap() {
            questions.insert(
                qid.clone(),
                serde_json::from_value::<kev_rs::api::Question>(serde_json::json!({
                    "type": q["type"],
                    "instructions": q["instructions"],
                    "criteria": q["criteria"],
                }))?,
            );
        }
        let req = SystemOneRequest {
            state: raw["state"].clone(),
            model: "kev-latest".to_string(),
            questions,
        };
        let (rec, _) = to_record(&req)?;
        let enc = encode::encode(&tok, &ids, &rec, MAX_STATE, MAX_BRANCH)?;
        let same = enc.ids == entries[i].ids;
        total += 1;
        ok += same as usize;
        if !same {
            let first_diff = enc
                .ids
                .iter()
                .zip(&entries[i].ids)
                .position(|(a, b)| a != b);
            println!(
                "record {i}: MISMATCH (first diff at {first_diff:?}, got {} ids, want {})",
                enc.ids.len(),
                entries[i].ids.len()
            );
        }
    }
    println!("encode-check: {ok}/{total} exact");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("kev_rs=info".parse().unwrap()),
        )
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve {
            checkpoint,
            host,
            port,
            paged,
            dtype,
            isq,
            run,
            release_date,
        } => {
            let checkpoint = checkpoint::resolve(&checkpoint).await?;
            let mut engine =
                KevEngine::load(&checkpoint, dtype.as_deref(), isq.as_deref(), paged, None).await?;
            engine.run = run;
            kev_rs::server::serve(engine, release_date, &host, port).await
        }
        Cmd::Parity {
            checkpoint,
            reference,
            paged,
            dtype,
            isq,
        } => {
            let checkpoint = checkpoint::resolve(&checkpoint).await?;
            parity(
                &checkpoint,
                &reference,
                dtype.as_deref(),
                isq.as_deref(),
                paged,
            )
            .await
        }
        Cmd::EncodeCheck {
            checkpoint,
            records,
            reference,
        } => {
            let checkpoint = checkpoint::resolve(&checkpoint).await?;
            encode_check(&checkpoint, &records, &reference)
        }
    }
}
