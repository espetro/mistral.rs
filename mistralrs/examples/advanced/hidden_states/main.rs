//! Prefill-only hidden-state requests: post-final-norm hidden tensors for every
//! prompt position, batched, with prefix caching.
//!
//! Run with: `cargo run --release --example hidden_states -p mistralrs -- <model_id> [--paged]`
//! Defaults to `Qwen/Qwen3.5-0.8B`.

use anyhow::Result;
use mistralrs::{Model, ModelBuilder, PagedAttentionMetaBuilder};

const BASE_TOKENS: usize = 40;
const EXTRA_TOKENS: usize = 10;
const CONCURRENT_LENS: [usize; 4] = [16, 24, 32, 48];

async fn build_model(model_id: &str, paged: bool) -> Result<Model> {
    let mut builder = ModelBuilder::new(model_id);
    if paged {
        builder = builder.with_paged_attn(PagedAttentionMetaBuilder::default().build()?);
    }
    builder.build().await
}

async fn request(model: &Model, tag: &str, tokens: Vec<u32>) -> Result<mistralrs::HiddenStates> {
    let res = model.send_hidden_states_request(tokens).await?;
    println!(
        "{tag}: hidden={:?} tokens={} prefix_cached_tokens={}",
        res.hidden.dims(),
        res.tokens.len(),
        res.prefix_cached_tokens
    );
    Ok(res)
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_id = args
        .next()
        .unwrap_or_else(|| "Qwen/Qwen3.5-0.8B".to_string());
    let paged = args.any(|a| a == "--paged");

    let base: Vec<u32> = (1..=BASE_TOKENS as u32).collect();
    let extended: Vec<u32> = base
        .iter()
        .cloned()
        .chain(1001..1001 + EXTRA_TOKENS as u32)
        .collect();

    // (d-baseline) full A++B in one shot on an unprimed engine
    let model = build_model(&model_id, paged).await?;
    println!("model={model_id} paged={paged}");
    let baseline = request(&model, "d-baseline", extended.clone()).await?;

    // fresh engine so (a)-(c) see a real prefix cache lifecycle
    let model2 = build_model(&model_id, paged).await?;

    // (a) fresh request: no prefix cached
    request(&model2, "a", base.clone()).await?;

    // (b) identical request: the whole prefix may be served from the cache
    request(&model2, "b", base.clone()).await?;

    // (c) base prefix + extra tokens: base tokens hit the prefix cache
    let c = request(&model2, "c", extended.clone()).await?;

    // (d) parity: c's rows vs the baseline's last EXTRA_TOKENS rows
    let suffix = baseline
        .hidden
        .narrow(0, baseline.hidden.dim(0)? - EXTRA_TOKENS, EXTRA_TOKENS)?;
    let diff = if c.hidden.dim(0)? == 0 {
        f32::NAN
    } else {
        (&c.hidden - &suffix)?
            .abs()?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?
    };
    println!("d parity: max|diff| of last {EXTRA_TOKENS} rows = {diff}");

    // concurrent requests of different lengths
    let futs: Vec<_> = CONCURRENT_LENS
        .iter()
        .map(|&n| model2.send_hidden_states_request((1..=n as u32).collect()))
        .collect();
    for (i, r) in futures::future::join_all(futs)
        .await
        .into_iter()
        .enumerate()
    {
        let r = r?;
        println!(
            "concurrent[{i}] len={}: hidden={:?} prefix_cached_tokens={}",
            CONCURRENT_LENS[i],
            r.hidden.dims(),
            r.prefix_cached_tokens
        );
    }

    Ok(())
}
