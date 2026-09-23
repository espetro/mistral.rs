//! Engine: encodes a record, seeds the state into the mistral.rs prefix cache, and
//! scores each question branch from the returned hidden states.

use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use mistralrs::{
    parse_isq_value, IsqBits, Model, ModelBuilder, ModelDType, PagedAttentionMetaBuilder,
};
use tokenizers::Tokenizer;

use crate::encode::{encode, Encoding, KevJson, SpecialIds, SERVE_MAX_BRANCH, SERVE_MAX_STATE};
use crate::head::{softmax, PointerHead};

const DEFAULT_PREFIX_CACHE_SIZE: usize = 4;
// state seed plus up to eight branch entries the cacher also admits
const ENGINE_SLOTS_PER_STATE: usize = 9;

pub struct KevJsonMeta {
    pub base: String,
    pub lora_head_dim: usize,
    pub temperature: f64,
    pub hidden_size: usize,
}

pub struct KevEngine {
    pub model: Model,
    pub tok: Tokenizer,
    pub head: PointerHead,
    pub ids: SpecialIds,
    pub meta: KevJsonMeta,
    seen_states: Mutex<SeenStates>,
    pub hits: std::sync::atomic::AtomicUsize,
    pub misses: std::sync::atomic::AtomicUsize,
    pub prefix_cache_size: usize,
    pub device: String,
    pub dtype: String,
    pub run: String,
}

struct SeenStates {
    set: std::collections::HashSet<u64>,
    order: VecDeque<u64>,
    cap: usize,
}

impl SeenStates {
    fn contains(&self, key: u64) -> bool {
        self.set.contains(&key)
    }
    fn insert(&mut self, key: u64) {
        if self.cap == 0 || self.set.contains(&key) {
            return;
        }
        self.order.push_back(key);
        self.set.insert(key);
        while self.order.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.set.remove(&old);
            }
        }
    }
    fn remove(&mut self, key: u64) {
        if self.set.remove(&key) {
            self.order.retain(|k| *k != key);
        }
    }
}

pub struct Meta {
    pub tokens: usize,
    pub state_tokens: usize,
    pub latency_ms: f64,
    pub prefix_cache_hit: bool,
}

fn hash_ids(ids: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    ids.hash(&mut h);
    h.finish()
}

impl KevEngine {
    pub async fn load(
        checkpoint: &Path,
        dtype: Option<&str>,
        isq: Option<&str>,
        paged: bool,
        prefix_cache_size: Option<usize>,
    ) -> Result<Self> {
        let kev: KevJson = serde_json::from_str(
            &std::fs::read_to_string(checkpoint.join("kev.json"))
                .context("kev.json missing; run scripts/export_checkpoint.py first")?,
        )?;
        if kev.option_isolation {
            bail!("option_isolation checkpoints are not supported");
        }
        let model_dir = checkpoint.join("model");
        let tok = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let head = PointerHead::load(
            &checkpoint.join("head.safetensors"),
            kev.head_dim,
            kev.temperature as f64,
        )?;
        let dtype = dtype.or(kev.dtype.as_deref()).unwrap_or("f32");
        let mut builder = ModelBuilder::new(model_dir.to_string_lossy().to_string());
        builder = match dtype {
            "auto" => builder.with_dtype(ModelDType::Auto),
            "f32" => builder.with_dtype(ModelDType::F32),
            "bf16" => builder.with_dtype(ModelDType::BF16),
            "f16" => builder.with_dtype(ModelDType::F16),
            other => bail!("unknown --dtype {other} (want auto|f32|bf16|f16)"),
        };
        if let Some(isq) = isq {
            builder = match IsqBits::try_from(isq) {
                Ok(bits) => builder.with_auto_isq(bits),
                Err(()) => {
                    builder.with_isq(parse_isq_value(isq, None).map_err(|e| anyhow::anyhow!(e))?)
                }
            };
        }
        if paged {
            builder = builder.with_paged_attn(PagedAttentionMetaBuilder::default().build()?);
        }
        let prefix_cache_size = prefix_cache_size.unwrap_or_else(|| {
            std::env::var("KEV_PREFIX_CACHE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_PREFIX_CACHE_SIZE)
        });
        // one engine slot minimum so branches in a request reuse the seeded state
        builder =
            builder.with_prefix_cache_n(Some(prefix_cache_size.max(1) * ENGINE_SLOTS_PER_STATE));
        let model = builder.build().await?;
        let device = match model.config()?.device.location() {
            candle_core::DeviceLocation::Cpu => "cpu".to_string(),
            candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
            candle_core::DeviceLocation::Metal { .. } => "metal".to_string(),
        };
        Ok(Self {
            model,
            tok,
            head,
            ids: kev.special_ids(),
            meta: KevJsonMeta {
                base: kev.base,
                lora_head_dim: kev.head_dim,
                temperature: kev.temperature as f64,
                hidden_size: kev.hidden_size,
            },
            seen_states: Mutex::new(SeenStates {
                set: Default::default(),
                order: VecDeque::new(),
                cap: prefix_cache_size,
            }),
            hits: 0.into(),
            misses: 0.into(),
            prefix_cache_size,
            device,
            dtype: dtype.to_string(),
            run: String::new(),
        })
    }

    /// The record -> ids path, for parity checks against a torch reference.
    pub fn encode(&self, rec: &crate::encode::Record) -> Result<Encoding> {
        encode(&self.tok, &self.ids, rec, SERVE_MAX_STATE, SERVE_MAX_BRANCH)
    }

    /// One state-only request seeds the engine prefix cache; later branch requests
    /// hit it via the shared token prefix.
    async fn seed_state(&self, state_ids: &[u32]) -> Result<()> {
        self.model
            .send_hidden_states_request(state_ids.to_vec())
            .await?;
        Ok(())
    }

    /// Probabilities per question for one encoded record.
    pub async fn probs(&self, enc: &Encoding) -> Result<(Vec<Vec<f32>>, Meta)> {
        let t = Instant::now();
        let state_ids = &enc.ids[..enc.state_len];
        let key = hash_ids(state_ids);
        let mut hit = {
            let mut seen = self.seen_states.lock().unwrap();
            if seen.contains(key) {
                true
            } else {
                seen.insert(key);
                false
            }
        };
        if !hit {
            self.seed_state(state_ids).await?;
        }
        let futs = enc.branches.iter().map(|br| {
            let mut toks = state_ids.to_vec();
            toks.extend_from_slice(&br.ids);
            self.model.send_hidden_states_request(toks)
        });
        let responses = futures::future::join_all(futs).await;
        let engine_hit = hit
            && responses.iter().all(|r| {
                r.as_ref()
                    .is_ok_and(|r| r.prefix_cached_tokens == enc.state_len)
            });
        if hit && !engine_hit {
            self.seen_states.lock().unwrap().remove(key);
            hit = false;
        }
        if hit {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let mut out = Vec::with_capacity(enc.branches.len());
        for (br, res) in enc.branches.iter().zip(responses) {
            let res = res?;
            let off = res.prefix_cached_tokens;
            let hidden = res.hidden.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
            let row = |idx: usize| -> Result<Tensor> {
                let r = enc.state_len + idx;
                if r < off {
                    bail!("row {r} lies inside the cached prefix (off={off})");
                }
                Ok(hidden.get(r - off)?)
            };
            let h_decide = row(br.decide)?;
            let opt_rows = br
                .opts
                .iter()
                .map(|&o| row(o))
                .collect::<Result<Vec<_>>>()?;
            let h_opts = Tensor::stack(&opt_rows, 0)?;
            let logits = self.head.logits(&h_decide, &h_opts)?;
            out.push(softmax(&logits));
        }
        let latency_ms = (t.elapsed().as_secs_f64() * 1000.0 * 10.0).round() / 10.0;
        Ok((
            out,
            Meta {
                tokens: enc.ids.len(),
                state_tokens: enc.state_len,
                latency_ms,
                prefix_cache_hit: hit && engine_hit,
            },
        ))
    }

    /// probs() for an un-encoded record (serving path: ContextOverflow -> 422).
    pub async fn probs_record(&self, rec: &crate::encode::Record) -> Result<(Vec<Vec<f32>>, Meta)> {
        let enc = self.encode(rec)?;
        self.probs(&enc).await
    }

    /// Number of state prefixes this engine has seeded (bounded by the LRU cap).
    pub fn cached_states(&self) -> usize {
        self.seen_states.lock().unwrap().set.len()
    }
}
