//! Token-level port of kev/model.py: user_tokens, encode, rows_of.

use anyhow::{bail, Result};
use serde::Deserialize;
use std::sync::LazyLock;
use tokenizers::Tokenizer;

// Serving context (kev.serve): per-branch cap mirrors Jev's ~32k, bounded by the base model window.
pub const SERVE_MAX_STATE: usize = 8192;
pub const SERVE_MAX_BRANCH: usize = 8192;
// The checkpoint's training context (kev.model MAX_STATE/MAX_BRANCH); parity runs use it.
pub const MAX_STATE: usize = 384;
pub const MAX_BRANCH: usize = 1024;

static SPECIAL_RE: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"<\|([A-Za-z0-9_]+)\|>").unwrap());

/// The five delimiter token ids, loaded from kev.json.
#[derive(Clone, Copy, Debug)]
pub struct SpecialIds {
    pub state: u32,
    pub q: u32,
    pub opt: u32,
    pub copt: u32,
    pub decide: u32,
}

#[derive(Deserialize)]
pub struct KevJson {
    pub base: String,
    pub base_revision: Option<String>,
    pub head_dim: usize,
    pub temperature: f32,
    pub hidden_size: usize,
    pub special_tokens: SpecialTokens,
    pub pad_id: u32,
    pub option_isolation: bool,
    /// Backbone precision recorded by export_checkpoint.py --dtype; absent on older exports.
    pub dtype: Option<String>,
}

#[derive(Deserialize)]
pub struct SpecialTokens {
    #[serde(rename = "<|fim_prefix|>")]
    pub fim_prefix: u32,
    #[serde(rename = "<|fim_middle|>")]
    pub fim_middle: u32,
    #[serde(rename = "<|box_start|>")]
    pub box_start: u32,
    #[serde(rename = "<|box_end|>")]
    pub box_end: u32,
    #[serde(rename = "<|fim_suffix|>")]
    pub fim_suffix: u32,
}

impl KevJson {
    pub fn special_ids(&self) -> SpecialIds {
        SpecialIds {
            state: self.special_tokens.fim_prefix,
            q: self.special_tokens.fim_middle,
            opt: self.special_tokens.box_start,
            copt: self.special_tokens.box_end,
            decide: self.special_tokens.fim_suffix,
        }
    }
}

/// A record in the internal Kev shape: rendered state + questions.
#[derive(Clone, Debug)]
pub struct Record {
    pub state: String,
    pub questions: Vec<RecordQuestion>,
}

#[derive(Clone, Debug)]
pub struct RecordQuestion {
    pub instr: String,
    pub options: Vec<String>,
    pub label: usize,
}

/// A record does not encode within its context; serving turns it into a 422.
#[derive(Debug)]
pub struct ContextOverflow(pub String);

impl std::fmt::Display for ContextOverflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ContextOverflow {}

/// Tokenize caller-supplied text so it can never produce delimiter tokens (option
/// boundaries are unforgeable). `<|name|>` is rewritten to `< broken bar name >`.
pub fn user_tokens(tok: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let rewritten = SPECIAL_RE.replace_all(text, "<\u{A6}$1\u{A6}>");
    let text: &str = rewritten.as_ref();
    Ok(tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec())
}

#[derive(Clone, Debug)]
pub struct Branch {
    /// The branch tokens (without the state prefix).
    pub ids: Vec<u32>,
    /// Index of the <decide> token relative to the branch start.
    pub decide: usize,
    /// Index of each option's </opt> token relative to the branch start.
    pub opts: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Encoding {
    /// The packed ids: state then all branches concatenated.
    pub ids: Vec<u32>,
    /// Number of state tokens (rows 0..state_len are the state prefix).
    pub state_len: usize,
    pub branches: Vec<Branch>,
}

/// Pack one record: [<state> ...] then per-question [<q> instr <opt> o </opt>... <decide>].
/// Mirrors kev.model.encode + rows_of; branch offsets are relative to the branch start
/// (row index in the full request = state_len + offset).
pub fn encode(
    tok: &Tokenizer,
    ids: &SpecialIds,
    rec: &Record,
    max_state: usize,
    max_branch: usize,
) -> Result<Encoding> {
    let state_tokens = user_tokens(tok, &rec.state)?;
    let mut state = Vec::with_capacity(state_tokens.len() + 1);
    state.push(ids.state);
    state.extend_from_slice(&state_tokens[..state_tokens.len().min(max_state - 1)]);

    let mut out_ids = state.clone();
    let mut branches = Vec::with_capacity(rec.questions.len());
    for q in &rec.questions {
        let mut br = Vec::new();
        br.push(ids.q);
        br.extend(user_tokens(tok, &q.instr)?);
        let instr_len = br.len();
        let mut spans = Vec::with_capacity(q.options.len());
        for o in &q.options {
            let mut sp = Vec::new();
            sp.push(ids.opt);
            sp.extend(user_tokens(tok, o)?);
            sp.push(ids.copt);
            spans.push(sp);
        }
        for sp in &spans {
            br.extend_from_slice(sp);
        }
        br.push(ids.decide);
        if br.len() > max_branch.saturating_sub(state.len()) {
            bail!(ContextOverflow(format!(
                "branch too long: {} tokens with a {}-token state (row limit {max_branch})",
                br.len(),
                state.len()
            )));
        }
        let mut ends = Vec::with_capacity(spans.len());
        let mut cursor = instr_len;
        for sp in &spans {
            cursor += sp.len();
            ends.push(cursor - 1);
        }
        let decide = br.len() - 1;
        branches.push(Branch {
            ids: br.clone(),
            decide,
            opts: ends,
        });
        out_ids.extend_from_slice(&br);
    }
    Ok(Encoding {
        ids: out_ids,
        state_len: state.len(),
        branches,
    })
}
