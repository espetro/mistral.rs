//! TypeSafe-compatible request/response shapes (POST /v1/systemone), ported from kev/api.py.

use anyhow::{bail, Result};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::encode::{Record, RecordQuestion};

pub const MAX_OPTIONS: usize = 255;

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Noul {
        #[serde(default)]
        instructions: Value,
        #[serde(default)]
        criteria: Option<IndexMap<String, Value>>,
    },
    Choice {
        #[serde(default)]
        instructions: Value,
        criteria: IndexMap<String, Value>,
    },
    Score {
        #[serde(default)]
        instructions: Value,
        criteria: Vec<Value>,
    },
}

impl Clone for SystemOneRequest {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            model: self.model.clone(),
            questions: self.questions.clone(),
        }
    }
}

impl Question {
    pub fn qtype(&self) -> &'static str {
        match self {
            Self::Noul { .. } => "noul",
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
        }
    }

    pub fn instructions(&self) -> &Value {
        match self {
            Self::Noul { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Choice { criteria, .. } if !(1..=MAX_OPTIONS).contains(&criteria.len()) => {
                bail!("criteria must have 1..{MAX_OPTIONS} options")
            }
            Self::Score { criteria, .. } if !(1..=MAX_OPTIONS).contains(&criteria.len()) => {
                bail!("criteria must have 1..{MAX_OPTIONS} options")
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SystemOneRequest {
    pub state: Value,
    #[serde(default = "default_model")]
    pub model: String,
    pub questions: IndexMap<String, Question>,
}

fn default_model() -> String {
    "kev-latest".to_string()
}

impl SystemOneRequest {
    pub fn validate(&self) -> Result<()> {
        if self.questions.is_empty() {
            bail!("questions must be non-empty");
        }
        for q in self.questions.values() {
            q.validate()?;
        }
        Ok(())
    }
}

/// Python repr(float) for finite values: shortest round-trip digits, fixed
/// notation for decimal exponents in [-4, 16), else 1-digit-lead scientific.
fn py_float_repr(f: f64) -> String {
    if !f.is_finite() {
        return format!("{f}");
    }
    let sci = format!("{f:e}");
    let (mant, exp) = sci.split_once('e').unwrap_or(("0", "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-4..16).contains(&exp) {
        let s = format!("{f}");
        if s.contains('.') {
            s
        } else {
            format!("{s}.0")
        }
    } else {
        let neg = mant.starts_with('-');
        let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
        let digits = digits.trim_end_matches('0');
        let digits = if digits.is_empty() { "0" } else { digits };
        let mant = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits.to_string()
        };
        let sign = if exp < 0 { '-' } else { '+' };
        format!(
            "{}{mant}e{sign}{:02}",
            if neg { "-" } else { "" },
            exp.abs()
        )
    }
}

/// Python repr for a JSON number: ints print plainly, floats print like repr(float).
fn py_num(n: &serde_json::Number) -> String {
    if n.is_f64() {
        py_float_repr(n.as_f64().unwrap_or(0.0))
    } else {
        n.to_string()
    }
}

/// Flatten str | object | array into text the model sees; field names are labels.
/// Ported from api.render.
pub fn render(v: &Value, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) => py_num(n),
        Value::String(s) => s.clone(),
        Value::Array(xs) => xs
            .iter()
            .map(|x| format!("{pad}- {}", render(x, indent + 1).trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Object(m) => m
            .iter()
            .map(|(k, x)| {
                if matches!(x, Value::Object(_) | Value::Array(_)) {
                    format!("{pad}{k}:\n{}", render(x, indent + 1))
                } else {
                    format!("{pad}{k}: {}", render(x, 0))
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

pub fn option_text(name: &str, desc: Option<&Value>) -> String {
    match desc {
        None => name.to_string(),
        Some(Value::Null) => name.to_string(),
        Some(Value::String(s)) if s.is_empty() => name.to_string(),
        Some(v) => format!("{name}: {}", render(v, 0)),
    }
}

/// The keys a question's probabilities are reported under, in option order.
pub fn question_keys(q: &Question) -> Vec<String> {
    match q {
        Question::Choice { criteria, .. } => criteria.keys().cloned().collect(),
        Question::Noul { .. } => vec!["false".to_string(), "true".to_string()],
        Question::Score { criteria, .. } => (0..criteria.len()).map(|i| i.to_string()).collect(),
    }
}

/// Internal record for encode(), plus per-question metadata for to_answers().
pub struct QuestionMeta {
    pub id: String,
    pub qtype: &'static str,
    pub keys: Vec<String>,
    pub legend: Option<IndexMap<String, String>>,
}

pub fn to_record(req: &SystemOneRequest) -> Result<(Record, Vec<QuestionMeta>)> {
    let mut qs = Vec::new();
    let mut meta = Vec::new();
    for (qid, q) in &req.questions {
        let keys = question_keys(q);
        let (opts, legend) = match q {
            Question::Noul { criteria, .. } => {
                let c = criteria.clone().unwrap_or_default();
                (
                    vec![
                        option_text("no", c.get("false")),
                        option_text("yes", c.get("true")),
                    ],
                    None,
                )
            }
            Question::Choice { criteria, .. } => (
                criteria
                    .iter()
                    .map(|(k, v)| option_text(k, Some(v)))
                    .collect(),
                None,
            ),
            Question::Score { criteria, .. } => {
                let opts: Vec<String> = criteria.iter().map(|x| render(x, 0)).collect();
                let legend = keys.iter().cloned().zip(opts.iter().cloned()).collect();
                (opts, Some(legend))
            }
        };
        meta.push(QuestionMeta {
            id: qid.clone(),
            qtype: q.qtype(),
            keys,
            legend,
        });
        qs.push(RecordQuestion {
            instr: render(q.instructions(), 0),
            options: opts,
            label: 0,
        });
    }
    Ok((
        Record {
            state: render(&req.state, 0),
            questions: qs,
        },
        meta,
    ))
}

pub fn choice_confidence(p: &[f32]) -> f32 {
    let k = p.len() as f32;
    if p.len() == 1 {
        1.0
    } else {
        (p.iter().cloned().fold(f32::NEG_INFINITY, f32::max) - 1.0 / k) / (1.0 - 1.0 / k)
    }
}

/// 1 - E|level - mode| / (L - 1).
pub fn score_confidence(p: &[f32]) -> f32 {
    let l = p.len() as f32;
    let mode = p
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    if p.len() == 1 {
        1.0
    } else {
        1.0 - p
            .iter()
            .enumerate()
            .map(|(i, pi)| pi * (i as f32 - mode as f32).abs())
            .sum::<f32>()
            / (l - 1.0)
    }
}

/// round(x, 4) with Python semantics (round-half-even on the decimal repr).
pub fn round_prob(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

fn argmax(p: &[f32]) -> usize {
    p.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// json.dumps with Python defaults (", " / ": " separators, ensure_ascii): the
/// output_tokens count tokenizes exactly this string.
pub fn py_dumps(v: &Value) -> String {
    let mut s = String::new();
    py_dump(v, &mut s);
    s
}

fn py_escape(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if c.is_ascii() => out.push(c),
            c => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    out.push('"');
}

fn py_dump(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&py_num(n)),
        Value::String(s) => py_escape(s, out),
        Value::Array(xs) => {
            out.push('[');
            for (i, x) in xs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_dump(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_escape(k, out);
                out.push_str(": ");
                py_dump(x, out);
            }
            out.push('}');
        }
    }
}

/// Billing-style figure: tokens of the serialised answers (kev.api.output_tokens).
pub fn output_tokens(tok: &tokenizers::Tokenizer, answers: &Value) -> Result<usize> {
    let s = py_dumps(answers);
    let enc = tok
        .encode(s.as_str(), false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;
    Ok(enc.get_ids().len())
}

/// The answers object, preserving Python's insertion and key order.
pub fn to_answers(probs: &[Vec<f32>], meta: &[QuestionMeta]) -> Value {
    let mut out = Map::new();
    for (p, m) in probs.iter().zip(meta) {
        let mut a = Map::new();
        match m.qtype {
            "noul" => {
                a.insert("type".into(), Value::from("noul"));
                a.insert("noul".into(), Value::from(round_prob(p[1] as f64)));
            }
            "choice" => {
                a.insert("type".into(), Value::from("choice"));
                a.insert("choice".into(), Value::from(m.keys[argmax(p)].clone()));
                a.insert(
                    "confidence".into(),
                    Value::from(round_prob(choice_confidence(p) as f64)),
                );
                let dist = m
                    .keys
                    .iter()
                    .zip(p)
                    .map(|(k, v)| (k.clone(), Value::from(round_prob(*v as f64))))
                    .collect::<Map<_, _>>();
                a.insert("probabilities".into(), Value::Object(dist));
            }
            _ => {
                let score: f64 = p
                    .iter()
                    .enumerate()
                    .map(|(i, pi)| i as f64 * *pi as f64)
                    .sum();
                a.insert("type".into(), Value::from("score"));
                a.insert("score".into(), Value::from(round_prob(score)));
                let legend = m
                    .legend
                    .as_ref()
                    .map(|l| {
                        l.iter()
                            .map(|(k, v)| (k.clone(), Value::from(v.clone())))
                            .collect::<Map<_, _>>()
                    })
                    .unwrap_or_default();
                a.insert("legend".into(), Value::Object(legend));
                let dist = m
                    .keys
                    .iter()
                    .zip(p)
                    .map(|(k, v)| (k.clone(), Value::from(round_prob(*v as f64))))
                    .collect::<Map<_, _>>();
                a.insert("probabilities".into(), Value::Object(dist));
                a.insert(
                    "confidence".into(),
                    Value::from(round_prob(score_confidence(p) as f64)),
                );
            }
        }
        out.insert(m.id.clone(), Value::Object(a));
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::py_float_repr;

    #[test]
    fn float_repr_matches_python() {
        for (f, want) in [
            (1e16, "1e+16"),
            (1e20, "1e+20"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (0.1, "0.1"),
            (1.0, "1.0"),
            (100.0, "100.0"),
            (-2.5e30, "-2.5e+30"),
            (123456789012345.0, "123456789012345.0"),
            (1e15, "1000000000000000.0"),
            (-0.0, "-0.0"),
        ] {
            assert_eq!(py_float_repr(f), want, "repr({f})");
        }
    }
}
