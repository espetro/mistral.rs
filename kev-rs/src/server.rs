//! Axum server mirroring kev/serve.py: /v1/systemone, /v1/models, /permute, /separate,
//! bearer auth via KEV_API_KEY, x-typesafe-request-id header.

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::State,
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use indexmap::IndexMap;
use rand::SeedableRng;
use serde::Deserialize;
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

use crate::api::{output_tokens, to_answers, to_record, Question, SystemOneRequest};
use crate::engine::KevEngine;

const MODEL_NAMES: [&str; 2] = ["kev-latest", "jev-latest"];

pub struct AppState {
    pub engine: Arc<KevEngine>,
    pub release_date: String,
}

type Shared = Arc<AppState>;

#[derive(Deserialize)]
struct PermuteRequest {
    request: SystemOneRequest,
    question: String,
    #[serde(default = "default_n_perm")]
    n_perm: usize,
    #[serde(default)]
    seed: u64,
}

fn default_n_perm() -> usize {
    6
}

fn api_err(status: StatusCode, detail: impl Into<String>) -> Response {
    (status, Json(json!({ "detail": detail.into() }))).into_response()
}

fn encode_err(e: &anyhow::Error) -> Response {
    api_err(StatusCode::UNPROCESSABLE_ENTITY, e.to_string())
}

async fn answer(engine: &KevEngine, req: &SystemOneRequest) -> Result<Value, Response> {
    req.validate()
        .map_err(|e| api_err(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;
    let (rec, meta) =
        to_record(req).map_err(|e| api_err(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;
    let (ps, m) = engine
        .probs_record(&rec)
        .await
        .map_err(|e| encode_err(&e))?;
    let answers = to_answers(&ps, &meta);
    Ok(json!({
        "model": req.model,
        "answers": answers,
        "usage": {
            "input_tokens": m.tokens,
            "output_tokens": output_tokens(&engine.tok, &answers).unwrap_or(0),
        },
        "latency_ms": m.latency_ms,
    }))
}

fn parse_body<T: for<'de> Deserialize<'de>>(body: &axum::body::Bytes) -> Result<T, Box<Response>> {
    serde_json::from_slice(body)
        .map_err(|e| Box::new(api_err(StatusCode::UNPROCESSABLE_ENTITY, e.to_string())))
}

async fn systemone(State(s): State<Shared>, body: axum::body::Bytes) -> Response {
    let req: SystemOneRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(r) => return *r,
    };
    match answer(&s.engine, &req).await {
        Ok(v) => Json(v).into_response(),
        Err(r) => r,
    }
}

async fn permute(State(s): State<Shared>, body: axum::body::Bytes) -> Response {
    let r: PermuteRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(r) => return *r,
    };
    let q = match r.request.questions.get(&r.question) {
        Some(q @ Question::Choice { .. }) => q,
        _ => {
            return api_err(
                StatusCode::UNPROCESSABLE_ENTITY,
                "question must be an existing choice question",
            )
        }
    };
    if !(1..=64).contains(&r.n_perm) {
        return api_err(StatusCode::UNPROCESSABLE_ENTITY, "n_perm must be in 1..64");
    }
    let criteria = match q {
        Question::Choice { criteria, .. } => criteria.clone(),
        _ => unreachable!(),
    };
    let keys: Vec<String> = criteria.keys().cloned().collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(r.seed);
    let mut runs = Vec::new();
    for i in 0..r.n_perm {
        use rand::seq::SliceRandom;
        let mut order = keys.clone();
        if i > 0 {
            order.shuffle(&mut rng);
        }
        let mut one = r.request.clone();
        one.questions = IndexMap::new();
        let mut crit = IndexMap::new();
        for k in &order {
            crit.insert(k.clone(), criteria[k].clone());
        }
        one.questions.insert(
            r.question.clone(),
            Question::Choice {
                instructions: q.instructions().clone(),
                criteria: crit,
            },
        );
        match answer(&s.engine, &one).await {
            Ok(resp) => {
                let a = &resp["answers"][&r.question];
                runs.push(json!({
                    "order": order,
                    "probabilities": a["probabilities"],
                    "choice": a["choice"],
                    "latency_ms": resp["latency_ms"],
                }));
            }
            Err(e) => return e,
        }
    }
    let mut spread = serde_json::Map::new();
    for k in &keys {
        let vals: Vec<f64> = runs
            .iter()
            .filter_map(|r| r["probabilities"][k].as_f64())
            .collect();
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        spread.insert(k.clone(), json!(max - min));
    }
    let argmax_stable = runs
        .iter()
        .map(|r| r["choice"].as_str().unwrap_or("").to_string())
        .collect::<std::collections::HashSet<_>>()
        .len()
        == 1;
    Json(json!({"runs": runs, "argmax_stable": argmax_stable, "spread": spread})).into_response()
}

async fn separate(State(s): State<Shared>, body: axum::body::Bytes) -> Response {
    let req: SystemOneRequest = match parse_body(&body) {
        Ok(r) => r,
        Err(r) => return *r,
    };
    let mut answers = serde_json::Map::new();
    let mut input_tokens = 0usize;
    let mut latency = 0f64;
    for (qid, q) in &req.questions {
        let mut one = req.clone();
        one.questions = IndexMap::new();
        one.questions.insert(qid.clone(), q.clone());
        match answer(&s.engine, &one).await {
            Ok(resp) => {
                input_tokens += resp["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize;
                latency += resp["latency_ms"].as_f64().unwrap_or(0.0);
                if let Some(a) = resp["answers"].get(qid) {
                    answers.insert(qid.clone(), a.clone());
                }
            }
            Err(e) => return e,
        }
    }
    let answers_v = Value::Object(answers);
    let out_tokens = output_tokens(&s.engine.tok, &answers_v).unwrap_or(0);
    Json(json!({
        "model": req.model,
        "answers": answers_v,
        "usage": {"input_tokens": input_tokens, "output_tokens": out_tokens},
        "latency_ms": (latency * 10.0).round() / 10.0,
    }))
    .into_response()
}

async fn models(State(s): State<Shared>) -> Response {
    let e = &s.engine;
    let card = json!({
        "description": format!(
            "Kev pointer head on {}, serving {} at temperature {:.2}",
            e.meta.base, e.run, e.meta.temperature
        ),
        "release_date": s.release_date,
        "run": e.run,
        "base": e.meta.base,
        "lora": null,
        "device": e.device,
        "backend": "mistralrs",
        "dtype": e.dtype,
        "temperature": e.meta.temperature,
        "prefix_cache": {
            "size": e.prefix_cache_size,
            "min_state_tokens": 0,
            "hits": e.hits.load(std::sync::atomic::Ordering::Relaxed),
            "misses": e.misses.load(std::sync::atomic::Ordering::Relaxed),
            "cached_states": e.cached_states(),
        },
    });
    let models: Vec<Value> = MODEL_NAMES
        .iter()
        .map(|name| {
            let mut c = card.clone();
            c["name"] = json!(name);
            c
        })
        .collect();
    Json(json!({"models": models})).into_response()
}

async fn typesafe(
    State(_s): State<Shared>,
    headers: HeaderMap,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let req_id = headers
        .get("x-typesafe-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let api_key = std::env::var("KEV_API_KEY").ok().filter(|k| !k.is_empty());
    let mut resp = if let Some(key) = api_key {
        let want = format!("Bearer {key}");
        let got = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let ok = got.len() == want.len()
            && got
                .bytes()
                .zip(want.bytes())
                .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                == 0;
        if !ok {
            (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"))],
                Json(json!({
                    "detail": "missing or invalid API key; send Authorization: Bearer <KEV_API_KEY>"
                })),
            )
                .into_response()
        } else {
            next.run(request).await
        }
    } else {
        next.run(request).await
    };
    resp.headers_mut().insert(
        "x-typesafe-request-id",
        HeaderValue::from_str(&req_id.unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()))
            .unwrap_or_else(|_| HeaderValue::from_static("invalid")),
    );
    resp
}

pub fn router(state: Shared) -> Router {
    Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/v1/systemone/permute", post(permute))
        .route("/v1/systemone/separate", post(separate))
        .route("/v1/models", get(models))
        .layer(middleware::from_fn_with_state(state.clone(), typesafe))
        .layer(
            CorsLayer::permissive()
                .expose_headers([header::HeaderName::from_static("x-typesafe-request-id")]),
        )
        .with_state(state)
}

pub async fn serve(engine: KevEngine, release_date: String, port: u16) -> Result<()> {
    let state = Arc::new(AppState {
        engine: Arc::new(engine),
        release_date,
    });
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    println!("serving on 127.0.0.1:{port}");
    axum::serve(listener, router(state)).await?;
    Ok(())
}
