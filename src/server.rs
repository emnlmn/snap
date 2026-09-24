//! Axum surface: the single POST /v1/systemone API (Jev wire + snap extras).

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::header::CONTENT_TYPE;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Json, Redirect};
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Value};

use crate::api::{from_native, SystemoneRequest};
use crate::engine::Engine;

struct Slot {
    engine: Engine,
    /// the tested-model spec the engine was built from (e.g. "spark-4b")
    name: String,
}

struct AppState {
    slot: Mutex<Slot>,
    /// serializes model swaps — only one load at a time
    switch: Mutex<()>,
    n_ctx: i32,
    n_threads: i32,
    started: Instant,
}

fn err422(e: anyhow::Error) -> (StatusCode, Json<Value>) {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({"error": e.to_string()})),
    )
}

async fn healthz(State(s): State<Arc<AppState>>) -> Json<Value> {
    let slot = s.slot.lock().unwrap();
    Json(json!({
        "status": "ok",
        "model": slot.engine.model_id,
        "name": slot.name,
        "uptime_s": s.started.elapsed().as_secs(),
    }))
}

async fn models(State(s): State<Arc<AppState>>) -> Json<Value> {
    let active = s.slot.lock().unwrap().name.clone();
    Json(json!({
        "object": "list",
        "data": crate::models::MODELS.iter().map(|(n, repo, file)| json!({
            "id": n, "object": "model", "created": 0, "owned_by": "snap",
            "repo": repo, "file": file, "active": *n == active,
        })).collect::<Vec<_>>(),
    }))
}

/// POST /v1/models {"model": "<spec>"} — load and hot-swap the resident
/// model. The old engine keeps serving until the new one is ready.
async fn switch_model(
    State(s): State<Arc<AppState>>,
    Json(req): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let name = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if name.is_empty() {
        return Err(err422(anyhow::anyhow!(
            "body must be {{\"model\": \"<name>\"}}"
        )));
    }
    let s2 = s.clone();
    let n_ctx = s.n_ctx;
    let n_threads = s.n_threads;
    let name2 = name.clone();
    let model_id = tokio::task::spawn_blocking(move || -> Result<String> {
        let _serial = s2.switch.lock().unwrap();
        if s2.slot.lock().unwrap().name == name2 {
            return Ok(s2.slot.lock().unwrap().engine.model_id.clone());
        }
        let path = crate::models::resolve(&name2)?;
        let mut eng = Engine::new(path.to_string_lossy().as_ref(), n_ctx, 1024, n_threads)?;
        if let Err(e) = eng.warmup() {
            eprintln!("snap: warmup failed: {e}");
        }
        let model_id = eng.model_id.clone();
        let mut slot = s2.slot.lock().unwrap();
        slot.engine = eng;
        slot.name = name2;
        Ok(model_id)
    })
    .await
    .map_err(|e| err422(anyhow::anyhow!(e)))?
    .map_err(err422)?;
    Ok(Json(
        json!({"status": "ok", "model": model_id, "name": name}),
    ))
}

async fn systemone(
    State(s): State<Arc<AppState>>,
    Json(req): Json<SystemoneRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let native = req.to_native();
    let out = tokio::task::spawn_blocking(move || s.slot.lock().unwrap().engine.decide(&native))
        .await
        .map_err(|e| err422(anyhow::anyhow!(e)))?
        .map_err(err422)?;
    Ok(Json(from_native(&out)))
}

// playground pages are baked into the binary — `snap serve` is self-contained
async fn pg_console() -> Html<&'static str> {
    Html(include_str!("web/playground.html"))
}
async fn pg_css() -> impl IntoResponse {
    (
        [(CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("web/app.css"),
    )
}
async fn pg_console_js() -> impl IntoResponse {
    (
        [(CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("web/console.js"),
    )
}
async fn pg_missile() -> Html<&'static str> {
    Html(include_str!("web/missile.html"))
}
async fn pg_missile_css() -> impl IntoResponse {
    (
        [(CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("web/missile.css"),
    )
}
async fn pg_missile_js() -> impl IntoResponse {
    (
        [(CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("web/missile.js"),
    )
}
async fn pg_logo() -> impl IntoResponse {
    (
        [(CONTENT_TYPE, "image/png")],
        include_bytes!("web/logo.png").as_slice(),
    )
}
/// SIGTERM (`snap stop`) or SIGINT — drain in-flight requests, then exit.
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}

pub async fn serve(
    engine: Engine,
    model: &str,
    n_ctx: i32,
    n_threads: i32,
    host: &str,
    port: u16,
) -> Result<()> {
    let state = Arc::new(AppState {
        slot: Mutex::new(Slot {
            engine,
            name: model.to_string(),
        }),
        switch: Mutex::new(()),
        n_ctx,
        n_threads,
        started: Instant::now(),
    });
    let app = Router::new()
        .route("/", get(|| async { Redirect::temporary("/playground") }))
        .route("/healthz", get(healthz))
        .route("/v1/models", get(models).post(switch_model))
        .route("/v1/systemone", post(systemone))
        .route("/playground", get(pg_console))
        .route("/playground/app.css", get(pg_css))
        .route("/playground/console.js", get(pg_console_js))
        .route("/playground/logo.png", get(pg_logo))
        .route("/playground/missile", get(pg_missile))
        .route("/playground/missile.css", get(pg_missile_css))
        .route("/playground/missile.js", get(pg_missile_js))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind((host, port))
        .await
        .with_context(|| format!("cannot listen on {host}:{port} — already running?"))?;
    let port = listener.local_addr().map(|a| a.port()).unwrap_or(port);
    crate::instances::bound_port(port);
    eprintln!("snap serving on http://{host}:{port}");
    eprintln!("  api         POST /v1/systemone");
    eprintln!("  playground  http://{host}:{port}/playground");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}
