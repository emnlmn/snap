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

struct AppState {
    engine: Mutex<Engine>,
    started: Instant,
}

fn err422(e: anyhow::Error) -> (StatusCode, Json<Value>) {
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        Json(json!({"error": e.to_string()})),
    )
}

async fn healthz(State(s): State<Arc<AppState>>) -> Json<Value> {
    let model = s.engine.lock().unwrap().model_id.clone();
    Json(json!({"status": "ok", "model": model, "uptime_s": s.started.elapsed().as_secs()}))
}

async fn models(State(s): State<Arc<AppState>>) -> Json<Value> {
    let model = s.engine.lock().unwrap().model_id.clone();
    Json(json!({
        "object": "list",
        "data": [{"id": model, "object": "model", "created": 0, "owned_by": "snap"}],
    }))
}

async fn systemone(
    State(s): State<Arc<AppState>>,
    Json(req): Json<SystemoneRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let native = req.to_native();
    let out = tokio::task::spawn_blocking(move || s.engine.lock().unwrap().decide(&native))
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


pub async fn serve(engine: Engine, host: &str, port: u16) -> Result<()> {
    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        started: Instant::now(),
    });
    let app = Router::new()
        .route("/", get(|| async { Redirect::temporary("/playground") }))
        .route("/healthz", get(healthz))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .route("/playground", get(pg_console))
        .route("/playground/app.css", get(pg_css))
        .route("/playground/console.js", get(pg_console_js))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind((host, port))
        .await
        .with_context(|| format!("cannot listen on {host}:{port} — already running?"))?;
    eprintln!("snap serving on http://{host}:{port}");
    eprintln!("  api         POST /v1/systemone");
    eprintln!("  playground  http://{host}:{port}/playground");
    axum::serve(listener, app).await?;
    Ok(())
}
