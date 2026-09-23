// snap playground — Missile Command. Every frame the whole sky goes to snap
// as one request: one threat score per missile plus one intercept choice,
// all read off a single shared-state pass. The battery fires on the answer.
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

/* ---------------- world constants ---------------- */
const W = 1000, H = 640, GROUND = 596;
const BATTERY = { x: 500, y: GROUND - 16 };
const CITY_X = [105, 200, 295, 705, 800, 895];
const LETTERS = "ABCDEF";
const MAX_ALIVE = 24;          // choice stays inside snap's 26 letter slots
const SHOT_SPEED = 420;        // world units / s
const BLAST = 48;
const RELOAD = 1.3;            // s between interceptors
const LEVELS = ["none", "low", "high", "critical"];
// the rubric lives in the shared state, read once; each per-missile suffix
// only carries the four level names
const SCALE = "none = falls on empty ground or ruins; low = impact in more than 10s; high = impact in 4 to 10s; critical = impact in under 4s";

const css = getComputedStyle(document.documentElement);
const C = Object.fromEntries(["bg", "surface", "raise", "line", "line-hi", "text", "soft", "muted", "faint",
  "accent", "num", "bad", "ok", "mono"].map((k) => [k, css.getPropertyValue(`--${k}`).trim()]));
const rgb = (hex) => { const n = parseInt(hex.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; };
const RGB = { muted: rgb(C.muted), num: rgb(C.num), bad: rgb(C.bad), accent: rgb(C.accent), faint: rgb(C.faint) };
const lerp = (a, b, t) => a + (b - a) * t;
const mix = (a, b, t) => a.map((v, i) => Math.round(lerp(v, b[i], t)));
const rgba = ([r, g, b], a = 1) => `rgba(${r},${g},${b},${a})`;
function threatRGB(t) {
  if (t == null) return RGB.muted;
  return t < .5 ? mix(RGB.muted, RGB.num, t * 2) : mix(RGB.num, RGB.bad, (t - .5) * 2);
}
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const rnd = (a, b) => a + Math.random() * (b - a);
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const pct = (p) => (p > 0 && p < .01 ? "<1%" : `${Math.round(p * 100)}%`);

/* ---------------- world ---------------- */
function skyline(seed) {
  const blocks = [];
  let x = -26;
  for (let k = 0; x < 22; k++) {
    const w = 7 + ((seed * 7 + k * 13) % 6);
    blocks.push({ x, w, h: 10 + ((seed * 11 + k * 17) % 16) });
    x += w + 1;
  }
  return blocks;
}

function newWorld() {
  return {
    t: 0, cities: CITY_X.map((x, i) => ({ x, name: LETTERS[i], alive: true, blocks: skyline(i + 3) })),
    missiles: [], shots: [], booms: [], pulses: [],
    nextId: 1, kills: 0, cooldown: 0, aim: -Math.PI / 2, queued: null, pick: null, over: false,
    trickle: 0, waveAt: .4, wave: 0,
  };
}

let G = newWorld();
let running = false;
let rain = true;
let mode = "shared";
let hoverId = null;

const alive = () => G.missiles;
const cityAt = (x) => G.cities.findIndex((c) => Math.abs(c.x - x) < 30);

function groundX() {
  for (;;) {
    const x = rnd(30, W - 30);
    if (cityAt(x) < 0 && Math.abs(x - BATTERY.x) > 50) return x;
  }
}

function makeMissile(tx, id) {
  if (tx == null) tx = Math.random() < .75 ? CITY_X[Math.floor(Math.random() * 6)] + rnd(-10, 10) : groundX();
  tx = clamp(tx, 20, W - 20);
  // enter from the visible top edge, even when the field is taller than the world
  const x0 = clamp(tx + rnd(-340, 340), 10, W - 10), y0 = Math.min(-8, -view.oy / view.s - 8);
  const dur = rnd(20, 32);
  const dx = tx - x0, dy = GROUND - y0;
  return { id, x0, y0, x: x0, y: y0, tx, vx: dx / dur, vy: dy / dur, target: cityAt(tx), threat: null, level: null, p: null, engaged: false };
}

function spawn(tx) {
  if (G.over || alive().length >= MAX_ALIVE) return false;
  G.missiles.push(makeMissile(tx, G.nextId++));
  return true;
}

const eta = (m) => Math.max(0, Math.round((GROUND - m.y) / m.vy));
function aimText(m) {
  if (m.target < 0) return "empty ground";
  const c = G.cities[m.target];
  return c.alive ? `city ${c.name}` : `ruins of city ${c.name}`;
}
const brief = (m) => `${aimText(m)}, impact in ${eta(m)}s`;

/* ---------------- interceptors ---------------- */
function fire(m) {
  const dx = m.x - BATTERY.x, dy = m.y - BATTERY.y;
  const a = m.vx * m.vx + m.vy * m.vy - SHOT_SPEED * SHOT_SPEED;
  const b = 2 * (dx * m.vx + dy * m.vy), c = dx * dx + dy * dy;
  const disc = b * b - 4 * a * c;
  let t = 0;
  if (disc >= 0) t = Math.max((-b - Math.sqrt(disc)) / (2 * a), (-b + Math.sqrt(disc)) / (2 * a), 0);
  const tx = m.x + m.vx * t, ty = Math.min(m.y + m.vy * t, GROUND - 24);
  const d = Math.hypot(tx - BATTERY.x, ty - BATTERY.y);
  G.shots.push({ x: BATTERY.x, y: BATTERY.y, tx, ty, left: d, ux: (tx - BATTERY.x) / d, uy: (ty - BATTERY.y) / d, target: m.id });
  G.aim = Math.atan2(ty - BATTERY.y, tx - BATTERY.x);
  G.cooldown = RELOAD;
  m.engaged = true;
}

/* ---------------- simulation ---------------- */
function step(dt) {
  G.t += dt;
  if (rain && !G.over) {
    // waves keep the sky crowded, so frames carry many missiles at once
    if (G.t >= G.waveAt) {
      const n = Math.min(6 + 2 * G.wave, 18);
      for (let k = 0; k < n; k++) spawn();
      G.wave++; G.waveAt = G.t + 10;
    }
    G.trickle += dt * .2;
    if (G.trickle >= 1) { G.trickle -= 1; spawn(); }
  }
  G.cooldown = Math.max(0, G.cooldown - dt);
  if (G.queued && G.cooldown === 0) {
    const m = G.missiles.find((q) => q.id === G.queued);
    if (m && !m.engaged) fire(m);
    G.queued = null;
  }

  for (const m of G.missiles) { m.x += m.vx * dt; m.y += m.vy * dt; }
  G.missiles = G.missiles.filter((m) => {
    if (m.y < GROUND) return true;
    G.booms.push({ x: m.tx, y: GROUND, t: 0, kind: "hit" });
    if (m.target >= 0 && G.cities[m.target].alive) G.cities[m.target].alive = false;
    return false;
  });

  G.shots = G.shots.filter((s) => {
    const d = SHOT_SPEED * dt;
    if (d < s.left) { s.x += s.ux * d; s.y += s.uy * d; s.left -= d; return true; }
    G.booms.push({ x: s.tx, y: s.ty, t: 0, kind: "ally", target: s.target });
    return false;
  });

  G.booms = G.booms.filter((b) => {
    b.t += dt;
    if (b.kind === "ally") {
      b.r = BLAST * (b.t < .28 ? 1 - (1 - b.t / .28) ** 3 : b.t < .7 ? 1 : 1 - (b.t - .7) / .5);
      if (b.t < .9) {
        const before = G.missiles.length;
        G.missiles = G.missiles.filter((m) => Math.hypot(m.x - b.x, m.y - b.y) > b.r);
        G.kills += before - G.missiles.length;
      }
      if (b.t >= 1.2) {
        const m = G.missiles.find((q) => q.id === b.target);
        if (m) m.engaged = false; // missed: back in the pool
        return false;
      }
      return true;
    }
    return b.t < 1.1;
  });

  G.pulses = G.pulses.filter((p) => (p.t += dt) < .5);

  if (!G.over && G.cities.every((c) => !c.alive)) gameOver();
}

/* ---------------- rendering: battlefield ---------------- */
const cv = $("cv"), ctx = cv.getContext("2d");
let view = { s: 1, ox: 0, oy: 0, dpr: 1, w: 0, h: 0 };

function resize() {
  const r = cv.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
  const s = Math.min(r.width / W, (r.height - 8) / H);
  view = { s, ox: (r.width - W * s) / 2, oy: r.height - H * s, dpr, w: r.width, h: r.height };
}
new ResizeObserver(() => { resize(); drawChart(); }).observe($("field"));

function toWorld(e) {
  const r = cv.getBoundingClientRect();
  return { x: (e.clientX - r.left - view.ox) / view.s, y: (e.clientY - r.top - view.oy) / view.s };
}

function brackets(x, y, r, color, lw = 1.5) {
  const k = r * .55;
  ctx.strokeStyle = color; ctx.lineWidth = lw;
  ctx.beginPath();
  for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
    ctx.moveTo(x + sx * r, y + sy * (r - k)); ctx.lineTo(x + sx * r, y + sy * r); ctx.lineTo(x + sx * (r - k), y + sy * r);
  }
  ctx.stroke();
}

function draw() {
  const { s, ox, oy, dpr, w } = view;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * ox, dpr * oy);
  const left = -ox / s, right = (w - ox) / s;
  const u = Math.max(1, .9 / s); // keeps labels and marks legible when the field is small

  // ground
  ctx.fillStyle = C.surface; ctx.fillRect(left, GROUND, right - left, 80);
  ctx.fillStyle = C["line-hi"]; ctx.fillRect(left, GROUND, right - left, 1);

  // cities
  ctx.font = `500 ${11 * u}px ${C.mono}`; ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const c of G.cities) {
    if (c.alive) {
      ctx.fillStyle = C["line-hi"];
      for (const b of c.blocks) ctx.fillRect(c.x + b.x, GROUND - b.h, b.w, b.h);
      ctx.fillStyle = C.soft;
      for (const b of c.blocks) ctx.fillRect(c.x + b.x, GROUND - b.h, b.w, 1);
    } else {
      ctx.fillStyle = C.line;
      ctx.beginPath(); ctx.moveTo(c.x - 24, GROUND);
      for (let k = 0; k <= 8; k++) ctx.lineTo(c.x - 24 + k * 6, GROUND - (k % 2 ? 5 : 2) - (k % 3));
      ctx.lineTo(c.x + 24, GROUND); ctx.fill();
    }
    ctx.fillStyle = c.alive ? C.muted : C.faint;
    ctx.fillText(c.name, c.x, GROUND + 8);
  }

  // battery
  ctx.fillStyle = C.raise;
  ctx.beginPath(); ctx.moveTo(BATTERY.x - 30, GROUND); ctx.lineTo(BATTERY.x - 14, BATTERY.y); ctx.lineTo(BATTERY.x + 14, BATTERY.y); ctx.lineTo(BATTERY.x + 30, GROUND); ctx.fill();
  ctx.strokeStyle = C.accent; ctx.lineWidth = 3; ctx.lineCap = "round";
  ctx.beginPath(); ctx.moveTo(BATTERY.x, BATTERY.y); ctx.lineTo(BATTERY.x + Math.cos(G.aim) * 14, BATTERY.y + Math.sin(G.aim) * 14); ctx.stroke();
  ctx.lineCap = "butt";
  if (G.cooldown > 0) {
    ctx.strokeStyle = rgba(RGB.accent, .5); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(BATTERY.x, BATTERY.y, 20, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * (1 - G.cooldown / RELOAD)); ctx.stroke();
  }

  // missile trails + heads
  ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.font = `${11 * u}px ${C.mono}`;
  for (const m of G.missiles) {
    const col = threatRGB(m.threat);
    const g = ctx.createLinearGradient(m.x0, m.y0, m.x, m.y);
    g.addColorStop(0, rgba(col, 0)); g.addColorStop(1, rgba(col, m.threat == null ? .45 : .8));
    ctx.strokeStyle = g; ctx.lineWidth = 1.5 * Math.min(u, 1.6);
    ctx.beginPath(); ctx.moveTo(m.x0, m.y0); ctx.lineTo(m.x, m.y); ctx.stroke();
    ctx.fillStyle = rgba(col); ctx.beginPath(); ctx.arc(m.x, m.y, 2.6 * u, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = rgba(col, m.engaged ? .45 : .9);
    ctx.fillText(`m${m.id}`, m.x + 7 * u, m.y - 7 * u);
  }

  // decision pulses: every scored missile lights up at the same instant
  for (const p of G.pulses) {
    const m = G.missiles.find((q) => q.id === p.id);
    if (!m) continue;
    const f = p.t / .5;
    ctx.strokeStyle = rgba(threatRGB(m.threat), (1 - f) * .8); ctx.lineWidth = 1.2 * Math.min(u, 1.6);
    ctx.beginPath(); ctx.arc(m.x, m.y, (4 + f * (8 + (m.threat ?? 0) * 12)) * u, 0, Math.PI * 2); ctx.stroke();
  }

  // snap's pick
  const pick = G.missiles.find((m) => m.id === G.pick);
  if (pick) {
    brackets(pick.x, pick.y, 11 * u, C.accent, 1.5 * Math.min(u, 1.6));
    if (pick.p != null) {
      ctx.fillStyle = C.accent; ctx.textAlign = "right";
      ctx.fillText(`shoot ${pct(pick.p)}`, pick.x - 16 * u, pick.y + 1);
      ctx.textAlign = "left";
    }
  }
  const hov = hoverId != null && hoverId !== G.pick && G.missiles.find((m) => m.id === hoverId);
  if (hov) brackets(hov.x, hov.y, 10 * u, C.soft, Math.min(u, 1.6));

  // interceptors
  for (const sh of G.shots) {
    ctx.strokeStyle = rgba(RGB.accent, .55); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(BATTERY.x, BATTERY.y); ctx.lineTo(sh.x, sh.y); ctx.stroke();
    ctx.fillStyle = C.text; ctx.beginPath(); ctx.arc(sh.x, sh.y, 2, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = rgba(RGB.accent, .5); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(sh.tx - 4, sh.ty - 4); ctx.lineTo(sh.tx + 4, sh.ty + 4); ctx.moveTo(sh.tx + 4, sh.ty - 4); ctx.lineTo(sh.tx - 4, sh.ty + 4); ctx.stroke();
  }

  // blasts
  for (const b of G.booms) {
    if (b.kind === "ally") {
      const fade = b.t < .7 ? 1 : Math.max(0, 1 - (b.t - .7) / .5);
      ctx.fillStyle = rgba(RGB.accent, .12 * fade); ctx.strokeStyle = rgba(RGB.accent, .85 * fade); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(b.x, b.y, Math.max(0, b.r), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    } else {
      const k = b.t / 1.1;
      ctx.fillStyle = rgba(RGB.bad, .35 * (1 - k));
      ctx.beginPath(); ctx.arc(b.x, b.y, 8 + k * 30, Math.PI, 0); ctx.fill();
      ctx.strokeStyle = rgba(RGB.bad, .9 * (1 - k)); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(b.x, b.y, 8 + k * 30, Math.PI, 0); ctx.stroke();
    }
  }
}

/* ---------------- snap decisions ---------------- */
let inflight = null;      // { t0, n, mode, reqStr }
let retryAt = 0;
let last = null;          // last decision, for the panel
let lastReq = null, lastRes = null;  // pretty-printed JSON, for the sheet
let sweeping = false;
const samples = [];       // { n, ms, mode }

function frameRequest(list, md) {
  const incoming = Object.fromEntries(list.map((m) => [`m${m.id}`, brief(m)]));
  const questions = {};
  for (const m of list)
    questions[`m${m.id}`] = { type: "score", instructions: `threat level of missile m${m.id}`, criteria: LEVELS };
  if (list.length > 1)
    questions.intercept = {
      type: "choice",
      instructions: "which missile should the battery shoot first? the most urgent one aimed at a standing city",
      criteria: incoming,
    };
  const standing = G.cities.filter((c) => c.alive).map((c) => c.name).join(" ");
  return {
    state: {
      defense: "one interceptor battery at the center; ruins cannot be saved",
      threat_scale: SCALE, cities_standing: standing || "none", incoming,
    },
    questions, mode: md, temperature: 1,
  };
}

async function post(req) {
  const res = await fetch("/v1/systemone", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(req),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
  return body;
}

async function decide() {
  const list = G.missiles.filter((m) => !m.engaged);
  if (!list.length) return;
  const world = G, md = mode;
  const req = frameRequest(list, md);
  const t0 = performance.now();
  inflight = { t0, n: list.length, mode: md, reqStr: JSON.stringify(req, null, 2) };
  renderSheet();
  try {
    const body = await post(req);
    const ms = performance.now() - t0;
    clearErr();
    lastReq = inflight.reqStr;
    lastRes = JSON.stringify(body, null, 2);
    if (world === G) apply(list, body, ms, md);
    renderSheet();
  } catch (e) {
    showErr(e);
    retryAt = performance.now() + 2500;
  } finally {
    inflight = null;
  }
}

function apply(list, body, rtt, md) {
  const x = body.x_snap || {};
  record(list.length, x.total_ms ?? rtt, md);
  for (const m of list) {
    const a = body.answers?.[`m${m.id}`];
    if (a) { m.threat = a.score; m.level = a.level; }
    if (!reduced) G.pulses.push({ id: m.id, t: 0 });
  }
  const probs = list.length > 1 ? body.answers?.intercept?.probabilities || {} : { [`m${list[0].id}`]: 1 };
  for (const m of list) m.p = probs[`m${m.id}`] ?? 0;

  // the answer is a distribution: if snap's top pick died while it was
  // thinking, walk down the map to the best missile still in the sky
  const ranked = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  let pick = null, fallback = 0;
  for (const [k] of ranked) {
    const m = G.missiles.find((q) => `m${q.id}` === k && !q.engaged);
    if (m) { pick = m; break; }
    fallback++;
  }
  G.pick = pick?.id ?? null;
  if (pick && running) G.queued = pick.id;

  const asJson = JSON.stringify({
    threat: Object.fromEntries(list.map((m) => [`m${m.id}`, LEVELS[m.level] ?? null])),
    ...(list.length > 1 ? { intercept: body.answers?.intercept?.choice ?? null } : {}),
  }).length;
  last = { n: list.length, x, rtt, md, pick, fallback, asJson };
  renderStats();
  renderBoard();
}

/* ---------------- panel ---------------- */
const stat = (v, unit, label, cls = "") =>
  `<div class="stat ${cls}"><b>${esc(v ?? "—")}${unit ? `<small>${unit}</small>` : ""}</b><span>${label}</span></div>`;

function renderStats() {
  if (!last) {
    $("stats").innerHTML = `<p class="stats-empty">Numbers from the last frame snap decided land here.</p>`;
    return;
  }
  const { x, n, md, asJson } = last;
  const ms = x.total_ms != null ? Math.round(x.total_ms) : Math.round(last.rtt);
  $("stats").innerHTML =
    stat(ms, "ms", `total · ${md}`, `lead ${md}`) +
    stat(x.prefill_ms != null ? Math.round(x.prefill_ms) : null, "ms", "prefill") +
    stat(n, "", n === 1 ? "missile scored" : "missiles scored") +
    stat(x.decoded_items, "", `items · ${x.suffix_decode ?? "—"}`) +
    stat(asJson.toLocaleString(), "ch", "as JSON text");
}

const shortTarget = (m) => {
  if (m.target < 0) return ["ground", "ground"];
  const c = G.cities[m.target];
  return c.alive ? [`city ${c.name}`, ""] : [`ruins ${c.name}`, "ruins"];
};

function renderBoard() {
  const ms = [...G.missiles].sort((a, b) =>
    (b.threat ?? -1) - (a.threat ?? -1) || (b.p ?? 0) - (a.p ?? 0) || a.id - b.id);
  $("board").hidden = !ms.length;
  $("board-empty").hidden = !!ms.length;
  $("rows").innerHTML = ms.map((m) => {
    const [tgt, tcls] = shortTarget(m);
    const col = rgba(threatRGB(m.threat));
    const cls = [m.id === G.pick ? "pick" : "", m.engaged ? "engaged" : "", m.id === hoverId ? "hover" : ""].join(" ");
    const thr = m.threat == null
      ? `<span class="pending">next frame</span>`
      : `<span class="track"><i style="--p:${m.threat.toFixed(3)};--c:${col}"></i></span><em style="--c:${col}">${LEVELS[m.level] ?? "—"}</em>`;
    const p = m.engaged ? "fired" : m.p == null ? "—" : pct(m.p);
    return `<div class="row ${cls}" role="row" data-id="${m.id}">
      <span class="id">m${m.id}</span><span class="tgt ${tcls}">${tgt}</span>
      <span class="r">${eta(m)}s</span><span class="thr">${thr}</span><span class="r p">${p}</span></div>`;
  }).join("");
}

$("rows").addEventListener("mouseover", (e) => {
  const r = e.target.closest("[data-id]");
  hoverId = r ? +r.dataset.id : null;
});
$("board").addEventListener("mouseleave", () => { hoverId = null; });
cv.addEventListener("mousemove", (e) => {
  const p = toWorld(e);
  let best = null, bd = 18;
  for (const m of G.missiles) { const d = Math.hypot(m.x - p.x, m.y - p.y); if (d < bd) { bd = d; best = m; } }
  hoverId = best?.id ?? null;
});
cv.addEventListener("mouseleave", () => { hoverId = null; });

function renderHud() {
  const standing = G.cities.filter((c) => c.alive).length;
  $("h-kills").textContent = G.kills;
  $("h-cities").textContent = `${standing}/6`;
  $("h-cities").classList.toggle("warn", standing <= 2);
  $("h-alive").textContent = G.missiles.length;
  const live = $("live");
  if (inflight) {
    live.textContent = `deciding ${Math.round(performance.now() - inflight.t0)} ms`;
    live.classList.add("busy");
  } else {
    live.textContent = sweeping ? "sweeping" : last ? `last ${Math.round(last.x.total_ms ?? last.rtt)} ms` : "";
    live.classList.remove("busy");
  }
  if (!$("sheet").hidden) renderSheet();
}

/* ---------------- last-frame sheet ---------------- */
function renderSheet() {
  if ($("sheet").hidden) return;
  if (lastReq && lastRes) {
    // the last completed pair stays on screen; the new frame gets the meta line
    $("req-json").textContent = lastReq;
    $("res-json").textContent = lastRes;
    $("res-head").textContent = `← 200 · ${Math.round(last.x.total_ms ?? last.rtt)} ms · ${last.md}`;
    $("sheet-meta").textContent = `${last.n} missiles · ${last.md}` +
      (inflight ? ` · next frame in flight · ${Math.round(performance.now() - inflight.t0)} ms` : "");
  } else if (inflight) {
    $("req-json").textContent = inflight.reqStr;
    $("res-json").textContent = `deciding ${inflight.n} missiles…`;
    $("res-head").textContent = `← pending · ${Math.round(performance.now() - inflight.t0)} ms`;
    $("sheet-meta").textContent = `${inflight.n} missiles · ${inflight.mode} · in flight`;
  } else {
    $("req-json").textContent = "No frame sent yet — press Start.";
    $("res-json").textContent = "—";
    $("res-head").textContent = "← response";
    $("sheet-meta").textContent = "";
  }
}

const frameBtn = $("frame-btn");
function setSheet(open) {
  $("sheet").hidden = !open;
  frameBtn.setAttribute("aria-expanded", open);
  if (open) renderSheet();
}
frameBtn.onclick = () => setSheet($("sheet").hidden);
$("sheet-close").onclick = () => setSheet(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") setSheet(false); });

function showErr(e) {
  let el = document.querySelector(".side-err");
  if (!el) {
    el = document.createElement("div");
    el.className = "err side-err";
    $("stats").after(el);
  }
  const msg = String(e?.message || e);
  el.innerHTML = `<b>snap didn't answer</b><span>${esc(msg)}${/fetch|network/i.test(msg) ? ". Is `snap serve` still running?" : ""} Retrying in a moment.</span>`;
}
const clearErr = () => document.querySelector(".side-err")?.remove();

/* ---------------- latency chart ---------------- */
const chart = $("chart"), cx = chart.getContext("2d");

function record(n, ms, md) {
  samples.push({ n, ms, md });
  if (samples.length > 600) samples.shift();
  drawChart();
  renderReadout();
}

function fit(md) {
  const pts = samples.filter((s) => s.md === md);
  if (new Set(pts.map((p) => p.n)).size < 2) return null;
  const mx = pts.reduce((a, p) => a + p.n, 0) / pts.length, my = pts.reduce((a, p) => a + p.ms, 0) / pts.length;
  let sxy = 0, sxx = 0;
  for (const p of pts) { sxy += (p.n - mx) * (p.ms - my); sxx += (p.n - mx) ** 2; }
  const slope = sxy / sxx;
  return { slope, at: (n) => my + slope * (n - mx) };
}

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

function renderReadout() {
  const sh = fit("shared"), di = fit("direct");
  const el = $("readout");
  if (!sh && !di) {
    el.innerHTML = samples.length
      ? "Keep playing: the line appears once frames with different missile counts come in."
      : "Each dot is one decision: missiles in the frame against server time.";
    return;
  }
  const per = (f, md) => `<b class="${md}">${f.slope >= 0 ? "+" : ""}${fmtMs(f.slope)}</b>`;
  if (sh && di) {
    const n = 16, ratio = di.at(n) / Math.max(1, sh.at(n));
    el.innerHTML = `Each extra missile costs ${per(sh, "shared")} batched, ${per(di, "direct")} on its own pass. At 16 missiles direct is <b>${ratio.toFixed(1)}×</b> slower.`;
  } else if (sh) {
    el.innerHTML = `Each extra missile costs ${per(sh, "shared")} when batched. Switch to <b class="direct">direct</b> or run the sweep to compare.`;
  } else {
    el.innerHTML = `Each extra missile costs ${per(di, "direct")} on its own pass. Switch to <b class="shared">shared</b> to compare.`;
  }
}

function drawChart() {
  const r = chart.getBoundingClientRect();
  if (!r.width) return;
  const dpr = Math.min(devicePixelRatio || 1, 2);
  chart.width = Math.round(r.width * dpr); chart.height = Math.round(r.height * dpr);
  cx.setTransform(dpr, 0, 0, dpr, 0, 0);
  cx.clearRect(0, 0, r.width, r.height);
  const L = 44, R = 8, T = 6, B = 20, w = r.width - L - R, h = r.height - T - B;
  const maxMs = Math.max(400, ...samples.map((s) => s.ms));
  const step = [100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000].find((v) => maxMs / v <= 4) || 20000;
  const top = Math.ceil(maxMs / step) * step;
  const X = (n) => L + (n / MAX_ALIVE) * w, Y = (ms) => T + h - (ms / top) * h;

  cx.font = `10.5px ${C.mono}`; cx.fillStyle = C.faint; cx.strokeStyle = C.line; cx.lineWidth = 1;
  cx.textAlign = "right"; cx.textBaseline = "middle";
  for (let v = 0; v <= top; v += step) {
    const y = Math.round(Y(v)) + .5;
    cx.beginPath(); cx.moveTo(L, y); cx.lineTo(L + w, y); cx.stroke();
    cx.fillText(fmtMs(v).replace(" ", ""), L - 8, y);
  }
  cx.textAlign = "center"; cx.textBaseline = "top";
  for (const n of [1, 4, 8, 12, 16, 20, 24]) cx.fillText(n, X(n), T + h + 6);

  for (const md of ["shared", "direct"]) {
    const col = md === "shared" ? RGB.accent : RGB.num;
    const pts = samples.filter((s) => s.md === md);
    cx.fillStyle = rgba(col, .45);
    for (const p of pts) { cx.beginPath(); cx.arc(X(p.n), Y(p.ms), 2.2, 0, Math.PI * 2); cx.fill(); }
    const byN = new Map();
    for (const p of pts) byN.set(p.n, [...(byN.get(p.n) || []), p.ms]);
    const line = [...byN].sort((a, b) => a[0] - b[0]).map(([n, v]) => [X(n), Y(median(v))]);
    if (line.length > 1) {
      cx.strokeStyle = rgba(col, .95); cx.lineWidth = 1.6; cx.lineJoin = "round";
      cx.beginPath(); line.forEach(([x, y], k) => (k ? cx.lineTo(x, y) : cx.moveTo(x, y))); cx.stroke();
    }
  }
}

/* ---------------- sweep: a controlled 1–24 benchmark ---------------- */
async function sweep() {
  const btn = $("sweep");
  if (sweeping) { sweeping = false; return; }
  sweeping = true;
  const resume = running;
  setRunning(false, true);
  btn.classList.add("busy");
  const plan = [1, 2, 4, 8, 12, 16, 24].flatMap((n) => ["shared", "direct"].map((md) => [n, md]));
  const cities = CITY_X.map((x, i) => ({ x, name: LETTERS[i], alive: true }));
  try {
    for (let k = 0; k < plan.length && sweeping; k++) {
      const [n, md] = plan[k];
      btn.textContent = `Stop · ${k + 1}/${plan.length}`;
      const list = Array.from({ length: n }, (_, j) => {
        const m = makeMissile(null, j + 1);
        const f = rnd(.1, .8);
        m.x = lerp(m.x0, m.tx, f); m.y = lerp(m.y0, GROUND, f);
        return m;
      });
      const t0 = performance.now();
      inflight = { t0, n, mode: md };
      const saved = G.cities;
      G.cities = cities; // time a pristine map, whatever the game's state
      const req = frameRequest(list, md);
      G.cities = saved;
      try {
        const body = await post(req);
        clearErr();
        record(n, body.x_snap?.total_ms ?? performance.now() - t0, md);
      } catch (e) { showErr(e); break; }
      finally { inflight = null; }
    }
  } finally {
    sweeping = false;
    btn.classList.remove("busy");
    btn.textContent = "Sweep 1–24";
    if (resume) setRunning(true);
  }
}
$("sweep").onclick = sweep;

/* ---------------- controls ---------------- */
function setRunning(on, quiet = false) {
  if (on && G.over) restart();
  running = on;
  $("go-label").textContent = on ? "Pause" : G.t > 0 ? "Resume" : "Start";
  $("go").classList.toggle("busy", false);
  if (on) $("veil").hidden = true;
  else if (!quiet && !G.over && G.t > 0) showVeil("Paused", `Frame loop stopped. <span class="k">${G.missiles.length}</span> missiles hang in the sky.`, "Resume");
}

function showVeil(title, body, cta) {
  $("v-title").textContent = title;
  $("v-body").innerHTML = body;
  $("v-go-label").textContent = cta;
  $("veil").hidden = false;
}

function gameOver() {
  G.over = true;
  running = false;
  G.queued = null;
  $("go-label").textContent = "Restart";
  const ms = samples.filter((s) => s.md === mode).map((s) => s.ms);
  showVeil("The last city fell",
    `<span class="k">${G.kills}</span> missiles intercepted in <span class="k">${Math.round(G.t)}s</span>` +
    (ms.length ? `, median frame <span class="k">${fmtMs(median(ms))}</span> in <b>${mode}</b> mode.` : "."),
    "Defend again");
}

function restart() {
  G = newWorld();
  last = null; lastReq = null; lastRes = null;
  renderStats(); renderBoard(); renderHud();
}

const toggle = () => setRunning(!running);
$("go").onclick = toggle;
$("v-go").onclick = () => setRunning(true);

$("salvo").onclick = () => { for (let k = 0; k < 5; k++) spawn(); renderBoard(); };
$("rain").onclick = (e) => {
  rain = !rain;
  e.currentTarget.setAttribute("aria-pressed", rain);
};

$("modes").addEventListener("click", (e) => {
  const b = e.target.closest("[data-m]");
  if (!b) return;
  mode = b.dataset.m;
  document.querySelectorAll("#modes button").forEach((x) => {
    x.classList.toggle("on", x === b);
    x.setAttribute("aria-checked", x === b);
  });
});

cv.addEventListener("click", (e) => {
  if (G.over) return;
  const p = toWorld(e);
  const tx = clamp(p.x, 20, W - 20);
  if (e.shiftKey) for (let k = 0; k < 5; k++) spawn(tx + rnd(-60, 60));
  else spawn(tx);
  renderBoard();
});

document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code === "Space" && !(e.target instanceof HTMLButtonElement)) { e.preventDefault(); toggle(); }
  else if (e.key === "s" || e.key === "S") { for (let k = 0; k < 5; k++) spawn(); renderBoard(); }
});

/* ---------------- frame loop ---------------- */
let prev = performance.now(), boardAt = 0;
function frame(now) {
  const dt = Math.min(.05, (now - prev) / 1000);
  prev = now;
  if (running) step(dt);
  else if (G.booms.length || G.pulses.length) { // let blasts finish under the veil
    G.booms = G.booms.filter((b) => (b.t += dt) < 1.2);
    G.pulses = G.pulses.filter((p) => (p.t += dt) < .5);
  }
  if (running && !inflight && !sweeping && now > retryAt && G.missiles.some((m) => !m.engaged)) decide();
  draw();
  renderHud();
  if (now - boardAt > 250) { boardAt = now; renderBoard(); }
  requestAnimationFrame(frame);
}

/* ---------------- health ---------------- */
async function health() {
  const srv = $("srv");
  try {
    const j = await (await fetch("/healthz")).json();
    srv.classList.remove("down", "wait");
    $("model").textContent = j.name || j.model;
    srv.title = j.model;
  } catch {
    srv.classList.remove("wait");
    srv.classList.add("down");
    $("model").textContent = "server unreachable";
  }
}

/* ---------------- init ---------------- */
resize();
renderStats();
renderBoard();
drawChart();
health();
setInterval(health, 5000);
requestAnimationFrame(frame);
