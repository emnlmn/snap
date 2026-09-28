// snap playground — API console. The editor (state box + question tree)
// is the source of truth; the JSON and cURL panes mirror it.
"use strict";

const $ = (id) => document.getElementById(id);

// the three Jev primitives lead; numeric is a snap extension
const PRIMS = [
  { t: "noul", desc: "How true is it — yes / no", ex: "is the customer asking for a refund?" },
  { t: "choice", desc: "Pick one key from a set", ex: "which team should handle this ticket?" },
  { t: "score", desc: "Grade on an ordered rubric", ex: "how severe is this bug report?" },
];
const EXT = [{ t: "numeric", desc: "Estimate a value in a range", ex: "how many minutes will this take?" }];
const PH = Object.fromEntries([...PRIMS, ...EXT].map((p) => [p.t, p.ex]));
PH.boolean = PH.noul;
const MODES = [
  ["shared", "state read once, questions batched"],
  ["direct", "each question on its own pass"],
];

const ICON = {
  x: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 4.5l7 7m0-7l-7 7"/></svg>',
  plus: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3.5v9M3.5 8h9"/></svg>',
  chev: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 4l4 4-4 4"/></svg>',
  caret: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 6.5l3 3 3-3"/></svg>',
  check: '<svg class="ck" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7"/></svg>',
  arrow: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8h9m-3.5-3.5L12.5 8 9 11.5"/></svg>',
};

/* ---------------- examples ---------------- */
const LESSONS = [
  {
    tag: "noul", title: "Is this lead hot?", sub: "Two probes on one inbound message",
    req: {
      state: { message: "Hi — we're a 40-person logistics company. Our current vendor's contract ends next month and we've shortlisted you. Can we get a demo this week? Budget is approved." },
      questions: {
        sales_ready: { type: "noul", instructions: "is this lead ready for a sales call?" },
        existing_customer: { type: "noul", instructions: "is this an existing customer?" },
      },
    },
  },
  {
    tag: "choice", title: "What language is this?", sub: "One key out of four",
    req: {
      state: { text: "Ich habe keine Ahnung, wo mein Regenschirm ist." },
      questions: {
        language: { type: "choice", instructions: "which language is the text written in?", criteria: {
          english: "English", german: "German", dutch: "Dutch", italian: "Italian" } },
      },
    },
  },
  {
    tag: "score", title: "How bad is this bug?", sub: "Four-level severity rubric",
    req: {
      state: { report: "Checkout button does nothing on Safari. About 30% of orders affected since this morning." },
      questions: {
        severity: { type: "score", instructions: "how severe is this bug report?", criteria: [
          "cosmetic — nobody is blocked",
          "annoying — there is a workaround",
          "blocking — some users can't finish",
          "outage — revenue is on fire"] },
      },
    },
  },
];

const CASES = [
  {
    title: "Ticket triage", sub: "route, urgency and two flags from one support ticket",
    req: {
      state: { ticket: "I've been charged twice this month and nobody answers my emails. Fix this today or I'm cancelling." },
      questions: {
        route: { type: "choice", criteria: {
          billing: "invoice, refund or charge problems",
          tech: "bug, outage, can't use the product",
          sales: "pricing, upgrade or new contract" } },
        urgency: { type: "score", criteria: [
          "low — can wait days",
          "medium — should be handled today",
          "critical — angry customer, churn risk"] },
        refund: { type: "noul", instructions: "is the customer asking for money back?" },
        human: { type: "noul", instructions: "does this need a human agent?" },
      },
    },
  },
  {
    title: "On-call triage", sub: "four alerts against one severity rubric, then which fire first",
    req: {
      state: {
        context: "Friday 18:40. A payments deploy went out at 18:05. One engineer on call.",
        severity_rubric: "sev1 = users blocked or losing money now; sev2 = degraded, will get worse; sev3 = noise, no user impact",
        alerts: {
          a1: "p95 latency on /checkout up from 300ms to 2.1s since the deploy, still rising",
          a2: "disk on logs-02 at 87%, projected full in ~36h",
          a3: "payment success rate down from 99.1% to 96.8% in the last 20 minutes",
          a4: "nightly analytics export ran 2h late, no downstream impact",
        },
      },
      questions: {
        sev_a1: { type: "score", instructions: "severity of alert a1", criteria: ["sev3", "sev2", "sev1"] },
        sev_a2: { type: "score", instructions: "severity of alert a2", criteria: ["sev3", "sev2", "sev1"] },
        sev_a3: { type: "score", instructions: "severity of alert a3", criteria: ["sev3", "sev2", "sev1"] },
        sev_a4: { type: "score", instructions: "severity of alert a4", criteria: ["sev3", "sev2", "sev1"] },
        first: { type: "choice", instructions: "which alert should the on-call handle first?", criteria: {
          a1: "checkout p95 latency",
          a2: "logs-02 disk filling",
          a3: "payment success rate dropping",
          a4: "late analytics export" } },
      },
    },
  },
  {
    title: "Policy check", sub: "four probes against a written refund policy, one may abstain",
    req: {
      state: {
        policy: "Refunds within 30 days of delivery, unused items in original packaging. " +
          "Final-sale items (gift cards, perishables, personalized goods) are never refundable. " +
          "Opened electronics are refundable only if defective. Shipping fees are never returned. " +
          "Anything the policy doesn't cover needs a store manager.",
        request: "Customer wants a refund for a wireless keyboard delivered 12 days ago — " +
          "keys stick after a coffee spill — and wants the €6 shipping fee back too.",
      },
      questions: {
        in_window: { type: "noul", instructions: "was the item delivered less than 30 days ago?" },
        damage_covered: { type: "noul", instructions: "does the written policy cover damage the customer caused?" },
        shipping_back: { type: "noul", instructions: "should the shipping fee be returned?" },
        needs_manager: { type: "noul", instructions: "does this request need a store manager?" },
        courier: { type: "choice", instructions: "which courier will handle the return shipment?", criteria: {
          ups: "UPS", fedex: "FedEx", dhl: "DHL" }, allow_abstain: true },
      },
    },
  },
  {
    title: "Review routing", sub: "sentiment, staff praise and owning team for one mixed review",
    req: {
      state: { review: "Waited 40 minutes and the pizza arrived cold. The driver was lovely though — even had a treat for my dog." },
      questions: {
        sentiment: { type: "score", instructions: "overall sentiment of the review", criteria: [
          "very negative", "negative", "mixed", "positive", "very positive"] },
        praises_driver: { type: "noul", instructions: "does the reviewer compliment the delivery driver?" },
        route: { type: "choice", instructions: "which team should own this feedback?", criteria: {
          logistics: "delivery times, drivers, couriers",
          kitchen: "food quality, temperature, preparation",
          support: "refunds, complaints, service recovery" } },
      },
    },
  },
  {
    title: "CV screen", sub: "seniority, lead experience and best-fit role from one résumé",
    req: {
      state: { resume: "8 years of Python (Django, FastAPI), PostgreSQL, Kafka. Led a team of 5 at a fintech. AWS. No mobile experience." },
      questions: {
        led_team: { type: "noul", instructions: "has the candidate led a team?" },
        best_role: { type: "choice", instructions: "which role fits the candidate best?", criteria: {
          backend: "server-side services, APIs, data pipelines",
          frontend: "web UI, client-side applications",
          mobile: "iOS and Android apps" } },
        seniority: { type: "score", instructions: "seniority level of the candidate", criteria: [
          "junior", "mid-level", "senior"] },
      },
    },
  },
  {
    title: "Effort estimate", sub: "a numeric answer in minutes, plus a risk flag",
    req: {
      state: { task: "export a 40k-row CSV without timing out the worker" },
      questions: {
        effort_min: { type: "numeric", min: 0, max: 480, step: 30, granularity: 16,
          instructions: "estimate the engineering effort, in minutes" },
        risky: { type: "noul", instructions: "is this likely to blow the estimate?" },
      },
    },
  },
];

/* ---------------- document model ---------------- */
// q = { name, type, instructions: string|null, criteria, min, max,
//       step: number|null, granularity, abstain: bool|undefined, open }
const doc = { state: "", questions: [], mode: "shared", temperature: 1 };
let activeTab = "builder";
let jsonDirty = false;
let running = false;
let last = null; // { at, ms }

function specToQ(name, s) {
  const type = s.type || "noul";
  return {
    name, type,
    instructions: s.instructions ?? null,
    criteria: type === "choice" ? Object.entries(s.criteria || {})
      : type === "score" ? [...(s.criteria || [])] : [],
    min: s.min ?? 0, max: s.max ?? 10, step: s.step ?? null,
    granularity: s.granularity ?? 8,
    abstain: s.allow_abstain,
    open: true,
  };
}

function loadRequest(req) {
  doc.state = typeof req.state === "string" ? req.state : JSON.stringify(req.state ?? "", null, 2);
  doc.questions = Object.entries(req.questions || {}).map(([n, s]) => specToQ(n, s));
  doc.mode = req.mode || "shared";
  doc.temperature = req.temperature ?? 1;
  $("state").value = doc.state;
  tempIn.value = doc.temperature;
  optsSum();
  paintState();
  renderTree();
  syncPanes();
}

function parseState(raw) {
  const t = raw.trim();
  if (!t) return "";
  try { return JSON.parse(t); } catch { return raw; }
}

function buildRequest() {
  const questions = {};
  for (const q of doc.questions) {
    const spec = { type: q.type };
    if (q.instructions != null && q.instructions !== "") spec.instructions = q.instructions;
    if (q.type === "choice") {
      spec.criteria = Object.fromEntries(q.criteria.filter(([k]) => k.trim()).map(([k, v]) => [k.trim(), v]));
    } else if (q.type === "score") {
      spec.criteria = q.criteria.filter((s) => s.trim());
    } else if (q.type === "numeric") {
      spec.min = q.min; spec.max = q.max;
      if (q.step != null) spec.step = q.step;
      spec.granularity = q.granularity;
    }
    if (q.abstain != null) spec.allow_abstain = q.abstain;
    questions[q.name || "q"] = spec;
  }
  return { state: parseState(doc.state), questions, temperature: doc.temperature, mode: doc.mode };
}

function setType(q, type) {
  if (type === q.type) return;
  const wasList = q.type === "choice" || q.type === "score";
  if (type === "choice") {
    q.criteria = q.type === "score" ? q.criteria.map((v, j) => [`opt${j + 1}`, v])
      : wasList ? q.criteria : [["yes", ""], ["no", ""]];
  } else if (type === "score") {
    q.criteria = q.type === "choice" ? q.criteria.map(([k, v]) => v || k)
      : wasList ? q.criteria : ["low", "medium", "high"];
  } else {
    q.criteria = [];
  }
  q.type = type;
}

function uniqueName(base = "q") {
  let n = doc.questions.length + 1;
  while (doc.questions.some((q) => q.name === `${base}${n}`)) n++;
  return `${base}${n}`;
}

function addQuestion(type) {
  const q = specToQ(uniqueName(), { type: "noul", instructions: "" });
  setType(q, type);
  if (type === "choice") q.criteria = [["opt1", ""], ["opt2", ""]];
  doc.questions.push(q);
  renderTree(); syncPanes();
  focusEdit(`[data-i="${doc.questions.length - 1}"][data-edit="instructions"]`);
}

/* ---------------- code fields ---------------- */
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

// JSON tokens -> spans; tolerant of half-typed input, it only colours
const TOK = /("(?:\\.|[^"\\\n])*")(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\btrue\b|\bfalse\b|\bnull\b)|([{}[\],:])/g;
function hl(src) {
  let out = "", at = 0;
  for (const m of src.matchAll(TOK)) {
    out += esc(src.slice(at, m.index));
    out += m[1] ? (m[2] ? `<span class="k">${esc(m[1])}</span><span class="p">${m[2]}</span>` : `<span class="s">${esc(m[1])}</span>`)
      : m[3] ? `<span class="n">${m[3]}</span>` : `<span class="p">${m[4]}</span>`;
    at = m.index + m[0].length;
  }
  return out + esc(src.slice(at));
}
// the trailing newline keeps the <pre> as tall as the textarea's last line
const paint = (ta, pre, json = true) => {
  pre.innerHTML = (json ? hl(ta.value) : esc(ta.value)) + "\n";
};
const follow = (ta, pre) => ta.addEventListener("scroll", () => {
  pre.scrollTop = ta.scrollTop; pre.scrollLeft = ta.scrollLeft;
});

// empty state: say what goes here, then show the shape
const STATE_PH = `<span class="ph">What every question is about: a ticket, an email, a log line, a record.
Plain text or JSON, read once and shared by all the questions below.
<span class="ph-ex">${hl('{"ticket": "Charged twice this month and nobody answers my emails."}')}</span></span>`;

// live count of what the state costs, with the resident model's own
// tokenizer on the state as the engine renders it (pretty-printing is free)
let tokTimer = 0, tokAbort = null;
function countState() {
  const el = $("state-tok");
  clearTimeout(tokTimer);
  tokAbort?.abort();
  if (!doc.state.trim()) { el.hidden = true; return; }
  el.classList.add("stale");
  tokTimer = setTimeout(async () => {
    tokAbort = new AbortController();
    try {
      const r = await fetch("/playground/tokenize", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ state: parseState(doc.state) }),
        signal: tokAbort.signal,
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const { tokens } = await r.json();
      el.innerHTML = `<b>${tokens.toLocaleString("en")}</b> ${tokens === 1 ? "token" : "tokens"}`;
      el.hidden = false;
      el.classList.remove("stale");
    } catch (e) {
      if (e.name !== "AbortError") el.hidden = true;
    }
  }, 250);
}

const stateEl = $("state"), stateHl = $("state-hl");
function paintState() {
  const v = stateEl.value;
  countState();
  if (!v) stateHl.innerHTML = STATE_PH;
  else paint(stateEl, stateHl, /^\s*[[{]/.test(v));
  let ok = true;
  try { JSON.parse(v); } catch { ok = false; }
  $("fmt").disabled = !ok;
}
follow(stateEl, stateHl);
stateEl.addEventListener("input", () => { doc.state = stateEl.value; paintState(); syncPanes(); });
stateEl.addEventListener("keydown", (e) => {
  if (e.key === "Tab" && !e.shiftKey) {
    e.preventDefault();
    document.execCommand("insertText", false, "  ");
  }
});
$("fmt").onclick = () => {
  try {
    stateEl.value = doc.state = JSON.stringify(JSON.parse(stateEl.value), null, 2);
    paintState(); syncPanes();
  } catch { /* button is disabled for plain text */ }
};

/* ---------------- question tree ---------------- */
const key = (text, attrs = "") =>
  `<span class="p">"</span><span class="k" contenteditable="plaintext-only" ${attrs}>${esc(text)}</span><span class="p">"</span><span class="p">:</span>`;
// the tooltip text also rides along, visually hidden, for screen readers
const fixedKey = (text, i) => `<span class="k fixed" data-tip="${esc(text)}" data-i="${i}">"${esc(text)}"</span><span class="p">:</span>` +
  `<span class="sr">${esc(TIPS[text]?.(doc.questions[i])[0] ?? "")}</span>`;
const str = (val, attrs, ph = "") =>
  `<span class="p">"</span><span class="s" contenteditable="plaintext-only" data-ph="${esc(ph)}" ${attrs}>${esc(val)}</span><span class="p">"</span>`;
const num = (val, attrs) =>
  `<span class="n" contenteditable="plaintext-only" inputmode="decimal" ${attrs}>${esc(val ?? "")}</span>`;
const enumv = (val, attrs) =>
  `<button class="e" ${attrs} aria-haspopup="menu">"${esc(val)}"${ICON.caret}</button>`;
const boolv = (val, attrs) =>
  `<button class="b ${val ? "on" : ""}" ${attrs} role="switch" aria-checked="${!!val}">${val ? "true" : "false"}</button>`;
const del = (attrs, label) =>
  `<button class="del" ${attrs} aria-label="${esc(label)}" title="${esc(label)}">${ICON.x}</button>`;
const add = (attrs, label) =>
  `<button class="add" ${attrs}>${ICON.plus}<span>${esc(label)}</span></button>`;
const line = (d, inner, cls = "") => `<div class="ln ${cls}" style="--d:${d}">${inner}</div>`;

// field tooltips: what each key means, read on hover of the key. Built at
// show time from the live question, so the numeric notes are never stale.
// numeric scale: `step` wins over `granularity` (schema.rs anchors(), 2–24 points)
function points(q) {
  const raw = q.step > 0 && q.max > q.min ? Math.round((q.max - q.min) / q.step) + 1 : null;
  return raw && `${Math.min(24, Math.max(2, raw))} points${raw > 24 ? ", capped at 24" : ""}`;
}
const TIPS = {
  type: () => ["The kind of answer: noul is yes or no, choice picks one key, score picks a level on a rubric, numeric estimates a value in a range."],
  instructions: () => ["The question, in words. Leave it out when the name already asks it."],
  criteria: (q) => [q.type === "score"
    ? "The rubric, lowest level first. The answer is one of these levels."
    : "The options. Each key is what comes back; its text tells the model when to pick it."],
  min: () => ["The lowest value on the scale."],
  max: () => ["The highest value on the scale."],
  step: (q) => ["The spacing between points on the scale. When set, it replaces granularity.", points(q)],
  granularity: (q) => ["How many points the scale is split into, evenly spaced from min to max: 2 to 24.",
    q.step != null ? "Ignored: step is set" : `${q.granularity} points`],
  allow_abstain: () => ["Adds an abstain option, so the model can decline instead of guessing."],
};

function optionalFields(q) {
  const f = [];
  if (q.instructions == null) f.push(["instructions", "the question, in words"]);
  if (q.abstain == null) f.push(["allow_abstain", "let the model decline instead of guessing"]);
  if (q.type === "numeric" && q.step == null) f.push(["step", "spacing between points, instead of granularity"]);
  return f;
}

function questionHTML(q, i) {
  const a = (extra) => `data-i="${i}" ${extra}`;
  const head = line(1,
    `<button class="fold ${q.open ? "open" : ""}" ${a('data-act="fold"')} aria-label="${q.open ? "Collapse" : "Expand"} ${esc(q.name)}" aria-expanded="${q.open}">${ICON.chev}</button>` +
    key(q.name, a('data-edit="qname" spellcheck="false"')) + ` <span class="p">{</span>` +
    (q.open ? "" : `<span class="fold-sum"><b>${esc(q.type)}</b>${q.instructions ? ` · ${esc(q.instructions)}` : ""}</span><span class="p">}</span>`) +
    del(a('data-act="delq"'), `Remove ${q.name}`), "qhead");
  if (!q.open) return `<div class="q">${head}</div>`;

  let body = line(2, fixedKey("type", i) + " " + enumv(q.type, a('data-act="pick-type"')));
  if (q.instructions != null)
    body += line(2, fixedKey("instructions", i) + " " +
      str(q.instructions, a('data-edit="instructions"'), PH[q.type] ? `e.g. ${PH[q.type]}` : "") +
      del(a('data-act="delfield" data-f="instructions"'), "Remove instructions"));

  if (q.type === "choice") {
    body += line(2, fixedKey("criteria", i) + ` <span class="p">{</span>`);
    q.criteria.forEach(([k, v], j) => {
      body += line(3, key(k, a(`data-j="${j}" data-edit="ck" spellcheck="false"`)) + " " +
        str(v, a(`data-j="${j}" data-edit="cv"`), "when to pick this key") +
        del(a(`data-j="${j}" data-act="delcrit"`), `Remove ${k}`));
    });
    body += line(3, add(a('data-act="addcrit"'), "option"));
    body += line(2, `<span class="p">}</span>`);
  } else if (q.type === "score") {
    body += line(2, fixedKey("criteria", i) + ` <span class="p">[</span>`);
    q.criteria.forEach((v, j) => {
      body += line(3, `<span class="idx">${j}</span>` +
        str(v, a(`data-j="${j}" data-edit="cv"`), j ? "a higher level" : "the lowest level") +
        del(a(`data-j="${j}" data-act="delcrit"`), `Remove level ${j}`));
    });
    body += line(3, add(a('data-act="addcrit"'), "level"));
    body += line(2, `<span class="p">]</span>`);
  } else if (q.type === "numeric") {
    body += line(2, fixedKey("min", i) + " " + num(q.min, a('data-edit="min"')));
    body += line(2, fixedKey("max", i) + " " + num(q.max, a('data-edit="max"')));
    if (q.step != null)
      body += line(2, fixedKey("step", i) + " " + num(q.step, a('data-edit="step"')) +
        del(a('data-act="delfield" data-f="step"'), "Remove step"));
    body += line(2, fixedKey("granularity", i) + " " + num(q.granularity, a('data-edit="granularity"')),
      q.step != null ? "off" : "");
  }

  if (q.abstain != null)
    body += line(2, fixedKey("allow_abstain", i) + " " + boolv(q.abstain, a('data-act="toggle-abstain"')) +
      del(a('data-act="delfield" data-f="abstain"'), "Remove allow_abstain"));
  if (optionalFields(q).length)
    body += line(2, add(a('data-act="addfield"'), "field"));
  body += line(1, `<span class="p">}</span>`);
  return `<div class="q open">${head}${body}</div>`;
}

const primRow = (p) => `
  <button class="prim" data-act="newq" data-t="${p.t}">
    <b>${p.t}</b><span>${esc(p.desc)}</span><i>e.g. “${esc(p.ex)}”</i>
  </button>`;

function renderTree() {
  const n = doc.questions.length;
  $("qcount").textContent = n;
  $("tree").innerHTML = line(0, `<span class="p">{</span>`) + (n
    ? doc.questions.map(questionHTML).join("")
    : `<div class="picker" style="--d:1">
         <p>Pick a primitive to add your first question</p>
         ${PRIMS.map(primRow).join("")}
         <p class="ext">snap extension</p>
         ${EXT.map(primRow).join("")}
       </div>`) + line(0, `<span class="p">}</span>`);
}

function focusEdit(sel) {
  requestAnimationFrame(() => {
    const el = $("tree").querySelector(sel);
    if (!el) return;
    el.focus();
    el.scrollIntoView({ block: "nearest" });
    const r = document.createRange();
    r.selectNodeContents(el);
    const s = getSelection(); s.removeAllRanges(); s.addRange(r);
  });
}

const tree = $("tree");

tree.addEventListener("input", (e) => {
  const el = e.target;
  const { i, j, edit } = el.dataset;
  if (!edit) return;
  const v = el.textContent;
  const q = doc.questions[+i];
  if (edit === "qname") q.name = v;
  else if (edit === "instructions") q.instructions = v;
  else if (edit === "ck") q.criteria[+j][0] = v;
  else if (edit === "cv") q.type === "choice" ? (q.criteria[+j][1] = v) : (q.criteria[+j] = v);
  else if (["min", "max", "step", "granularity"].includes(edit)) q[edit] = v.trim() === "" ? null : +v;
  el.classList.toggle("bad", el.classList.contains("n") && v.trim() !== "" && !Number.isFinite(+v));
  if (edit === "step") tree.querySelector(`[data-i="${i}"][data-edit="granularity"]`)?.closest(".ln").classList.toggle("off", q.step != null);
  syncPanes();
});

tree.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && e.target.isContentEditable && !(e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    e.target.blur();
  }
});

tree.addEventListener("focusout", (e) => {
  if (e.target.dataset?.edit !== "qname") return;
  const q = doc.questions[+e.target.dataset.i];
  if (q && !q.name.trim()) { q.name = uniqueName(); renderTree(); syncPanes(); }
});

tree.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const { i, j, act, f, t } = btn.dataset;
  const q = doc.questions[+i];
  switch (act) {
    case "newq": return addQuestion(t);
    case "fold": q.open = !q.open; break;
    case "delq": doc.questions.splice(+i, 1); break;
    case "delcrit": q.criteria.splice(+j, 1); break;
    case "delfield": f === "abstain" ? (q.abstain = undefined) : (q[f] = null); break;
    case "toggle-abstain": q.abstain = !q.abstain; break;
    case "addcrit": {
      const n = q.criteria.length;
      q.criteria.push(q.type === "choice" ? [`opt${n + 1}`, ""] : "");
      renderTree(); syncPanes();
      return focusEdit(`[data-i="${i}"][data-j="${n}"][data-edit="${q.type === "choice" ? "ck" : "cv"}"]`);
    }
    case "pick-type":
      return openPop(btn, typeItems(q.type, (tp) => { setType(q, tp); renderTree(); syncPanes(); }));
    case "addfield":
      return openPop(btn, optionalFields(q).map(([name, hint]) => ({
        label: name, hint, code: true,
        on: () => {
          if (name === "instructions") q.instructions = "";
          if (name === "allow_abstain") q.abstain = true;
          if (name === "step") q.step = 1;
          renderTree(); syncPanes();
          if (name !== "allow_abstain") focusEdit(`[data-i="${i}"][data-edit="${name}"]`);
        },
      })));
    default: return;
  }
  renderTree();
  syncPanes();
});

/* ---------------- field tooltips ---------------- */
// Linear-style: a short delay the first time, then instant while the pointer
// moves from key to key; gone on scroll, click or keypress
const tip = $("tip");
let tipTimer = 0, tipWarm = 0, hot = null; // hot: the numeric bar under the pointer
function showTip(k) {
  const q = doc.questions[+k.dataset.i];
  const [text, note] = q && TIPS[k.dataset.tip] ? TIPS[k.dataset.tip](q) : [];
  if (text) placeTip(k, `<b>${esc(k.dataset.tip)}</b><p>${esc(text)}</p>${note ? `<span>${esc(note)}</span>` : ""}`);
}
function placeTip(k, html) {
  tip.innerHTML = html;
  const r = k.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
  const up = r.top - h - 8 > 8;
  tip.style.left = `${Math.max(8, Math.min(r.left - 4, innerWidth - w - 8))}px`;
  tip.style.top = `${up ? r.top - h - 8 : r.bottom + 8}px`;
  tip.style.setProperty("--to", up ? "bottom left" : "top left");
  tip.classList.add("on");
}
function hideTip() {
  clearTimeout(tipTimer);
  if (tip.classList.contains("on")) tipWarm = performance.now();
  tip.classList.remove("on");
  hot?.classList.remove("hot");
  hot = null;
}
tree.addEventListener("mouseover", (e) => {
  const k = e.target.closest(".k.fixed[data-tip]");
  if (!k) return;
  clearTimeout(tipTimer);
  tipTimer = setTimeout(() => showTip(k), performance.now() - tipWarm < 400 ? 0 : 450);
});
tree.addEventListener("mouseout", (e) => { if (e.target.closest(".k.fixed[data-tip]")) hideTip(); });
["mousedown", "keydown", "scroll"].forEach((ev) => addEventListener(ev, hideTip, true));

// numeric bars read out in the same tooltip, instantly: the whole column is
// the hit target, so a 1% sliver is as easy to point at as the peak
$("answers").addEventListener("pointermove", (e) => {
  const h = e.target.closest(".hist"), r = h?.getBoundingClientRect();
  const b = h?.children[Math.min(h.children.length - 1, Math.floor((e.clientX - r.left) / r.width * h.children.length))];
  if (b === hot) return;
  hideTip();
  if (!b?.dataset.k) return;
  hot = b;
  b.classList.add("hot");
  placeTip(b, `<b>${esc(b.dataset.k)}</b><p>${pct(+b.dataset.p)} of the probability</p>`);
});
$("answers").addEventListener("pointerleave", hideTip);

function typeItems(current, pick) {
  const item = (p) => ({ label: p.t, hint: p.desc, ex: p.ex, code: true, current: p.t === current || (current === "boolean" && p.t === "noul"), on: () => pick(p.t) });
  return [...PRIMS.map(item), { sep: "snap extension" }, ...EXT.map(item)];
}

$("addq").onclick = (e) => openPop(e.currentTarget, typeItems(null, addQuestion));

/* ---------------- run settings ---------------- */
// mode and temperature sit behind one quiet key next to Run: the key says
// what they are now, the popover says what they mean
const tempRow = $("temp-row"), tempIn = $("temp");
tempRow.remove();
const optsSum = () => { $("opts-sum").textContent = `${doc.mode} · temp ${doc.temperature}`; };
optsSum();
$("opts").onclick = (e) => openPop(e.currentTarget, [
  { sep: "mode" },
  ...MODES.map(([m, hint]) => ({
    label: m, hint, code: true, current: m === doc.mode, keep: true,
    on: () => { doc.mode = m; optsSum(); syncPanes(); },
  })),
  { sep: "temperature" },
], tempRow);
tempIn.addEventListener("input", () => {
  const v = parseFloat(tempIn.value);
  const ok = Number.isFinite(v) && v > 0 && v <= 10;
  tempIn.classList.toggle("bad", !ok);
  if (ok) { doc.temperature = v; optsSum(); syncPanes(); }
});
tempIn.addEventListener("blur", () => {
  tempIn.value = doc.temperature;
  tempIn.classList.remove("bad");
});

// the platform's own run chord
const MOD = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent) ? "⌘ ↵" : "Ctrl ↵";
document.querySelectorAll("kbd.mod").forEach((k) => { k.textContent = MOD; });

/* ---------------- popover ---------------- */
const pop = $("pop");
let popAnchor = null;

// `extra` is a live node (a field, not a choice) appended under the items;
// `keep` items apply in place and leave the popover open
function openPop(anchor, items, extra) {
  if (popAnchor === anchor) return closePop();
  popAnchor = anchor;
  pop.setAttribute("role", extra ? "dialog" : "menu");
  pop.innerHTML = items.map((it, k) => it.sep
    ? `<div class="sep">${esc(it.sep)}</div>`
    : `<button role="menuitem" data-k="${k}" class="${it.current ? "cur" : ""} ${it.ex ? "rich" : ""}">
        <b${it.code ? ' class="code"' : ""}>${esc(it.label)}</b>${it.hint ? `<span>${esc(it.hint)}</span>` : ""}${it.ex ? `<i>e.g. “${esc(it.ex)}”</i>` : ""}${it.current ? ICON.check : ""}
      </button>`).join("");
  if (extra) pop.append(extra);
  pop.hidden = false;
  const r = anchor.getBoundingClientRect();
  const pw = pop.offsetWidth, ph = pop.offsetHeight;
  const left = Math.max(12, Math.min(r.left, innerWidth - pw - 12));
  const up = r.bottom + 6 + ph > innerHeight && r.top - ph - 6 > 0;
  pop.style.left = `${left}px`;
  pop.style.top = `${Math.max(12, up ? r.top - ph - 6 : r.bottom + 6)}px`;
  pop.style.setProperty("--o", `${up ? "bottom" : "top"} ${r.left + r.width / 2 - left}px`);
  pop.onclick = (e) => {
    const b = e.target.closest("[data-k]");
    if (!b) return;
    const it = items[+b.dataset.k];
    if (it.keep) {
      pop.querySelector(".cur")?.classList.remove("cur");
      b.classList.add("cur");
      const ck = pop.querySelector(".ck");
      if (ck) b.append(ck);
    } else closePop(false);
    it.on();
  };
  (pop.querySelector(".cur") || pop.querySelector("button"))?.focus();
}

function closePop(refocus = true) {
  if (pop.hidden) return;
  pop.hidden = true;
  const a = popAnchor;
  popAnchor = null;
  if (refocus && a?.isConnected) a.focus();
}

document.addEventListener("mousedown", (e) => {
  if (!pop.hidden && !pop.contains(e.target) && !e.target.closest("[aria-haspopup]")) closePop(false);
});
pop.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const btns = [...pop.querySelectorAll("button")];
  const k = btns.indexOf(document.activeElement);
  if (e.key === "ArrowDown") { e.preventDefault(); btns[(k + 1) % btns.length].focus(); }
  if (e.key === "ArrowUp") { e.preventDefault(); btns[(k - 1 + btns.length) % btns.length].focus(); }
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePop(); });
addEventListener("resize", () => closePop(false));
document.querySelectorAll(".pane-body, .tree").forEach((p) => p.addEventListener("scroll", () => closePop(false)));

/* ---------------- examples ---------------- */
function useExample(ex) {
  loadRequest(ex.req);
  switchTab("builder");
  run();
}

// one row per example: title, what it asks, and what comes back
const exRow = (k, title, sub, aside) => `
  <button class="ex" data-k="${k}"><b>${esc(title)}</b><span class="cs">${esc(sub)}</span><span class="qn">${aside}</span>${ICON.arrow}</button>`;
$("lessons").innerHTML = LESSONS.map((l, k) => exRow(k, l.title, l.sub, `<code>${l.tag}</code>`)).join("");
$("cases").innerHTML = CASES.map((c, k) => exRow(k, c.title, c.sub, `${Object.keys(c.req.questions).length} questions`)).join("");
$("lessons").onclick = (e) => { const b = e.target.closest("[data-k]"); if (b) useExample(LESSONS[+b.dataset.k]); };
$("cases").onclick = (e) => { const b = e.target.closest("[data-k]"); if (b) useExample(CASES[+b.dataset.k]); };

$("examples").onclick = (e) => openPop(e.currentTarget, [
  { sep: "the three primitives" },
  ...LESSONS.map((l) => ({ label: l.title, hint: l.tag, on: () => useExample(l) })),
  { sep: "use cases" },
  ...CASES.map((c) => ({ label: c.title, hint: "", on: () => useExample(c) })),
]);

/* ---------------- tabs ---------------- */
// segmented controls: one plate slides under the selected label
function thumb(seg) {
  const b = seg.querySelector("button.on");
  if (!b) return;
  seg.style.setProperty("--x", `${b.offsetLeft}px`);
  seg.style.setProperty("--w", `${b.offsetWidth}px`);
}
function select(seg, b) {
  seg.querySelectorAll("button").forEach((x) => {
    x.classList.toggle("on", x === b);
    x.setAttribute("aria-selected", x === b);
  });
  thumb(seg);
}
const SEGS = [$("tabs"), $("rtabs")];
SEGS.forEach(thumb);
document.fonts?.ready.then(() => {
  SEGS.forEach(thumb);
  requestAnimationFrame(() => SEGS.forEach((s) => s.classList.add("ready")));
});
addEventListener("resize", () => SEGS.forEach(thumb));

function switchTab(t) {
  if (activeTab === "json" && jsonDirty && t !== "json") {
    try { jsonDirty = false; loadRequest(JSON.parse($("json").value)); }
    catch (err) {
      jsonDirty = true;
      $("json-note").textContent = `Invalid JSON, not synced: ${err.message}`;
      $("json-note").classList.add("warn");
      return;
    }
  }
  activeTab = t;
  select($("tabs"), $("tabs").querySelector(`[data-t="${t}"]`));
  $("pane-builder").hidden = t !== "builder";
  $("pane-json").hidden = t !== "json";
  $("pane-curl").hidden = t !== "curl";
  syncPanes();
}

$("tabs").addEventListener("click", (e) => { if (e.target.dataset.t) switchTab(e.target.dataset.t); });
const jsonEl = $("json"), jsonHl = $("json-hl");
follow(jsonEl, jsonHl);
jsonEl.addEventListener("input", () => {
  jsonDirty = true;
  paint(jsonEl, jsonHl);
  $("json-note").classList.remove("warn");
  $("json-note").innerHTML = "Body for <code>POST /v1/systemone</code>. Edits sync back to the editor when you switch tabs.";
});

let curlText = "";
$("copy-curl").onclick = async (e) => {
  const b = e.currentTarget, label = b.querySelector("span");
  try { await navigator.clipboard.writeText(curlText); label.textContent = "Copied"; b.classList.add("ok"); }
  catch {
    const r = document.createRange(); r.selectNodeContents($("curl"));
    const s = getSelection(); s.removeAllRanges(); s.addRange(r);
    label.textContent = `Press ${MOD.startsWith("⌘") ? "⌘" : "Ctrl+"}C`;
  }
  setTimeout(() => { label.textContent = "Copy"; b.classList.remove("ok"); }, 1600);
};

function syncPanes() {
  const req = buildRequest();
  if (activeTab === "json" && !jsonDirty) { jsonEl.value = JSON.stringify(req, null, 2); paint(jsonEl, jsonHl); }
  const body = JSON.stringify(req).replace(/'/g, "'\\''");
  const url = `http://${location.host}/v1/systemone`;
  curlText = `curl -s ${url} \\\n  -H 'content-type: application/json' \\\n  -d '${body}'`;
  $("curl").innerHTML = `<span class="k">curl</span> <span class="p">-s</span> ${esc(url)} <span class="p">\\</span>\n` +
    `  <span class="p">-H</span> <span class="s">'content-type: application/json'</span> <span class="p">\\</span>\n` +
    `  <span class="p">-d '</span>${hl(body)}<span class="p">'</span>`;
  $("run").disabled = running || (activeTab !== "json" && !doc.questions.length);
}

$("rtabs").addEventListener("click", (e) => {
  const r = e.target.dataset.r;
  if (!r) return;
  select($("rtabs"), e.target);
  $("result").hidden = r !== "result";
  $("raw").hidden = r !== "raw";
});

/* ---------------- run ---------------- */
// byte-identical request to the one on screen: dim it and tween from it.
// anything else — a different state, options, instructions — is a fresh run.
let lastReqJson = null;
const sameShape = (req) => !!lastDist && !!$("answers").querySelector(".ans:not(.pending)") &&
  JSON.stringify(req) === lastReqJson;

async function run() {
  if (running) return;
  let req;
  if (activeTab === "json") {
    try { req = JSON.parse(jsonEl.value); }
    catch (e) { return showErr("That JSON doesn't parse", e.message); }
  } else {
    if (!doc.questions.length) return;
    req = buildRequest();
  }
  running = true;
  $("run").classList.add("busy"); $("run").disabled = true; $("run-label").textContent = "Running";
  document.body.classList.add("running");

  const fresh = !sameShape(req);
  if (fresh) $("answers").innerHTML = Object.entries(req.questions || {}).map(([n, s]) => ansHTML(n, s)).join("");
  $("empty").hidden = true;
  $("dial").hidden = false;
  if (fresh) $("facts").innerHTML = "";
  $("clock-l").textContent = "in flight";
  const clock = $("clock");
  const t0 = performance.now();
  let raf = 0;
  const tick = () => { clock.textContent = Math.round(performance.now() - t0); raf = requestAnimationFrame(tick); };
  tick();

  try {
    const res = await fetch("/v1/systemone", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    const ms = performance.now() - t0;
    const body = await res.json().catch(() => ({ error: res.statusText || `HTTP ${res.status}` }));
    cancelAnimationFrame(raf);
    setHttp(res.status, res.ok);
    $("raw").innerHTML =
      `<span class="h">→ POST /v1/systemone</span>\n${hl(JSON.stringify(req, null, 2))}\n\n` +
      `<span class="h">← <b class="${res.ok ? "" : "bad"}">${res.status}</b> · ${ms.toFixed(0)} ms</span>\n${hl(JSON.stringify(body, null, 2))}`;
    if (!res.ok) showErr(`The server rejected the request (${res.status})`, body.error || JSON.stringify(body));
    else renderResult(req, body, ms, fresh);
  } catch (e) {
    cancelAnimationFrame(raf);
    setHttp(null);
    showErr("Couldn't reach snap", `${e.message}. Is \`snap serve\` still running?`);
  } finally {
    running = false;
    $("run").classList.remove("busy"); $("run-label").textContent = "Run";
    document.body.classList.remove("running");
    syncPanes();
  }
}

$("run").onclick = run;
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); run(); }
});

function setHttp(status, ok) {
  const h = $("http");
  h.hidden = status == null;
  h.textContent = ok ? `${status} OK` : String(status);
  h.classList.toggle("bad", !ok);
}

function showErr(title, detail) {
  $("empty").hidden = true;
  $("dial").hidden = true;
  lastDist = null;
  $("answers").innerHTML = `<div class="err"><b>${esc(title)}</b><span>${esc(detail)}</span></div>`;
}

/* ---------------- result ---------------- */
const pct = (p) => `${Math.round(p * 100)}%`;
const num2 = (x) => String(+(+x).toFixed(2)); // 8.571428571428571 → 8.57
let lastDist = null; // name -> {key: p} — feeds the run-to-run tween and the ± deltas

const SPECIAL = { __abstain__: "abstain", __below__: "below", __above__: "above" };
const STATUS = { abstained: "abstained", contested: "contested", out_of_bounds: "out of bounds" };
// what each status means (decisions.rs), for the badge's tooltip
const STATUS_WHY = {
  contested: "The top two answers are less than 15 points apart: treat this one as uncertain.",
  abstained: "No option fit, so the model declined instead of guessing (allow_abstain).",
  out_of_bounds: "Most of the probability fell outside the min–max range.",
};

// the status said in this answer's own numbers, first thing in the detail line
function statusNote(a, rows, type) {
  const lab = (k) => SPECIAL[k] || (type === "numeric" ? num2(k) : k);
  const [p1, p2] = [...rows].sort((x, y) => y[1] - x[1]);
  if (a.status === "contested" && p2) return `too close to call: ${lab(p1[0])} ${pct(p1[1])} vs ${lab(p2[0])} ${pct(p2[1])}`;
  if (a.status === "abstained") return "the model declined to answer";
  if (a.status === "out_of_bounds") return `${pct(p1[1])} of the probability is ${p1[0] === "__below__" ? "below" : "above"} the range`;
  return "";
}
const MAX_ROWS = 12;

function renderResult(req, body, ms, fresh) {
  last = { at: Date.now(), ms };
  renderRan();

  const x = body.x_snap || {};
  const f = (label, v, unit, title, cls = "") =>
    `<span class="${cls}" title="${esc(title)}">${esc(label)}<b>${esc(v ?? "—")}</b>${unit ? `<small>${unit}</small>` : ""}</span>`;
  const seen = (x.cache_hits ?? 0) + (x.cache_misses ?? 0);
  $("facts").innerHTML =
    f("decode", x.decode_ms, "ms", "time inside batched decode calls") +
    f("decoded", body.usage?.input_tokens, "tok", "tokens actually decoded") +
    f("prompt", x.prompt_tokens, "tok", "prompt tokens across all items") +
    (x.shared_prefix_tokens > 0 ? f("shared", x.shared_prefix_tokens, "tok", "prefix shared by every item") : "") +
    f("cached", x.cached_head_tokens, "tok", "resident template head tokens") +
    (seen ? f("cache", `${x.cache_hits}/${seen}`, "", "items that forked off a span cached by an earlier request") : "") +
    f("items", x.decoded_items, "", "decode items") +
    f("waves", x.waves, "", "batched decode waves") +
    f("generated", body.usage?.output_tokens ?? 0, "tok", "nothing is generated: every answer is read off letter logits", "zero");
  // the clock lands on the engine's own time in the frame the bars rise
  $("clock").textContent = x.total_ms == null ? Math.round(ms) : x.total_ms < 100 ? x.total_ms.toFixed(1) : Math.round(x.total_ms);
  $("clock-l").textContent = x.total_ms == null ? "round-trip" : "engine time";

  const prev = fresh ? {} : lastDist || {};
  const dist = {};
  $("answers").innerHTML = Object.entries(req.questions || {}).map(([name, spec]) => {
    const a = body.answers?.[name];
    if (!a) return "";
    const html = ansHTML(name, spec, a, prev[name]);
    dist[name] = Object.fromEntries(answerView(a, spec).rows.map(([k, p]) => [k, p]));
    return html;
  }).join("");
  lastDist = dist;
  lastReqJson = JSON.stringify(req);
  tweenDist();
  fitAxes();
}

// one answer article; without `a` it is the pending shape of the question
function ansHTML(name, spec, a, prev) {
  const type = a?.type || (spec.type === "boolean" ? "noul" : spec.type || "noul");
  const status = a && STATUS[a.status];
  const v = a ? answerView(a, spec) : { rows: pendingRows(spec, type) };
  // the answer is a chip in the winning bar's colour, heat when the status qualifies
  // it, with how sure right beside it; where on the scale reads under the bars
  const verdict = a
    ? `<span class="conf" title="How far the top answer stands above an even split: 100% is certain, 0% a coin toss.">confidence<b>${pct(a.confidence ?? 0)}</b></span><b title="answer">${esc(v.headline)}</b>`
    : `<b class="sk"></b>`;
  const note = a ? statusNote(a, v.rows, type) : "";
  const detail = [note && `<span class="why-st">${esc(note)}</span>`, v.sub && esc(v.sub)].filter(Boolean).join(" · ");
  return `
    <article class="ans${a ? "" : " pending"}${status ? " warn" : ""}">
      <header>
        <h3>${esc(name)}${status ? `<span class="tag" title="${esc(STATUS_WHY[a.status])}">${status}</span>` : ""}</h3>
        <div class="verdict">${verdict}</div>
        ${spec.instructions ? `<p>${esc(spec.instructions)}</p>` : ""}
      </header>
      ${type === "numeric" ? histHTML(v.rows, prev, !a, v.mark) : distHTML(v.rows, prev, !a)}
      ${detail ? `<p class="detail">${detail}</p>` : ""}
    </article>`;
}

function pendingRows(spec, type) {
  const crit = spec.criteria || {};
  const keys = type === "noul" ? ["yes", "no"]
    : type === "choice" ? Object.keys(crit)
    : type === "score" ? [...crit].map(String)
    : Array.from({ length: Math.min(64, spec.granularity ?? 8) }, (_, j) => `bin${j}`);
  if (spec.allow_abstain) keys.push("__abstain__");
  return keys.map((k) => [k, 0, false, type === "choice" ? crit[k] : undefined]);
}

function distHTML(rows, prev, pending) {
  let shown = rows, rest = [];
  if (rows.length > MAX_ROWS) {
    const keep = new Set([...rows].sort((p, q) => q[1] - p[1]).slice(0, MAX_ROWS).map((r) => r[0]));
    shown = rows.filter((r) => keep.has(r[0]));
    rest = rows.filter((r) => !keep.has(r[0]));
  }
  const more = !rest.length ? ""
    : `<p class="more">${rest.length} more ${rest.length === 1 ? "option" : "options"}${pending ? "" : `, ${pct(rest.reduce((s, r) => s + r[1], 0))} combined`}</p>`;
  return `<div class="dist">${shown.map(([k, p, win, desc]) => {
    const lbl = SPECIAL[k] || k;
    const old = prev?.[k] ?? 0;
    const d = prev?.[k] == null ? 0 : Math.round((p - old) * 100);
    const fill = pending ? "<i></i>" : `<i data-f="${old}" data-t="${p.toFixed(4)}" style="transform:scaleX(${old})"></i>`;
    const val = pending ? "—"
      : `<b class="v" data-f="${old}" data-t="${p.toFixed(4)}">${pct(old)}</b>${d ? `<i class="d">${d > 0 ? "+" : "−"}${Math.abs(d)}</i>` : ""}`;
    return `
      <div class="bar${win ? " win" : ""}">
        <span class="lbl" title="${esc(desc ? `${k} — ${desc}` : k)}">${esc(lbl)}${desc && desc !== k ? `<em>${esc(desc)}</em>` : ""}</span>
        <span class="track" role="meter" aria-label="${esc(lbl)}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(p * 100)}">${fill}</span>
        <span class="val">${val}</span>
      </div>`;
  }).join("")}${more}</div>`;
}

// numeric bins as bars scaled to the peak, the out-of-range slots apart at
// the edges; the tween starts from the last run's shape at its own peak
function histHTML(rows, prev, pending, mark) {
  const peak = Math.max(...rows.map((r) => r[1])) || 1;
  const oldPeak = prev ? Math.max(...Object.values(prev)) || 1 : 1;
  const lbl = (k) => SPECIAL[k] || num2(k);
  const bar = ([k, p, win]) => {
    if (pending) return `<i style="transform:scaleY(.04)"></i>`;
    const old = (prev?.[k] ?? 0) / oldPeak;
    return `<i${win ? ' class="w"' : ""} data-k="${esc(lbl(k))}" data-p="${p}" data-ax="y" data-f="${old}" data-t="${(p / peak).toFixed(4)}" style="transform:scaleY(${old})"></i>`;
  };
  const edge = (k) => {
    const r = rows.find((x) => x[0] === k);
    return r ? `<div class="x"><div class="hist">${bar(r)}</div><span>${SPECIAL[k]}</span></div>` : "";
  };
  // JS lists integer-like keys ("0", "10") first: put the bins back in value order
  const inRange = rows.filter((r) => !SPECIAL[r[0]]).sort((p, q) => parseFloat(p[0]) - parseFloat(q[0]));
  const n = inRange.length;
  // every bar gets its value under its own tick; fitAxes thins them to what fits
  const axis = pending || !n ? ""
    : `<div class="hist-axis" style="grid-template-columns:repeat(${n},minmax(0,1fr))">${
      inRange.map(([k, , win]) => `<span${win ? ' class="w"' : ""}><b>${num2(k)}</b></span>`).join("")}</div>`;
  const said = [...rows.filter((r) => r[0] === "__below__"), ...inRange, ...rows.filter((r) => r[0] === "__above__" || r[0] === "__abstain__")]
    .map(([k, p]) => `${lbl(k)} ${pct(p)}`).join(", ");
  // the answer is the probability-weighted mean: mark where it falls on the scale
  const at = mark && n > 1 && mark.max > mark.min
    ? Math.min(100, Math.max(0, (.5 + (mark.value - mark.min) / (mark.max - mark.min) * (n - 1)) / n * 100)) : null;
  const m = pending || at == null ? "" : `<i class="mark" style="left:${at.toFixed(2)}%"><span>mean</span></i>`;
  return `<div class="histw" role="img" aria-label="${pending ? "pending" : esc(said)}">` +
    `${edge("__below__")}<div class="in${m ? " marked" : ""}">${m}<div class="hist">${inRange.map(bar).join("")}</div>${axis}</div>${edge("__above__")}${edge("__abstain__")}</div>`;
}

// label as many bars as the width allows: the stride grows until the widest
// value fits, and each label keeps the tick that ties it to its own bar
function fitAxes() {
  for (const ax of $("answers").querySelectorAll(".hist-axis")) {
    const cells = [...ax.children], col = ax.clientWidth / cells.length;
    if (!col) continue;
    const k = Math.ceil((Math.max(...cells.map((c) => c.firstChild.offsetWidth)) + 8) / col);
    cells.forEach((c, i) => c.classList.toggle("off", i % k > 0));
  }
}
new ResizeObserver(fitAxes).observe($("answers"));

// one eased pass drives every fill and its readout together — ink and digits
// move in lock-step from the previous run's values to the new ones
function tweenDist() {
  const els = $("answers").querySelectorAll("[data-t]");
  const apply = (e) => els.forEach((el) => {
    const f = +el.dataset.f, t = +el.dataset.t;
    const v = f + (t - f) * e;
    if (el.classList.contains("v")) el.textContent = pct(v);
    else el.style.transform = el.dataset.ax === "y" ? `scaleY(${v})` : `scaleX(${v})`;
  });
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return apply(1);
  const D = 560, t0 = performance.now();
  (function step(now) {
    const k = Math.min(1, (now - t0) / D);
    apply(1 - (1 - k) ** 4);
    if (k < 1) requestAnimationFrame(step);
  })(t0);
}

function answerView(a, spec) {
  const probs = Object.entries(a.probabilities || {});
  const top = probs.reduce((m, [k, p]) => (p > m[1] ? [k, p] : m), ["", -1])[0];
  if (a.type === "noul") {
    const p = a.noul ?? 0;
    return { headline: a.boolean ? "Yes" : "No",
      rows: [["yes", p, a.boolean], ["no", 1 - p, !a.boolean]] };
  }
  if (a.type === "choice") {
    const crit = spec.criteria || {};
    return { headline: SPECIAL[a.choice] || (a.choice ?? "—"), rows: probs.map(([k, p]) => [k, p, k === a.choice, crit[k]]) };
  }
  if (a.type === "score") {
    return { headline: top || "—", sub: `level ${(a.level ?? 0) + 1} of ${probs.length} · score ${(a.score ?? 0).toFixed(2)}`,
      rows: probs.map(([k, p]) => [k, p, k === top]) };
  }
  // out of bounds: the answer is the edge the engine picked, not the mean of
  // what little mass stayed in range — that reads as a detail instead
  const rows = probs.map(([k, p]) => [k, p, k === top]);
  const edge = { __below__: `Below ${num2(spec.min ?? 0)}`, __above__: `Above ${num2(spec.max ?? 10)}`, __abstain__: "abstain" }[top];
  if (edge) return { headline: edge, sub: a.value == null ? "" : `in-range mean ${num2(a.value)}`, rows };
  if (a.value == null) return { headline: "—", rows };
  // the headline is the mean; the tallest bar is only the single most likely point
  return { headline: num2(a.value), rows,
    sub: `mean of the distribution · most likely ${num2(top)} at ${pct(a.probabilities[top])}`,
    mark: { value: +a.value, min: +(spec.min ?? 0), max: +(spec.max ?? 10) } };
}

function renderRan() {
  if (!last) return;
  const s = Math.round((Date.now() - last.at) / 1000);
  const ago = s < 5 ? "just now" : s < 60 ? `${s}s ago` : `${Math.round(s / 60)}m ago`;
  $("ran").textContent = `${ago} · ${Math.round(last.ms)} ms round-trip`;
}
setInterval(renderRan, 5000);

/* ---------------- model picker + health ---------------- */
let currentName = null; // tested-model spec the engine is running
let wantModel = null;   // spec a switch is in flight for

$("srv").onclick = async (e) => {
  const anchor = e.currentTarget;
  let items = [];
  try {
    const j = await (await fetch("/v1/models")).json();
    // one row per HF repo — quant variants stay resolvable by name,
    // the picker shows each model at its default quantization only
    const seen = new Set();
    const rows = (j.data || []).filter((m) => !seen.has(m.repo) && seen.add(m.repo));
    const activeRepo = (j.data || []).find((m) => m.active)?.repo;
    items = rows.map((m) => ({
      label: m.id, code: true,
      hint: (m.file || "").replace(/\.gguf$/i, ""),
      current: m.repo === activeRepo || (m.active ?? m.id === currentName),
      on: () => switchModel(m.id),
    }));
  } catch { /* fall through to the single-item menu */ }
  if (!items.length)
    items = [{ label: currentName || $("model").textContent, hint: "", code: true, current: true, on: () => {} }];
  openPop(anchor, items);
};

async function switchModel(name) {
  if (wantModel || name === currentName) return;
  wantModel = name;
  const srv = $("srv");
  srv.classList.add("loading");
  $("model").textContent = `loading ${name}…`;
  try {
    const r = await fetch("/v1/models", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: name }),
    });
    if (!r.ok) throw new Error((await r.json()).error || `HTTP ${r.status}`);
    const j = await r.json();
    wantModel = null;
    currentName = j.name || name;
    srv.classList.remove("loading", "down", "wait");
    $("model").textContent = currentName;
    srv.title = j.model || currentName;
    countState(); // new model, new tokenizer
    return;
  } catch (e) {
    wantModel = null;
    srv.classList.remove("loading");
    srv.classList.add("down");
    $("model").textContent = "switch failed";
    srv.title = String(e.message || e);
  }
  // the health poll confirms the new resident model
}

async function health() {
  try {
    const j = await (await fetch("/healthz")).json();
    if (currentName && j.name && j.name !== currentName) countState(); // new model, new tokenizer
    currentName = j.name || null;
    if (wantModel && j.name !== wantModel) return; // still loading
    wantModel = null;
    $("srv").classList.remove("down", "wait", "loading");
    $("model").textContent = j.name || j.model;
    $("srv").title = j.model;
  } catch {
    if (wantModel) return;
    $("model").textContent = "server unreachable";
    $("srv").classList.remove("wait");
    $("srv").classList.add("down");
    $("srv").title = "GET /healthz is not answering";
  }
}

/* ---------------- init ---------------- */
// the console opens blank: start from an example or write the first question
loadRequest({ state: "", questions: {} });
health();
setInterval(health, 5000);
if (location.hash === "#run") useExample(CASES[0]);
