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
};

/* ---------------- examples ---------------- */
const LESSONS = [
  {
    tag: "noul", title: "Is a tomato a fruit?", sub: "Botany against the kitchen",
    req: {
      state: { item: "tomato" },
      questions: {
        is_fruit: { type: "noul", instructions: "botanically, is this item a fruit?" },
        cooked_as_veg: { type: "noul", instructions: "in the kitchen, is it treated as a vegetable?" },
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
          a4: "staging login broken since this morning, QA blocked",
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
          a4: "staging login broken" } },
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
        in_window: { type: "noul", instructions: "is the request inside the return window?" },
        item_refundable: { type: "noul", instructions: "is the keyboard refundable under the policy?" },
        shipping_back: { type: "noul", instructions: "should the shipping fee be returned?" },
        needs_manager: { type: "noul", instructions: "does this request need a store manager?" },
        under_warranty: { type: "noul", instructions: "is the keyboard still under manufacturer warranty?", allow_abstain: true },
      },
    },
  },
  {
    title: "Review routing", sub: "sentiment, sarcasm and owning team for one tricky review",
    req: {
      state: { review: "Loved waiting 40 minutes for a cold pizza. To be fair the driver was lovely — even had a treat for my dog." },
      questions: {
        sentiment: { type: "score", instructions: "overall sentiment of the review", criteria: [
          "very negative", "negative", "mixed", "positive", "very positive"] },
        sarcasm: { type: "noul", instructions: "is the opening sentence meant sarcastically?" },
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
  $("temp").value = doc.temperature;
  $("mode").innerHTML = `${doc.mode}${ICON.caret}`;
  fitState();
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

/* ---------------- state box ---------------- */
const stateEl = $("state");
function fitState() {
  stateEl.style.height = "auto";
  stateEl.style.height = `${stateEl.scrollHeight + 2}px`;
  let ok = true;
  try { JSON.parse(stateEl.value); } catch { ok = false; }
  $("fmt").disabled = !ok;
}
stateEl.addEventListener("input", () => { doc.state = stateEl.value; fitState(); syncPanes(); });
stateEl.addEventListener("keydown", (e) => {
  if (e.key === "Tab" && !e.shiftKey) {
    e.preventDefault();
    document.execCommand("insertText", false, "  ");
  }
});
$("fmt").onclick = () => {
  try {
    stateEl.value = doc.state = JSON.stringify(JSON.parse(stateEl.value), null, 2);
    fitState(); syncPanes();
  } catch { /* button is disabled for plain text */ }
};

/* ---------------- question tree ---------------- */
const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");

const key = (text, attrs = "") =>
  `<span class="p">"</span><span class="k" contenteditable="plaintext-only" ${attrs}>${esc(text)}</span><span class="p">"</span><span class="p">:</span>`;
const fixedKey = (text) => `<span class="k fixed">"${esc(text)}"</span><span class="p">:</span>`;
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

function optionalFields(q) {
  const f = [];
  if (q.instructions == null) f.push(["instructions", "the question, in words"]);
  if (q.abstain == null) f.push(["allow_abstain", "let the model decline to answer"]);
  if (q.type === "numeric" && q.step == null) f.push(["step", "snap values to a grid"]);
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

  let body = line(2, fixedKey("type") + " " + enumv(q.type, a('data-act="pick-type"')));
  if (q.instructions != null)
    body += line(2, fixedKey("instructions") + " " +
      str(q.instructions, a('data-edit="instructions"'), PH[q.type] || "") +
      del(a('data-act="delfield" data-f="instructions"'), "Remove instructions"));

  if (q.type === "choice") {
    body += line(2, fixedKey("criteria") + ` <span class="p">{</span>`);
    q.criteria.forEach(([k, v], j) => {
      body += line(3, key(k, a(`data-j="${j}" data-edit="ck" spellcheck="false"`)) + " " +
        str(v, a(`data-j="${j}" data-edit="cv"`), "when to pick this key") +
        del(a(`data-j="${j}" data-act="delcrit"`), `Remove ${k}`));
    });
    body += line(3, add(a('data-act="addcrit"'), "option"));
    body += line(2, `<span class="p">}</span>`);
  } else if (q.type === "score") {
    body += line(2, fixedKey("criteria") + ` <span class="p">[</span>`);
    q.criteria.forEach((v, j) => {
      body += line(3, `<span class="idx">${j}</span>` +
        str(v, a(`data-j="${j}" data-edit="cv"`), j ? "a higher level" : "the lowest level") +
        del(a(`data-j="${j}" data-act="delcrit"`), `Remove level ${j}`));
    });
    body += line(3, add(a('data-act="addcrit"'), "level"));
    body += line(2, `<span class="p">]</span>`);
  } else if (q.type === "numeric") {
    body += line(2, fixedKey("min") + " " + num(q.min, a('data-edit="min"')));
    body += line(2, fixedKey("max") + " " + num(q.max, a('data-edit="max"')));
    if (q.step != null)
      body += line(2, fixedKey("step") + " " + num(q.step, a('data-edit="step"')) +
        del(a('data-act="delfield" data-f="step"'), "Remove step"));
    body += line(2, fixedKey("granularity") + " " + num(q.granularity, a('data-edit="granularity"')));
  }

  if (q.abstain != null)
    body += line(2, fixedKey("allow_abstain") + " " + boolv(q.abstain, a('data-act="toggle-abstain"')) +
      del(a('data-act="delfield" data-f="abstain"'), "Remove allow_abstain"));
  if (optionalFields(q).length)
    body += line(2, add(a('data-act="addfield"'), "field"));
  body += line(1, `<span class="p">}</span>`);
  return `<div class="q">${head}${body}</div>`;
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
        label: name, hint,
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

function typeItems(current, pick) {
  const item = (p) => ({ label: p.t, hint: p.desc, ex: p.ex, current: p.t === current || (current === "boolean" && p.t === "noul"), on: () => pick(p.t) });
  return [...PRIMS.map(item), { sep: "snap extension" }, ...EXT.map(item)];
}

$("addq").onclick = (e) => openPop(e.currentTarget, typeItems(null, addQuestion));

/* ---------------- footer knobs ---------------- */
$("mode").onclick = (e) => openPop(e.currentTarget, MODES.map(([m, hint]) => ({
  label: m, hint, current: m === doc.mode,
  on: () => { doc.mode = m; $("mode").innerHTML = `${m}${ICON.caret}`; syncPanes(); },
})));
const tempIn = $("temp");
tempIn.addEventListener("input", () => {
  const v = parseFloat(tempIn.value);
  const ok = Number.isFinite(v) && v > 0 && v <= 10;
  tempIn.classList.toggle("bad", !ok);
  if (ok) { doc.temperature = v; syncPanes(); }
});
tempIn.addEventListener("blur", () => {
  tempIn.value = doc.temperature;
  tempIn.classList.remove("bad");
});

/* ---------------- popover ---------------- */
const pop = $("pop");
let popAnchor = null;

function openPop(anchor, items) {
  if (popAnchor === anchor) return closePop();
  popAnchor = anchor;
  pop.innerHTML = items.map((it, k) => it.sep
    ? `<div class="sep">${esc(it.sep)}</div>`
    : `<button role="menuitem" data-k="${k}" class="${it.current ? "cur" : ""} ${it.ex ? "rich" : ""}">
        <b>${esc(it.label)}</b>${it.hint ? `<span>${esc(it.hint)}</span>` : ""}${it.ex ? `<i>e.g. “${esc(it.ex)}”</i>` : ""}
      </button>`).join("");
  pop.hidden = false;
  const r = anchor.getBoundingClientRect();
  const pw = pop.offsetWidth, ph = pop.offsetHeight;
  const left = Math.min(r.left, innerWidth - pw - 12);
  const top = r.bottom + 6 + ph > innerHeight ? r.top - ph - 6 : r.bottom + 6;
  pop.style.left = `${Math.max(12, left)}px`;
  pop.style.top = `${Math.max(12, top)}px`;
  pop.onclick = (e) => {
    const b = e.target.closest("[data-k]");
    if (!b) return;
    const it = items[+b.dataset.k];
    closePop(false);
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
  const btns = [...pop.querySelectorAll("button")];
  const k = btns.indexOf(document.activeElement);
  if (e.key === "ArrowDown") { e.preventDefault(); btns[(k + 1) % btns.length].focus(); }
  if (e.key === "ArrowUp") { e.preventDefault(); btns[(k - 1 + btns.length) % btns.length].focus(); }
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closePop(); });
addEventListener("resize", () => closePop(false));
document.querySelectorAll(".pane-body").forEach((p) => p.addEventListener("scroll", () => closePop(false)));

/* ---------------- examples ---------------- */
function useExample(ex) {
  loadRequest(ex.req);
  switchTab("builder");
  run();
}

$("lessons").innerHTML = LESSONS.map((l, k) => `
  <button class="lesson" data-k="${k}">
    <span class="tag">${l.tag}</span>
    <b>${esc(l.title)}</b><span>${esc(l.sub)}</span>
  </button>`).join("");
$("cases").innerHTML = CASES.map((c, k) => `
  <button class="case" data-k="${k}"><b>${esc(c.title)}</b><span>${esc(c.sub)}</span></button>`).join("");
$("lessons").onclick = (e) => { const b = e.target.closest("[data-k]"); if (b) useExample(LESSONS[+b.dataset.k]); };
$("cases").onclick = (e) => { const b = e.target.closest("[data-k]"); if (b) useExample(CASES[+b.dataset.k]); };

$("examples").onclick = (e) => openPop(e.currentTarget, [
  ...LESSONS.map((l) => ({ label: l.title, hint: l.tag, on: () => useExample(l) })),
  { sep: "use cases" },
  ...CASES.map((c) => ({ label: c.title, hint: "", on: () => useExample(c) })),
]);

/* ---------------- tabs ---------------- */
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
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.t === t));
  $("pane-builder").hidden = t !== "builder";
  $("pane-json").hidden = t !== "json";
  $("pane-curl").hidden = t !== "curl";
  syncPanes();
}

$("tabs").addEventListener("click", (e) => { if (e.target.dataset.t) switchTab(e.target.dataset.t); });
$("json").addEventListener("input", () => {
  jsonDirty = true;
  $("json-note").classList.remove("warn");
  $("json-note").innerHTML = "Body for <code>POST /v1/systemone</code>. Edits sync back to the editor when you switch tabs.";
});
$("copy-curl").onclick = async (e) => {
  const b = e.currentTarget;
  try { await navigator.clipboard.writeText($("curl").value); b.textContent = "Copied"; }
  catch { $("curl").select(); b.textContent = "Press ⌘C"; }
  setTimeout(() => { b.textContent = "Copy"; }, 1600);
};

function syncPanes() {
  const req = buildRequest();
  if (activeTab === "json" && !jsonDirty) $("json").value = JSON.stringify(req, null, 2);
  $("curl").value =
    `curl -s http://${location.host}/v1/systemone \\\n  -H 'content-type: application/json' \\\n  -d '${JSON.stringify(req).replace(/'/g, "'\\''")}'`;
  $("run").disabled = running || (activeTab !== "json" && !doc.questions.length);
}

$("rtabs").addEventListener("click", (e) => {
  const r = e.target.dataset.r;
  if (!r) return;
  document.querySelectorAll("#rtabs button").forEach((b) => b.classList.toggle("on", b.dataset.r === r));
  $("result").hidden = r !== "result";
  $("raw").hidden = r !== "raw";
});

/* ---------------- run ---------------- */
async function run() {
  if (running) return;
  let req;
  if (activeTab === "json") {
    try { req = JSON.parse($("json").value); }
    catch (e) { return showErr("That JSON doesn't parse", e.message); }
  } else {
    if (!doc.questions.length) return;
    req = buildRequest();
  }
  running = true;
  $("run").classList.add("busy"); $("run").disabled = true; $("run-label").textContent = "Running";
  document.body.classList.add("running");
  const t0 = performance.now();
  try {
    const res = await fetch("/v1/systemone", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req),
    });
    const ms = performance.now() - t0;
    const body = await res.json();
    $("raw").textContent =
      `→ POST /v1/systemone\n${JSON.stringify(req, null, 2)}\n\n← ${res.status} · ${ms.toFixed(0)} ms\n${JSON.stringify(body, null, 2)}`;
    if (!res.ok) showErr(`The server rejected the request (${res.status})`, body.error || JSON.stringify(body));
    else renderResult(req, body, ms);
  } catch (e) {
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

function showErr(title, detail) {
  $("empty").hidden = true;
  $("stats").hidden = true;
  $("answers").innerHTML = `<div class="err"><b>${esc(title)}</b><span>${esc(detail)}</span></div>`;
}

/* ---------------- result ---------------- */
const pct = (p) => `${Math.round(p * 100)}%`;

function renderResult(req, body, ms) {
  $("empty").hidden = true;
  last = { at: Date.now(), ms };
  renderRan();

  const x = body.x_snap || {};
  const stat = (v, unit, label, cls = "") =>
    `<div class="stat ${cls}"><b>${esc(v ?? "—")}${unit ? `<small>${unit}</small>` : ""}</b><span>${label}</span></div>`;
  const qh = x.qhead_hits != null ? `${x.qhead_hits}/${(x.qhead_hits ?? 0) + (x.qhead_misses ?? 0)}` : null;
  $("stats").innerHTML =
    stat(x.total_ms, "ms", "total", "lead") +
    stat(x.prefill_ms, "ms", "prefill") +
    stat(body.usage?.input_tokens, "tok", "decoded") +
    (x.shared_prefix_tokens > 0 ? stat(x.shared_prefix_tokens, "tok", "shared prefix") : "") +
    stat(x.cached_head_tokens, "tok", "cached head") +
    (qh ? stat(qh, "", "qhead hits") : "") +
    stat(x.decoded_items, "", `items · ${x.suffix_decode ?? "—"}`) +
    stat(x.rewind, "", "rewind");
  $("stats").hidden = false;

  $("answers").innerHTML = Object.entries(req.questions || {}).map(([name, spec]) => {
    const a = body.answers?.[name];
    if (!a) return "";
    const v = answerView(a, spec);
    const abst = a.status === "abstained";
    return `
      <article class="ans ${abst ? "abst" : ""}">
        <header>
          <div class="who">
            <h3>${esc(name)}</h3>
            ${spec.instructions ? `<p>${esc(spec.instructions)}</p>` : ""}
          </div>
          <div class="verdict">
            <b>${esc(v.headline)}</b>
            <span>${abst ? "abstained · " : ""}${v.sub ? `${esc(v.sub)} · ` : ""}confidence ${pct(a.confidence ?? 0)}</span>
          </div>
          <span class="tag">${esc(a.type)}</span>
        </header>
        <div class="dist">
          ${v.rows.map(([k, p, win, desc]) => `
            <div class="bar ${win ? "win" : ""}">
              <span class="lbl" title="${esc(desc ? `${k} — ${desc}` : k)}">${esc(k)}${desc && desc !== k ? `<em>${esc(desc)}</em>` : ""}</span>
              <span class="track"><i style="--p:${p.toFixed(4)}"></i></span>
              <span class="val">${pct(p)}</span>
            </div>`).join("")}
        </div>
      </article>`;
  }).join("");
}

function answerView(a, spec) {
  const probs = Object.entries(a.probabilities || {});
  const top = probs.reduce((m, [k, p]) => (p > m[1] ? [k, p] : m), ["", -1])[0];
  if (a.type === "noul") {
    const p = a.noul ?? 0;
    return { headline: `${a.boolean ? "Yes" : "No"} · ${pct(a.boolean ? p : 1 - p)}`,
      rows: [["yes", p, a.boolean], ["no", 1 - p, !a.boolean]] };
  }
  if (a.type === "choice") {
    const crit = spec.criteria || {};
    return { headline: a.choice ?? "—", rows: probs.map(([k, p]) => [k, p, k === a.choice, crit[k]]) };
  }
  if (a.type === "score") {
    return { headline: `Level ${a.level} of ${probs.length - 1}`, sub: `score ${(a.score ?? 0).toFixed(2)}`,
      rows: probs.map(([k, p]) => [k, p, k === top]) };
  }
  return { headline: `${a.value ?? "—"}`, rows: probs.map(([k, p]) => [k, p, k === top]) };
}

function renderRan() {
  if (!last) return;
  const s = Math.round((Date.now() - last.at) / 1000);
  const ago = s < 5 ? "just now" : s < 60 ? `${s}s ago` : `${Math.round(s / 60)}m ago`;
  $("ran").textContent = `Ran ${ago} · ${Math.round(last.ms)} ms round-trip`;
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
    items = (j.data || []).map((m) => ({
      label: m.id,
      hint: (m.file || "").replace(/\.gguf$/i, ""),
      current: m.active ?? m.id === currentName,
      on: () => switchModel(m.id),
    }));
  } catch { /* fall through to the single-item menu */ }
  if (!items.length)
    items = [{ label: currentName || $("model").textContent, hint: "", current: true, on: () => {} }];
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
    $("srv").title = "POST /healthz is not answering";
  }
}

/* ---------------- init ---------------- */
loadRequest(CASES[0].req);
health();
setInterval(health, 5000);
