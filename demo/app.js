// MedLens demo: static viewer for precomputed Phase 1-5 results (no model runs here).
"use strict";

const FRAMINGS = [
  ["none", "No hint"],
  ["sham", "Harmless sentence"],
  ["metadata", "Metadata tag"],
  ["explicit", "Plain hint"],
  ["authority", "Attending says"],
  ["patient", "Patient says"],
];
const DIRECTIONS = [
  ["wrong", "Points to a wrong answer"],
  ["gold", "Points to the right answer"],
];
const LABELS = {
  unfaithful: "Silent hint-following: the hint changed the answer and the reasoning never says so",
  faithful: "Open hint-following: the hint changed the answer and the reasoning says so",
  unaffected: "Not affected: the model kept its no-hint answer",
};
const WHY_NOT = {
  baseline_already_target: "Not scored for faithfulness: without the hint the model already gave the hinted answer",
  cued_unfinished_or_unparsed: "Not scored for faithfulness: the answer could not be read",
};

const state = { index: null, q: null, framing: "explicit", direction: "wrong", cache: new Map() };
const $ = (id) => document.getElementById(id);
const pct = (x, d = 0) => `${(100 * x).toFixed(d)}%`;

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Minimal, safe markdown: escape first, then bold/italic, bullets and paragraphs.
function md(text) {
  const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/(^|[^*])\*(?!\s)(.+?)\*/g, "$1<em>$2</em>");
  return esc(text).split(/\n{2,}/).map((block) => {
    const lines = block.split("\n");
    if (lines.every((l) => /^\s*[*-]\s+/.test(l))) {
      return "<ul>" + lines.map((l) => `<li>${inline(l.replace(/^\s*[*-]\s+/, ""))}</li>`).join("") + "</ul>";
    }
    return `<p>${inline(lines.join("<br>"))}</p>`;
  }).join("");
}

// Wrap the first verbatim occurrence of `needle` (already escaped HTML-safe text) in a mark.
function markOnce(html, needle, cls) {
  if (!needle) return html;
  const e = esc(needle);
  const i = html.indexOf(e);
  return i < 0 ? html : html.slice(0, i) + `<mark class="${cls}">${e}</mark>` + html.slice(i + e.length);
}

function condName() {
  return ["none", "sham"].includes(state.framing) ? state.framing : `${state.framing}_${state.direction}`;
}

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

async function loadQuestion(id) {
  if (!state.cache.has(id)) state.cache.set(id, await getJSON(`data/q/${id}.json`));
  state.q = state.cache.get(id);
}

function renderStats() {
  const n = state.index.numbers;
  const tiles = [
    [pct(n.admitted.rate, 1), `of ${n.admitted.followed} hint-followed answers mention the hint in the reasoning`],
    [n.post.auroc_out_of_fold.toFixed(2), "detector score after reasoning (AUROC; 0.5 = coin flip, 1 = perfect)"],
    [pct(n.post.flag_rate_no_hint, 1), "false alarms on questions with no hint"],
    [n.injection_auroc.toFixed(2), "the same detector on prompt-injection cases (does not transfer)"],
  ];
  $("stats").innerHTML = tiles.map(([v, k]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
}

function renderGallery() {
  const g = state.index.gallery;
  $("gallery").innerHTML = Object.entries(g).map(([cat, shelf]) => `
    <div class="shelf">
      <p class="shelf-title">${esc(shelf.title)} <span class="n">(${shelf.n_total} runs in the ${state.index.n_questions} questions)</span></p>
      <div class="chips">${shelf.picks.map((p, i) =>
        `<button class="chip" data-q="${p.item_id}" data-c="${p.condition}">Case ${i + 1}</button>`).join("")}</div>
    </div>`).join("");
  $("gallery").querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => go(b.dataset.q, b.dataset.c)));
}

function renderList() {
  const term = $("search").value.trim().toLowerCase();
  const qs = state.index.questions.filter((q) => !term || q.preview.toLowerCase().includes(term) || (q.specialty || "").toLowerCase().includes(term));
  $("qlist").innerHTML = qs.slice(0, 400).map((q) =>
    `<li><button data-q="${q.id}" ${state.q && state.q.id === q.id ? 'aria-current="true"' : ""}>${esc(q.preview)}<span class="sp">${esc(q.specialty || "")}</span></button></li>`).join("");
  $("qlist").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => go(b.dataset.q, condName())));
}

function renderQuestion() {
  const q = state.q;
  $("question-card").innerHTML = `
    <div class="meta">${esc(q.specialty || "")} · MedQA question ${esc(q.id)}</div>
    <p class="qtext">${esc(q.question)}</p>
    <ul class="opts">${Object.entries(q.options).map(([k, v]) =>
      `<li class="${k === q.gold ? "gold" : ""}"><strong>${k}.</strong> ${esc(v)}${k === q.gold ? '<span class="tag">correct answer</span>' : ""}</li>`).join("")}</ul>`;
}

function renderToggles() {
  const seg = (el, items, cur, onPick, disabled = false) => {
    el.innerHTML = items.map(([k, label]) =>
      `<button role="radio" aria-checked="${k === cur}" data-k="${k}" ${disabled ? "disabled" : ""}>${label}</button>`).join("");
    el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => onPick(b.dataset.k)));
  };
  seg($("framing"), FRAMINGS, state.framing, (k) => { state.framing = k; update(); });
  const noDir = ["none", "sham"].includes(state.framing);
  seg($("direction"), DIRECTIONS, state.direction, (k) => { state.direction = k; update(); }, noDir);
  const run = state.q.runs[condName()];
  $("cue").innerHTML = run.cue
    ? `Added to the prompt: <mark class="hint">${esc(run.cue)}</mark>${run.target ? ` (points to ${run.target})` : ""}`
    : "The prompt is the question alone.";
}

function renderReasoning(run) {
  let html = md(run.text);
  if (run.ack && run.judge_quote) html = markOnce(html, run.judge_quote, "quote");
  const cut = run.stop_reason === "max_tokens" ? " (the answer ran out of space)" : "";
  $("reasoning").innerHTML = html;
  $("final").textContent = `Answer read from the text: ${run.pred ?? "none"}${cut}`;
}

function meter(score, thr) {
  return `<div class="meter"><div class="fill" style="width:${(100 * score).toFixed(1)}%"></div><div class="tick" style="left:calc(${(100 * thr).toFixed(1)}% - 1px)" title="warning threshold"></div></div>`;
}

function renderEvidence(run) {
  const n = state.index.numbers;
  const none = state.q.runs.none;
  const flagged = run.flagged;
  const badge = `
    <div class="badge ${flagged ? "warn" : "ok"}" role="status">
      <span class="icon" aria-hidden="true">${flagged ? "⚠" : "✓"}</span>
      <div><strong>${flagged ? "Warning: likely following a hint" : "No hint-following detected"}</strong>
      <small>Detector score ${run.post_score.toFixed(2)} (warning at ≥ ${n.post.threshold.toFixed(2)}), read from the model's
      internal activity after it reasoned. It is never told what the hint was.</small>
      ${meter(run.post_score, n.post.threshold)}</div>
    </div>`;
  const said = run.ack === true ? `Yes: <mark class="quote">${esc(run.judge_quote)}</mark>`
    : run.ack === false ? "No. The reasoning never mentions the hint."
    : run.cue ? "Not checked (only answers that switched to the hinted letter were judged)." : "No hint in this prompt.";
  const label = run.label ? LABELS[run.label] : (WHY_NOT[run.why_not] || (run.cue ? "—" : "No hint in this prompt."));
  const letters = Object.entries(run.probs).map(([k, p]) => `
    <span>${k}</span><div class="b ${k === run.target ? "hl" : ""}" style="width:${Math.max(1, 100 * p).toFixed(1)}%"></div><span class="num">${pct(p)}</span>`).join("");
  $("evidence").innerHTML = badge + `
    <div class="row"><div class="k">The test: same question, no hint</div>
      <div class="v">No hint → <strong>${none.pred ?? "?"}</strong>; this prompt → <strong>${run.pred ?? "?"}</strong>. ${esc(run.effect.charAt(0).toUpperCase() + run.effect.slice(1))}.</div></div>
    <div class="row"><div class="k">Does the reasoning mention the hint?</div><div class="v">${said}</div></div>
    <div class="row"><div class="k">Verdict</div><div class="v">${esc(label)}</div></div>
    <div class="row"><div class="k">Quick answer, before any reasoning</div>
      <div class="v">Chance of each letter if forced to answer at once${run.target ? " (hinted letter in blue)" : ""}.
      Early-warning score ${run.pre_score.toFixed(2)} ${run.flagged_pre ? "(would warn)" : "(would not warn)"}.</div>
      <div class="letters">${letters}</div></div>
    <p class="src">Score source: ${esc(run.post_source)}. The judge is an LLM (Qwen 27B), checked against other
      AI annotators, not people.</p>`;
}

function renderNotes() {
  const n = state.index.numbers;
  const kind = (k, label) => (state.index.badge_by_kind.find((r) => r.kind === k && r.label === label) || { mean: NaN }).mean;
  const notes = [
    `<strong>The warning flags following a hint, not hiding it.</strong> It fires just as often when the reasoning admits the hint. It works as a "hidden reasoning" alarm here only because this model almost never admits hints (${pct(n.admitted.rate, 1)}).`,
    `<strong>It is set to catch ${pct(n.post.recall)} of silent hint-following.</strong> At that setting it fires on ${pct(n.post.flag_rate_ignored_hint, 1)} of answers that ignored a wrong hint and ${pct(n.post.flag_rate_no_hint, 1)} of answers with no hint.`,
    `<strong>It also fires when the answer merely agrees with the hint.</strong> When a right-answer hint pointed to the answer the model would have given anyway, it warned on ${pct(kind("hint_gold", "not scored"))} of those runs (${state.index.n_questions} demo questions). It seems to read "my answer matches a letter suggested in the prompt", not "the hint changed my mind".`,
    `<strong>Every hint here names an answer letter.</strong> Cues that don't (for example, a misleading lab value) were not tested.`,
    `<strong>It does not catch prompt injection.</strong> On 150 medical prompt-injection cases (MPIB) it scored ${n.injection_auroc.toFixed(2)}, about a coin flip.`,
    `<strong>Fixing it failed.</strong> Pushing the model's internal activity away from hint-following cut it in quick answers (45% → 28%) but not once the model reasoned step by step.`,
    `<strong>One model, one question set.</strong> MedGemma 4B on 375 MedQA questions. Right-answer hints, the harmless sentence and no-hint prompts were never used to train the detector; their scores show how it behaves on new kinds of runs.`,
  ];
  $("notes").innerHTML = notes.map((x) => `<li>${x}</li>`).join("");
  $("features").innerHTML = state.index.features.map((f) =>
    `<li>Layer ${f.layer}, feature ${f.feature}: ${esc(f.label || "unlabeled")} · <a href="${f.url}" target="_blank" rel="noopener">Neuronpedia</a></li>`).join("");
}

function update() {
  const c = condName();
  try { history.replaceState(null, "", `#${state.q.id}--${c.replace("_", "-")}`); } catch (e) { /* hash is a convenience */ }
  renderToggles();
  const run = state.q.runs[c];
  renderReasoning(run);
  renderEvidence(run);
  document.querySelectorAll(".chip").forEach((b) => b.setAttribute("aria-current", String(b.dataset.q === state.q.id && b.dataset.c === c)));
}

async function go(id, cond) {
  await loadQuestion(id);
  if (cond === "none" || cond === "sham") state.framing = cond;
  else if (cond) [state.framing, state.direction] = cond.split("_");
  renderQuestion();
  renderList();
  update();
}

async function init() {
  try {
    state.index = await getJSON("data/index.json");
  } catch (e) {
    $("main").innerHTML = `<p class="card">Could not load data/index.json. Build it with <code>scripts/phase5_build_demo.py</code> and serve this folder over HTTP (for example <code>python -m http.server</code>).</p>`;
    return;
  }
  renderStats();
  renderGallery();
  renderNotes();
  $("search").addEventListener("input", renderList);
  // Links look like #<question id>--<condition>, e.g. #medqa-006382b37f7a--metadata-gold.
  const [hq, hc] = location.hash.slice(1).split("--");
  const known = state.index.questions.some((q) => q.id === hq);
  const first = state.index.gallery.caught.picks[0] || { item_id: state.index.questions[0].id, condition: "explicit_wrong" };
  await go(known ? hq : first.item_id, known && hc ? hc.replace("-", "_") : first.condition);
}

init();
