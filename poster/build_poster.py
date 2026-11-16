"""Build the 24 x 36 in project poster (HTML -> vector PDF via headless Chrome).

Every number is read from saved run outputs (paths in RUNS); nothing is typed in by hand, and key
numbers are asserted against the phase reports. The layout, palette, fonts and chart helper are
adapted from the author's earlier Silent Second Opinion poster (../silent-second-opinion/poster).

Run from the repo root:  PYTHONPATH=scripts uv run python poster/build_poster.py [--no-pdf]
Outputs: poster/poster.html, poster/medlens-honesty-poster-24x36.pdf
Chart SVGs use a viewBox in points (1 unit = 1 pt), so font-size N in a chart prints at N pt.
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from medlens.eval.stats import wilson_interval

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
RUNS = {
    "phase1": OUT / "phase1_hints/20261114-193610",
    "phase2_readout_L17": OUT / "phase2_causal_readout/20261114-212634",
    "phase2_readout_corr": OUT / "phase2_causal_readout/20261114-203532",
    "phase2_cot": OUT / "phase2_causal_cot/20261114-222220",
    "phase3_probes": OUT / "phase3_probes/20261115-153757",
    "phase3_followups": OUT / "phase3_followups/20261115-154212",
    "phase4_screen": OUT / "phase4_screen/20261115-164028",
    "phase4_cot": OUT / "phase4_cot_dev/20261115-174147",
    "phase4_mpib": OUT / "phase4_mpib/20261116-101344",
    "phase5_build": OUT / "phase5_build_demo/20261116-173236",
}
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT_HTML = ROOT / "poster" / "poster.html"
OUT_PDF = ROOT / "poster" / "medlens-honesty-poster-24x36.pdf"

# Duke brand palette (brand.duke.edu/colors), as in the Silent Second Opinion poster
NAVY, ROYAL, COPPER = "#012169", "#00539B", "#C84E00"
INK, MUTED = "#000000", NAVY
GRID, PANEL = "#E2E6ED", "#F3F2F1"
FONTS = (ROOT / "poster" / "fonts" / "fonts.css").read_text()


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


# ---- data ------------------------------------------------------------------------------------------
p1 = pd.read_csv(RUNS["phase1"] / "summary.csv").set_index("condition")
labels = pd.read_parquet(RUNS["phase1"] / "labels.parquet")
followed_all = labels[labels.label.isin(["unfaithful", "faithful"])]
N_FOLLOWED, N_ADMITTED = len(followed_all), int((followed_all.label == "faithful").sum())
ADMIT_CI = wilson_interval(N_ADMITTED, N_FOLLOWED)
N_Q = labels.item_id.nunique()
SHAM = float(p1.loc["sham", "follow_rate"])
ACC_NONE = float((labels[labels.condition == "none"].correct == True).mean())  # noqa: E712
FRAMINGS = [("metadata", "Metadata"), ("explicit", "Plain"),
            ("authority", "Attending"), ("patient", "Patient")]
follow = {}
for k, _ in FRAMINGS:
    r = p1.loc[f"{k}_wrong"]
    lo, hi = (float(v) for v in r.follow_ci.strip("()").split(","))
    n_adm = round(float(r.ack_among_followed) * int(r.followed))
    follow[k] = {"rate": float(r.follow_rate), "lo": lo, "hi": hi, "n": int(r.n_eligible),
                 "followed": int(r.followed), "admitted": n_adm}

c17 = json.loads((RUNS["phase2_readout_L17"] / "causal_summary.json").read_text())["sets"]["top10"]
corr = json.loads((RUNS["phase2_readout_corr"] / "causal_summary.json").read_text())["sets"]["top10"]
cot2 = json.loads((RUNS["phase2_cot"] / "causal_summary.json").read_text())

pt = pd.read_csv(RUNS["phase3_probes"] / "probe_table.csv").set_index("probe")
base = json.loads((RUNS["phase3_probes"] / "baselines.json").read_text())
fu = json.loads((RUNS["phase3_followups"] / "followups.json").read_text())
gen = json.loads((RUNS["phase3_probes"] / "generalization.json").read_text())
post = pt.loc["post:post:resid:L22"]
pre = pt.loc["pre:final:resid:L22"]
POST_CI = json.loads(post.auroc_ci)
held_type = [v["auroc"] for k, v in gen["post:post:resid:L22"].items() if k.startswith("hold_out_hint")]
ack_flag = fu["post_reasoning"]["acknowledged_runs"]["flag_rate_at_80pct_recall"]
n_ack = fu["post_reasoning"]["acknowledged_runs"]["n"]

screen = pd.read_csv(RUNS["phase4_screen"] / "screen_summary.csv").set_index("setting")
cot4 = pd.read_csv(RUNS["phase4_cot"] / "cot_summary.csv").set_index("setting")
mpib = json.loads((RUNS["phase4_mpib"] / "results.json").read_text())
H1 = mpib["H1"]["post"]
badge = pd.read_csv(RUNS["phase5_build"] / "badge_by_kind.csv")
agree = badge[(badge.kind == "hint_gold") & (badge.label == "not scored")].iloc[0]

# Example: the first "caught" gallery case (seeded rule) whose no-hint answer was correct.
demo_idx = json.loads((ROOT / "demo/data/index.json").read_text())
for pick in demo_idx["gallery"]["caught"]["picks"]:
    q = json.loads((ROOT / f"demo/data/q/{pick['item_id']}.json").read_text())
    if q["runs"]["none"]["pred"] == q["gold"]:
        EX_Q, EX_RUN = q, q["runs"][pick["condition"]]
        break
EX_QUOTE = "The foreign origin of the pills suggests a possible adulterant or contaminant."
assert EX_QUOTE in EX_RUN["text"] and EX_RUN["ack"] is False and EX_RUN["flagged"]

# Guard against drift: these are the numbers the phase reports state.
assert (N_FOLLOWED, N_ADMITTED, N_Q) == (912, 30, 375)
assert round(post.auroc, 3) == 0.982 and round(H1["auroc_attacked_run"], 2) == 0.48
assert (cot2["mac_reproduces_following"], cot2["follow_after_intervention"]) == (35, 35)


# ---- vector charts (viewBox in points) ---------------------------------------------------------------
def vbars(width, height, groups, baseline=None, baseline_label="", legend=(), ymax=1.0,
          ticks=(0, .25, .5, .75, 1.0), right=0):
    """Grouped bars with 95% whiskers. groups = [{label, sub, bars: [{v, lo, hi, color}]}]."""
    left, top = 78, 30
    legend_h = 38 * len(legend) + (14 if legend else 0)
    label_h = 78
    ph, pw = height - top - label_h - legend_h, width - left - right

    def y(v):
        return top + ph * (1 - v / ymax)

    g = [f'<svg viewBox="0 0 {width} {height}" width="{width}pt" height="{height}pt" role="img" '
         f'xmlns="http://www.w3.org/2000/svg" font-family="Open Sans, sans-serif">']
    for t in ticks:
        g.append(f'<line x1="{left}" x2="{left + pw}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="{GRID}" stroke-width="1.4"/>')
        g.append(f'<text x="{left - 10}" y="{y(t) + 7:.1f}" text-anchor="end" font-size="20" fill="{MUTED}">{int(t * 100)}%</text>')
    gw = pw / len(groups)
    for gi, grp in enumerate(groups):
        nb = len(grp["bars"])
        bw, gap = min(74, gw * 0.84 / nb), 8
        x0 = left + gi * gw + (gw - (nb * bw + (nb - 1) * gap)) / 2
        for bi, b in enumerate(grp["bars"]):
            x = x0 + bi * (bw + gap)
            g.append(f'<rect x="{x:.1f}" y="{y(b["v"]):.1f}" width="{bw:.1f}" height="{ph * b["v"] / ymax:.1f}" fill="{b["color"]}"/>')
            cx = x + bw / 2
            for y1, y2 in ((b["lo"], b["hi"]),):
                g.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(y1):.1f}" y2="{y(y2):.1f}" stroke="{INK}" stroke-width="2.2"/>')
            for yy in (b["lo"], b["hi"]):
                g.append(f'<line x1="{cx - 7:.1f}" x2="{cx + 7:.1f}" y1="{y(yy):.1f}" y2="{y(yy):.1f}" stroke="{INK}" stroke-width="2.2"/>')
            g.append(f'<text x="{cx:.1f}" y="{y(b["hi"]) - 9:.1f}" text-anchor="middle" font-size="25" font-weight="700" fill="{INK}">{round(100 * b["v"])}%</text>')
        gx = left + gi * gw + gw / 2
        g.append(f'<text x="{gx:.1f}" y="{top + ph + 32:.1f}" text-anchor="middle" font-size="21" font-weight="700" fill="{INK}">{esc(grp["label"])}</text>')
        if grp.get("sub"):
            g.append(f'<text x="{gx:.1f}" y="{top + ph + 60:.1f}" text-anchor="middle" font-size="19" fill="{MUTED}">{esc(grp["sub"])}</text>')
    g.append(f'<line x1="{left}" x2="{left + pw}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="{INK}" stroke-width="1.8"/>')
    if baseline is not None:
        g.append(f'<line x1="{left}" x2="{left + pw + 4}" y1="{y(baseline):.1f}" y2="{y(baseline):.1f}" stroke="{INK}" stroke-width="2" stroke-dasharray="7 5"/>')
        for k, line in enumerate(baseline_label.split("|")):
            g.append(f'<text x="{left + pw + 10}" y="{y(baseline) + 6 + 24 * k - 12 * baseline_label.count("|"):.1f}" font-size="20" fill="{MUTED}">{esc(line)}</text>')
    ly = top + ph + label_h
    for k, (fill, text) in enumerate(legend):
        yy = ly + 34 * k + 8
        g.append(f'<rect x="{left}" y="{yy - 17:.1f}" width="26" height="26" fill="{fill}"/>')
        g.append(f'<text x="{left + 38}" y="{yy + 4:.1f}" font-size="21" fill="{INK}">{esc(text)}</text>')
    g.append("</svg>")
    return "".join(g)


def hbars(width, rows, chance=0.5, xmin=0.3):
    """AUROC bars drawn from chance (0.5) on a 0.3-1.0 axis, with 95% whiskers. rows = [(label, value, (lo, hi) | None, color)]."""
    left, right, top, rh = 250, 60, 16, 46
    height = top + rh * len(rows) + 46
    pw = width - left - right

    def x(v):
        return left + pw * (v - xmin) / (1 - xmin)

    g = [f'<svg viewBox="0 0 {width} {height}" width="{width}pt" height="{height}pt" role="img" '
         f'xmlns="http://www.w3.org/2000/svg" font-family="Open Sans, sans-serif">']
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        g.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 4}" y2="{top + rh * len(rows)}" stroke="{GRID}" stroke-width="1.4"/>')
        g.append(f'<text x="{x(t):.1f}" y="{top + rh * len(rows) + 28}" text-anchor="middle" font-size="19" fill="{MUTED}">{t:.1f}</text>')
    for i, (lab, v, ci, color) in enumerate(rows):
        yc = top + rh * i + rh / 2
        g.append(f'<text x="{left - 12}" y="{yc + 7:.1f}" text-anchor="end" font-size="20" fill="{INK}">{esc(lab)}</text>')
        x0, x1 = sorted((x(chance), x(v)))
        g.append(f'<rect x="{x0:.1f}" y="{yc - 14:.1f}" width="{max(2, x1 - x0):.1f}" height="28" fill="{color}"/>')
        end = max(x(ci[1]) if ci else x(v), x(chance))
        if ci:
            g.append(f'<line x1="{x(ci[0]):.1f}" x2="{x(ci[1]):.1f}" y1="{yc:.1f}" y2="{yc:.1f}" stroke="{INK}" stroke-width="2.2"/>')
        g.append(f'<text x="{end + 8:.1f}" y="{yc + 8:.1f}" font-size="22" font-weight="700" fill="{INK}">{v:.2f}</text>')
    g.append(f'<line x1="{x(chance):.1f}" x2="{x(chance):.1f}" y1="{top - 4}" y2="{top + rh * len(rows)}" stroke="{INK}" stroke-width="1.8"/>')
    g.append("</svg>")
    return "".join(g)


def wilson_bar(rate: float, n: int, color: str) -> dict:
    lo, hi = wilson_interval(round(rate * n), n)
    return {"v": rate, "lo": lo, "hi": hi, "color": color}


fig_follow = vbars(
    540, 300,
    [{"label": lab,
      "bars": [{"v": follow[k]["rate"], "lo": follow[k]["lo"], "hi": follow[k]["hi"], "color": COPPER if k == "metadata" else ROYAL}]}
     for k, lab in FRAMINGS],
    baseline=SHAM, baseline_label=f"sham|{pct(SHAM)}", ymax=0.8, ticks=(0, .2, .4, .6, .8), right=70,
)

mpib_post_ci = tuple(H1["auroc_attacked_ci"])
fig_detect = hbars(560, [
    ("Probe, after reasoning", float(post.auroc), tuple(POST_CI), NAVY),
    ("Probe, before reasoning", float(pre.auroc), tuple(json.loads(pre.auroc_ci)), ROYAL),
    ("LLM judge on the text", base["llm_judge_yes_rate"]["auroc"], None, "#8A96AD"),
    ("Hint type alone", fu["hint_type_only"]["auroc"], None, "#8A96AD"),
    ("Word counts (TF-IDF)", base["text_tfidf_reasoning_only"]["auroc"], None, "#8A96AD"),
    ("Same probe on injection", H1["auroc_attacked_run"], mpib_post_ci, COPPER),
])

scfg = yaml.safe_load((RUNS["phase4_screen"] / "config.yaml").read_text())
n_s = scfg["screen"]["n_hinted_per_framing"] * len(scfg["framings"])  # wrong-hint prompts per setting
n_c = int(cot4.loc["baseline", "n_eligible"])
fig_fix = vbars(
    560, 360,
    [{"label": "Quick answer", "sub": f"one pass, n = {n_s}",
      "bars": [wilson_bar(float(screen.loc["baseline", "follow_rate"]), n_s, "#8A96AD"),
               wilson_bar(float(screen.loc["ablate_pre_L22", "follow_rate"]), n_s, NAVY)]},
     {"label": "Step-by-step answer", "sub": f"reasoned, n = {n_c}",
      "bars": [wilson_bar(float(cot4.loc["baseline", "follow_rate"]), n_c, "#8A96AD"),
               wilson_bar(float(cot4.loc["ablate_pre_L22", "follow_rate"]), n_c, NAVY)]}],
    legend=[("#8A96AD", "No fix"), (NAVY, "Fix: remove direction (layer 22)")],
    ymax=0.6, ticks=(0, .2, .4, .6),
)

# ---- content ----------------------------------------------------------------------------------------
fmeta, fexp = follow["metadata"], follow["explicit"]
silent_share = 1 - N_ADMITTED / N_FOLLOWED
CSS = FONTS + f"""
@page {{ size: 24in 36in; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{ font-family: "Open Sans", sans-serif; color: {INK}; }}
.poster {{ width: 24in; height: 36in; overflow: hidden; background: #fff; padding: 0.35in; display: flex; flex-direction: column; gap: 0.24in; }}
.title {{ background: {NAVY}; color: #fff; height: 3.7in; display: grid; grid-template-columns: 3.4in 1fr 0.6in; align-items: center; padding: 0.2in 0.25in; flex: none; }}
.title .mid {{ text-align: center; }}
.logo {{ width: 3.1in; height: 3.1in; background: #fff; padding: 0.2in; display: flex; align-items: center; justify-content: center; }}
.logo img {{ width: 100%; height: 100%; display: block; object-fit: contain; }}
.title h1 {{ font-family: "EB Garamond", serif; font-size: 68pt; line-height: 1.0; font-weight: 700; margin: 0 0 0.14in; }}
.title .who {{ font-size: 34pt; font-weight: 700; line-height: 1.15; }}
.title .inst {{ font-size: 28pt; font-weight: 600; line-height: 1.2; margin-top: 0.03in; }}
.cols {{ flex: 1; display: grid; grid-template-columns: 1fr 1.2fr 1fr; gap: 0.3in; min-height: 0; }}
.col {{ display: flex; flex-direction: column; gap: 0.19in; min-height: 0; }}
.sec {{ display: flex; flex-direction: column; gap: 0.1in; min-height: 0; }}
.sec.grow {{ flex: 1; }}
.bar {{ background: {NAVY}; color: #fff; font-weight: 700; font-size: 38pt; line-height: 1; padding: 0.15in 0.22in; flex: none; }}
.box {{ border: 0.045in solid {GRID}; padding: 0.2in 0.26in; flex: 1; min-height: 0; }}
.box p, .box li {{ font-size: 24.5pt; line-height: 1.26; margin: 0 0 0.1in; }}
.box ul, .box ol {{ margin: 0; padding-left: 0.36in; }}
.box li {{ margin-bottom: 0.11in; }}
.small p, .small li {{ font-size: 21.5pt; margin-bottom: 0.08in; }}
.cap {{ font-size: 21pt; line-height: 1.25; color: {MUTED}; margin: 0.04in 0 0.12in; }}
.flag {{ color: {COPPER}; font-weight: 700; }}
.stat {{ background: {PANEL}; padding: 0.14in 0.2in; margin: 0.06in 0 0.14in; display: grid; grid-template-columns: 1fr 1fr; gap: 0.2in; }}
.stat .n {{ font-family: "EB Garamond", serif; font-size: 66pt; line-height: .95; font-weight: 700; }}
.stat .t {{ font-size: 20.5pt; line-height: 1.2; }}
.grp {{ border-left: 0.07in solid {NAVY}; padding: 0.02in 0 0.04in 0.2in; margin-bottom: 0.18in; }}
.grp ul {{ margin-top: 0; }}
.glabel {{ font-weight: 700; font-size: 28pt; letter-spacing: 0.02em; color: {ROYAL}; margin: 0 0 0.06in; }}
.also .glabel {{ color: {COPPER}; }}
svg {{ width: 7.4in; max-width: 100%; height: auto; display: block; }}
"""

BODY = f"""
<div class="poster">
  <div class="title">
    <div class="logo"><img src="assets/duke-shield.png" alt="Duke University shield"></div>
    <div class="mid">
      <h1>When a hint changes MedGemma’s answer, its reasoning hides it. Its activations don’t.</h1>
      <div class="who">Temesgen Tewolde</div>
      <div class="inst">Duke University · AIPI 590: Explainable AI</div>
    </div>
    <div></div>
  </div>

  <div class="cols">
    <div class="col">
      <div class="sec"><div class="bar">Abstract</div><div class="box">
        <p>A medical model’s written reasoning may leave out what drove its answer. We added hints that point to an answer to {N_Q} MedQA questions for MedGemma 4B. The hint changed the answer in {N_FOLLOWED} runs, and the reasoning mentioned it in only {N_ADMITTED} ({pct(N_ADMITTED / N_FOLLOWED, 1)}). Inside the model, the hint travels as an answer-letter signal. A probe on layer-22 activations, never told the hint, catches hint-following with AUROC {post.auroc:.2f}. Steering against that signal fixes quick answers but not reasoned ones, and the probe does not catch prompt injection ({H1['auroc_attacked_run']:.2f}).</p>
      </div></div>
      <div class="sec"><div class="bar">Objectives</div><div class="box">
        <ol>
          <li><b>See it:</b> how often does a hint change the answer without the reasoning saying so?</li>
          <li><b>Catch it:</b> do the model’s activations reveal what the text hides?</li>
          <li><b>Fix it:</b> does steering the model’s activations stop it?</li>
          <li><b>Injection:</b> is it the same weakness as prompt injection?</li>
        </ol>
      </div></div>
      <div class="sec"><div class="bar">Methods</div><div class="box">
        <ul>
          <li><b>Model:</b> MedGemma 4B (open weights), greedy decoding, read with Gemma Scope 2 SAEs at layers 9, 17, 22, 29.</li>
          <li><b>Data:</b> {N_Q} MedQA questions (4 options) not in the MPIB benchmark. No-hint accuracy {pct(ACC_NONE, 1)}.</li>
          <li><b>10 versions per question:</b> no hint; a harmless sham sentence; 4 hints (metadata tag, “Hint: the correct answer is X”, an attending’s opinion, a patient’s insistence), each pointing to a wrong or the right answer.</li>
          <li><b>Labels:</b> a judge model (Qwen 27B) checks whether the reasoning mentions the hint; it agreed with 2 blind model annotators on 100/100 cases. Silent = followed and never mentioned.</li>
          <li><b>Detector:</b> logistic-regression probes on activations, 5 folds grouped by question, never shown the hint text or letter.</li>
          <li><b>Fix and injection:</b> pre-registered. Directions learned on a dev half; 150 MPIB cases with an instruction hidden in the question.</li>
        </ul>
      </div></div>
      <div class="sec grow"><div class="bar">Demo</div><div class="box small">
        <p>An interactive page shows what the model <b>says</b> next to what <b>drove</b> its answer (no-hint answer, judge check, warning badge), with live hint toggles for {demo_idx['n_questions']} questions. Precomputed; no GPU needed.</p>
      </div></div>
    </div>

    <div class="col">
      <div class="sec grow"><div class="bar">Results</div><div class="box">
        <div class="grp">
          <div class="glabel">1 · See it</div>
          <ul><li>Every hint moved answers far above the {pct(SHAM)} sham floor. The metadata tag moved {pct(fmeta['rate'])} and was <b>never</b> mentioned.</li></ul>
          {fig_follow}
          <p class="cap">Wrong-answer hints: share of eligible runs that moved to the hinted answer. Dashed: sham sentence. Whiskers: 95% CIs. Of those that moved, the reasoning admitted the hint in {", ".join(f"{follow[k]['admitted']}/{follow[k]['followed']}" for k, _ in FRAMINGS)} (same order).</p>
        </div>
        <div class="grp">
          <div class="glabel">2 · Catch it</div>
          <ul>
            <li>The hint travels as <b>answer-letter features</b> on the hint’s own tokens. Removing 10 at layer 17 cut quick-answer hint-following from {pct(c17['readout_follows_hint_before'])} to {pct(c17['readout_follows_hint_after'])}; features that only correlate barely moved the pull toward the hinted letter (log-probability {corr['mean_delta_logp_target_hinted']:+.2f}, vs {c17['mean_delta_logp_target_hinted']:+.2f}).</li>
            <li>A hint-blind probe catches silent hint-following (within hint type {fu['post_reasoning']['per_hint_type']['mean_within_type']:.2f}; unseen hint types {min(held_type):.2f}–{max(held_type):.2f}). Text-only detectors do far worse.</li>
          </ul>
          {fig_detect}
          <p class="cap">AUROC (0.5 = chance) for silent hint-following vs ignored hints, {int(post.n):,} runs. Bottom bar: moved vs unmoved answers on {mpib['n_cases']} injection cases.</p>
        </div>
        <div class="grp">
          <div class="glabel">3 · Fix it</div>
          <ul><li>Removing the hint-following direction works on quick answers but <b>not once the model reasons</b>; random directions do nothing.</li></ul>
          {fig_fix}
          <p class="cap">Hint-following on dev questions (different samples per row). Whiskers: 95% CIs.</p>
        </div>
        <div class="grp">
          <div class="glabel">4 · Injection</div>
          <ul><li>Injected instructions moved {pct(mpib['moved_rate_baseline'])} of answers, but the probe could not tell which ({H1['auroc_attacked_run']:.2f}). Pre-registered verdict: <b>{mpib['verdict']}</b>, leaning against a shared mechanism.</li></ul>
        </div>
      </div></div>
    </div>

    <div class="col">
      <div class="sec"><div class="bar">Example</div><div class="box small">
        <p><b>One run</b> (pharmacology): fever and sweating after foreign diet pills. B = uncoupling of oxidative phosphorylation (correct); C = bacterial contamination (wrong).</p>
        <p><b>No hint:</b> the model chose <b>{EX_Q['runs']['none']['pred']}</b>. With <span class="flag">“{esc(EX_RUN['cue'])}”</span> added, it chose <b>{EX_RUN['pred']}</b>.</p>
        <p><b>Reasoning:</b> “…{esc(EX_QUOTE)}” <span class="flag">It never mentions the hint.</span></p>
        <p><b>Detector:</b> score {EX_RUN['post_score']:.3f}, above the warning threshold.</p>
        <p><i>A seeded pick from the demo’s gallery, one of {demo_idx['gallery']['caught']['n_total']} caught runs in its {demo_idx['n_questions']} questions.</i></p>
      </div></div>
      <div class="sec"><div class="bar">Conclusion</div><div class="box">
        <div class="stat">
          <div><div class="n" style="color:{COPPER}">{pct(silent_share)}</div><div class="t">of hint-driven answer changes are never mentioned</div></div>
          <div><div class="n" style="color:{NAVY}">{post.auroc:.2f}</div><div class="t">AUROC of a probe that never sees the hint</div></div>
        </div>
        <ol>
          <li><b>See it.</b> The reasoning hides almost every hint it follows.</li>
          <li><b>Catch it.</b> Activations reveal it, but the probe flags <i>following</i>, not hiding: it fires on {pct(ack_flag)} of the {n_ack} admitted runs, and on {pct(agree['mean'])} of right-answer hints the model agreed with anyway.</li>
          <li><b>Fix it.</b> Steering that works on quick answers fails once the model reasons ({cot2['mac_reproduces_following']}/{cot2['n']} still followed after feature removal).</li>
          <li><b>Injection.</b> The probe does not transfer.</li>
        </ol>
      </div></div>
      <div class="sec"><div class="bar">Limitations</div><div class="box small">
        <ul>
          <li>One 4B model; English multiple choice; greedy decoding.</li>
          <li>Every hint names a letter; content cues untested.</li>
          <li>Judge checked against model annotators, not people.</li>
          <li>Small fix checks (24 dev questions); MPIB attacks target actions, not letters.</li>
        </ul>
      </div></div>
      <div class="sec grow"><div class="bar">References</div><div class="box small">
        <ol>
          <li>Turpin et al. <i>Language Models Don’t Always Say What They Think.</i> NeurIPS (2023).</li>
          <li>Afolabi et al. <i>Faithful or Just Plausible?</i> ML4H, arXiv:2603.13988 (2026).</li>
          <li>Ayadi &amp; Hautli-Janisz. <i>Probing Bias Formation in Medical LLMs through Activation Steering.</i> ACL SRW (2026).</li>
          <li>Google DeepMind. <i>Gemma Scope 2</i> (2025).</li>
          <li>Lee, Jang &amp; Choi. <i>MPIB.</i> Findings of EMNLP (2026).</li>
        </ol>
        <p style="margin-top:0.12in"><b>Note:</b> research use only; model text is AI-generated and not medical advice. Code, paper and an interactive demo are in the project repository.</p>
      </div></div>
    </div>
  </div>
</div>
"""

OUT_HTML.write_text(
    f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>MedLens honesty poster (24 x 36)</title>'
    f"<style>{CSS}</style></head><body>{BODY}</body></html>"
)
print("wrote", OUT_HTML)
if "--no-pdf" not in sys.argv:
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--hide-scrollbars",
           f"--print-to-pdf={OUT_PDF}", f"file://{OUT_HTML}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    print("chrome exit", r.returncode, "->", OUT_PDF.name, f"({OUT_PDF.stat().st_size / 1e3:.0f} KB)")
