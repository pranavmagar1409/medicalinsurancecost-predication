"""
Medical insurance cost estimator: Flask app for Render.

Files needed in the repo root:
    app.py
    requirements.txt
    xgboost.pkl        (your trained XGBRegressor)

Render settings:
    Build command:  pip install -r requirements.txt
    Start command:  gunicorn app:app --workers 1 --threads 4 --timeout 120
"""

import math
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
APP_NAME = "Covercast"  # change this to rename the app

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "xgboost.pkl"))

# Column order the model was trained on (read from the pickle itself).
FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]

# Integer codes the model was trained with (alphabetical label encoding).
# If you ever retrain with different encodings, change them here only.
SEX_CODES = {"female": 0, "male": 1}
SMOKER_CODES = {"no": 0, "yes": 1}
REGION_CODES = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

# Fallback "average" shown if the model can't report its own baseline.
FALLBACK_AVERAGE = 13030.0
HEALTHY_BMI = 24.9

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
app = Flask(__name__)

MODEL = None
MODEL_ERROR = None


def load_model():
    global MODEL, MODEL_ERROR
    try:
        with open(MODEL_PATH, "rb") as fh:
            MODEL = pickle.load(fh)
        try:
            MODEL.set_params(n_jobs=1)
        except Exception:
            pass
        MODEL_ERROR = None
    except Exception as exc:  # keep the site up and report the problem clearly
        MODEL = None
        MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        app.logger.error("Could not load model from %s: %s", MODEL_PATH, MODEL_ERROR)


load_model()


# --------------------------------------------------------------------------
# Prediction helpers
# --------------------------------------------------------------------------
class BadInput(ValueError):
    pass


def parse_payload(data):
    if not isinstance(data, dict):
        raise BadInput("Send the details as a JSON object.")
    try:
        age = int(round(float(data["age"])))
        bmi = float(data["bmi"])
        children = int(data["children"])
        sex = SEX_CODES[str(data["sex"]).strip().lower()]
        smoker = SMOKER_CODES[str(data["smoker"]).strip().lower()]
        region = REGION_CODES[str(data["region"]).strip().lower()]
    except (KeyError, TypeError, ValueError):
        raise BadInput("Some details are missing or not valid.")
    if not math.isfinite(bmi):
        raise BadInput("BMI must be a number.")
    return {
        "age": min(max(age, 18), 64),
        "sex": sex,
        "bmi": round(min(max(bmi, 15.0), 55.0), 1),
        "children": min(max(children, 0), 5),
        "smoker": smoker,
        "region": region,
    }


def make_frame(rows):
    df = pd.DataFrame(rows)[FEATURES]
    return df.astype(
        {
            "age": "int64",
            "sex": "int64",
            "bmi": "float64",
            "children": "int64",
            "smoker": "int64",
            "region": "int64",
        }
    )


def predict_many(rows):
    preds = np.asarray(MODEL.predict(make_frame(rows)), dtype=float)
    return np.maximum(preds, 0.0)


def contributions(row):
    """Per-feature contribution (SHAP values) in dollars, plus the baseline."""
    try:
        import xgboost as xgb

        booster = MODEL.get_booster() if hasattr(MODEL, "get_booster") else MODEL
        matrix = xgb.DMatrix(make_frame([row]))
        values = booster.predict(matrix, pred_contribs=True)[0]
        drivers = [
            {"key": name, "value": float(values[i])} for i, name in enumerate(FEATURES)
        ]
        return drivers, float(values[-1])
    except Exception as exc:
        app.logger.warning("Contributions unavailable: %s", exc)
        return None, None


def build_scenarios(row, current):
    rows, meta = [], []
    if row["smoker"] == 1:
        rows.append({**row, "smoker": 0})
        meta.append({"key": "quit_smoking"})
    if row["bmi"] > HEALTHY_BMI:
        rows.append({**row, "bmi": HEALTHY_BMI})
        meta.append({"key": "healthy_bmi", "bmi": HEALTHY_BMI})
    if not rows:
        return []
    preds = predict_many(rows)
    out = []
    for info, pred in zip(meta, preds):
        delta = float(pred) - current
        if delta <= -100:
            out.append({**info, "delta": delta, "estimate": float(pred)})
    out.sort(key=lambda item: item["delta"])
    return out


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    html = INDEX_HTML.replace("__APP_NAME__", APP_NAME)
    resp = Response(html, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/healthz")
def healthz():
    if MODEL is None:
        return jsonify(status="error", detail=MODEL_ERROR), 500
    return jsonify(status="ok")


@app.post("/api/predict")
def api_predict():
    if MODEL is None:
        return (
            jsonify(error="The model isn't available right now. Please try again shortly."),
            503,
        )
    try:
        row = parse_payload(request.get_json(silent=True))
    except BadInput as exc:
        return jsonify(error=str(exc)), 400

    try:
        current = float(predict_many([row])[0])
        drivers, baseline = contributions(row)
        scenarios = build_scenarios(row, current)
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify(error="Something went wrong while estimating. Please try again."), 500

    return jsonify(
        prediction=round(current, 2),
        monthly=round(current / 12, 2),
        baseline=round(baseline if baseline is not None else FALLBACK_AVERAGE, 2),
        drivers=drivers,
        scenarios=scenarios,
    )


# --------------------------------------------------------------------------
# Front end (single page, no build step)
# --------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__APP_NAME__: medical insurance cost estimator</title>
<meta name="description" content="Estimate your yearly medical insurance charges from your age, BMI, smoking status, family size and region.">
<meta name="theme-color" content="#0B5F5A">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='18' fill='%230B5F5A'/%3E%3Cpath d='M32 14v36M14 32h36' stroke='%23F2B134' stroke-width='9' stroke-linecap='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wdth,wght@12..96,75..100,200..800&family=Figtree:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --mist:#E8F0EE;
  --paper:#F9FCFB;
  --ink:#0E2A2B;
  --ink-2:#47615F;
  --line:rgba(14,42,43,.14);
  --lagoon:#0B5F5A;
  --lagoon-deep:#084540;
  --marigold:#F2B134;
  --mint:#98E3BF;
  --display:"Bricolage Grotesque","Segoe UI",system-ui,sans-serif;
  --body:"Figtree","Segoe UI",system-ui,sans-serif;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--mist);color:var(--ink);font:400 1rem/1.55 var(--body);-webkit-font-smoothing:antialiased}
button,input{font:inherit;color:inherit}
.wrap{max-width:1120px;margin:0 auto;padding:28px 20px 96px}

/* header */
.brand{display:flex;align-items:center;gap:10px;font:600 1.1rem var(--display);letter-spacing:-.01em}
.mark{width:30px;height:30px;border-radius:10px;background:var(--lagoon);display:grid;place-items:center}
.mark svg{width:16px;height:16px}
h1{margin:44px 0 12px;font:700 clamp(2.3rem,5.4vw,4.2rem)/1.02 var(--display);font-variation-settings:"wdth" 86;letter-spacing:-.025em;max-width:15ch;text-wrap:balance}
.lede{margin:0;max-width:46ch;color:var(--ink-2);font-size:1.1rem}

/* layout */
.grid{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:28px;align-items:start;margin-top:40px}
@media (max-width:960px){.grid{grid-template-columns:1fr}}

/* form card */
.card{background:var(--paper);border:1px solid var(--line);border-radius:28px;padding:6px 28px}
@media (max-width:520px){.card{padding:4px 20px;border-radius:24px}}
.field{padding:24px 0;border-top:1px solid var(--line)}
.field:first-child{border-top:0}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:12px}
.name{font-weight:600}
.val{font:700 1.9rem/1 var(--display);font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.val small{font:500 .9rem var(--body);color:var(--ink-2);margin-left:4px;letter-spacing:0}
.hint{margin:8px 0 0;color:var(--ink-2);font-size:.92rem}

/* sliders */
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:28px;margin:0;background:transparent;cursor:pointer;--p:50%}
input[type=range]::-webkit-slider-runnable-track{height:8px;border-radius:99px;background:linear-gradient(to right,var(--lagoon) var(--p),rgba(14,42,43,.12) var(--p))}
input[type=range]::-moz-range-track{height:8px;border-radius:99px;background:rgba(14,42,43,.12)}
input[type=range]::-moz-range-progress{height:8px;border-radius:99px;background:var(--lagoon)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:26px;height:26px;margin-top:-9px;border-radius:50%;background:var(--marigold);border:3px solid var(--ink);transition:transform .15s}
input[type=range]::-moz-range-thumb{width:20px;height:20px;border-radius:50%;background:var(--marigold);border:3px solid var(--ink)}
input[type=range]:active::-webkit-slider-thumb{transform:scale(1.12)}
input[type=range]:focus-visible{outline:none}
input[type=range]:focus-visible::-webkit-slider-thumb{outline:3px solid var(--lagoon);outline-offset:3px}
input[type=range]:focus-visible::-moz-range-thumb{outline:3px solid var(--lagoon);outline-offset:3px}

/* bmi chip */
.chip{display:inline-flex;align-items:center;gap:8px;margin-top:10px;padding:7px 14px;border-radius:99px;background:rgba(11,95,90,.1);font-weight:500;font-size:.95rem}
.chip i{width:9px;height:9px;border-radius:50%;background:var(--lagoon)}
.chip[data-tone=warn] i{background:var(--marigold)}
.chip[data-tone=warn]{background:rgba(242,177,52,.22)}

/* segmented radios */
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media (max-width:520px){.pair{grid-template-columns:1fr}}
.seg{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:6px;padding:5px;border-radius:16px;background:rgba(14,42,43,.07)}
.seg.two{grid-auto-flow:row;grid-template-columns:1fr 1fr}
.seg label{position:relative;display:block}
.seg input{position:absolute;inset:0;opacity:0;margin:0;cursor:pointer}
.seg span{display:block;text-align:center;padding:11px 8px;border-radius:12px;font-weight:600;color:var(--ink-2);transition:background .2s,color .2s}
.seg input:checked+span{background:var(--ink);color:var(--mist)}
.seg input:focus-visible+span{outline:3px solid var(--lagoon);outline-offset:2px}
.seg label:hover input:not(:checked)+span{background:rgba(14,42,43,.08)}

/* stepper */
.row{display:flex;align-items:center;justify-content:space-between;gap:16px}
.stepper{display:flex;align-items:center;gap:6px;padding:5px;border-radius:99px;background:rgba(14,42,43,.07)}
.stepper button{width:42px;height:42px;border:0;border-radius:50%;background:var(--paper);font:600 1.4rem/1 var(--display);cursor:pointer;transition:background .15s,transform .15s}
.stepper button:hover:not(:disabled){background:var(--marigold)}
.stepper button:active:not(:disabled){transform:scale(.94)}
.stepper button:disabled{opacity:.35;cursor:not-allowed}
.stepper button:focus-visible{outline:3px solid var(--lagoon);outline-offset:2px}
.stepper output{min-width:38px;text-align:center;font:700 1.5rem/1 var(--display);font-variant-numeric:tabular-nums}

/* quote panel */
.quote{position:sticky;top:20px;max-height:calc(100vh - 40px);overflow:auto;border-radius:32px;padding:30px 30px 26px;background:var(--lagoon);color:var(--mist)}
@media (max-width:960px){.quote{position:static;max-height:none}}
@media (max-width:520px){.quote{padding:24px 20px;border-radius:26px}}
.q-label{margin:0;font-weight:500;opacity:.85}
.q-num{margin:6px 0 2px;font:800 clamp(3.6rem,8.4vw,6rem)/1 var(--display);font-variation-settings:"wdth" 82;letter-spacing:-.035em;font-variant-numeric:tabular-nums;color:#fff}
.q-sub{margin:0;opacity:.85}

.meter{margin-top:26px}
.track{position:relative;height:12px;border-radius:99px;background:rgba(255,255,255,.18)}
.fill{position:absolute;inset:0 auto 0 0;width:0;border-radius:99px;background:var(--marigold);transition:width .7s cubic-bezier(.2,.8,.2,1)}
.tick{position:absolute;top:-6px;bottom:-6px;width:2px;background:#fff;border-radius:2px}
.scale{position:relative;display:flex;justify-content:space-between;margin-top:10px;font-size:.85rem;opacity:.85}
.scale .avg{position:absolute;transform:translateX(-50%);white-space:nowrap}
.q-compare{margin:22px 0 0;font-size:1.05rem}

.sec{margin-top:26px;padding-top:22px;border-top:1px solid rgba(255,255,255,.2)}
.sec h2{margin:0;font:700 1.25rem/1.2 var(--display);letter-spacing:-.01em}
.sec .sub{margin:4px 0 14px;opacity:.8;font-size:.92rem}
.list{list-style:none;margin:0;padding:0;display:grid;gap:10px}
.d{display:grid;grid-template-columns:84px 1fr 74px;align-items:center;gap:12px;font-size:.95rem}
.d .bar{position:relative;height:10px;border-radius:99px;background:rgba(255,255,255,.12)}
.d .bar::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:rgba(255,255,255,.45)}
.d .bar i{position:absolute;top:0;bottom:0;border-radius:99px;transition:width .6s cubic-bezier(.2,.8,.2,1)}
.d .bar i.up{left:50%;background:var(--marigold)}
.d .bar i.down{right:50%;background:var(--mint)}
.d .amt{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.s{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;padding:14px 16px;border-radius:18px;background:rgba(255,255,255,.1)}
.s b{display:block;font-weight:600}
.s small{display:block;opacity:.8;margin-top:2px}
.s .save{font:700 1.3rem/1.1 var(--display);color:var(--mint);white-space:nowrap;font-variant-numeric:tabular-nums}
.none{margin:0;opacity:.85}

.status{margin:18px 0 0;font-size:.92rem}
.status button{margin-left:6px;padding:6px 14px;border:0;border-radius:99px;background:var(--marigold);color:var(--ink);font-weight:600;cursor:pointer}
.status button:focus-visible{outline:3px solid #fff;outline-offset:2px}
.quote[aria-busy=true] .q-num{opacity:.75}

.note{margin:28px 0 0;max-width:70ch;color:var(--ink-2);font-size:.9rem}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}

/* mobile estimate bar */
.mini{display:none}
@media (max-width:960px){
  .mini{position:fixed;left:12px;right:12px;bottom:12px;z-index:5;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 18px;border:0;border-radius:99px;background:var(--lagoon-deep);color:var(--mist);cursor:pointer;box-shadow:0 8px 24px rgba(8,69,64,.35);transition:transform .3s,opacity .3s}
  .mini span{font-size:.92rem}
  .mini strong{font:800 1.4rem/1 var(--display);color:#fff;font-variant-numeric:tabular-nums}
  .mini.hide{transform:translateY(140%);opacity:0;pointer-events:none}
}
.mini:focus-visible{outline:3px solid var(--marigold);outline-offset:3px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto!important}}
</style>
</head>
<body>
<div class="wrap">
  <div class="brand">
    <span class="mark" aria-hidden="true"><svg viewBox="0 0 16 16"><path d="M8 2v12M2 8h12" stroke="#F2B134" stroke-width="3" stroke-linecap="round"/></svg></span>
    __APP_NAME__
  </div>

  <h1>What might your medical insurance cost this year?</h1>
  <p class="lede">Adjust the details below. The estimate updates as you go.</p>

  <div class="grid">
    <form class="card" id="form" onsubmit="return false" autocomplete="off">
      <div class="field">
        <div class="head"><label class="name" for="age">Age</label><div class="val"><span id="ageOut">35</span><small>years</small></div></div>
        <input type="range" id="age" min="18" max="64" step="1" value="35">
      </div>

      <div class="field">
        <div class="head"><label class="name" for="height">Height</label><div class="val"><span id="heightOut">170</span><small>cm</small></div></div>
        <input type="range" id="height" min="120" max="210" step="1" value="170">
        <div class="head" style="margin-top:18px"><label class="name" for="weight">Weight</label><div class="val"><span id="weightOut">70</span><small>kg</small></div></div>
        <input type="range" id="weight" min="35" max="180" step="1" value="70">
        <div class="chip" id="bmiChip"><i></i><span id="bmiText"></span></div>
      </div>

      <div class="field">
        <div class="pair">
          <div>
            <div class="head"><span class="name" id="l-sex">Sex</span></div>
            <div class="seg" role="radiogroup" aria-labelledby="l-sex">
              <label><input type="radio" name="sex" value="female" checked><span>Female</span></label>
              <label><input type="radio" name="sex" value="male"><span>Male</span></label>
            </div>
          </div>
          <div>
            <div class="head"><span class="name" id="l-smoker">Do you smoke?</span></div>
            <div class="seg" role="radiogroup" aria-labelledby="l-smoker">
              <label><input type="radio" name="smoker" value="no" checked><span>No</span></label>
              <label><input type="radio" name="smoker" value="yes"><span>Yes</span></label>
            </div>
          </div>
        </div>
      </div>

      <div class="field">
        <div class="row">
          <div>
            <span class="name" id="l-kids">Children covered</span>
            <p class="hint" style="margin-top:2px">Dependents on the plan</p>
          </div>
          <div class="stepper" role="group" aria-labelledby="l-kids">
            <button type="button" id="kidsMinus" aria-label="One fewer child">&minus;</button>
            <output id="kidsOut" aria-live="polite">0</output>
            <button type="button" id="kidsPlus" aria-label="One more child">+</button>
          </div>
        </div>
      </div>

      <div class="field">
        <div class="head"><span class="name" id="l-region">Region</span></div>
        <div class="seg two" role="radiogroup" aria-labelledby="l-region">
          <label><input type="radio" name="region" value="northeast" checked><span>Northeast</span></label>
          <label><input type="radio" name="region" value="northwest"><span>Northwest</span></label>
          <label><input type="radio" name="region" value="southeast"><span>Southeast</span></label>
          <label><input type="radio" name="region" value="southwest"><span>Southwest</span></label>
        </div>
      </div>
    </form>

    <aside class="quote" id="quote" aria-labelledby="q-title" aria-busy="true">
      <p class="q-label" id="q-title">Estimated yearly charges</p>
      <div class="q-num" id="num">&ndash;</div>
      <p class="q-sub" id="sub">&nbsp;</p>

      <div class="meter">
        <div class="track"><div class="fill" id="fill"></div><div class="tick" id="tick"></div></div>
        <div class="scale"><span>$0</span><span class="avg" id="avgLabel">average</span><span>$60k+</span></div>
      </div>
      <p class="q-compare" id="compare"></p>

      <section class="sec">
        <h2>What&rsquo;s driving this estimate</h2>
        <p class="sub" id="driversSub">Each bar shows how much a detail moves the estimate up or down.</p>
        <ul class="list" id="drivers"></ul>
      </section>

      <section class="sec">
        <h2>What could lower it</h2>
        <p class="sub">Changes that would reduce your estimate, holding everything else the same.</p>
        <ul class="list" id="scen"></ul>
      </section>

      <p class="status" id="status" role="status"></p>
    </aside>
  </div>

  <p class="note">Estimates come from an XGBoost model trained on historical US insurance data. They&rsquo;re illustrations rather than quotes, and don&rsquo;t replace advice from an insurer or a doctor.</p>
  <div class="sr" id="live" aria-live="polite"></div>
</div>

<button class="mini hide" id="mini" type="button" aria-label="Jump to your estimate">
  <span>Estimated yearly charges</span><strong id="miniNum">&ndash;</strong>
</button>

<script>
(() => {
  const $ = s => document.querySelector(s);
  const money = new Intl.NumberFormat('en-US', {style:'currency', currency:'USD', maximumFractionDigits:0});
  const signed = v => (v < 0 ? '\u2212' : '+') + money.format(Math.abs(Math.round(v)));
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const METER_MAX = 60000;
  const NAMES = {age:'Age', bmi:'BMI', smoker:'Smoking', children:'Children', region:'Region', sex:'Sex'};

  const el = {
    age:$('#age'), height:$('#height'), weight:$('#weight'),
    ageOut:$('#ageOut'), heightOut:$('#heightOut'), weightOut:$('#weightOut'),
    chip:$('#bmiChip'), bmiText:$('#bmiText'),
    kidsOut:$('#kidsOut'), minus:$('#kidsMinus'), plus:$('#kidsPlus'),
    num:$('#num'), miniNum:$('#miniNum'), sub:$('#sub'), fill:$('#fill'), tick:$('#tick'), avgLabel:$('#avgLabel'),
    compare:$('#compare'), drivers:$('#drivers'), driversSub:$('#driversSub'), scen:$('#scen'),
    status:$('#status'), quote:$('#quote'), live:$('#live'), mini:$('#mini')
  };
  let kids = 0, shown = 0, raf = 0, timer = 0, ctrl = null, hasResult = false;

  const radio = name => document.querySelector(`input[name="${name}"]:checked`).value;
  const bmiOf = (h, w) => +(w / Math.pow(h / 100, 2)).toFixed(1);

  function paintRange(input){
    const p = (input.value - input.min) / (input.max - input.min) * 100;
    input.style.setProperty('--p', p + '%');
  }

  function bmiLabel(b){
    if (b < 18.5) return ['below the healthy range', 'warn'];
    if (b < 25)   return ['in the healthy range', 'ok'];
    if (b < 30)   return ['above the healthy range', 'warn'];
    return ['in the obesity range', 'warn'];
  }

  function syncLabels(){
    el.ageOut.textContent = el.age.value;
    el.heightOut.textContent = el.height.value;
    el.weightOut.textContent = el.weight.value;
    [el.age, el.height, el.weight].forEach(paintRange);
    const b = bmiOf(+el.height.value, +el.weight.value);
    const [txt, tone] = bmiLabel(b);
    el.bmiText.textContent = `BMI ${b.toFixed(1)}, ${txt}`;
    el.chip.dataset.tone = tone;
    el.kidsOut.textContent = kids;
    el.minus.disabled = kids <= 0;
    el.plus.disabled = kids >= 5;
  }

  function setNumber(v){
    shown = v;
    const t = money.format(Math.round(v));
    el.num.textContent = t;
    el.miniNum.textContent = t;
  }

  function tween(to){
    cancelAnimationFrame(raf);
    if (reduce || !hasResult){ setNumber(to); return; }
    const from = shown, t0 = performance.now(), dur = 600;
    const step = now => {
      const k = Math.min((now - t0) / dur, 1), e = 1 - Math.pow(1 - k, 3);
      setNumber(from + (to - from) * e);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  }

  function render(r, s){
    tween(r.prediction);
    el.sub.textContent = `about ${money.format(Math.round(r.monthly))} a month`;
    el.fill.style.width = Math.min(r.prediction / METER_MAX, 1) * 100 + '%';
    const avgPct = Math.min(r.baseline / METER_MAX, 1) * 100;
    el.tick.style.left = `calc(${avgPct}% - 1px)`;
    el.avgLabel.style.left = avgPct + '%';
    el.avgLabel.textContent = `average ${money.format(Math.round(r.baseline))}`;

    const pct = Math.round(Math.abs(r.prediction - r.baseline) / r.baseline * 100);
    el.compare.textContent = pct < 3
      ? 'That is about the same as the average.'
      : `That is ${pct}% ${r.prediction > r.baseline ? 'above' : 'below'} the average.`;

    // drivers
    el.drivers.innerHTML = '';
    if (r.drivers && r.drivers.length){
      const max = Math.max(...r.drivers.map(d => Math.abs(d.value)), 1);
      r.drivers.slice().sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).forEach(d => {
        const li = document.createElement('li');
        li.className = 'd';
        const w = Math.max(Math.abs(d.value) / max * 50, 1.5);
        li.innerHTML = `<span>${NAMES[d.key] || d.key}</span><span class="bar"><i class="${d.value >= 0 ? 'up' : 'down'}" style="width:${w}%"></i></span><span class="amt">${signed(d.value)}</span>`;
        el.drivers.appendChild(li);
      });
      el.driversSub.textContent = `Compared with the ${money.format(Math.round(r.baseline))} average. Bars to the right raise the estimate; bars to the left lower it.`;
      el.drivers.parentElement.hidden = false;
    } else {
      el.drivers.parentElement.hidden = true;
    }

    // scenarios
    el.scen.innerHTML = '';
    if (!r.scenarios.length){
      el.scen.innerHTML = '<li class="none">Nothing you can change here would lower this estimate noticeably.</li>';
    }
    r.scenarios.forEach(sc => {
      let title, detail;
      if (sc.key === 'quit_smoking'){
        title = 'If you didn\u2019t smoke';
        detail = `Estimate would be ${money.format(Math.round(sc.estimate))}`;
      } else {
        const kg = Math.round(sc.bmi * Math.pow(s.height / 100, 2));
        title = `If your BMI were ${sc.bmi}`;
        detail = `About ${kg} kg at ${s.height} cm. Estimate would be ${money.format(Math.round(sc.estimate))}`;
      }
      const li = document.createElement('li');
      li.className = 's';
      li.innerHTML = `<div><b>${title}</b><small>${detail}</small></div><div class="save">${signed(sc.delta)}</div>`;
      el.scen.appendChild(li);
    });

    el.status.textContent = '';
    el.quote.setAttribute('aria-busy', 'false');
    hasResult = true;
    clearTimeout(el.liveTimer);
    el.liveTimer = setTimeout(() => {
      el.live.textContent = `Estimated yearly charges ${money.format(Math.round(r.prediction))}`;
    }, 500);
  }

  function showError(msg){
    el.quote.setAttribute('aria-busy', 'false');
    el.status.innerHTML = '';
    el.status.append(document.createTextNode(msg + ' '));
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = 'Try again';
    b.onclick = run;
    el.status.appendChild(b);
  }

  async function run(){
    const s = {
      age:+el.age.value, height:+el.height.value, weight:+el.weight.value,
      sex:radio('sex'), smoker:radio('smoker'), region:radio('region'), children:kids
    };
    const body = {age:s.age, sex:s.sex, bmi:bmiOf(s.height, s.weight), children:s.children, smoker:s.smoker, region:s.region};
    if (ctrl) ctrl.abort();
    ctrl = new AbortController();
    el.quote.setAttribute('aria-busy', 'true');
    try{
      const res = await fetch('/api/predict', {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify(body), signal:ctrl.signal
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || 'The estimate could not be updated.');
      render(data, s);
    } catch (e){
      if (e.name === 'AbortError') return;
      showError(e instanceof TypeError ? 'Could not reach the server. Check your connection.' : e.message);
    }
  }

  const schedule = () => { syncLabels(); clearTimeout(timer); timer = setTimeout(run, 130); };

  ['age', 'height', 'weight'].forEach(id => el[id].addEventListener('input', schedule));
  document.querySelectorAll('input[type=radio]').forEach(r => r.addEventListener('change', schedule));
  el.minus.addEventListener('click', () => { if (kids > 0){ kids--; schedule(); } });
  el.plus.addEventListener('click',  () => { if (kids < 5){ kids++; schedule(); } });

  // mobile estimate bar: visible only while the full panel is off screen
  el.mini.addEventListener('click', () => el.quote.scrollIntoView({behavior:'smooth', block:'start'}));
  if ('IntersectionObserver' in window){
    new IntersectionObserver(([e]) => el.mini.classList.toggle('hide', e.isIntersecting || !hasResult),
      {threshold:0.15}).observe(el.quote);
  }

  syncLabels();
  run();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
