# app.py — Statute-Barred Debt Predictor — Advanced Edition
# Streamlit frontend for the sklearn bundle pickle. No TensorFlow.
#   Pages:  Overview · Predict · Batch · Explorer · Insights · Settings
#   Bonus:  Demo mode with guided callouts for presenting.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

MODEL_CANDIDATES = [
    "statute_barred_model.pkl",
    "statute_barred_model (1).pkl",
    "statute_barred_model(1).pkl",
]
DATA_PATH = BASE / "accounts_sample.parquet"

NAVY_BG  = "#070b16"; PANEL_BG = "#0f1729"; PANEL_2 = "#141d33"
BORDER   = "rgba(120,150,200,0.18)"
TEXT     = "#eaf0fa"; MUTED = "#8ea3c2"
TEAL     = "#38e0c4"; CORAL = "#ff6b7a"; PURPLE = "#a78bfa"
AMBER    = "#ffb547"; GREEN = "#2ee6a8"; RED = "#ff5c6c"
GRID     = "rgba(120,150,200,0.10)"

st.set_page_config(
    page_title="Statute-Barred Debt Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family:'Inter', system-ui, sans-serif; }}
.stApp {{
  background:
    radial-gradient(900px 500px at 10% -10%, #16305c 0%, transparent 55%),
    radial-gradient(800px 600px at 95% 0%, #3a1f66 0%, transparent 50%),
    {NAVY_BG};
  color:{TEXT};
}}
#MainMenu, footer, header {{visibility:hidden;}}
.block-container {{padding-top:1.4rem; padding-bottom:3rem; max-width:1500px;}}

.hero {{
  position:relative; padding:30px 34px; border-radius:20px; margin-bottom:22px;
  overflow:hidden;
  background:linear-gradient(120deg,#101a35 0%,#1c2452 45%,#2a1d54 100%);
  border:1px solid {BORDER}; box-shadow:0 20px 60px rgba(0,0,0,.5);
}}
.hero::before {{
  content:""; position:absolute; inset:-2px;
  background:linear-gradient(120deg,
    rgba(56,224,196,.15), rgba(167,139,250,.15),
    rgba(255,107,122,.15), rgba(56,224,196,.15));
  background-size:300% 300%; animation:sweep 14s ease infinite;
  z-index:0; filter:blur(30px);
}}
@keyframes sweep {{
  0%{{background-position:0% 50%;}}
  50%{{background-position:100% 50%;}}
  100%{{background-position:0% 50%;}}
}}
.hero-content {{position:relative; z-index:1;}}
.hero h1 {{
  margin:0 0 6px; font-size:30px; font-weight:800; letter-spacing:-.6px;
  background:linear-gradient(90deg,{TEAL},{PURPLE},{CORAL});
  -webkit-background-clip:text; background-clip:text; color:transparent;
}}
.hero p {{margin:0; color:{MUTED}; font-size:14px;}}
.hero .badge {{
  display:inline-block; margin-top:12px; padding:4px 11px;
  font-size:11px; font-weight:600; letter-spacing:.5px;
  color:{TEAL}; background:rgba(56,224,196,.10);
  border:1px solid rgba(56,224,196,.30); border-radius:99px;
}}

.metric-card {{
  position:relative; padding:20px 22px; border-radius:16px;
  background:linear-gradient(180deg,{PANEL_BG},{PANEL_2});
  border:1px solid {BORDER}; box-shadow:0 12px 30px rgba(0,0,0,.4);
  overflow:hidden; height:100%;
}}
.metric-card::after {{
  content:""; position:absolute; top:0; left:0; height:3px; width:100%;
  background:linear-gradient(90deg,var(--accent,{TEAL}),transparent 90%);
}}
.metric-label {{
  font-size:11px; font-weight:600; text-transform:uppercase;
  letter-spacing:1.2px; color:{MUTED}; margin-bottom:6px;
}}
.metric-value {{
  font-size:28px; font-weight:800; letter-spacing:-.5px;
  color:{TEXT}; line-height:1.1;
}}
.metric-hint {{font-size:11.5px; color:{MUTED}; margin-top:6px;}}

.section-title {{font-size:15px; font-weight:700; margin:6px 0 2px; color:{TEXT};}}
.section-sub {{font-size:12px; color:{MUTED}; margin-bottom:12px;}}

.panel {{
  padding:20px 22px; border-radius:16px;
  background:linear-gradient(180deg,{PANEL_BG},{PANEL_2});
  border:1px solid {BORDER}; box-shadow:0 12px 30px rgba(0,0,0,.35);
  margin-bottom:18px;
}}

.demo-tip {{
  padding:12px 16px; border-radius:12px; margin:0 0 14px;
  background:linear-gradient(90deg, rgba(255,181,71,.12), rgba(255,181,71,.04));
  border:1px solid rgba(255,181,71,.35);
  color:#ffd894; font-size:12.5px; line-height:1.55;
}}
.demo-tip b {{color:#ffecb8;}}

section[data-testid="stSidebar"] {{
  background:linear-gradient(180deg,#0a1226 0%,#0c1428 100%);
  border-right:1px solid {BORDER};
}}
.stButton > button {{
  border-radius:11px; font-weight:600; border:1px solid {BORDER};
  background:linear-gradient(135deg, rgba(56,224,196,.18), rgba(167,139,250,.14));
  color:{TEXT}; transition:all .15s ease; padding:10px 18px;
}}
.stButton > button:hover {{
  border-color:{TEAL}; box-shadow:0 8px 24px rgba(56,224,196,.25);
  transform:translateY(-1px);
}}
.stDataFrame {{border:1px solid {BORDER} !important; border-radius:14px !important; overflow:hidden;}}
.stTabs [data-baseweb="tab-list"] {{gap:6px;}}
.stTabs [data-baseweb="tab"] {{
  background:rgba(120,150,200,0.06); border-radius:10px; padding:8px 16px;
  border:1px solid transparent; color:{MUTED}; font-weight:600; font-size:12.5px;
}}
.stTabs [aria-selected="true"] {{
  background:linear-gradient(135deg, rgba(56,224,196,.16), rgba(167,139,250,.12)) !important;
  border-color:rgba(56,224,196,.35) !important; color:{TEXT} !important;
}}
hr {{border-color:{BORDER}; opacity:.4;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def _find_model_path() -> Path:
    for name in MODEL_CANDIDATES:
        p = BASE / name
        if p.exists():
            return p
    st.error("**Model file not found.** Expected one of:\n\n"
             + "\n".join(f"- `{n}`" for n in MODEL_CANDIDATES))
    st.stop()


@st.cache_resource(show_spinner="Loading model bundle…")
def load_bundle() -> dict:
    obj = joblib.load(_find_model_path())
    if isinstance(obj, dict) and "model" in obj:
        required = ["encoder", "scaler", "train_medians",
                    "onehot_cols", "numerical_cols", "binary_features"]
        missing = [k for k in required if k not in obj]
        if missing:
            st.error(f"**Pickle missing keys:** {missing}")
            st.stop()
        obj.setdefault("contact_cols", ["NumPhones", "NumEmails", "NumAddresses"])
        obj.setdefault("threshold", 0.5)
        return obj
    if hasattr(obj, "predict_proba"):
        return {"model": obj, "kind": "bare_estimator", "threshold": 0.5}
    st.error(f"**Unrecognized pickle:** `{type(obj).__name__}`")
    st.stop()


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(DATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df_raw: pd.DataFrame, bundle: dict) -> np.ndarray:
    df = df_raw.copy()

    contact_cols = bundle["contact_cols"]
    if all(c in df.columns for c in contact_cols):
        df["TotalContactInfo"] = df[contact_cols].sum(axis=1)

    if "CurrentBalance" in df.columns and "DebtLoadPrincipal" in df.columns:
        denom = df["DebtLoadPrincipal"].replace(0, np.nan)
        df["BalanceToDebtRatio"] = (
            df["CurrentBalance"] / denom
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in bundle["binary_features"]:
        if col in df.columns:
            df[col] = df[col].map({"N": 0, "Y": 1, 1: 1, 0: 0})

    if "EntityID" in df.columns:
        df["EntityID"] = df["EntityID"].astype(str)

    num = (df[bundle["numerical_cols"]]
             .apply(pd.to_numeric, errors="coerce")
             .fillna(bundle["train_medians"]))
    X_num = bundle["scaler"].transform(num).astype(np.float32)
    X_bin = df[bundle["binary_features"]].to_numpy(dtype=np.float32)
    X_oh  = bundle["encoder"].transform(df[bundle["onehot_cols"]]).astype(np.float32)
    return np.hstack([X_num, X_bin, X_oh]).astype(np.float32)


def predict_proba(df_raw: pd.DataFrame, bundle: dict) -> np.ndarray:
    if df_raw.empty:
        return np.array([])
    if bundle.get("kind") == "bare_estimator":
        X = df_raw.copy()
        if "EntityID" in X.columns:
            X["EntityID"] = X["EntityID"].astype(str)
        for c in ("InBankruptcy", "IsLegal"):
            if c in X.columns:
                X[c] = X[c].map({"N": 0, "Y": 1, 1: 1, 0: 0})
        return np.asarray(bundle["model"].predict_proba(X)[:, 1])
    X = preprocess(df_raw, bundle)
    return np.asarray(bundle["model"].predict_proba(X)[:, 1])


def explain_row(row: pd.Series, bundle: dict, base_prob: float, top_k: int = 8) -> pd.DataFrame:
    """Local ablation: replace each feature with its median/mode, measure Δp."""
    df = pd.DataFrame([row])
    effects = []

    num_cols = bundle["numerical_cols"]
    cat_cols = bundle["onehot_cols"]

    for col in num_cols:
        if col not in df.columns:
            continue
        med = bundle["train_medians"].get(col, df[col].iloc[0])
        if pd.isna(df[col].iloc[0]):
            continue
        trial = df.copy()
        trial[col] = med
        new_p = float(predict_proba(trial, bundle)[0])
        effects.append({
            "feature": col,
            "value": row.get(col),
            "replacement": med,
            "delta": base_prob - new_p,
        })

    for col in cat_cols:
        if col not in df.columns:
            continue
        trial = df.copy()
        trial[col] = "MISSING"
        new_p = float(predict_proba(trial, bundle)[0])
        effects.append({
            "feature": col,
            "value": row.get(col),
            "replacement": "(missing)",
            "delta": base_prob - new_p,
        })

    out = pd.DataFrame(effects)
    out["abs_delta"] = out["delta"].abs()
    out = out.sort_values("abs_delta", ascending=False).head(top_k).drop(columns="abs_delta")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def hero(title: str, subtitle: str, badge: str = "scikit-learn · in-process inference"):
    st.markdown(f"""
        <div class="hero">
          <div class="hero-content">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <span class="badge">⚡ {badge}</span>
          </div>
        </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, hint="", accent=TEAL):
    st.markdown(f"""
        <div class="metric-card" style="--accent:{accent};">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-hint">{hint}</div>
        </div>
    """, unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="section-title">{title}</div>'
                f'<div class="section-sub">{sub}</div>',
                unsafe_allow_html=True)


def panel_open(): st.markdown('<div class="panel">', unsafe_allow_html=True)
def panel_close(): st.markdown('</div>', unsafe_allow_html=True)


def demo_tip(text: str, key: str = ""):
    if st.session_state.get("demo_mode"):
        st.markdown(f'<div class="demo-tip">🎬 <b>Demo:</b> {text}</div>',
                    unsafe_allow_html=True)


def style_fig(fig, height=340):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=PANEL_2, bordercolor=BORDER,
                        font=dict(color=TEXT, family="Inter, sans-serif")),
    )
    return fig


def prob_color(p):
    if p >= 0.75: return RED
    if p >= 0.50: return AMBER
    if p >= 0.25: return "#eab308"
    return GREEN


# ─────────────────────────────────────────────────────────────────────────────
# BOOT
# ─────────────────────────────────────────────────────────────────────────────
BUNDLE = load_bundle()
SAMPLE = load_sample()

if SAMPLE.empty:
    st.error("**`accounts_sample.parquet` missing.** Generate it in `train.py`.")
    st.stop()

THRESHOLD = float(BUNDLE.get("threshold", 0.5))

if "scored_df" not in st.session_state:
    with st.spinner("Scoring portfolio…"):
        probs = predict_proba(SAMPLE, BUNDLE)
    df = SAMPLE.copy()
    df["_prob"] = probs
    st.session_state["scored_df"] = df

DF = st.session_state["scored_df"].copy()

# Cached diagnostics (fit once per session)
if "diag" not in st.session_state:
    y_true = DF["IsStatBarred"].astype(int).values
    y_prob = DF["_prob"].values
    y_pred = (y_prob >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    # Threshold sweep (precision/recall/f1 vs threshold)
    thresholds = np.linspace(0.05, 0.95, 40)
    sweep = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        if pred.sum() == 0:
            continue
        sweep.append({
            "threshold": float(t),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall":    recall_score(y_true, pred, zero_division=0),
            "f1":        f1_score(y_true, pred, zero_division=0),
            "flagged":   int(pred.sum()),
        })
    sweep_df = pd.DataFrame(sweep)

    st.session_state["diag"] = {
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "auc": float(roc_auc_score(y_true, y_prob)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sweep": sweep_df,
    }

DIAG = st.session_state["diag"]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:6px 0 18px;">
              <div style="width:42px;height:42px;border-radius:12px;
                          background:linear-gradient(135deg,{TEAL},{PURPLE});
                          display:grid;place-items:center;font-size:20px;">⚖️</div>
              <div>
                <div style="font-size:15px;font-weight:800;letter-spacing:-.3px;">DebtScope</div>
                <div style="font-size:11px;color:{MUTED};">Statute-barred analytics</div>
              </div>
            </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["Overview", "Predict", "Batch Score", "Accounts Explorer",
             "Model Insights", "Settings"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            f"<div style='font-size:11px;font-weight:700;letter-spacing:1.2px;"
            f"text-transform:uppercase;color:{MUTED};margin-bottom:6px;'>Global threshold</div>",
            unsafe_allow_html=True,
        )
        new_t = st.slider("Decision threshold", 0.05, 0.95, THRESHOLD, 0.01,
                          format="%.2f", label_visibility="collapsed")
        if new_t != THRESHOLD:
            BUNDLE["threshold"] = new_t
            st.session_state["scored_df"]["_pred"] = (
                st.session_state["scored_df"]["_prob"] >= new_t
            ).astype(int)
            st.rerun()

        st.markdown("---")
        demo_on = st.toggle("🎬 Demo mode", value=st.session_state.get("demo_mode", False))
        st.session_state["demo_mode"] = demo_on
        if demo_on:
            st.caption("Guided callouts appear throughout the app.")

        st.markdown("---")
        st.markdown(
            f"<div style='color:{GREEN};font-weight:600;font-size:12.5px;'>● Model loaded</div>"
            f"<div style='color:{MUTED};font-size:11px;margin-top:4px;'>"
            f"{len(SAMPLE):,} accounts · {type(BUNDLE['model']).__name__}</div>"
            f"<div style='color:{MUTED};font-size:11px;'>"
            f"AUC {DIAG['auc']:.3f} · Acc {DIAG['acc']*100:.1f}%</div>",
            unsafe_allow_html=True,
        )

        return page


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_overview():
    hero("Predict Statute-Barred Debt with Deep Learning",
         "Identify time-barred accounts and optimize collection strategy.")
    demo_tip("Start here. Show KPIs → then hover the histogram → then the creditor chart.")

    total   = len(DF)
    barred  = int((DF["_prob"] >= THRESHOLD).sum())
    rate    = barred / total if total else 0
    high    = int((DF["_prob"] >= 0.80).sum())
    balance = float(DF["CurrentBalance"].sum())
    recover = balance * (1 - rate) * 0.35

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: metric_card("Total Accounts", f"{total:,}", "In sample", TEAL)
    with c2: metric_card("Statute-Barred", f"{rate*100:.1f}%",
                         f"≥ {THRESHOLD:.2f} threshold", CORAL)
    with c3: metric_card("High-Risk Accounts", f"{high:,}", "Probability ≥ 0.80", PURPLE)
    with c4: metric_card("Recoverable (est.)", f"${recover:,.0f}", "Net of risk", AMBER)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    r1l, r1r = st.columns([1.2, 1], gap="large")

    with r1l:
        panel_open()
        section("Probability distribution", "Model confidence across the sample")
        hist = go.Figure(go.Histogram(
            x=DF["_prob"], nbinsx=30,
            marker=dict(color=DF["_prob"],
                        colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
                        line=dict(width=0)),
            hovertemplate="Probability %{x:.2f}<br>Count %{y}<extra></extra>",
        ))
        hist.add_vline(x=THRESHOLD, line_dash="dash", line_color=TEAL,
                       annotation_text=f"threshold {THRESHOLD:.2f}",
                       annotation_position="top right")
        hist.update_layout(xaxis_title="Probability", yaxis_title="Accounts", bargap=.04)
        st.plotly_chart(style_fig(hist, 320), use_container_width=True)
        panel_close()

    with r1r:
        panel_open()
        section("Risk buckets", "Accounts grouped by probability band")
        buckets = pd.cut(DF["_prob"], [0, .25, .5, .75, 1.0],
                         labels=["Low", "Medium", "High", "Critical"],
                         include_lowest=True)
        counts = (buckets.value_counts()
                        .reindex(["Low","Medium","High","Critical"]).reset_index())
        counts.columns = ["Band", "Count"]
        bar = go.Figure(go.Bar(
            x=counts["Band"], y=counts["Count"],
            marker=dict(color=[GREEN, "#eab308", AMBER, RED]),
            text=counts["Count"], textposition="outside",
        ))
        bar.update_layout(xaxis_title="", yaxis_title="Accounts", showlegend=False)
        st.plotly_chart(style_fig(bar, 320), use_container_width=True)
        panel_close()

    r2l, r2r = st.columns([1.3, 1], gap="large")

    with r2l:
        panel_open()
        section("By creditor", "Account count coloured by average probability")
        if "OriginalCreditor[Redacted]" in DF.columns:
            grp = (DF.groupby("OriginalCreditor[Redacted]")
                     .agg(accounts=("_prob", "size"), avg=("_prob", "mean"))
                     .reset_index()
                     .sort_values("accounts", ascending=False).head(12))
            cb = go.Figure(go.Bar(
                x=grp["OriginalCreditor[Redacted]"], y=grp["accounts"],
                marker=dict(color=grp["avg"],
                            colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
                            cmin=0, cmax=1, line=dict(width=0)),
                hovertemplate="%{x}<br>Accounts %{y}<br>Avg prob %{marker.color:.2f}<extra></extra>",
            ))
            cb.update_layout(xaxis_title="", yaxis_title="Accounts",
                             coloraxis_showscale=False)
            st.plotly_chart(style_fig(cb, 320), use_container_width=True)
        else:
            st.info("No creditor column.")
        panel_close()

    with r2r:
        panel_open()
        section("By collection status", "Barred rate per status")
        if "CollectionStatus" in DF.columns:
            d = DF.copy()
            d["_barred"] = (d["_prob"] >= THRESHOLD).astype(int)
            grp = (d.groupby("CollectionStatus")
                     .agg(rate=("_barred", "mean"), n=("_barred", "size"))
                     .reset_index().sort_values("n", ascending=False))
            fig = go.Figure(go.Bar(
                x=grp["CollectionStatus"], y=grp["rate"] * 100,
                marker=dict(color=grp["rate"],
                            colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
                            cmin=0, cmax=1, line=dict(width=0)),
                hovertemplate="%{x}<br>%{y:.1f}% barred<br>n=%{customdata}<extra></extra>",
                customdata=grp["n"],
            ))
            fig.update_layout(xaxis_title="", yaxis_title="Barred %",
                              yaxis=dict(range=[0, 100]))
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
        else:
            st.info("No status column.")
        panel_close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def page_predict():
    hero("Live Prediction", "Score a single account interactively.", "Real-time inference")
    demo_tip("Fill the form or click 'Random sample' → hit Predict → show the gauge + local explanation.")

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        panel_open()
        section("Account details", "Every field is optional — defaults fill in automatically")

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.caption("Or load a random account from the sample to auto-fill")
        with col_b:
            if st.button("🎲 Random sample", use_container_width=True):
                row = DF.sample(1).iloc[0]
                for col in DF.columns:
                    if col.startswith("_"):
                        continue
                    st.session_state[f"inp_{col}"] = row[col]
                st.rerun()

        cols = list(DF.columns)
        input_cols = [c for c in cols if not c.startswith("_") and c != "IsStatBarred"]

        with st.form("predict_form", clear_on_submit=False):
            values = {}
            g1, g2 = st.columns(2)
            for i, col in enumerate(input_cols):
                target = g1 if i % 2 == 0 else g2
                key = f"inp_{col}"
                sample_val = DF[col].iloc[0]
                default = st.session_state.get(key, sample_val)
                with target:
                    if pd.api.types.is_numeric_dtype(DF[col]):
                        values[col] = st.number_input(
                            col,
                            value=float(default) if pd.notna(default) else 0.0,
                            step=1.0 if "Num" in col or "Age" in col or "Parties" in col else 0.01,
                            key=key,
                        )
                    else:
                        options = sorted(DF[col].dropna().astype(str).unique())
                        if default not in options and pd.notna(default):
                            options = [str(default)] + options
                        idx = options.index(str(default)) if str(default) in options else 0
                        values[col] = st.selectbox(col, options, index=idx, key=key)

            submitted = st.form_submit_button("Predict →", use_container_width=True)

        panel_close()

        if submitted:
            row_df = pd.DataFrame([values])
            try:
                p = float(predict_proba(row_df, BUNDLE)[0])
                st.session_state["last_prediction"] = {
                    "input": values,
                    "prob": p,
                    "row": row_df.iloc[0],
                }
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with right:
        panel_open()
        section("Result", "Model verdict and probability")

        last = st.session_state.get("last_prediction")
        if not last:
            st.info("Fill the form on the left and click **Predict** to see results here.")
            panel_close()
            return

        p = last["prob"]
        barred = p >= THRESHOLD
        color = prob_color(p)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number=dict(suffix="%", font=dict(size=42, color=TEXT, family="Inter")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=MUTED,
                          tickfont=dict(color=MUTED, size=10)),
                bar=dict(color=color, thickness=0.28),
                bgcolor="rgba(120,150,200,0.10)", borderwidth=0,
                steps=[
                    dict(range=[0, 25],   color="rgba(46,230,168,0.10)"),
                    dict(range=[25, 50],  color="rgba(234,179,8,0.10)"),
                    dict(range=[50, 75],  color="rgba(255,181,71,0.10)"),
                    dict(range=[75, 100], color="rgba(255,92,108,0.10)"),
                ],
            ),
        ))
        gauge.update_layout(
            height=240, margin=dict(l=20, r=20, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color=TEXT),
        )
        st.plotly_chart(gauge, use_container_width=True)

        verdict = "STATUTE-BARRED" if barred else "NOT statute-barred"
        st.markdown(f"""
            <div style="text-align:center;padding:12px;border-radius:12px;
                        background:linear-gradient(90deg,{color}22,transparent);
                        border:1px solid {color}55;">
              <div style="font-size:13.5px;font-weight:700;color:{color};
                          letter-spacing:.4px;">{verdict}</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        with st.expander("🔍 Local explanation (what drove this prediction)"):
            expl = explain_row(last["row"], BUNDLE, p, top_k=8)
            fig = go.Figure(go.Bar(
                x=expl["delta"],
                y=expl["feature"],
                orientation="h",
                marker=dict(color=expl["delta"],
                            colorscale=[[0, GREEN], [0.5, MUTED], [1, RED]],
                            cmid=0, line=dict(width=0)),
                hovertemplate="%{y}<br>Δp = %{x:+.3f}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Δp (positive = pushes toward barred)",
                              yaxis_title="", showlegend=False,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)

        panel_close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — BATCH SCORE
# ─────────────────────────────────────────────────────────────────────────────
def page_batch():
    hero("Batch Scoring", "Upload a CSV and score every row with the model.", "Server-side inference")
    demo_tip("Upload a CSV (any columns) → see how many are flagged → download the scored output.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV with the same columns as the training data.")
        return

    try:
        df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    panel_open()
    section("Preview", f"{len(df_in):,} rows × {len(df_in.columns)} columns")
    st.dataframe(df_in.head(10), use_container_width=True)
    panel_close()

    if st.button("Score uploaded data", use_container_width=True):
        try:
            probs = predict_proba(df_in, BUNDLE)
        except Exception as e:
            st.error(f"Scoring failed: {e}")
            return

        out = df_in.copy()
        out["statute_barred_probability"] = probs.round(4)
        out["predicted_is_statute_barred"] = (probs >= THRESHOLD)
        out["risk_band"] = pd.cut(probs, [0, .25, .5, .75, 1.0],
                                  labels=["Low","Medium","High","Critical"],
                                  include_lowest=True).astype(str)

        panel_open()
        section("Summary",
                f"Flagged {int(out['predicted_is_statute_barred'].sum())} of {len(out)} rows "
                f"at threshold {THRESHOLD:.2f}")

        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Rows scored", f"{len(out):,}", "", TEAL)
        with c2: metric_card("Flagged",
                             f"{int(out['predicted_is_statute_barred'].sum()):,}",
                             f"{out['predicted_is_statute_barred'].mean()*100:.1f}% of rows",
                             CORAL)
        with c3: metric_card("Mean probability", f"{probs.mean():.3f}", "", PURPLE)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.dataframe(out.head(50), use_container_width=True)

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download scored CSV", csv,
                           file_name="scored_accounts.csv", mime="text/csv",
                           use_container_width=True)
        panel_close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def page_explorer():
    hero("Accounts Explorer", "Filter, sort, and inspect scored accounts.",
         "Live filtering")
    demo_tip("Narrow the probability range → then search a creditor → click into a row.")

    panel_open()
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    with f1: search = st.text_input("Search", placeholder="Account ID or creditor…")
    with f2: band = st.selectbox("Risk band", ["All", "Low", "Medium", "High", "Critical"])
    with f3: only_barred = st.checkbox("Only flagged", value=False)
    with f4: size = st.select_slider("Rows", [25, 50, 100, 200], value=50)

    d = DF.copy()
    if search:
        s = search.lower()
        d = d[d.astype(str).apply(lambda r: r.str.lower().str.contains(s).any(), axis=1)]
    if band != "All":
        buckets = pd.cut(d["_prob"], [0, .25, .5, .75, 1.0],
                         labels=["Low", "Medium", "High", "Critical"],
                         include_lowest=True).astype(str)
        d = d[buckets == band]
    if only_barred:
        d = d[d["_prob"] >= THRESHOLD]

    total = len(d)
    page = st.session_state.get("exp_page", 1)
    pages = max(1, (total + size - 1) // size)
    page = min(page, pages)
    rows = d.iloc[(page - 1) * size : page * size].copy()

    section(f"{total:,} accounts match", f"Page {page} of {pages}")

    if rows.empty:
        st.info("No accounts match.")
        panel_close()
        return

    display = pd.DataFrame({
        "Account ID":  rows["AccountID"] if "AccountID" in rows.columns else rows.index.astype(str),
        "Creditor":    rows.get("OriginalCreditor[Redacted]"),
        "Balance":     rows["CurrentBalance"].apply(lambda v: f"${v:,.2f}"),
        "Status":      rows.get("CollectionStatus"),
        "Age":         rows.get("CustomerAge"),
        "Probability": rows["_prob"].astype(float),
        "Flagged":     rows["_prob"].apply(lambda p: "Yes" if p >= THRESHOLD else "No"),
    })

    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability", format="%.2f", min_value=0.0, max_value=1.0,
            ),
            "Balance": st.column_config.TextColumn("Balance", width="small"),
        },
        height=min(650, 60 + 38 * len(display)),
    )

    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if st.button("← Previous", disabled=page <= 1, use_container_width=True):
            st.session_state["exp_page"] = page - 1
            st.rerun()
    with c2:
        st.markdown(
            f"<div style='text-align:center;color:{MUTED};font-size:12.5px;padding-top:8px;'>"
            f"Page <b style='color:{TEXT}'>{page}</b> of {pages}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        if st.button("Next →", disabled=page >= pages, use_container_width=True):
            st.session_state["exp_page"] = page + 1
            st.rerun()
    panel_close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def page_insights():
    hero("Model Insights", "Diagnostics, threshold tuning, and feature importance.",
         "Evaluation on the sample")
    demo_tip("This page sells the model. Show the ROC curve → threshold sweep → feature importance.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Performance", "🎯 Threshold sweep", "🔬 Feature importance", "🧩 Confusion matrix"]
    )

    with tab1:
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1: metric_card("ROC-AUC", f"{DIAG['auc']:.4f}", "Discrimination", TEAL)
        with c2: metric_card("Accuracy", f"{DIAG['acc']*100:.2f}%", "Correct predictions", GREEN)
        with c3: metric_card("Precision", f"{DIAG['precision']:.3f}", "Flagged correctly", AMBER)
        with c4: metric_card("Recall", f"{DIAG['recall']:.3f}", "Caught of all barred", CORAL)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        panel_open()
        section("ROC curve", "True positive rate vs. false positive rate")
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(
            x=DIAG["fpr"], y=DIAG["tpr"], mode="lines",
            line=dict(color=TEAL, width=3), name=f"AUC = {DIAG['auc']:.4f}",
        ))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=MUTED, width=1, dash="dash"), name="Random",
        ))
        roc_fig.update_layout(
            xaxis_title="False positive rate", yaxis_title="True positive rate",
            xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(style_fig(roc_fig, 380), use_container_width=True)
        panel_close()

    with tab2:
        panel_open()
        section("Precision / Recall vs. threshold",
                "Trade-off as you move the decision threshold")
        sweep = DIAG["sweep"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["precision"],
                                 mode="lines+markers", name="Precision",
                                 line=dict(color=TEAL, width=2.5)))
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["recall"],
                                 mode="lines+markers", name="Recall",
                                 line=dict(color=CORAL, width=2.5)))
        fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["f1"],
                                 mode="lines+markers", name="F1",
                                 line=dict(color=PURPLE, width=2, dash="dot")))
        fig.add_vline(x=THRESHOLD, line_dash="dash", line_color=AMBER,
                      annotation_text=f"current {THRESHOLD:.2f}",
                      annotation_position="top right")
        fig.update_layout(xaxis_title="Threshold", yaxis_title="Score",
                          yaxis=dict(range=[0, 1]))
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

        st.caption(f"At the current threshold of **{THRESHOLD:.2f}**, "
                   f"**{int((DF['_prob']>=THRESHOLD).sum()):,}** of {len(DF):,} accounts are flagged.")
        panel_close()

    with tab3:
        panel_open()
        section("Permutation feature importance",
                "Drop in ROC-AUC when each feature is shuffled (higher = more important)")

        if st.button("Compute importances (may take ~10s)", use_container_width=True):
            with st.spinner("Running permutation importance…"):
                sub = DF.sample(min(300, len(DF)), random_state=42).copy()
                X = preprocess(sub, BUNDLE)
                y = sub["IsStatBarred"].astype(int).values
                result = permutation_importance(
                    BUNDLE["model"], X, y,
                    n_repeats=5, random_state=42, n_jobs=-1, scoring="roc_auc",
                )
                feature_names = (
                    list(BUNDLE["numerical_cols"])
                    + list(BUNDLE["binary_features"])
                    + list(BUNDLE["encoder"].get_feature_names_out(BUNDLE["onehot_cols"]))
                )
                imp = pd.DataFrame({
                    "feature": feature_names,
                    "mean": result.importances_mean,
                    "std":  result.importances_std,
                }).sort_values("mean", ascending=False).head(20)
                st.session_state["importance"] = imp

        imp = st.session_state.get("importance")
        if imp is not None:
            fig = go.Figure(go.Bar(
                x=imp["mean"], y=imp["feature"], orientation="h",
                marker=dict(color=imp["mean"],
                            colorscale=[[0, MUTED], [1, TEAL]], line=dict(width=0)),
                error_x=dict(type="data", array=imp["std"], color=MUTED),
                hovertemplate="%{y}<br>Δ AUC = %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Drop in ROC-AUC", yaxis_title="",
                              yaxis=dict(autorange="reversed"), showlegend=False)
            st.plotly_chart(style_fig(fig, 480), use_container_width=True)
        panel_close()

    with tab4:
        panel_open()
        section("Confusion matrix",
                f"Evaluated at threshold {THRESHOLD:.2f} on the sample")
        cm = DIAG["cm"]
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted: N", "Predicted: Y"],
            y=["Actual: N", "Actual: Y"],
            colorscale=[[0, "rgba(120,150,200,0.05)"], [1, TEAL]],
            text=cm, texttemplate="%{text}",
            hovertemplate="%{y} · %{x}<br>Count %{z}<extra></extra>",
            showscale=False,
        ))
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        st.markdown(
            f"**True negatives** {tn:,} · **False positives** {fp:,} · "
            f"**False negatives** {fn:,} · **True positives** {tp:,}"
        )
        panel_close()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE — SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
def page_settings():
    hero("Settings & About", "Configuration, model info, and reset options.",
         "Runtime")

    left, right = st.columns([1, 1], gap="large")

    with left:
        panel_open()
        section("Runtime configuration", "Values loaded in this session")
        st.markdown(f"""
            <div style="padding:14px;border-radius:12px;background:rgba(120,150,200,.06);
                        border:1px solid {BORDER};margin-bottom:10px;">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
                          color:{MUTED};">Threshold</div>
              <div style="font-size:26px;font-weight:800;color:{TEAL};">{THRESHOLD:.2f}</div>
            </div>
            <div style="padding:14px;border-radius:12px;background:rgba(120,150,200,.06);
                        border:1px solid {BORDER};margin-bottom:10px;">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
                          color:{MUTED};">Model</div>
              <div style="font-size:13.5px;color:{TEXT};">
                {type(BUNDLE['model']).__name__}<br>
                <span style="color:{MUTED};font-size:12px;">
                  {len(BUNDLE['encoder'].get_feature_names_out(BUNDLE['onehot_cols']))} one-hot features
                </span>
              </div>
            </div>
            <div style="padding:14px;border-radius:12px;background:rgba(120,150,200,.06);
                        border:1px solid {BORDER};">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
                          color:{MUTED};">Sample loaded</div>
              <div style="font-size:13.5px;color:{TEXT};">
                {len(SAMPLE):,} accounts · {len(SAMPLE.columns)} columns
              </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Re-score portfolio", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k in ("scored_df", "diag"):
                    del st.session_state[k]
            st.success("Cache cleared. Rebooting…")
            st.rerun()
        panel_close()

    with right:
        panel_open()
        section("About", "What this app does")
        st.markdown(f"""
        <div style="font-size:13px;line-height:1.65;color:{MUTED};">
          <b style="color:{TEXT}">DebtScope</b> predicts whether a debt account is
          likely to be <b style="color:{TEXT}">statute-barred</b> — meaning the
          legal limitation period may have expired and enforcement could be
          restricted.<br><br>

          <b style="color:{TEXT}">Model:</b> gradient-boosted classifier trained on
          historical account data with features covering balances, contact
          information, collection status, and customer attributes.<br><br>

          <b style="color:{TEXT}">Pipeline:</b> all preprocessing (imputation,
          one-hot encoding, scaling) is bundled into the pickle and applied at
          inference time so predictions match training exactly.<br><br>

          <b style="color:{TEXT}">Disclaimer:</b> this is a statistical estimate
          for portfolio prioritisation only — not legal advice. Always confirm
          limitation periods under the applicable jurisdiction.
        </div>
        """, unsafe_allow_html=True)
        panel_close()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.session_state.setdefault("exp_page", 1)
    st.session_state.setdefault("demo_mode", False)

    page = sidebar()

    if page == "Overview":
        page_overview()
    elif page == "Predict":
        page_predict()
    elif page == "Batch Score":
        page_batch()
    elif page == "Accounts Explorer":
        page_explorer()
    elif page == "Model Insights":
        page_insights()
    elif page == "Settings":
        page_settings()


if __name__ == "__main__":
    main()
