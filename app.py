# app.py — Statute-Barred Debt Predictor
# Loads statute_barred_model.pkl and runs all inference in-process.
# Deploy directly on Streamlit Cloud — no backend required.

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
MODEL_PATH = BASE / "statute_barred_model.pkl"
PREP_PATH  = BASE / "preprocessor.pkl"          # optional fallback
DATA_PATH  = BASE / "accounts_sample.parquet"

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
.block-container {{padding-top:1.6rem; padding-bottom:3rem; max-width:1500px;}}

.hero {{
  position:relative; padding:34px 38px; border-radius:20px; margin-bottom:26px;
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
  margin:0 0 8px; font-size:32px; font-weight:800; letter-spacing:-.6px;
  background:linear-gradient(90deg,{TEAL},{PURPLE},{CORAL});
  -webkit-background-clip:text; background-clip:text; color:transparent;
}}
.hero p {{margin:0; color:{MUTED}; font-size:14.5px;}}
.hero .badge {{
  display:inline-block; margin-top:14px; padding:5px 12px;
  font-size:11.5px; font-weight:600; letter-spacing:.5px;
  color:{TEAL}; background:rgba(56,224,196,.10);
  border:1px solid rgba(56,224,196,.30); border-radius:99px;
}}

.metric-card {{
  position:relative; padding:22px 24px; border-radius:16px;
  background:linear-gradient(180deg,{PANEL_BG},{PANEL_2});
  border:1px solid {BORDER}; box-shadow:0 12px 30px rgba(0,0,0,.4);
  overflow:hidden; height:100%;
}}
.metric-card::after {{
  content:""; position:absolute; top:0; left:0; height:3px; width:100%;
  background:linear-gradient(90deg,var(--accent,{TEAL}),transparent 90%);
}}
.metric-label {{
  font-size:11.5px; font-weight:600; text-transform:uppercase;
  letter-spacing:1.2px; color:{MUTED}; margin-bottom:8px;
}}
.metric-value {{
  font-size:30px; font-weight:800; letter-spacing:-.5px;
  color:{TEXT}; line-height:1.1;
}}
.metric-hint {{font-size:12px; color:{MUTED}; margin-top:6px;}}

.section-title {{font-size:15px; font-weight:700; margin:6px 0 2px; color:{TEXT};}}
.section-sub {{font-size:12.5px; color:{MUTED}; margin-bottom:14px;}}

.panel {{
  padding:22px 24px; border-radius:16px;
  background:linear-gradient(180deg,{PANEL_BG},{PANEL_2});
  border:1px solid {BORDER}; box-shadow:0 12px 30px rgba(0,0,0,.35);
  margin-bottom:20px;
}}

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
hr {{border-color:{BORDER}; opacity:.4;}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model bundle…")
def load_bundle():
    """Load statute_barred_model.pkl and detect its shape."""
    if not MODEL_PATH.exists():
        st.error(
            f"**Model file missing:** `{MODEL_PATH.name}`\n\n"
            "Commit `statute_barred_model.pkl` to the repo root."
        )
        st.stop()

    obj = joblib.load(MODEL_PATH)

    # Case 1: dict bundle — {"model": ..., "encoder": ..., "scaler": ...}
    if isinstance(obj, dict) and "model" in obj:
        bundle = obj
        model  = bundle["model"]
        prep   = bundle          # the dict itself carries prep info
        return {"model": model, "prep": prep, "kind": "dict_bundle"}

    # Case 2: sklearn Pipeline or bare estimator — preprocessing inside
    if hasattr(obj, "predict_proba") or hasattr(obj, "predict"):
        model = obj
        # Try to load a separate preprocessor if present
        if PREP_PATH.exists():
            prep = joblib.load(PREP_PATH)
        else:
            prep = None
        return {"model": model, "prep": prep, "kind": "pipeline"}

    # Case 3: Keras model pickled (unusual)
    if hasattr(obj, "predict") and hasattr(obj, "save"):
        model = obj
        prep = joblib.load(PREP_PATH) if PREP_PATH.exists() else None
        return {"model": model, "prep": prep, "kind": "keras"}

    st.error(
        f"**Unrecognized model format:** {type(obj).__name__}\n\n"
        "Expected either a dict `{'model': ...}`, an sklearn Pipeline, "
        "or a Keras model."
    )
    st.stop()

@st.cache_data(show_spinner=False)
def load_sample():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(DATA_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# Mirrors train.py exactly when a prep dict is available.
# If the pickle is a full sklearn Pipeline, we skip straight to predict.
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df_raw: pd.DataFrame, prep: dict) -> np.ndarray:
    df = df_raw.copy()

    # Feature engineering
    contact_cols = prep.get("contact_cols", ["NumPhones", "NumEmails", "NumAddresses"])
    if all(c in df.columns for c in contact_cols):
        df["TotalContactInfo"] = df[contact_cols].sum(axis=1)

    if "CurrentBalance" in df.columns and "DebtLoadPrincipal" in df.columns:
        denom = df["DebtLoadPrincipal"].replace(0, np.nan)
        df["BalanceToDebtRatio"] = (
            df["CurrentBalance"] / denom
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Binary encoding
    for col in prep.get("binary_features", ["InBankruptcy", "IsLegal"]):
        if col in df.columns:
            df[col] = df[col].map({"N": 0, "Y": 1, 1: 1, 0: 0})

    # EntityID as string
    if "EntityID" in df.columns:
        df["EntityID"] = df["EntityID"].astype(str)

    # Numerical + scaling
    num = (
        df[prep["numerical_cols"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(prep["train_medians"])
    )
    X_num = prep["scaler"].transform(num).astype(np.float32)

    # Binary
    X_bin = df[prep["binary_features"]].to_numpy(dtype=np.float32)

    # One-hot
    X_oh = prep["encoder"].transform(df[prep["onehot_cols"]]).astype(np.float32)

    return np.hstack([X_num, X_bin, X_oh]).astype(np.float32)


def score_dataframe(df_raw: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Score a DataFrame using whatever form the pickle takes."""
    if df_raw.empty:
        return np.array([])

    model = bundle["model"]
    prep  = bundle["prep"]
    kind  = bundle["kind"]

    if kind == "pipeline":
        # sklearn Pipeline handles preprocessing internally
        X = df_raw.copy()
        if "EntityID" in X.columns:
            X["EntityID"] = X["EntityID"].astype(str)
        for col in ("InBankruptcy", "IsLegal"):
            if col in X.columns:
                X[col] = X[col].map({"N": 0, "Y": 1, 1: 1, 0: 0})
        # Feature engineering must match training if the Pipeline expects them
        if all(c in X.columns for c in ["NumPhones", "NumEmails", "NumAddresses"]):
            X["TotalContactInfo"] = X[["NumPhones", "NumEmails", "NumAddresses"]].sum(axis=1)
        if "CurrentBalance" in X.columns and "DebtLoadPrincipal" in X.columns:
            denom = X["DebtLoadPrincipal"].replace(0, np.nan)
            X["BalanceToDebtRatio"] = (
                X["CurrentBalance"] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0)
        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception:
            # Some Keras-wrapped pickles only expose .predict
            probs = model.predict(X).ravel()
        return np.asarray(probs)

    # dict_bundle or keras → use the prep dict
    X = preprocess(df_raw, prep)
    try:
        probs = model.predict_proba(X)[:, 1]
    except AttributeError:
        probs = model.predict(X, verbose=0).ravel() if hasattr(model, "predict") else model.predict(X)
        probs = np.asarray(probs).ravel()
    return np.asarray(probs)

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def hero(title: str, subtitle: str, badge: str = "Pickle · in-process inference"):
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
    st.markdown(
        f'<div class="section-title">{title}</div>'
        f'<div class="section-sub">{sub}</div>',
        unsafe_allow_html=True,
    )

def panel_open(): st.markdown('<div class="panel">', unsafe_allow_html=True)
def panel_close(): st.markdown('</div>', unsafe_allow_html=True)

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
    st.error(
        "**accounts_sample.parquet missing.** Generate it at the end of `train.py`:\n\n"
        "```python\n"
        'df["AccountID"] = ["ACC" + str(i).zfill(6) for i in range(len(df))]\n'
        'sample = df.groupby("IsStatBarred", group_keys=False).apply(\n'
        '    lambda g: g.sample(min(len(g), 500), random_state=42)\n'
        ').reset_index(drop=True)\n'
        'sample.to_parquet("accounts_sample.parquet", index=False)\n'
        "```"
    )
    st.stop()

THRESHOLD = BUNDLE["prep"].get("threshold", 0.5) if isinstance(BUNDLE["prep"], dict) else 0.5

# Score the sample once per session
cache_key = f"scored_{BUNDLE['kind']}_{id(BUNDLE['model'])}_{THRESHOLD}"
if cache_key not in st.session_state:
    with st.spinner("Scoring portfolio…"):
        probs = score_dataframe(SAMPLE, BUNDLE)
    df_scored = SAMPLE.copy()
    df_scored["_prob"] = probs
    st.session_state[cache_key] = df_scored
    st.session_state["_current_threshold"] = THRESHOLD
    st.session_state["_active_cache_key"] = cache_key

DF = st.session_state[st.session_state["_active_cache_key"]].copy()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:6px 0 20px;">
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
            ["Dashboard", "Accounts Explorer", "Account Details", "Model & Settings"],
            label_visibility="collapsed",
        )

        filters = {}
        if page == "Accounts Explorer":
            st.markdown("---")
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;letter-spacing:1.2px;"
                f"text-transform:uppercase;color:{MUTED};margin-bottom:8px;'>Filters</div>",
                unsafe_allow_html=True,
            )
            filters["search"] = st.text_input("Search", placeholder="Account ID or creditor…")
            filters["min_prob"], filters["max_prob"] = st.slider(
                "Probability range", 0.0, 1.0, (0.0, 1.0), 0.01
            )
            filters["min_balance"] = st.number_input("Min balance", 0.0, value=0.0, step=100.0)
            filters["max_balance"] = st.number_input("Max balance", 0.0, value=1_000_000.0, step=1000.0)
            filters["only_barred"] = st.checkbox("Show only predicted statute-barred", value=False)
            filters["page_size"]   = st.select_slider("Rows per page", [25, 50, 100], value=50)

        st.markdown("---")
        st.markdown(
            f"<div style='color:{GREEN};font-weight:600;font-size:12.5px;'>● Model loaded</div>"
            f"<div style='color:{MUTED};font-size:11px;margin-top:4px;'>"
            f"{len(SAMPLE):,} accounts · format: {BUNDLE['kind'].replace('_',' ')}</div>",
            unsafe_allow_html=True,
        )

        return page, filters

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard():
    hero(
        "Predict Statute-Barred Debt with Deep Learning",
        "Identify time-barred accounts and optimize collection strategy.",
    )

    total   = len(DF)
    barred  = int((DF["_prob"] >= THRESHOLD).sum())
    rate    = barred / total if total else 0
    high    = int((DF["_prob"] >= 0.80).sum())
    balance = float(DF["CurrentBalance"].sum())
    recover = balance * (1 - rate) * 0.35

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1: metric_card("Total Accounts", f"{total:,}", "In sample", TEAL)
    with c2: metric_card("Statute-Barred", f"{rate*100:.1f}%", "Above threshold", CORAL)
    with c3: metric_card("High-Risk Accounts", f"{high:,}", "Probability ≥ 0.80", PURPLE)
    with c4: metric_card("Recoverable (est.)", f"${recover:,.0f}", "Net of risk", AMBER)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    r1l, r1r = st.columns([1.2, 1], gap="large")

    with r1l:
        panel_open()
        section("Probability distribution", "Model confidence across the sample")
        hist = go.Figure(go.Histogram(
            x=DF["_prob"], nbinsx=30,
            marker=dict(
                color=DF["_prob"],
                colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
                line=dict(width=0),
            ),
            hovertemplate="Probability %{x:.2f}<br>Count %{y}<extra></extra>",
        ))
        hist.update_layout(xaxis_title="Probability", yaxis_title="Accounts", bargap=.04)
        st.plotly_chart(style_fig(hist, 340), use_container_width=True)
        panel_close()

    with r1r:
        panel_open()
        section("Risk buckets", "Accounts grouped by probability band")
        buckets = pd.cut(DF["_prob"], [0, .25, .5, .75, 1.0],
                         labels=["Low", "Medium", "High", "Critical"], include_lowest=True)
        counts = buckets.value_counts().reindex(["Low","Medium","High","Critical"]).reset_index()
        counts.columns = ["Band", "Count"]
        bar = go.Figure(go.Bar(
            x=counts["Band"], y=counts["Count"],
            marker=dict(color=[GREEN, "#eab308", AMBER, RED]),
            text=counts["Count"], textposition="outside",
        ))
        bar.update_layout(xaxis_title="", yaxis_title="Accounts", showlegend=False)
        st.plotly_chart(style_fig(bar, 340), use_container_width=True)
        panel_close()

    r2l, r2r = st.columns([1.3, 1], gap="large")

    with r2l:
        panel_open()
        section("By creditor", "Average probability per creditor")
        if "OriginalCreditor[Redacted]" in DF.columns:
            grp = (DF.groupby("OriginalCreditor[Redacted]")
                     .agg(accounts=("_prob", "size"), avg=("_prob", "mean"))
                     .reset_index().sort_values("accounts", ascending=False).head(12))
            cb = go.Figure(go.Bar(
                x=grp["OriginalCreditor[Redacted]"], y=grp["accounts"],
                marker=dict(color=grp["avg"],
                            colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
                            cmin=0, cmax=1, line=dict(width=0)),
                hovertemplate="%{x}<br>Accounts %{y}<br>Avg prob %{marker.color:.2f}<extra></extra>",
            ))
            cb.update_layout(xaxis_title="", yaxis_title="Accounts", coloraxis_showscale=False)
            st.plotly_chart(style_fig(cb, 340), use_container_width=True)
        else:
            st.info("No creditor column in sample.")
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
                hovertemplate="%{x}<br>%{y:.1f}% barred<extra></extra>",
            ))
            fig.update_layout(xaxis_title="", yaxis_title="Barred %", yaxis=dict(range=[0, 100]))
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
        else:
            st.info("No status column in sample.")
        panel_close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def page_explorer(filters):
    hero("Accounts Explorer", "Search, filter, and inspect scored accounts.", "Live inference")

    d = DF.copy()
    if filters.get("search"):
        s = filters["search"].lower()
        d = d[d.astype(str).apply(lambda r: r.str.lower().str.contains(s).any(), axis=1)]
    d = d[(d["_prob"] >= filters.get("min_prob", 0)) & (d["_prob"] <= filters.get("max_prob", 1))]
    d = d[(d["CurrentBalance"] >= filters.get("min_balance", 0)) &
          (d["CurrentBalance"] <= filters.get("max_balance", 10**9))]
    if filters.get("only_barred"):
        d = d[d["_prob"] >= THRESHOLD]

    size  = filters.get("page_size", 50)
    total = len(d)
    page  = st.session_state.get("explorer_page", 1)
    pages = max(1, (total + size - 1) // size)
    page  = min(page, pages)
    rows  = d.iloc[(page - 1) * size : page * size].copy()

    panel_open()
    section(f"{total:,} accounts match", f"Page {page} of {pages}")

    if rows.empty:
        st.info("No accounts match the current filters.")
        panel_close()
        return

    display = pd.DataFrame({
        "Account ID":  rows["AccountID"] if "AccountID" in rows.columns else rows.index.astype(str),
        "Creditor":    rows.get("OriginalCreditor[Redacted]"),
        "Balance":     rows["CurrentBalance"].apply(lambda v: f"${v:,.2f}"),
        "Status":      rows.get("CollectionStatus"),
        "Age":         rows.get("CustomerAge"),
        "Probability": rows["_prob"].astype(float),
        "Predicted":   rows["_prob"].apply(
            lambda p: "Statute-Barred" if p >= THRESHOLD else "Active"
        ),
    })

    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability", format="%.2f", min_value=0.0, max_value=1.0,
            ),
            "Balance": st.column_config.TextColumn("Balance", width="small"),
        },
        height=min(700, 60 + 42 * len(display)),
    )

    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if st.button("← Previous", disabled=page <= 1, use_container_width=True):
            st.session_state["explorer_page"] = page - 1
            st.rerun()
    with c2:
        st.markdown(
            f"<div style='text-align:center;color:{MUTED};font-size:12.5px;padding-top:8px;'>"
            f"Page <b style='color:{TEXT}'>{page}</b> of {pages}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        if st.button("Next →", disabled=page >= pages, use_container_width=True):
            st.session_state["explorer_page"] = page + 1
            st.rerun()
    panel_close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — DETAILS
# ─────────────────────────────────────────────────────────────────────────────
def page_details():
    hero("Account Details", "Inspect a single account and its prediction.", "Explainable inference")

    col1, col2 = st.columns([3, 1])
    with col1:
        aid = st.text_input("Account ID", placeholder="e.g. ACC000123")
    with col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        load_btn = st.button("Load account", use_container_width=True)

    if not (aid and load_btn):
        st.info("Enter an Account ID and press **Load account**.")
        return

    if "AccountID" not in DF.columns:
        st.error("Sample data has no AccountID column.")
        return

    match = DF[DF["AccountID"].astype(str).str.lower() == aid.strip().lower()]
    if match.empty:
        st.warning(f"No account found with ID **{aid}**.")
        return

    row = match.iloc[0]
    p   = float(row["_prob"])
    barred = p >= THRESHOLD

    panel_open()
    st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-start;
                    flex-wrap:wrap;gap:16px;">
          <div>
            <div style="font-size:12px;text-transform:uppercase;letter-spacing:1.4px;color:{MUTED};">
              Account</div>
            <div style="font-size:26px;font-weight:800;letter-spacing:-.5px;">{row.get('AccountID','—')}</div>
            <div style="color:{MUTED};font-size:13px;margin-top:4px;">
              {row.get('OriginalCreditor[Redacted]','—')} · {row.get('ProductOrDebtType','—')}
            </div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:12px;text-transform:uppercase;letter-spacing:1.4px;color:{MUTED};">
              Current balance</div>
            <div style="font-size:26px;font-weight:800;letter-spacing:-.5px;color:{TEAL};">
              ${float(row['CurrentBalance']):,.2f}</div>
          </div>
        </div>
    """, unsafe_allow_html=True)
    panel_close()

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        metric_card("Debt load principal",
                    f"${float(row.get('DebtLoadPrincipal', 0)):,.2f}",
                    "Original amount", TEAL)
    with k2:
        age = row.get("CustomerAge")
        metric_card("Customer age", f"{int(age)}" if pd.notna(age) else "—", "Years", CORAL)
    with k3:
        metric_card("Phones / Emails / Addrs",
                    f"{int(row.get('NumPhones',0))}/{int(row.get('NumEmails',0))}/{int(row.get('NumAddresses',0))}",
                    "Contact channels", PURPLE)
    with k4:
        metric_card("Probability", f"{p*100:.1f}%", "Model confidence", prob_color(p))

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        panel_open()
        section("Prediction", "Model verdict")
        color = prob_color(p)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number=dict(suffix="%", font=dict(size=40, color=TEXT, family="Inter")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED, size=10)),
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
            height=260, margin=dict(l=20, r=20, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color=TEXT),
        )
        st.plotly_chart(gauge, use_container_width=True)

        verdict = "STATUTE-BARRED" if barred else "NOT statute-barred"
        st.markdown(f"""
            <div style="text-align:center;padding:12px;border-radius:12px;
                        background:linear-gradient(90deg,{color}22,transparent);
                        border:1px solid {color}55;">
              <div style="font-size:13.5px;font-weight:700;color:{color};letter-spacing:.4px;">
                {verdict}</div>
            </div>
        """, unsafe_allow_html=True)
        panel_close()

    with right:
        panel_open()
        section("Raw record", "All fields for this account")
        rec = pd.DataFrame([row]).T.reset_index()
        rec.columns = ["Field", "Value"]
        st.dataframe(rec, use_container_width=True, hide_index=True, height=420)
        panel_close()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE — SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
def page_settings():
    hero("Model & Settings", "Tune the decision threshold that classifies accounts.", "Runtime configuration")

    left, right = st.columns([1, 1], gap="large")

    with left:
        panel_open()
        section("Decision threshold", "Accounts above this are flagged statute-barred")
        new_t = st.slider("Threshold", 0.05, 0.95, float(THRESHOLD), 0.01, format="%.2f")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        preview = go.Figure(go.Indicator(
            mode="number+delta", value=new_t,
            delta=dict(reference=float(THRESHOLD), valueformat=".2f",
                       increasing=dict(color=CORAL), decreasing=dict(color=TEAL)),
            number=dict(font=dict(size=56, color=TEAL, family="Inter")),
        ))
        preview.update_layout(height=160, paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(preview, use_container_width=True)

        st.markdown(f"""
            <div style="padding:12px 14px;border-radius:11px;background:rgba(120,150,200,.06);
                        border:1px solid {BORDER};font-size:12.5px;color:{MUTED};line-height:1.55;">
              <b style="color:{TEXT}">Lower</b> → more accounts flagged (higher recall).<br>
              <b style="color:{TEXT}">Higher</b> → stricter flag (higher precision).
            </div>
        """, unsafe_allow_html=True)

        if st.button("Apply threshold", use_container_width=True):
            if isinstance(BUNDLE["prep"], dict):
                BUNDLE["prep"]["threshold"] = new_t
            for k in list(st.session_state.keys()):
                if k.startswith("scored_"):
                    del st.session_state[k]
            st.session_state["_current_threshold"] = new_t
            st.success(f"Threshold set to **{new_t:.2f}**")
            st.rerun()
        panel_close()

    with right:
        panel_open()
        section("Current configuration", "Loaded at runtime")
        impact = (DF["_prob"] >= THRESHOLD).mean() * 100

        st.markdown(f"""
            <div style="padding:16px;border-radius:12px;
                        background:linear-gradient(180deg,{PANEL_BG},{PANEL_2});
                        border:1px solid {BORDER};margin-bottom:12px;">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
                          color:{MUTED};margin-bottom:6px;">Threshold</div>
              <div style="font-size:34px;font-weight:800;color:{TEAL};letter-spacing:-.5px;">
                {THRESHOLD:.2f}</div>
            </div>

            <div style="padding:16px;border-radius:12px;
                        background:linear-gradient(180deg,{PANEL_BG},{PANEL_2});
                        border:1px solid {BORDER};margin-bottom:12px;">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
                          color:{MUTED};margin-bottom:6px;">Portfolio flagged</div>
              <div style="font-size:26px;font-weight:800;color:{CORAL};letter-spacing:-.5px;">
                {impact:.1f}%</div>
            </div>

            <div style="padding:16px;border-radius:12px;
                        background:linear-gradient(180deg,{PANEL_BG},{PANEL_2});
                        border:1px solid {BORDER};">
              <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
                          color:{MUTED};margin-bottom:6px;">Model</div>
              <div style="font-size:13.5px;color:{TEXT};">
                {type(BUNDLE['model']).__name__}<br>
                <span style="color:{MUTED};font-size:12px;">format: {BUNDLE['kind'].replace('_',' ')}</span>
              </div>
            </div>
        """, unsafe_allow_html=True)
        panel_close()

# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.session_state.setdefault("explorer_page", 1)
    page, filters = sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Accounts Explorer":
        page_explorer(filters)
    elif page == "Account Details":
        page_details()
    elif page == "Model & Settings":
        page_settings()

if __name__ == "__main__":
    main()
