# app.py
"""
Statute-Barred Debt Predictor — Streamlit frontend
All data processing, ML inference, and business logic live in FastAPI.
This file only calls APIs and visualizes the results.
"""

from __future__ import annotations

import os
from datetime import datetime, date, timedelta
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 15

# Palette
NAVY_BG   = "#070b16"
PANEL_BG  = "#0f1729"
PANEL_2   = "#141d33"
BORDER    = "rgba(120,150,200,0.18)"
TEXT      = "#eaf0fa"
MUTED     = "#8ea3c2"
TEAL      = "#38e0c4"
CORAL     = "#ff6b7a"
PURPLE    = "#a78bfa"
AMBER     = "#ffb547"
GREEN     = "#2ee6a8"
RED       = "#ff5c6c"
GRID      = "rgba(120,150,200,0.10)"

st.set_page_config(
    page_title="Statute-Barred Debt Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(900px 500px at 10% -10%, #16305c 0%, transparent 55%),
            radial-gradient(800px 600px at 95% 0%, #3a1f66 0%, transparent 50%),
            {NAVY_BG};
        color: {TEXT};
    }}

    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}}

    /* ── Hero banner ─────────────────────────────────────────── */
    .hero {{
        position: relative;
        padding: 34px 38px;
        border-radius: 20px;
        margin-bottom: 26px;
        overflow: hidden;
        background: linear-gradient(120deg, #101a35 0%, #1c2452 45%, #2a1d54 100%);
        border: 1px solid {BORDER};
        box-shadow: 0 20px 60px rgba(0,0,0,.5);
    }}
    .hero::before {{
        content: "";
        position: absolute;
        inset: -2px;
        background: linear-gradient(120deg,
            rgba(56,224,196,.15), rgba(167,139,250,.15), rgba(255,107,122,.15), rgba(56,224,196,.15));
        background-size: 300% 300%;
        animation: sweep 14s ease infinite;
        z-index: 0;
        filter: blur(30px);
    }}
    @keyframes sweep {{
        0%   {{background-position: 0% 50%;}}
        50%  {{background-position: 100% 50%;}}
        100% {{background-position: 0% 50%;}}
    }}
    .hero-content {{position: relative; z-index: 1;}}
    .hero h1 {{
        margin: 0 0 8px;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.6px;
        background: linear-gradient(90deg, {TEAL}, {PURPLE}, {CORAL});
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }}
    .hero p {{
        margin: 0;
        color: {MUTED};
        font-size: 14.5px;
        font-weight: 400;
    }}
    .hero .badge {{
        display: inline-block;
        margin-top: 14px;
        padding: 5px 12px;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: .5px;
        color: {TEAL};
        background: rgba(56,224,196,.10);
        border: 1px solid rgba(56,224,196,.30);
        border-radius: 99px;
    }}

    /* ── Metric cards ────────────────────────────────────────── */
    .metric-card {{
        position: relative;
        padding: 22px 24px;
        border-radius: 16px;
        background: linear-gradient(180deg, {PANEL_BG}, {PANEL_2});
        border: 1px solid {BORDER};
        box-shadow: 0 12px 30px rgba(0,0,0,.4);
        overflow: hidden;
        height: 100%;
    }}
    .metric-card::after {{
        content: "";
        position: absolute; top: 0; left: 0; height: 3px; width: 100%;
        background: linear-gradient(90deg, var(--accent, {TEAL}), transparent 90%);
    }}
    .metric-label {{
        font-size: 11.5px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1.2px;
        color: {MUTED}; margin-bottom: 8px;
    }}
    .metric-value {{
        font-size: 30px; font-weight: 800; letter-spacing: -0.5px;
        color: {TEXT}; line-height: 1.1;
    }}
    .metric-hint {{
        font-size: 12px; color: {MUTED}; margin-top: 6px;
    }}

    /* ── Section titles ─────────────────────────────────────── */
    .section-title {{
        font-size: 15px; font-weight: 700; letter-spacing: -0.2px;
        margin: 6px 0 2px;
        color: {TEXT};
    }}
    .section-sub {{
        font-size: 12.5px; color: {MUTED}; margin-bottom: 14px;
    }}

    /* ── Panel wrapper ──────────────────────────────────────── */
    .panel {{
        padding: 22px 24px;
        border-radius: 16px;
        background: linear-gradient(180deg, {PANEL_BG}, {PANEL_2});
        border: 1px solid {BORDER};
        box-shadow: 0 12px 30px rgba(0,0,0,.35);
        margin-bottom: 20px;
    }}

    /* ── Streamlit overrides ────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0a1226 0%, #0c1428 100%);
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .stRadio > label {{
        font-size: 12px; font-weight: 600; letter-spacing: 1px;
        text-transform: uppercase; color: {MUTED};
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
        padding: 9px 12px; border-radius: 10px; margin-bottom: 4px;
        transition: all .15s ease; cursor: pointer;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        background: rgba(120,150,200,.08);
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {{
        background: linear-gradient(90deg, rgba(56,224,196,.14), rgba(167,139,250,.10));
        border: 1px solid rgba(56,224,196,.30);
    }}

    .stButton > button {{
        border-radius: 11px;
        font-weight: 600;
        border: 1px solid {BORDER};
        background: linear-gradient(135deg, rgba(56,224,196,.18), rgba(167,139,250,.14));
        color: {TEXT};
        transition: all .15s ease;
        padding: 10px 18px;
    }}
    .stButton > button:hover {{
        border-color: {TEAL};
        box-shadow: 0 8px 24px rgba(56,224,196,.25);
        transform: translateY(-1px);
    }}

    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {TEAL} !important;
        box-shadow: 0 0 0 3px rgba(56,224,196,.20) !important;
    }}

    .stDataFrame {{
        border: 1px solid {BORDER} !important;
        border-radius: 14px !important;
        overflow: hidden;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, {PANEL_BG}, {PANEL_2});
        padding: 16px; border-radius: 14px;
        border: 1px solid {BORDER};
    }}
    div[data-testid="stMetricLabel"] > div {{
        color: {MUTED} !important; font-size: 12px !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TEXT} !important; font-weight: 700 !important;
    }}

    hr {{border-color: {BORDER}; opacity: .4;}}

    .status-online {{color:{GREEN}; font-weight:600; font-size:12.5px;}}
    .status-offline {{color:{RED}; font-weight:600; font-size:12.5px;}}
    .status-check {{color:{MUTED}; font-weight:600; font-size:12.5px;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. API HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _url(path: str) -> str:
    return f"{API_BASE.rstrip('/')}{path}"

def api_get(path: str, params: Optional[dict] = None) -> tuple[Optional[Any], Optional[str]]:
    try:
        r = requests.get(_url(path), params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:180]}"
        return r.json(), None
    except requests.RequestException as e:
        return None, f"Network error: {e}"

def api_post(path: str, body: dict) -> tuple[Optional[Any], Optional[str]]:
    try:
        r = requests.post(_url(path), json=body, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {r.text[:180]}"
        return r.json(), None
    except requests.RequestException as e:
        return None, f"Network error: {e}"

def check_health() -> bool:
    data, err = api_get("/api/dashboard/summary")
    return err is None

# ─────────────────────────────────────────────────────────────────────────────
# 4. SHARED UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def hero(title: str, subtitle: str, badge: str = "Served by FastAPI · real-time inference"):
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-content">
            <h1>{title}</h1>
            <p>{subtitle}</p>
            <span class="badge">⚡ {badge}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def metric_card(label: str, value: str, hint: str = "", accent: str = TEAL):
    st.markdown(
        f"""
        <div class="metric-card" style="--accent:{accent};">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section(title: str, sub: str = ""):
    st.markdown(
        f"""<div class="section-title">{title}</div>
            <div class="section-sub">{sub}</div>""",
        unsafe_allow_html=True,
    )

def panel_open():
    st.markdown('<div class="panel">', unsafe_allow_html=True)

def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
def style_fig(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(
            bgcolor=PANEL_2,
            bordercolor=BORDER,
            font=dict(color=TEXT, family="Inter, sans-serif"),
        ),
    )
    return fig

def prob_color(p: float) -> str:
    if p >= 0.75: return RED
    if p >= 0.50: return AMBER
    if p >= 0.25: return "#eab308"
    return GREEN

# ─────────────────────────────────────────────────────────────────────────────
# 6. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px; padding:6px 0 20px;">
              <div style="width:42px;height:42px;border-radius:12px;
                          background:linear-gradient(135deg,{TEAL},{PURPLE});
                          display:grid;place-items:center;font-size:20px;">⚖️</div>
              <div>
                <div style="font-size:15px;font-weight:800;letter-spacing:-.3px;">DebtScope</div>
                <div style="font-size:11px;color:{MUTED};">Statute-barred analytics</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            ["Dashboard", "Accounts Explorer", "Account Details", "Model & Settings"],
            label_visibility="collapsed",
        )

        filters: dict = {}

        if page == "Accounts Explorer":
            st.markdown("---")
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;letter-spacing:1.2px;"
                f"text-transform:uppercase;color:{MUTED};margin-bottom:8px;'>Filters</div>",
                unsafe_allow_html=True,
            )
            filters["search"] = st.text_input("Search", placeholder="Account ID or creditor…")
            filters["min_prob"], filters["max_prob"] = st.slider(
                "Statute-barred probability", 0.0, 1.0, (0.0, 1.0), 0.01,
            )
            filters["min_balance"] = st.number_input("Min balance", min_value=0.0, value=0.0, step=100.0)
            filters["max_balance"] = st.number_input("Max balance", min_value=0.0, value=100000.0, step=100.0)

            today = date.today()
            filters["from_date"] = st.date_input("Activity from", value=today - timedelta(days=365 * 3))
            filters["to_date"]   = st.date_input("Activity to", value=today)

            filters["is_stat_barred"] = st.checkbox("Show only predicted statute-barred", value=False)
            filters["page_size"] = st.select_slider("Rows per page", options=[25, 50, 100, 200], value=50)

        st.markdown("---")
        if st.session_state.get("_online") is None:
            st.session_state["_online"] = check_health()
        online = st.session_state["_online"]
        cls = "status-online" if online else "status-offline"
        txt = "● API online" if online else "● API offline"
        st.markdown(f"<div class='{cls}'>{txt}</div>", unsafe_allow_html=True)
        st.caption(f"`{API_BASE}`")

        if st.button("↻ Reconnect", use_container_width=True):
            st.session_state["_online"] = check_health()
            st.rerun()

    return {"page": page, "filters": filters}

# ─────────────────────────────────────────────────────────────────────────────
# 7. PAGE — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def page_dashboard():
    hero(
        "Predict Statute-Barred Debt with Deep Learning",
        "Identify time-barred accounts and optimize collection strategy.",
    )

    summary, err = api_get("/api/dashboard/summary")
    if err or not summary:
        st.error(f"Could not load dashboard summary — {err}")
        return

    total       = summary.get("total_accounts", 0)
    barred_pct  = summary.get("stat_barred_percent", 0.0)
    high_risk   = summary.get("high_risk_accounts", 0)
    recoverable = summary.get("estimated_recoverable_amount", 0.0)
    last_run    = summary.get("last_scoring_run", "—")

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        metric_card("Total Accounts", f"{total:,}", "Across all portfolios", TEAL)
    with c2:
        metric_card("Statute-Barred", f"{barred_pct:.1f}%",
                    "Share of portfolio at risk", CORAL)
    with c3:
        metric_card("High-Risk Accounts", f"{high_risk:,}",
                    "Probability above threshold", PURPLE)
    with c4:
        metric_card("Recoverable Amount", f"${recoverable:,.0f}",
                    "Estimated net of risk", AMBER)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    # ── Fetch a page of accounts for chart building ────────────────
    accounts, aerr = api_get("/api/accounts", params={"page": 1, "page_size": 1000})
    df = pd.DataFrame(accounts.get("items", [])) if accounts else pd.DataFrame()

    if aerr:
        st.warning(f"Charts limited — {aerr}")

    row1_l, row1_r = st.columns([1.2, 1], gap="large")

    # ── Probability distribution ──────────────────────────────────
    with row1_l:
        panel_open()
        section("Probability distribution",
                "Density of predicted statute-barred probability across the portfolio")
        if not df.empty and "stat_barred_probability" in df.columns:
            probs = df["stat_barred_probability"].astype(float)
            hist = go.Figure()
            hist.add_trace(go.Histogram(
                x=probs, nbinsx=30,
                marker=dict(
                    color=probs,
                    colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
                    line=dict(width=0),
                ),
                hovertemplate="Probability %{x:.2f}<br>Count %{y}<extra></extra>",
            ))
            hist.update_layout(
                xaxis_title="Probability", yaxis_title="Accounts",
                bargap=0.04,
            )
            st.plotly_chart(style_fig(hist, 340), use_container_width=True)
        else:
            st.info("No probability data available.")
        panel_close()

    # ── Risk buckets ──────────────────────────────────────────────
    with row1_r:
        panel_open()
        section("Risk bucket distribution",
                "Accounts grouped by probability band")
        if not df.empty and "stat_barred_probability" in df.columns:
            probs = df["stat_barred_probability"].astype(float)
            buckets = pd.cut(probs, [0, .25, .5, .75, 1.0],
                             labels=["Low (0–25%)", "Medium (25–50%)",
                                     "High (50–75%)", "Critical (75–100%)"],
                             include_lowest=True)
            counts = buckets.value_counts().reindex(
                ["Low (0–25%)", "Medium (25–50%)",
                 "High (50–75%)", "Critical (75–100%)"]
            ).reset_index()
            counts.columns = ["Band", "Count"]
            bar = go.Figure(go.Bar(
                x=counts["Band"], y=counts["Count"],
                marker=dict(color=[GREEN, "#eab308", AMBER, RED]),
                text=counts["Count"], textposition="outside",
                hovertemplate="%{x}<br>%{y} accounts<extra></extra>",
            ))
            bar.update_layout(xaxis_title="", yaxis_title="Accounts", showlegend=False)
            st.plotly_chart(style_fig(bar, 340), use_container_width=True)
        else:
            st.info("No bucket data available.")
        panel_close()

    row2_l, row2_r = st.columns([1.3, 1], gap="large")

    # ── Creditor breakdown ────────────────────────────────────────
    with row2_l:
        panel_open()
        section("Portfolio breakdown by creditor",
                "Average statute-barred probability per creditor")
        if not df.empty and "creditor" in df.columns:
            grp = (df.groupby("creditor")
                     .agg(accounts=("creditor", "size"),
                          avg_prob=("stat_barred_probability", "mean"))
                     .reset_index()
                     .sort_values("accounts", ascending=False)
                     .head(12))
            cb = go.Figure()
            cb.add_trace(go.Bar(
                x=grp["creditor"], y=grp["accounts"],
                marker=dict(
                    color=grp["avg_prob"],
                    colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
                    cmin=0, cmax=1,
                    line=dict(width=0),
                ),
                hovertemplate="%{x}<br>Accounts %{y}<br>Avg prob %{marker.color:.2f}<extra></extra>",
                name="Accounts",
            ))
            cb.update_layout(
                xaxis_title="Creditor", yaxis_title="Accounts",
                coloraxis_showscale=False,
            )
            st.plotly_chart(style_fig(cb, 340), use_container_width=True)
        else:
            st.info("No creditor data available.")
        panel_close()

    # ── Monthly trend (derived from last_activity_date) ───────────
    with row2_r:
        panel_open()
        section("Monthly trend",
                "% statute-barred by last activity month")
        if not df.empty and "last_activity_date" in df.columns:
            d = df.copy()
            d["last_activity_date"] = pd.to_datetime(d["last_activity_date"], errors="coerce")
            d = d.dropna(subset=["last_activity_date"])
            if not d.empty:
                d["month"] = d["last_activity_date"].dt.to_period("M").dt.to_timestamp()
                trend = (d.groupby("month")
                           .agg(rate=("predicted_is_stat_barred", "mean"),
                                count=("predicted_is_stat_barred", "size"))
                           .reset_index())
                line = go.Figure()
                line.add_trace(go.Scatter(
                    x=trend["month"], y=trend["rate"] * 100,
                    mode="lines+markers",
                    line=dict(color=TEAL, width=3),
                    marker=dict(size=7, color=TEAL,
                                line=dict(color=NAVY_BG, width=2)),
                    fill="tozeroy",
                    fillcolor="rgba(56,224,196,0.10)",
                    hovertemplate="%{x|%b %Y}<br>%{y:.1f}% barred<extra></extra>",
                    name="Barred rate",
                ))
                line.update_layout(
                    xaxis_title="", yaxis_title="Statute-barred %",
                    yaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(style_fig(line, 340), use_container_width=True)
            else:
                st.info("No dated activity to plot.")
        else:
            st.info("No activity-date data available.")
        panel_close()

    st.markdown(
        f"<div style='color:{MUTED};font-size:12px;text-align:right;'>"
        f"Last scoring run: <b style='color:{TEXT}'>{last_run}</b></div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 8. PAGE — ACCOUNTS EXPLORER
# ─────────────────────────────────────────────────────────────────────────────
def page_explorer(filters: dict):
    hero(
        "Accounts Explorer",
        "Search, filter, and inspect accounts scored by the model.",
        badge="Live API query",
    )

    params = {
        "page": st.session_state.get("explorer_page", 1),
        "page_size": filters.get("page_size", 50),
    }
    if filters.get("search"):
        params["search"] = filters["search"]
    if filters.get("min_prob", 0) > 0:
        params["min_prob"] = filters["min_prob"]
    if filters.get("max_prob", 1) < 1:
        params["max_prob"] = filters["max_prob"]
    if filters.get("min_balance", 0) > 0:
        params["min_balance"] = filters["min_balance"]
    if filters.get("max_balance", 100000) < 100000:
        params["max_balance"] = filters["max_balance"]
    if filters.get("from_date"):
        params["from_date"] = filters["from_date"].isoformat()
    if filters.get("to_date"):
        params["to_date"] = filters["to_date"].isoformat()
    if filters.get("is_stat_barred"):
        params["is_stat_barred"] = "true"

    data, err = api_get("/api/accounts", params=params)
    if err or not data:
        st.error(f"Could not load accounts — {err}")
        return

    items = data.get("items", [])
    total = data.get("total", 0)
    page_size = params["page_size"]
    total_pages = max(1, (total + page_size - 1) // page_size)

    panel_open()
    section(
        f"{total:,} accounts match",
        f"Page {params['page']} of {total_pages}",
    )

    if not items:
        st.info("No accounts match the current filters.")
        panel_close()
        return

    df = pd.DataFrame(items)
    display = pd.DataFrame({
        "Account ID":   df.get("account_id"),
        "Creditor":     df.get("creditor"),
        "Balance":      df.get("current_balance").apply(
                            lambda v: f"${v:,.2f}" if pd.notna(v) else "—"),
        "Purchased":    df.get("purchase_date"),
        "Probability":  df.get("stat_barred_probability").astype(float),
        "Predicted":    df.get("predicted_is_stat_barred").apply(
                            lambda b: "Statute-Barred" if b else "Active"),
        "Last activity": df.get("last_activity_date"),
    })

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability",
                help="Model confidence that this debt is statute-barred",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "Predicted": st.column_config.TextColumn(
                "Predicted class", width="medium",
            ),
            "Balance": st.column_config.TextColumn("Current balance", width="small"),
        },
        height=min(700, 60 + 42 * len(display)),
    )

    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if st.button("← Previous", disabled=params["page"] <= 1, use_container_width=True):
            st.session_state["explorer_page"] = params["page"] - 1
            st.rerun()
    with c2:
        st.markdown(
            f"<div style='text-align:center;color:{MUTED};font-size:12.5px;padding-top:8px;'>"
            f"Page <b style='color:{TEXT}'>{params['page']}</b> of {total_pages}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        if st.button("Next →", disabled=params["page"] >= total_pages, use_container_width=True):
            st.session_state["explorer_page"] = params["page"] + 1
            st.rerun()

    panel_close()

    panel_open()
    section("Filtered distribution",
            "Probability spread of the currently matched accounts")
    probs = df["stat_barred_probability"].astype(float)
    fig = go.Figure(go.Histogram(
        x=probs, nbinsx=25,
        marker=dict(
            color=probs,
            colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
            line=dict(width=0),
        ),
        hovertemplate="%{x:.2f}<br>%{y} accounts<extra></extra>",
    ))
    fig.update_layout(xaxis_title="Probability", yaxis_title="Accounts", bargap=.05)
    st.plotly_chart(style_fig(fig, 260), use_container_width=True)
    panel_close()

# ─────────────────────────────────────────────────────────────────────────────
# 9. PAGE — ACCOUNT DETAILS
# ─────────────────────────────────────────────────────────────────────────────
def page_details():
    hero(
        "Account Details",
        "Inspect a single account, its prediction, and the model's reasoning.",
        badge="Explainable inference",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        account_id = st.text_input("Account ID", placeholder="e.g. ACC123")
    with c2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        go_btn = st.button("Load account", use_container_width=True)

    if not (account_id and go_btn):
        st.info("Enter an Account ID and press **Load account**.")
        return

    detail, err = api_get(f"/api/accounts/{account_id}")
    if err or not detail:
        st.error(f"Could not load account — {err}")
        return

    pred, perr = api_get(f"/api/accounts/{account_id}/prediction")

    panel_open()
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;">
          <div>
            <div style="font-size:12px;text-transform:uppercase;letter-spacing:1.4px;color:{MUTED};">
              Account
            </div>
            <div style="font-size:26px;font-weight:800;letter-spacing:-.5px;">
              {detail.get('account_id','—')}
            </div>
            <div style="color:{MUTED};font-size:13px;margin-top:4px;">
              {detail.get('creditor','—')} · {detail.get('jurisdiction','—')}
            </div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:12px;text-transform:uppercase;letter-spacing:1.4px;color:{MUTED};">
              Current balance
            </div>
            <div style="font-size:26px;font-weight:800;letter-spacing:-.5px;color:{TEAL};">
              ${detail.get('current_balance',0):,.2f}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    panel_close()

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        metric_card("Original balance",
                    f"${detail.get('original_balance',0):,.2f}",
                    "At account open", TEAL)
    with k2:
        metric_card("Days since last payment",
                    f"{detail.get('days_since_last_payment',0):,}",
                    "Key limitation signal", CORAL)
    with k3:
        metric_card("Account age",
                    f"{detail.get('account_age_days',0):,} days",
                    f"Limitation period {detail.get('limitation_period_years','—')}y",
                    PURPLE)
    with k4:
        if pred:
            p = pred.get("stat_barred_probability", 0)
            metric_card("Statute-barred probability",
                        f"{p*100:.1f}%",
                        "Model confidence",
                        prob_color(p))
        else:
            metric_card("Statute-barred probability", "—", "Prediction unavailable", MUTED)

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        panel_open()
        section("Prediction", "Model verdict and confidence")
        if perr or not pred:
            st.warning(f"Prediction unavailable — {perr}")
        else:
            p = pred.get("stat_barred_probability", 0)
            barred = pred.get("predicted_is_stat_barred", False)
            color = prob_color(p)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p * 100,
                number=dict(suffix="%", font=dict(size=40, color=TEXT, family="Inter")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=MUTED,
                              tickfont=dict(color=MUTED, size=10)),
                    bar=dict(color=color, thickness=0.28),
                    bgcolor="rgba(120,150,200,0.10)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0, 25],   color="rgba(46,230,168,0.10)"),
                        dict(range=[25, 50],  color="rgba(234,179,8,0.10)"),
                        dict(range=[50, 75],  color="rgba(255,181,71,0.10)"),
                        dict(range=[75, 100], color="rgba(255,92,108,0.10)"),
                    ],
                ),
            ))
            gauge.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color=TEXT),
            )
            st.plotly_chart(gauge, use_container_width=True)

            verdict = "STATUTE-BARRED" if barred else "NOT statute-barred"
            st.markdown(
                f"""
                <div style="text-align:center;padding:12px;border-radius:12px;
                            background:linear-gradient(90deg,{color}22,transparent);
                            border:1px solid {color}55;margin-top:6px;">
                  <div style="font-size:13.5px;font-weight:700;color:{color};letter-spacing:.4px;">
                    {verdict}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            action = pred.get("recommended_action")
            if action:
                st.markdown(
                    f"""
                    <div style="margin-top:14px;padding:14px;border-radius:12px;
                                background:rgba(120,150,200,0.06);border:1px solid {BORDER};">
                      <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
                                  color:{MUTED};margin-bottom:6px;">Recommended action</div>
                      <div style="font-size:13.5px;color:{TEXT};line-height:1.5;">{action}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        panel_close()

    with right:
        panel_open()
        section("Feature impact", "What drove this prediction")
        if perr or not pred:
            st.warning("Explanation unavailable.")
        else:
            feats = pred.get("top_features", []) or []
            if not feats:
                st.info("No feature contributions reported.")
            else:
                fdf = pd.DataFrame(feats)
                impact_color = {"high": RED, "medium": AMBER, "low": TEAL}
                fdf["color"] = fdf["impact"].map(impact_color).fillna(TEAL)

                bar = go.Figure(go.Bar(
                    x=fdf["value"].astype(str).str.slice(0, 22),
                    y=fdf["name"],
                    orientation="h",
                    marker=dict(color=fdf["color"], line=dict(width=0)),
                    text=fdf["impact"],
                    textposition="outside",
                    hovertemplate="%{y}<br>value: %{x}<br>impact: %{text}<extra></extra>",
                ))
                bar.update_layout(
                    xaxis_title="", yaxis_title="",
                    showlegend=False,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(style_fig(bar, 300), use_container_width=True)

                st.dataframe(
                    fdf[["name", "value", "impact"]].rename(columns={
                        "name": "Feature", "value": "Value", "impact": "Impact",
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=180,
                )
        panel_close()

    panel_open()
    section("Raw record", "Everything returned by the backend")
    raw = pd.DataFrame([detail]).T.reset_index()
    raw.columns = ["Field", "Value"]
    st.dataframe(raw, use_container_width=True, hide_index=True, height=320)
    panel_close()

# ─────────────────────────────────────────────────────────────────────────────
# 10. PAGE — MODEL & SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
def page_settings():
    hero(
        "Model & Settings",
        "Tune the decision threshold that separates statute-barred from active accounts.",
        badge="Runtime configuration",
    )

    current, err = api_get("/api/settings/threshold")
    cur_val = current.get("threshold", 0.5) if current else 0.5

    if err:
        st.warning(f"Could not read current threshold — {err}")

    left, right = st.columns([1, 1], gap="large")

    with left:
        panel_open()
        section("Decision threshold",
                "Accounts with probability ≥ threshold are flagged statute-barred")
        new_val = st.slider(
            "Threshold",
            min_value=0.05, max_value=0.95,
            value=float(cur_val), step=0.01,
            format="%.2f",
        )

        preview = go.Figure(go.Indicator(
            mode="number+delta",
            value=new_val,
            delta=dict(reference=cur_val, valueformat=".2f",
                       increasing=dict(color=CORAL),
                       decreasing=dict(color=TEAL)),
            number=dict(font=dict(size=56, color=TEAL, family="Inter")),
        ))
        preview.update_layout(
            height=160,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(preview, use_container_width=True)

        st.markdown(
            f"""
            <div style="padding:12px 14px;border-radius:11px;background:rgba(120,150,200,.06);
                        border:1px solid {BORDER};font-size:12.5px;color:{MUTED};line-height:1.55;">
              <b style="color:{TEXT}">Lower</b> → more accounts flagged (higher recall, more false positives).<br>
              <b style="color:{TEXT}">Higher</b> → stricter flag (higher precision, may miss some barred debts).
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Apply threshold", use_container_width=True):
            res, perr = api_post("/api/settings/threshold", {"threshold": new_val})
            if perr:
                st.error(f"Failed to update — {perr}")
            else:
                st.success(
                    f"Threshold updated to {res.get('threshold', new_val):.2f} "
                    f"at {res.get('updated_at', 'now')}"
                )
        panel_close()

    with right:
        panel_open()
        section("Current configuration", "Live values served by the backend")
        cfg, cerr = api_get("/api/settings/threshold")
        if cerr or not cfg:
            st.warning(f"Unavailable — {cerr}")
        else:
            st.markdown(
                f"""
                <div style="padding:16px;border-radius:12px;
                            background:linear-gradient(180deg,{PANEL_BG},{PANEL_2});
                            border:1px solid {BORDER};">
                  <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
                              color:{MUTED};margin-bottom:6px;">Threshold</div>
                  <div style="font-size:34px;font-weight:800;color:{TEAL};letter-spacing:-.5px;">
                    {cfg.get('threshold', 0.5):.2f}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div style="padding:16px;border-radius:12px;
                            background:linear-gradient(180deg,{PANEL_BG},{PANEL_2});
                            border:1px solid {BORDER};">
                  <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
                              color:{MUTED};margin-bottom:6px;">API base URL</div>
                  <div style="font-size:13.5px;font-family:monospace;color:{TEXT};">
                    {API_BASE}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div style="padding:16px;border-radius:12px;
                            background:linear-gradient(180deg,{PANEL_BG},{PANEL_2});
                            border:1px solid {BORDER};">
                  <div style="font-size:11px;text-transform:uppercase;letter-spacing:1.2px;
                              color:{MUTED};margin-bottom:8px;">OpenAPI docs</div>
                  <a href="{API_BASE.rstrip('/')}/docs" target="_blank"
                     style="color:{TEAL};text-decoration:none;font-size:13.5px;font-weight:600;">
                     {API_BASE.rstrip('/')}/docs →
                  </a>
                </div>
                """,
                unsafe_allow_html=True,
            )
        panel_close()

# ─────────────────────────────────────────────────────────────────────────────
# 11. ROUTER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.session_state.setdefault("explorer_page", 1)
    st.session_state.setdefault("_online", None)

    ctx = sidebar()
    page = ctx["page"]
    filters = ctx["filters"]

    if page == "Accounts Explorer":
        fp = tuple(str(v) for v in filters.values())
        if st.session_state.get("_last_filter_key") != fp:
            st.session_state["_last_filter_key"] = fp
            st.session_state["explorer_page"] = 1

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
