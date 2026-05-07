import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from model import (
    predict_win_probability,
    get_all_options,
    get_historical_data,
    get_job_forecast,
    PRICE_RANGES,
    CLIENT_RETURN,
)

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuoteIQ · Tar Surfacing Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #161923;
    --border:    #242836;
    --accent:    #f0a500;
    --accent2:   #e05a2b;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --won:       #22c55e;
    --lost:      #ef4444;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

.quoteiq-header {
    background: linear-gradient(135deg, #0d0f14 0%, #1a1f2e 100%);
    border-bottom: 2px solid var(--accent);
    padding: 1.5rem 2rem 1rem;
    margin: -1rem -1rem 2rem -1rem;
}
.quoteiq-header h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3rem !important;
    letter-spacing: 4px;
    color: var(--accent) !important;
    margin: 0 !important;
    line-height: 1;
}
.quoteiq-header p {
    color: var(--muted) !important;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0.25rem 0 0 0 !important;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    color: var(--accent);
    line-height: 1;
}
.metric-sub {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 3px;
    color: var(--text);
    border-left: 3px solid var(--accent);
    padding-left: 0.75rem;
    margin: 2rem 0 1rem 0;
}

.badge-won  { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid #22c55e; border-radius: 6px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }
.badge-lost { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid #ef4444; border-radius: 6px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }

.prob-bar-wrap { background: var(--border); border-radius: 999px; height: 12px; width: 100%; margin: 0.5rem 0; }
.prob-bar-fill { height: 12px; border-radius: 999px; background: linear-gradient(90deg, var(--accent2), var(--accent)); transition: width 0.6s ease; }

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label { color: var(--muted) !important; font-size: 0.78rem !important; letter-spacing: 1px; text-transform: uppercase; }

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stNumberInput"] input {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

div[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
    cursor: pointer !important;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88 !important; }

div[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px; overflow: hidden; }

.forecast-row {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.forecast-date { font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem; color: var(--accent); }
.forecast-job  { font-size: 0.82rem; color: var(--text); }
.forecast-val  { font-size: 0.85rem; font-weight: 600; color: var(--won); }
.forecast-conf { font-size: 0.75rem; color: var(--muted); }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="quoteiq-header">
    <h1>QuoteIQ</h1>
    <p>Tar Surfacing Intelligence · Kempton Park, East Rand · Powered by ML</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar inputs ───────────────────────────────────────────────────────────────
opts = get_all_options()

with st.sidebar:
    st.markdown("### 🛣️ Quote Parameters")
    st.markdown("---")

    job_type    = st.selectbox("Job Type", opts["job_types"])
    area        = st.selectbox("Area", opts["areas"])
    client_type = st.selectbox("Client Type", opts["client_types"])

    st.markdown("---")
    quantity   = st.number_input("Quantity (m²)", min_value=100, max_value=5000, value=1000, step=50)
    unit_price = st.number_input("Unit Price (R/m²)", min_value=30, max_value=200, value=int(PRICE_RANGES[job_type]["sweet"]), step=1)

    st.markdown("---")
    analyse = st.button("Analyse Quote")

    st.markdown("---")
    st.markdown(f"<span style='font-size:0.7rem;color:#6b7280;letter-spacing:1px;'>QUOTE TOTAL</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='font-family:Bebas Neue;font-size:1.8rem;color:#f0a500;'>R{quantity * unit_price:,.0f}</span>", unsafe_allow_html=True)

# ── Main tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊  Win Analysis", "💰  Price Optimiser", "📅  Job Forecast", "📁  Quote History"])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 · Win Analysis
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    result = predict_win_probability(job_type, area, client_type, quantity, unit_price)
    prob   = result["win_probability"]

    col1, col2, col3 = st.columns(3)

    with col1:
        color = "#22c55e" if prob >= 60 else "#f0a500" if prob >= 40 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Win Probability</div>
            <div class="metric-value" style="color:{color};">{prob}%</div>
            <div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:{prob}%;"></div></div>
            <div class="metric-sub">{'Strong position' if prob >= 60 else 'Competitive — review price' if prob >= 40 else 'Price may be too high'}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Quote Total</div>
            <div class="metric-value">R{result['quote_amount']:,.0f}</div>
            <div class="metric-sub">{quantity:,} m² × R{unit_price}/m²</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        cr = CLIENT_RETURN[client_type]
        badge_color = {"High": "#22c55e", "Medium": "#f0a500", "Low": "#ef4444"}[cr["label"]]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Client Return Likelihood</div>
            <div class="metric-value" style="color:{badge_color};">{cr['label']}</div>
            <div class="metric-sub">~{cr['days']} days · {cr['note']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Probability Gauge</div>", unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        delta={"reference": 60, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#6b7280", "tickfont": {"color": "#6b7280"}},
            "bar":  {"color": "#f0a500"},
            "steps": [
                {"range": [0, 40],   "color": "rgba(239,68,68,0.2)"},
                {"range": [40, 60],  "color": "rgba(240,165,0,0.2)"},
                {"range": [60, 100], "color": "rgba(34,197,94,0.2)"},
            ],
            "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": 0.75, "value": 60},
        },
        number={"suffix": "%", "font": {"color": "#f0a500", "family": "Bebas Neue", "size": 48}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eaf0",
        height=280,
        margin=dict(t=20, b=0, l=20, r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("<div class='section-title'>Win Rate by Job Type</div>", unsafe_allow_html=True)
    df = get_historical_data()
    win_by_job = df.groupby("job_type")["outcome"].apply(lambda x: (x == "Won").mean() * 100).reset_index()
    win_by_job.columns = ["Job Type", "Win Rate %"]

    fig_bar = px.bar(win_by_job, x="Job Type", y="Win Rate %",
                     color="Win Rate %", color_continuous_scale=["#ef4444", "#f0a500", "#22c55e"],
                     range_color=[0, 100])
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eaf0", height=300,
        margin=dict(t=10, b=10, l=0, r=0),
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(size=10)),
    )
    fig_bar.update_traces(marker_line_width=0)
    st.plotly_chart(fig_bar, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 · Price Optimiser
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    pr      = PRICE_RANGES[job_type]
    result2 = predict_win_probability(job_type, area, client_type, quantity, unit_price)

    st.markdown("<div class='section-title'>Recommended Price Range (R/m²)</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Floor Price</div>
            <div class="metric-value" style="color:#ef4444;">R{pr['min']}/m²</div>
            <div class="metric-sub">Minimum to stay profitable</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sweet Spot ⭐</div>
            <div class="metric-value">R{pr['sweet']}/m²</div>
            <div class="metric-sub">Best win probability</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Ceiling Price</div>
            <div class="metric-value" style="color:#f0a500;">R{pr['max']}/m²</div>
            <div class="metric-sub">Max before win rate drops</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Win Probability vs Unit Price</div>", unsafe_allow_html=True)

    prices = list(range(pr["min"] - 10, pr["max"] + 15, 2))
    probs  = [predict_win_probability(job_type, area, client_type, quantity, p)["win_probability"] for p in prices]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=prices, y=probs, mode="lines+markers",
        line=dict(color="#f0a500", width=3),
        marker=dict(color="#f0a500", size=5),
        fill="tozeroy", fillcolor="rgba(240,165,0,0.08)",
        name="Win %",
    ))
    fig_line.add_vline(x=unit_price, line_dash="dash", line_color="#e05a2b",
                       annotation_text=f"Your price R{unit_price}/m²", annotation_font_color="#e05a2b")
    fig_line.add_vline(x=pr["sweet"], line_dash="dot", line_color="#22c55e",
                       annotation_text=f"Sweet spot R{pr['sweet']}/m²", annotation_font_color="#22c55e")
    fig_line.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eaf0", height=320,
        margin=dict(t=20, b=10, l=0, r=0),
        xaxis_title="Unit Price (R/m²)", yaxis_title="Win Probability (%)",
        yaxis=dict(range=[0, 100]),
        showlegend=False,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("<div class='section-title'>Total Quote Impact</div>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Your Quote</div>
            <div class="metric-value">R{quantity * unit_price:,.0f}</div>
            <div class="metric-sub">At R{unit_price}/m² · {result2['win_probability']}% win chance</div>
        </div>""", unsafe_allow_html=True)
    with cb:
        sweet_total = quantity * pr["sweet"]
        sweet_prob  = predict_win_probability(job_type, area, client_type, quantity, pr["sweet"])["win_probability"]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sweet Spot Quote</div>
            <div class="metric-value" style="color:#22c55e;">R{sweet_total:,.0f}</div>
            <div class="metric-sub">At R{pr['sweet']}/m² · {sweet_prob}% win chance</div>
        </div>""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 · Job Forecast
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-title'>Forecasted Jobs · Next 30 Days</div>", unsafe_allow_html=True)

    forecast = get_job_forecast()
    for f in forecast:
        conf_num  = int(f["confidence"].replace("%", ""))
        bar_color = "#22c55e" if conf_num >= 80 else "#f0a500" if conf_num >= 70 else "#ef4444"
        st.markdown(f"""
        <div class="forecast-row">
            <div>
                <div class="forecast-date">{f['date']}</div>
                <div class="forecast-job">{f['job_type']} · {f['area']} · {f['client_type']}</div>
            </div>
            <div style="text-align:right;">
                <div class="forecast-val">{f['estimated_value']}</div>
                <div class="forecast-conf" style="color:{bar_color};">Confidence {f['confidence']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Clients Likely to Need Work Again</div>", unsafe_allow_html=True)

    for ct, info in CLIENT_RETURN.items():
        color = {"High": "#22c55e", "Medium": "#f0a500", "Low": "#ef4444"}[info["label"]]
        st.markdown(f"""
        <div class="forecast-row">
            <div>
                <div style="font-weight:600;font-size:0.95rem;">{ct} Clients</div>
                <div class="forecast-job">{info['note']}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'Bebas Neue';font-size:1.2rem;color:{color};">{info['label']} Likelihood</div>
                <div class="forecast-conf">Follow up in ~{info['days']} days</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 · Quote History
# ───────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-title'>Historical Quotes</div>", unsafe_allow_html=True)

    df_display = get_historical_data().copy()
    df_display["Quote (R)"]   = df_display["quote_amount"].apply(lambda x: f"R{x:,.0f}")
    df_display["Unit Price"]  = df_display["unit_price"].apply(lambda x: f"R{x}/m²")
    df_display = df_display.rename(columns={
        "job_type": "Job Type", "area": "Area",
        "client_type": "Client Type", "quantity": "m²",
        "outcome": "Outcome",
    })

    st.dataframe(
        df_display[["Job Type", "Area", "Client Type", "m²", "Unit Price", "Quote (R)", "Outcome"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div class='section-title'>Outcome Breakdown</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        outcome_counts = df_display["Outcome"].value_counts().reset_index()
        fig_pie = go.Figure(go.Pie(
            labels=outcome_counts["Outcome"],
            values=outcome_counts["count"],
            marker_colors=["#22c55e", "#ef4444"],
            hole=0.55,
            textfont=dict(family="DM Sans", size=13),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eaf0",
            height=280,
            margin=dict(t=10, b=10, l=0, r=0),
            showlegend=True,
            legend=dict(font=dict(color="#e8eaf0")),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        area_win = df_display.copy()
        area_win["Won"] = area_win["Outcome"] == "Won"
        area_summary = area_win.groupby("Area")["Won"].mean().mul(100).reset_index()
        area_summary.columns = ["Area", "Win Rate %"]
        fig_area = px.bar(area_summary, x="Area", y="Win Rate %",
                          color="Win Rate %", color_continuous_scale=["#ef4444", "#f0a500", "#22c55e"],
                          range_color=[0, 100])
        fig_area.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eaf0", height=280,
            margin=dict(t=10, b=10, l=0, r=0),
            coloraxis_showscale=False,
        )
        fig_area.update_traces(marker_line_width=0)
        st.plotly_chart(fig_area, use_container_width=True)
