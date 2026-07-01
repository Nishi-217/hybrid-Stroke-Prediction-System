"""UI theme, styling, and reusable dashboard components."""

from __future__ import annotations

import streamlit as st

# Design tokens
THEME = {
    "light": {
        "bg": "#f0f4f8",
        "bg_gradient": "linear-gradient(135deg, #f0f9ff 0%, #f0f4f8 50%, #ecfeff 100%)",
        "card": "#ffffff",
        "card_border": "#e2e8f0",
        "text": "#0f172a",
        "text_muted": "#64748b",
        "primary": "#0369a1",
        "primary_light": "#e0f2fe",
        "accent": "#0891b2",
        "sidebar": "#ffffff",
        "shadow": "0 1px 3px rgba(15,23,42,0.06), 0 4px 12px rgba(15,23,42,0.04)",
    },
    "dark": {
        "bg": "#0c1222",
        "bg_gradient": "linear-gradient(135deg, #0c1222 0%, #111827 50%, #0f172a 100%)",
        "card": "#151d2e",
        "card_border": "#1e293b",
        "text": "#f1f5f9",
        "text_muted": "#94a3b8",
        "primary": "#38bdf8",
        "primary_light": "#0c4a6e",
        "accent": "#22d3ee",
        "sidebar": "#111827",
        "shadow": "0 4px 16px rgba(0,0,0,0.35)",
    },
}

NAV_ITEMS = [
    ("Home", "🏠"),
    ("Predict Stroke", "🩺"),
    ("Dataset Analysis", "📊"),
    ("Model Performance", "📈"),
    ("Explainable AI", "🔬"),
    ("About", "ℹ️"),
]


def inject_global_css(theme: str = "light") -> None:
    """Inject professional healthcare dashboard styles."""
    t = THEME[theme]
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background: {t["bg_gradient"]};
        }}

        /* Hide default Streamlit chrome */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{
            background: transparent;
        }}

        /* Main content container */
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {t["sidebar"]};
            border-right: 1px solid {t["card_border"]};
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            padding: 0.55rem 0.75rem;
            border-radius: 8px;
            transition: background 0.15s;
        }}
        section[data-testid="stSidebar"] .stRadio label:hover {{
            background: {t["primary_light"]};
        }}

        /* Buttons */
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {t["primary"]}, {t["accent"]});
            border: none;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.6rem 1.5rem;
            transition: transform 0.15s, box-shadow 0.15s;
        }}
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 14px rgba(3,105,161,0.35);
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: transparent;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 0.5rem 1.25rem;
            font-weight: 500;
        }}

        /* Metrics */
        div[data-testid="stMetric"] {{
            background: {t["card"]};
            border: 1px solid {t["card_border"]};
            border-radius: 12px;
            padding: 1rem 1.25rem;
            box-shadow: {t["shadow"]};
        }}
        div[data-testid="stMetric"] label {{
            color: {t["text_muted"]} !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
        }}
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            color: {t["text"]} !important;
            font-weight: 700 !important;
        }}

        /* Forms */
        div[data-testid="stForm"] {{
            background: {t["card"]};
            border: 1px solid {t["card_border"]};
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: {t["shadow"]};
        }}

        /* Dataframes */
        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid {t["card_border"]};
        }}

        /* Custom components */
        .page-hero {{
            background: {t["card"]};
            border: 1px solid {t["card_border"]};
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.75rem;
            box-shadow: {t["shadow"]};
            border-left: 4px solid {t["primary"]};
        }}
        .page-hero h1 {{
            margin: 0 0 0.35rem 0;
            font-size: 1.85rem;
            font-weight: 700;
            color: {t["text"]};
            letter-spacing: -0.02em;
        }}
        .page-hero p {{
            margin: 0;
            color: {t["text_muted"]};
            font-size: 1rem;
            line-height: 1.5;
        }}

        .stat-card {{
            background: {t["card"]};
            border: 1px solid {t["card_border"]};
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            box-shadow: {t["shadow"]};
            height: 100%;
        }}
        .stat-card .label {{
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {t["text_muted"]};
            margin-bottom: 0.35rem;
        }}
        .stat-card .value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: {t["primary"]};
            line-height: 1.2;
        }}
        .stat-card .sub {{
            font-size: 0.82rem;
            color: {t["text_muted"]};
            margin-top: 0.25rem;
        }}

        .section-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: {t["text"]};
            margin: 1.75rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {t["primary_light"]};
        }}

        .feature-card {{
            background: {t["card"]};
            border: 1px solid {t["card_border"]};
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            box-shadow: {t["shadow"]};
        }}
        .feature-card .icon {{
            font-size: 1.25rem;
            line-height: 1;
        }}
        .feature-card .text {{
            color: {t["text"]};
            font-size: 0.92rem;
            line-height: 1.45;
        }}

        .workflow-card {{
            background: {t["card"]};
            border: 1px solid {t["card_border"]};
            border-radius: 12px;
            padding: 1.1rem 0.75rem;
            text-align: center;
            box-shadow: {t["shadow"]};
        }}
        .workflow-card .step-num {{
            display: inline-block;
            width: 28px;
            height: 28px;
            line-height: 28px;
            border-radius: 50%;
            background: {t["primary_light"]};
            color: {t["primary"]};
            font-weight: 700;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }}
        .workflow-card .step-name {{
            font-size: 0.82rem;
            font-weight: 600;
            color: {t["text"]};
        }}

        .risk-panel {{
            background: {t["card"]};
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            box-shadow: {t["shadow"]};
            border: 1px solid {t["card_border"]};
        }}
        .risk-gauge {{
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1;
            margin: 0.5rem 0;
        }}
        .risk-label {{
            font-size: 1.15rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }}
        .risk-badge-lg {{
            display: inline-block;
            padding: 0.5rem 1.25rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.95rem;
            margin-top: 0.75rem;
        }}

        .recommendation-box {{
            background: {t["primary_light"]};
            border-left: 4px solid {t["primary"]};
            border-radius: 0 12px 12px 0;
            padding: 1.1rem 1.35rem;
            margin: 1.25rem 0;
            color: {t["text"]};
            font-size: 0.95rem;
            line-height: 1.55;
        }}

        .form-section-label {{
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {t["primary"]};
            margin: 1rem 0 0.5rem 0;
            padding-bottom: 0.25rem;
        }}

        .sidebar-brand {{
            text-align: center;
            padding: 0.5rem 0 1rem 0;
        }}
        .sidebar-brand h2 {{
            margin: 0.5rem 0 0.15rem 0;
            font-size: 1.15rem;
            font-weight: 700;
            color: {t["text"]};
        }}
        .sidebar-brand p {{
            margin: 0;
            font-size: 0.78rem;
            color: {t["text_muted"]};
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }}
        .status-ready {{
            background: #dcfce7;
            color: #166534;
        }}
        .status-warn {{
            background: #fef3c7;
            color: #92400e;
        }}

        .info-card {{
            background: {t["card"]};
            border: 1px solid {t["card_border"]};
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: {t["shadow"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_hero(title: str, subtitle: str) -> None:
    """Render a consistent page header banner."""
    st.markdown(
        f"""
        <div class="page-hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(label: str, value: str, sub: str = "") -> None:
    """Render a custom statistics card."""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title: str) -> None:
    """Render a styled section heading."""
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def render_feature_card(icon: str, text: str) -> None:
    """Render a feature list item as a card."""
    st.markdown(
        f"""
        <div class="feature-card">
            <span class="icon">{icon}</span>
            <span class="text">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_step(number: int, name: str) -> None:
    """Render a workflow pipeline step."""
    st.markdown(
        f"""
        <div class="workflow-card">
            <div class="step-num">{number}</div>
            <div class="step-name">{name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_result(
    probability: float,
    risk_category: str,
    color: str,
    emoji: str,
    prediction_label: str,
) -> None:
    """Render the main prediction result panel."""
    st.markdown(
        f"""
        <div class="risk-panel" style="border-top: 4px solid {color};">
            <div style="color:#64748b;font-size:0.85rem;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.06em;">
                Stroke Probability
            </div>
            <div class="risk-gauge" style="color:{color};">{probability * 100:.1f}%</div>
            <div class="risk-label">{prediction_label}</div>
            <div class="risk-badge-lg" style="background:{color}18;color:{color};">
                {emoji} {risk_category}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation(text: str) -> None:
    """Render clinical recommendation callout."""
    st.markdown(
        f'<div class="recommendation-box">💡 <strong>Recommendation:</strong> {text}</div>',
        unsafe_allow_html=True,
    )


def render_form_section(label: str) -> None:
    """Render a form subsection label."""
    st.markdown(f'<div class="form-section-label">{label}</div>', unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """Render branded sidebar header."""
    st.markdown(
        """
        <div class="sidebar-brand">
            <div style="font-size:2.5rem;line-height:1;">🫀</div>
            <h2>Stroke Risk AI</h2>
            <p>Clinical Decision Support</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_status(ready: bool) -> None:
    """Render model readiness indicator."""
    if ready:
        st.markdown(
            '<div class="status-pill status-ready">● Model Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-warn">● Model Not Trained</div>',
            unsafe_allow_html=True,
        )
