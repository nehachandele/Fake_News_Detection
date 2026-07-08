"""
ui.py
======
Shared visual theme and reusable UI components for the AI-Powered Fake News
Intelligence Platform.

Design direction: "Evidence Desk" -- the app treats every analyzed article
like a case file receiving a verdict stamp, borrowing its visual language
from newsroom wire services and investigative case files rather than a
generic dashboard look. Centralizing the theme here means every page in
`pages/` renders identically without copy-pasted CSS.
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
COLOR_BG = "#12151B"
COLOR_PANEL = "#1A2027"
COLOR_PANEL_ALT = "#1F2630"
COLOR_BORDER = "#2A313D"
COLOR_TEXT = "#ECE9E2"
COLOR_TEXT_MUTED = "#8891A0"
COLOR_AMBER = "#E8A33D"
COLOR_REAL = "#2FB88A"
COLOR_FAKE = "#D6455A"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {COLOR_TEXT};
}}

.stApp {{
    background-color: {COLOR_BG};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {COLOR_PANEL};
    border-right: 1px solid {COLOR_BORDER};
}}

/* Headings use the mono "wire service" face */
h1, h2, h3 {{
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.02em;
}}

h1 {{
    font-weight: 700;
    border-bottom: 2px solid {COLOR_BORDER};
    padding-bottom: 0.5rem;
}}

/* Eyebrow label style, used above section headers */
.eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {COLOR_AMBER};
    margin-bottom: 0.25rem;
}}

/* Case-file style card */
.case-card {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}}

.case-card-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {COLOR_TEXT_MUTED};
    margin-bottom: 0.4rem;
}}

/* Wire ticker strip */
.wire-ticker {{
    display: flex;
    gap: 0;
    border-top: 1px solid {COLOR_BORDER};
    border-bottom: 1px solid {COLOR_BORDER};
    margin: 1rem 0 1.5rem 0;
    overflow-x: auto;
}}

.wire-item {{
    flex: 1;
    padding: 0.9rem 1.2rem;
    border-right: 1px solid {COLOR_BORDER};
    font-family: 'IBM Plex Mono', monospace;
    min-width: 140px;
}}

.wire-item:last-child {{ border-right: none; }}

.wire-label {{
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: {COLOR_TEXT_MUTED};
}}

.wire-value {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {COLOR_TEXT};
    margin-top: 0.15rem;
}}

/* The signature element: rubber-stamp verdict */
.stamp-wrap {{
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 1.5rem 0;
}}

.stamp {{
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 2.1rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.55rem 1.8rem;
    border: 4px double currentColor;
    border-radius: 6px;
    transform: rotate(-6deg);
    display: inline-block;
    opacity: 0.92;
}}

.stamp-real {{ color: {COLOR_REAL}; }}
.stamp-fake {{ color: {COLOR_FAKE}; }}

/* Risk / clickbait tag pill */
.tag-pill {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    border: 1px solid currentColor;
    margin: 0.15rem 0.25rem 0.15rem 0;
}}

.tag-amber {{ color: {COLOR_AMBER}; }}
.tag-real {{ color: {COLOR_REAL}; }}
.tag-fake {{ color: {COLOR_FAKE}; }}
.tag-muted {{ color: {COLOR_TEXT_MUTED}; }}

/* Contribution word chips (explainability) */
.word-chip {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    padding: 0.25rem 0.55rem;
    border-radius: 3px;
    margin: 0.15rem;
}}

hr {{
    border-color: {COLOR_BORDER};
}}

/* Streamlit button restyle */
.stButton > button {{
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.8rem;
    background-color: {COLOR_AMBER};
    color: {COLOR_BG};
    border: none;
    border-radius: 3px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
}}

.stButton > button:hover {{
    background-color: #f0b45c;
    color: {COLOR_BG};
}}

/* Metric widget */
[data-testid="stMetric"] {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER};
    border-radius: 4px;
    padding: 0.8rem 1rem;
}}
</style>
"""


def inject_theme() -> None:
    """Inject the shared CSS theme. Call once at the top of every page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar(artifacts) -> None:
    """Render the shared sidebar: model info, dataset info, performance
    metrics, and version -- present on every page for consistent context.

    Parameters
    ----------
    artifacts:
        A loaded `utils.model_utils.ModelArtifacts` instance.
    """
    metrics = artifacts.evaluation_metrics
    test_metrics = metrics.get("test_metrics", {})
    vec_config = metrics.get("vectorizer_config", {})

    with st.sidebar:
        st.markdown(
            '<div class="eyebrow">Fake News Intelligence</div>'
            '<h3 style="margin-top:0;">Evidence Desk</h3>',
            unsafe_allow_html=True,
        )
        st.caption("v1.0.0 &middot; Platform build")

        st.markdown("---")
        eyebrow("Model")
        st.write(f"**Algorithm:** {metrics.get('best_model_name', 'N/A')}")
        st.write(f"**Vectorizer:** TF-IDF ({vec_config.get('ngram_range', ['?', '?'])[0]}-{vec_config.get('ngram_range', ['?', '?'])[1]} grams)")
        st.write(f"**Vocabulary size:** {vec_config.get('vocabulary_size', 'N/A'):,}")

        st.markdown("---")
        eyebrow("Performance (test set)")
        col1, col2 = st.columns(2)
        col1.metric("F1", f"{test_metrics.get('f1_score', 0):.3f}")
        col2.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")
        col1.metric("Precision", f"{test_metrics.get('precision', 0):.3f}")
        col2.metric("Recall", f"{test_metrics.get('recall', 0):.3f}")

        st.markdown("---")
        eyebrow("Dataset")
        st.write(f"**Training rows:** {metrics.get('train_size', 0):,}")
        st.write(f"**Test rows:** {metrics.get('test_size', 0):,}")
        st.write(f"**Total articles:** {metrics.get('dataset_size', 0):,}")

        st.markdown("---")
        st.caption("Built with scikit-learn, NLTK, and Streamlit.")


def eyebrow(text: str) -> None:
    """Render a small amber uppercase label above a section header."""
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def wire_ticker(items: list[tuple[str, str]]) -> None:
    """Render a horizontal wire-service style ticker strip of stat items.

    Parameters
    ----------
    items:
        List of (label, value) string pairs, e.g. [("MODEL", "Passive Aggressive"),
        ("TEST F1", "0.970")].
    """
    cells = "".join(
        f'<div class="wire-item"><div class="wire-label">{label}</div>'
        f'<div class="wire-value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="wire-ticker">{cells}</div>', unsafe_allow_html=True)


def case_card_open(title: str) -> None:
    st.markdown(f'<div class="case-card"><div class="case-card-title">{title}</div>', unsafe_allow_html=True)


def case_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def verdict_stamp(prediction: str) -> None:
    """Render the signature rubber-stamp verdict element."""
    css_class = "stamp-real" if prediction.lower() == "real" else "stamp-fake"
    st.markdown(
        f'<div class="stamp-wrap"><div class="stamp {css_class}">{prediction}</div></div>',
        unsafe_allow_html=True,
    )


def tag_pill(text: str, kind: str = "muted") -> str:
    """Return HTML for a small tag pill. kind: amber|real|fake|muted."""
    return f'<span class="tag-pill tag-{kind}">{text}</span>'


def word_chip(word: str, contribution: float, max_abs: float) -> str:
    """Return HTML for a single explainability word chip, colored and sized
    by the magnitude/direction of its contribution.
    """
    intensity = min(abs(contribution) / max_abs, 1.0) if max_abs else 0.0
    color = COLOR_FAKE if contribution > 0 else COLOR_REAL
    opacity = 0.25 + 0.65 * intensity
    return (
        f'<span class="word-chip" style="background-color:{color}{int(opacity*255):02x};'
        f'color:{COLOR_TEXT};">{word} ({contribution:+.3f})</span>'
    )
