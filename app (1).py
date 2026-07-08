"""
app.py
=======
AI-Powered Fake News Intelligence Platform -- landing dashboard.

Run with: streamlit run app.py
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.model_utils import ModelLoadError, load_artifacts, warmup
from utils.ui import case_card_close, case_card_open, eyebrow, inject_theme, render_sidebar, wire_ticker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Fake News Intelligence Platform",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_theme()


# --------------------------------------------------------------------------
# Cached resource loading -- the model is loaded from disk exactly once per
# server process, and the (~2.6s) cold-start cost is paid here at startup
# rather than on a user's first prediction.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model artifacts...")
def get_artifacts():
    artifacts = load_artifacts(models_dir="models", reports_dir="reports")
    warmup(artifacts)
    return artifacts


try:
    artifacts = get_artifacts()
except ModelLoadError as exc:
    st.error(
        "**Model artifacts not found.**\n\n"
        f"{exc}\n\n"
        "Run the Phase 1 notebook (`notebooks/01_fake_news_intelligence.ipynb`) "
        "first to generate `models/` and `reports/evaluation_metrics.json`."
    )
    st.stop()

render_sidebar(artifacts)

metrics = artifacts.evaluation_metrics
test_metrics = metrics.get("test_metrics", {})
model_comparison = metrics.get("model_comparison", [])

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
eyebrow("AI-Powered Fake News Intelligence Platform")
st.markdown("# Evidence Desk")
st.markdown(
    "Paste an article, upload a batch of headlines, or just watch the "
    "wire below. Every verdict on this desk comes from a model trained "
    "and evaluated in the open — nothing here is a guess dressed up as a fact."
)

# Wire ticker -- key live stats pulled straight from evaluation_metrics.json
wire_ticker([
    ("Model", metrics.get("best_model_name", "N/A")),
    ("Test F1", f"{test_metrics.get('f1_score', 0):.3f}"),
    ("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}"),
    ("Training Articles", f"{metrics.get('train_size', 0):,}"),
    ("Vocabulary", f"{metrics.get('vectorizer_config', {}).get('vocabulary_size', 0):,}"),
])

st.markdown("")

# --------------------------------------------------------------------------
# Navigation cards
# --------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    case_card_open("01 &middot; Single Article")
    st.write("Paste one article and get an instant verdict, confidence score, and word-level explanation.")
    st.page_link("pages/1_Predict.py", label="Open Predict →")
    case_card_close()

with col2:
    case_card_open("02 &middot; Bulk Analysis")
    st.write("Upload a CSV of headlines or articles and screen all of them at once, with a downloadable report.")
    st.page_link("pages/2_Bulk_Analysis.py", label="Open Bulk Analysis →")
    case_card_close()

with col3:
    case_card_open("03 &middot; Model Info")
    st.write("Training dataset, algorithm, vectorizer, and the full performance breakdown behind every verdict.")
    st.page_link("pages/3_Model_Info.py", label="Open Model Info →")
    case_card_close()

st.markdown("---")

# --------------------------------------------------------------------------
# Model comparison snapshot (Plotly, matching the notebook's model bake-off)
# --------------------------------------------------------------------------
eyebrow("Behind the Verdict")
st.markdown("### How the model was chosen")

if model_comparison:
    comp_df = pd.DataFrame(model_comparison)
    fig = go.Figure()
    metric_cols = [("test_accuracy", "Accuracy"), ("test_precision", "Precision"),
                   ("test_recall", "Recall"), ("test_f1", "F1"), ("test_roc_auc", "ROC-AUC")]
    colors = ["#8891A0", "#5B9BD5", "#E8A33D", "#2FB88A", "#D6455A"]
    for (col, label), color in zip(metric_cols, colors):
        fig.add_trace(go.Bar(name=label, x=comp_df["model"], y=comp_df[col], marker_color=color))

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        plot_bgcolor="#1A2027",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", color="#ECE9E2"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(range=[0.5, 1.0], gridcolor="#2A313D"),
        margin=dict(t=20, b=20),
        height=420,
    )
    st.plotly_chart(fig, width='stretch')
    st.caption(
        f"Five candidate models were trained and cross-validated on the same "
        f"{metrics.get('train_size', 0):,}-article training split. "
        f"**{metrics.get('best_model_name', 'N/A')}** won on held-out F1-score, "
        f"then had its hyperparameters tuned with RandomizedSearchCV."
    )
else:
    st.info("Model comparison data not found in evaluation_metrics.json.")

st.markdown("---")
st.caption(
    "This dashboard reads every number directly from `reports/evaluation_metrics.json`, "
    "generated by the Phase 1 training notebook. No metric on this page is hard-coded."
)
