"""
pages/2_Bulk_Analysis.py
==========================
Bulk News Prediction page: upload a CSV, screen every row, visualize results
with Plotly, and download a CSV or PDF report.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.model_utils import ModelLoadError, load_artifacts, predict_batch, warmup
from utils.report_generator import build_bulk_prediction_pdf, dataframe_to_csv_bytes
from utils.ui import case_card_close, case_card_open, eyebrow, inject_theme, render_sidebar

st.set_page_config(page_title="Bulk Analysis | Fake News Intelligence", page_icon="📊", layout="wide")
inject_theme()


@st.cache_resource(show_spinner="Loading model artifacts...")
def get_artifacts():
    artifacts = load_artifacts(models_dir="models", reports_dir="reports")
    warmup(artifacts)
    return artifacts


try:
    artifacts = get_artifacts()
except ModelLoadError as exc:
    st.error(f"Model artifacts not found: {exc}")
    st.stop()

render_sidebar(artifacts)

eyebrow("Batch Intake")
st.markdown("# Bulk Analysis")
st.markdown(
    "Upload a CSV containing a column of article text (or headlines). "
    "Every row is screened independently and results can be exported as CSV or PDF."
)

MAX_ROWS = 5000

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # ---- Robust validation: invalid / malformed CSV ----
    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(
            f"Could not read this file as a CSV: {exc}\n\n"
            "Make sure it's a valid, comma-separated CSV file with a header row."
        )
        st.stop()

    if raw_df.empty:
        st.warning("The uploaded CSV has no rows.")
        st.stop()

    if len(raw_df) > MAX_ROWS:
        st.warning(f"This file has {len(raw_df):,} rows; only the first {MAX_ROWS:,} will be processed.")
        raw_df = raw_df.head(MAX_ROWS)

    # ---- Column selection ----
    candidate_cols = [c for c in raw_df.columns if raw_df[c].dtype == object]
    if not candidate_cols:
        st.error("No text columns found in this CSV. At least one column must contain article text.")
        st.stop()

    default_idx = candidate_cols.index("text") if "text" in candidate_cols else 0
    text_column = st.selectbox("Which column contains the article text?", candidate_cols, index=default_idx)

    st.dataframe(raw_df.head(5), width='stretch')

    if st.button("Run Bulk Prediction", width='content'):
        texts = raw_df[text_column].fillna("").astype(str).tolist()

        progress = st.progress(0.0, text="Scoring articles...")
        results = []
        batch_size = 200
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            results.extend(predict_batch(chunk, artifacts))
            progress.progress(min((start + batch_size) / len(texts), 1.0), text=f"Scored {min(start + batch_size, len(texts)):,} / {len(texts):,}")
        progress.empty()

        # ---- Assemble results dataframe ----
        out_df = raw_df.copy()
        out_df["prediction"] = [r.prediction if r else "FAILED" for r in results]
        out_df["confidence"] = [r.confidence if r else None for r in results]
        out_df["probability_fake"] = [r.probability_fake if r else None for r in results]
        out_df["risk_level"] = [r.risk_level if r else None for r in results]

        st.session_state["bulk_results_df"] = out_df
        st.session_state["bulk_results_meta"] = {"text_column": text_column, "raw_texts": texts}

    # --------------------------------------------------------------------
    # Render results if we have them (persisted in session_state so download
    # buttons don't wipe the results on click, which is a Streamlit rerun quirk)
    # --------------------------------------------------------------------
    if "bulk_results_df" in st.session_state:
        out_df = st.session_state["bulk_results_df"]
        valid_df = out_df[out_df["prediction"] != "FAILED"]
        failed_count = (out_df["prediction"] == "FAILED").sum()

        st.markdown("---")
        eyebrow("Results")
        st.markdown("### Batch Summary")

        fake_count = int((valid_df["prediction"] == "Fake").sum())
        real_count = int((valid_df["prediction"] == "Real").sum())
        avg_conf = float(valid_df["confidence"].mean()) if len(valid_df) else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", f"{len(out_df):,}")
        m2.metric("Predicted Fake", f"{fake_count:,}")
        m3.metric("Predicted Real", f"{real_count:,}")
        m4.metric("Failed Rows", f"{failed_count:,}")

        st.dataframe(out_df, width='stretch', height=320)

        # ---- Plotly visualizations ----
        st.markdown("### Visualizations")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            fig_dist = go.Figure(data=[go.Pie(
                labels=["Fake", "Real"], values=[fake_count, real_count], hole=0.55,
                marker_colors=["#D6455A", "#2FB88A"],
            )])
            fig_dist.update_layout(
                title="Prediction Distribution", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="IBM Plex Mono", color="#ECE9E2"),
                height=360,
            )
            st.plotly_chart(fig_dist, width='stretch')

        with viz_col2:
            fig_conf = go.Figure(data=[go.Histogram(
                x=valid_df["confidence"], nbinsx=20, marker_color="#E8A33D",
            )])
            fig_conf.update_layout(
                title="Confidence Histogram", template="plotly_dark",
                plot_bgcolor="#1A2027", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Mono", color="#ECE9E2"),
                xaxis_title="Confidence", yaxis_title="Count", height=360,
            )
            st.plotly_chart(fig_conf, width='stretch')

        viz_col3, viz_col4 = st.columns(2)

        with viz_col3:
            risk_counts = valid_df["risk_level"].value_counts()
            fig_class = go.Figure(data=[go.Bar(
                x=risk_counts.index, y=risk_counts.values, marker_color="#5B9BD5",
            )])
            fig_class.update_layout(
                title="Class / Risk-Level Distribution", template="plotly_dark",
                plot_bgcolor="#1A2027", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Mono", color="#ECE9E2"), height=360,
            )
            st.plotly_chart(fig_class, width='stretch')

        with viz_col4:
            # Top influential words across the batch, via the explainer model
            from utils.model_utils import explain_prediction
            word_counter = Counter()
            sample_texts = st.session_state["bulk_results_meta"]["raw_texts"][:300]  # cap for latency
            for t in sample_texts:
                try:
                    exp = explain_prediction(t, artifacts, top_k=5)
                    for w in exp.top_contributing_words:
                        word_counter[w.word] += abs(w.contribution)
                except Exception:
                    continue
            top_words = word_counter.most_common(15)
            if top_words:
                words, weights = zip(*top_words)
                fig_words = go.Figure(data=[go.Bar(
                    x=list(weights)[::-1], y=list(words)[::-1], orientation="h", marker_color="#8891A0",
                )])
                fig_words.update_layout(
                    title="Top Influential Words (batch)", template="plotly_dark",
                    plot_bgcolor="#1A2027", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="IBM Plex Mono", color="#ECE9E2"), height=360,
                )
                st.plotly_chart(fig_words, width='stretch')
            else:
                st.info("Not enough data to compute top influential words.")

        # ---- Downloads ----
        st.markdown("### Download Reports")
        dl_col1, dl_col2 = st.columns(2)

        csv_bytes = dataframe_to_csv_bytes(out_df)
        with dl_col1:
            st.download_button(
                "Download CSV", data=csv_bytes, file_name="fake_news_bulk_predictions.csv",
                mime="text/csv", width='stretch',
            )

        summary = {
            "total_rows": len(out_df), "fake_count": fake_count, "real_count": real_count,
            "failed_count": int(failed_count), "avg_confidence": avg_conf,
        }
        pdf_bytes = build_bulk_prediction_pdf(summary, out_df, artifacts.model_name)
        with dl_col2:
            st.download_button(
                "Download PDF Report", data=pdf_bytes, file_name="fake_news_bulk_report.pdf",
                mime="application/pdf", width='stretch',
            )
else:
    st.info("Upload a CSV file to begin. It should have at least one column containing article text.")
    case_card_open("Expected Format")
    st.code("id,title,text\n1,\"Some headline\",\"Full article text goes here...\"", language="csv")
    case_card_close()
