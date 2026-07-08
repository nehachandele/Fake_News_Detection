"""
pages/1_Predict.py
====================
Single News Prediction page: textarea input, verdict stamp, confidence,
explainable AI word contributions, news statistics, and clickbait detection.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.feature_engineering import compute_text_stats, detect_clickbait
from utils.logging_config import log_prediction
from utils.model_utils import ModelLoadError, PredictionError, explain_prediction, load_artifacts, predict, warmup
from utils.report_generator import build_single_prediction_pdf
from utils.ui import case_card_close, case_card_open, eyebrow, inject_theme, render_sidebar, tag_pill, verdict_stamp, word_chip

st.set_page_config(page_title="Predict | Fake News Intelligence", page_icon="🔍", layout="wide")
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

eyebrow("Case Intake")
st.markdown("# Predict")
st.markdown("Paste an article below. The model reads only the text — no source, byline, or metadata influences the verdict.")

# --------------------------------------------------------------------------
# Input + validation
# --------------------------------------------------------------------------
MAX_CHARS = 50_000
MIN_CHARS = 20

article_text = st.text_area(
    "Article text",
    height=260,
    max_chars=MAX_CHARS,
    placeholder="Paste the full article text here (or just a headline)...",
    label_visibility="collapsed",
)

col_btn, col_hint = st.columns([1, 4])
with col_btn:
    run_clicked = st.button("Analyze Article", width='stretch')
with col_hint:
    st.caption(f"{len(article_text):,} / {MAX_CHARS:,} characters")

if run_clicked:
    stripped = article_text.strip()

    # ---- Robust validation ----
    if not stripped:
        st.warning("Enter some article text before analyzing — the field is currently empty.")
        st.stop()
    if len(stripped) < MIN_CHARS:
        st.warning(f"That's only {len(stripped)} characters. Enter at least {MIN_CHARS} for a meaningful prediction.")
        st.stop()

    # ---- Run prediction + explanation ----
    try:
        with st.spinner("Scoring article..."):
            result = predict(stripped, artifacts)
            explanation = explain_prediction(stripped, artifacts, top_k=12)
            stats = compute_text_stats(stripped)
            clickbait = detect_clickbait(stripped)
    except PredictionError as exc:
        st.error(f"Could not score this article: {exc}")
        st.stop()

    log_prediction(
        text=stripped,
        prediction=result.prediction,
        confidence=result.confidence,
        inference_time_ms=result.inference_time_ms,
        source="single",
        extra={"risk_level": result.risk_level},
    )

    st.markdown("---")

    # ---- Verdict + confidence ----
    verdict_col, meta_col = st.columns([1, 1.4])
    with verdict_col:
        verdict_stamp(result.prediction)
    with meta_col:
        st.markdown(f"**Confidence:** {result.confidence * 100:.1f}%")
        st.progress(result.confidence)
        m1, m2, m3 = st.columns(3)
        m1.metric("P(Fake)", f"{result.probability_fake * 100:.1f}%")
        m2.metric("P(Real)", f"{result.probability_real * 100:.1f}%")
        m3.metric("Inference Time", f"{result.inference_time_ms:.2f} ms")
        risk_kind = "fake" if "risk" in result.risk_level.lower() and result.prediction == "Fake" else (
            "real" if result.prediction == "Real" else "amber"
        )
        st.markdown(tag_pill(result.risk_level, kind="amber"), unsafe_allow_html=True)

    st.markdown("---")

    # ---- Explainable AI ----
    eyebrow("Explainable AI")
    st.markdown("### Why the model reached this verdict")
    st.caption(
        "Each word below is weighted by (its TF-IDF score) × (its learned coefficient in the "
        "explainer model). Red pushes toward Fake, green pushes toward Real. Intensity = strength of evidence."
    )

    if explanation.top_contributing_words:
        max_abs = max(abs(w.contribution) for w in explanation.top_contributing_words)
        chips_html = "".join(
            word_chip(w.word, w.contribution, max_abs) for w in explanation.top_contributing_words
        )
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.info("No individually weighted words were found in this text (it may be too short or too generic).")

    st.markdown("---")

    # ---- Download report ----
    eyebrow("Export")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        single_csv = pd.DataFrame([{
            "text_excerpt": stripped[:200],
            "prediction": result.prediction,
            "confidence": result.confidence,
            "probability_fake": result.probability_fake,
            "probability_real": result.probability_real,
            "risk_level": result.risk_level,
            "inference_time_ms": result.inference_time_ms,
        }])
        st.download_button(
            "Download CSV", data=single_csv.to_csv(index=False).encode("utf-8"),
            file_name="fake_news_prediction.csv", mime="text/csv", width='stretch',
        )
    with dl_col2:
        pdf_bytes = build_single_prediction_pdf(
            article_excerpt=stripped,
            prediction=result.prediction,
            confidence=result.confidence,
            probability_fake=result.probability_fake,
            risk_level=result.risk_level,
            top_words=[(w.word, w.contribution) for w in explanation.top_contributing_words],
            model_name=artifacts.model_name,
        )
        st.download_button(
            "Download PDF Report", data=pdf_bytes, file_name="fake_news_prediction_report.pdf",
            mime="application/pdf", width='stretch',
        )

    st.markdown("---")

    # ---- News Statistics + Clickbait side by side ----
    stat_col, click_col = st.columns(2)

    with stat_col:
        case_card_open("News Statistics")
        s1, s2 = st.columns(2)
        s1.metric("Characters", f"{stats.char_count:,}")
        s1.metric("Sentences", f"{stats.sentence_count:,}")
        s1.metric("Reading Time", f"{stats.reading_time_minutes:.1f} min")
        s2.metric("Words", f"{stats.word_count:,}")
        s2.metric("Avg Word Length", f"{stats.avg_word_length:.2f}")
        s2.metric("Vocabulary Richness", f"{stats.vocabulary_richness:.2f}")
        st.caption(f"Flesch Reading Ease: {stats.reading_ease_score:.0f}/100 (higher = easier to read)")
        case_card_close()

    with click_col:
        case_card_open("Clickbait Detection")
        st.markdown(f"**Clickbait Score:** {clickbait.normalized_score:.0f}/100")
        st.progress(clickbait.normalized_score / 100)
        st.markdown(tag_pill(f"{clickbait.level} Clickbait Signal", kind="amber"), unsafe_allow_html=True)
        if clickbait.matched_phrases:
            st.caption("Matched phrases:")
            st.markdown(
                "".join(tag_pill(p, kind="fake") for p in clickbait.matched_phrases),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No known clickbait phrases detected.")
        case_card_close()

else:
    st.info("Paste an article above and click **Analyze Article** to see the verdict, explanation, and statistics.")
