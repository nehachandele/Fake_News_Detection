"""
model_utils.py
===============
Model loading, prediction, and explainability utilities for the AI-Powered
Fake News Intelligence Platform.

This module is intentionally framework-agnostic (no Streamlit imports) so it
can be unit-tested and reused outside the app -- e.g. in a batch scoring
script or a future API service. The Streamlit app wraps `load_artifacts()`
with `st.cache_resource` so the model is loaded from disk exactly once per
server process, per the "load model only once" requirement.

Artifacts expected (produced by the Phase 1 notebook)
------------------------------------------------------
- models/best_model.pkl        -- the deployed production classifier
- models/vectorizer.pkl        -- the fitted TF-IDF vectorizer
- models/explainer_model.pkl   -- a Logistic Regression model sharing the
                                   same vectorizer, used purely for
                                   word-level explainability
- models/label_mapping.json    -- {"0": "Real", "1": "Fake"}
- reports/evaluation_metrics.json -- metrics captured at training time,
                                   displayed as-is in the app's Model Info page
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from utils.text_preprocessing import clean_text

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Raised when required model artifacts cannot be found or loaded."""


class PredictionError(Exception):
    """Raised when a prediction cannot be produced for a given input."""


# --------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------
@dataclass
class ModelArtifacts:
    """Bundle of everything needed to serve predictions and explanations."""

    model: Any
    vectorizer: Any
    explainer_model: Any
    label_mapping: Dict[str, str]
    evaluation_metrics: Dict[str, Any]
    model_name: str = field(init=False)

    def __post_init__(self) -> None:
        self.model_name = self.evaluation_metrics.get("best_model_name", type(self.model).__name__)


def warmup(artifacts: "ModelArtifacts") -> float:
    """Run one throwaway prediction to force all lazy-loaded dependencies
    (notably the `contractions` dictionary, which costs ~2-3s on its very
    first use per process) to initialize before the first real user request.

    Call this once, immediately after `load_artifacts()`, at app startup.
    Returns the elapsed warmup time in milliseconds, purely for logging.
    """
    start = time.perf_counter()
    try:
        predict("Warmup request to initialize lazy-loaded NLP dependencies.", artifacts)
    except PredictionError:
        pass  # warmup text is only meant to trigger imports, result is discarded
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("Model warmup completed in %.1f ms", elapsed_ms)
    return round(elapsed_ms, 2)


def load_artifacts(models_dir: str = "models", reports_dir: str = "reports") -> ModelArtifacts:
    """Load all model artifacts from disk.

    Parameters
    ----------
    models_dir:
        Directory containing best_model.pkl, vectorizer.pkl,
        explainer_model.pkl, label_mapping.json.
    reports_dir:
        Directory containing evaluation_metrics.json.

    Returns
    -------
    ModelArtifacts

    Raises
    ------
    ModelLoadError
        If any required file is missing or fails to deserialize. This is
        raised eagerly, at startup, rather than surfacing as a confusing
        error on the first prediction request.
    """
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)

    required_files = {
        "model": models_path / "best_model.pkl",
        "vectorizer": models_path / "vectorizer.pkl",
        "explainer_model": models_path / "explainer_model.pkl",
        "label_mapping": models_path / "label_mapping.json",
        "evaluation_metrics": reports_path / "evaluation_metrics.json",
    }

    missing = [str(p) for p in required_files.values() if not p.exists()]
    if missing:
        raise ModelLoadError(
            "Missing required model artifact(s): " + ", ".join(missing) +
            ". Re-run the Phase 1 notebook to generate them."
        )

    try:
        model = joblib.load(required_files["model"])
        vectorizer = joblib.load(required_files["vectorizer"])
        explainer_model = joblib.load(required_files["explainer_model"])
        with open(required_files["label_mapping"]) as f:
            label_mapping = json.load(f)
        with open(required_files["evaluation_metrics"]) as f:
            evaluation_metrics = json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Failed to load model artifacts: {exc}") from exc

    logger.info("Model artifacts loaded successfully (model=%s)", type(model).__name__)
    return ModelArtifacts(
        model=model,
        vectorizer=vectorizer,
        explainer_model=explainer_model,
        label_mapping=label_mapping,
        evaluation_metrics=evaluation_metrics,
    )


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _decision_score(model: Any, features) -> np.ndarray:
    """Return a probability-like score in [0, 1] for the positive (Fake) class,
    regardless of whether the underlying model exposes predict_proba or only
    decision_function (e.g. LinearSVC, PassiveAggressiveClassifier).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]

    if hasattr(model, "decision_function"):
        raw = model.decision_function(features)
        # Squash the unbounded decision function into (0, 1) for display
        # purposes only. This is a monotonic transform (sigmoid), so it
        # preserves ranking/ordering exactly -- it does not change which
        # class wins, only how the confidence is displayed.
        return 1.0 / (1.0 + np.exp(-raw))

    raise PredictionError(
        f"Model {type(model).__name__} exposes neither predict_proba nor decision_function."
    )


def risk_level(probability_fake: float) -> str:
    """Bucket a fake-probability into a human-readable risk level for the UI."""
    if probability_fake >= 0.80:
        return "High Risk"
    if probability_fake >= 0.55:
        return "Moderate Risk"
    if probability_fake >= 0.45:
        return "Uncertain"
    if probability_fake >= 0.20:
        return "Likely Reliable"
    return "Low Risk"


@dataclass
class PredictionResult:
    """Everything the Streamlit prediction panel needs to render a result."""

    prediction: str            # "Fake" or "Real"
    confidence: float           # 0-1, probability of the winning class
    probability_fake: float     # 0-1
    probability_real: float     # 0-1
    risk_level: str
    inference_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def predict(text: str, artifacts: ModelArtifacts) -> PredictionResult:
    """Run the full pipeline (clean -> vectorize -> predict) on a single article.

    Parameters
    ----------
    text:
        Raw article text as submitted by the user.
    artifacts:
        Loaded `ModelArtifacts` (see `load_artifacts`).

    Returns
    -------
    PredictionResult

    Raises
    ------
    PredictionError
        If `text` is empty/whitespace-only after cleaning, or if the
        underlying model fails to score it.
    """
    start = time.perf_counter()

    cleaned = clean_text(text)
    if not cleaned:
        raise PredictionError(
            "Input text produced no usable content after cleaning "
            "(it may be empty, too short, or contain no alphabetic words)."
        )

    try:
        features = artifacts.vectorizer.transform([cleaned])
        pred_label = int(artifacts.model.predict(features)[0])
        proba_fake = float(_decision_score(artifacts.model, features)[0])
    except Exception as exc:  # noqa: BLE001
        raise PredictionError(f"Model failed to score input: {exc}") from exc

    proba_real = 1.0 - proba_fake
    confidence = proba_fake if pred_label == 1 else proba_real
    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictionResult(
        prediction=artifacts.label_mapping.get(str(pred_label), str(pred_label)),
        confidence=round(confidence, 4),
        probability_fake=round(proba_fake, 4),
        probability_real=round(proba_real, 4),
        risk_level=risk_level(proba_fake),
        inference_time_ms=round(elapsed_ms, 2),
    )


def predict_batch(texts: List[str], artifacts: ModelArtifacts) -> List[Optional[PredictionResult]]:
    """Predict for a list of texts (used by the bulk CSV upload feature).

    Rows that fail (e.g. empty text) yield `None` at that position rather
    than raising, so one bad row in a large CSV upload doesn't abort the
    whole batch. Callers should filter/report `None` entries to the user.
    """
    results: List[Optional[PredictionResult]] = []
    for i, text in enumerate(texts):
        try:
            results.append(predict(text, artifacts))
        except PredictionError as exc:
            logger.warning("Row %d failed prediction: %s", i, exc)
            results.append(None)
    return results


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
@dataclass
class WordContribution:
    word: str
    contribution: float  # signed: positive pushes toward Fake, negative toward Real


@dataclass
class ExplanationResult:
    prediction: str
    confidence: float
    probability_fake: float
    top_contributing_words: List[WordContribution]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probability_fake": self.probability_fake,
            "top_contributing_words": [
                {"word": w.word, "contribution": w.contribution} for w in self.top_contributing_words
            ],
        }


def explain_prediction(text: str, artifacts: ModelArtifacts, top_k: int = 10) -> ExplanationResult:
    """Explain a prediction via per-word TF-IDF-weight x coefficient contributions.

    Uses `artifacts.explainer_model` (a Logistic Regression model sharing the
    same vectorizer) rather than the deployed `artifacts.model`, since only a
    linear model with `coef_` gives a directly interpretable per-word
    contribution score. This keeps explanations available and correct
    regardless of which model wins the "best model" selection.

    Parameters
    ----------
    text:
        Raw article text.
    artifacts:
        Loaded `ModelArtifacts`.
    top_k:
        Number of top (by absolute contribution) words to return.

    Returns
    -------
    ExplanationResult
    """
    cleaned = clean_text(text)
    if not cleaned:
        raise PredictionError("Input text produced no usable content after cleaning.")

    vec = artifacts.vectorizer.transform([cleaned])
    proba_fake = float(artifacts.explainer_model.predict_proba(vec)[0, 1])
    pred_label = int(artifacts.explainer_model.predict(vec)[0])

    feature_names = np.array(artifacts.vectorizer.get_feature_names_out())
    coefs = artifacts.explainer_model.coef_[0]

    nonzero_indices = vec.nonzero()[1]
    contributions = [
        WordContribution(word=feature_names[i], contribution=round(float(vec[0, i] * coefs[i]), 5))
        for i in nonzero_indices
    ]
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

    confidence = proba_fake if pred_label == 1 else (1.0 - proba_fake)

    return ExplanationResult(
        prediction=artifacts.label_mapping.get(str(pred_label), str(pred_label)),
        confidence=round(confidence, 4),
        probability_fake=round(proba_fake, 4),
        top_contributing_words=contributions[:top_k],
    )


if __name__ == "__main__":
    # Manual smoke test: `python -m utils.model_utils` (run from project root,
    # requires models/ and reports/ to already exist from the Phase 1 notebook)
    logging.basicConfig(level=logging.INFO)
    art = load_artifacts()
    print(f"Loaded model: {art.model_name}")
    warmup_ms = warmup(art)
    print(f"Warmup took {warmup_ms} ms (one-time cost, absorbed at startup)")

    sample_text = (
        "WASHINGTON (Reuters) - The Senate voted on Tuesday to advance the "
        "infrastructure bill after months of negotiation between both parties."
    )
    result = predict(sample_text, art)
    print("Prediction:", result.to_dict())

    explanation = explain_prediction(sample_text, art)
    print("Explanation:", explanation.to_dict())
