"""
logging_config.py
===================
Prediction logging for the AI-Powered Fake News Intelligence Platform.

Every prediction made through the Streamlit app is appended to a local
log file (JSON Lines format, one record per line) so usage can be
audited, debugged, or analyzed later -- e.g. to spot drift in the kinds
of articles being submitted over time.

This intentionally does not log full article text by default (to avoid
bloating the log with potentially large/sensitive content) -- only a
short excerpt, a hash, and the prediction outcome.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path("logs")
PREDICTION_LOG_FILE = LOG_DIR / "predictions.jsonl"

logger = logging.getLogger(__name__)


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_prediction(
    text: str,
    prediction: str,
    confidence: float,
    inference_time_ms: float,
    source: str = "single",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one prediction record to the prediction log.

    Parameters
    ----------
    text:
        The (raw) article text that was scored. Only a short excerpt and a
        SHA-256 hash are persisted, not the full text.
    prediction:
        "Fake" or "Real".
    confidence:
        Model confidence in [0, 1].
    inference_time_ms:
        How long the prediction took, in milliseconds.
    source:
        "single" or "bulk", to distinguish which page the prediction came from.
    extra:
        Any additional fields to merge into the log record (e.g. risk_level).

    Notes
    -----
    Logging failures are caught and logged to the standard logger rather
    than raised -- a broken log write should never break a user-facing
    prediction.
    """
    try:
        _ensure_log_dir()
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "text_excerpt": text[:120],
            "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "text_length": len(text),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "inference_time_ms": round(inference_time_ms, 2),
        }
        if extra:
            record.update(extra)

        with open(PREDICTION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write prediction log: %s", exc)


def read_recent_predictions(limit: int = 100) -> list[Dict[str, Any]]:
    """Read the most recent `limit` prediction log records, newest first.

    Returns an empty list if the log file doesn't exist yet (e.g. on a
    freshly deployed instance with no predictions made).
    """
    if not PREDICTION_LOG_FILE.exists():
        return []

    try:
        with open(PREDICTION_LOG_FILE, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read prediction log: %s", exc)
        return []

    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(records))
