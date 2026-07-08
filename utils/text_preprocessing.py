"""
text_preprocessing.py
======================
Reusable, deterministic text-cleaning pipeline for the AI-Powered Fake News
Intelligence Platform.

This module is the single source of truth for text normalization. It is
imported both by the training notebook (Phase 1) and by the Streamlit
application (Phase 2), so training-time and inference-time preprocessing can
never drift apart -- a common and hard-to-debug source of production model
degradation ("train/serve skew").

Pipeline order
--------------
1. Lowercase
2. Strip HTML tags
3. Strip URLs
4. Strip email addresses
5. Expand contractions ("don't" -> "do not")
6. Strip numbers
7. Strip special characters / punctuation
8. Remove stopwords
9. Lemmatize remaining tokens
10. Normalize whitespace

Example
-------
>>> from utils.text_preprocessing import TextPreprocessor
>>> pre = TextPreprocessor()
>>> pre.clean("BREAKING: Visit http://fake.com or email us at a@b.com!! 100% true???")
'break visit email true'
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Third-party dependencies. Imported lazily-safe: if NLTK data is missing we
# raise a clear, actionable error instead of a cryptic LookupError deep in a
# request handler.
# --------------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "nltk is required for text_preprocessing.py. Install it with "
        "`pip install nltk`."
    ) from exc

try:
    import contractions
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "the 'contractions' package is required for text_preprocessing.py. "
        "Install it with `pip install contractions`."
    ) from exc


class PreprocessingError(Exception):
    """Raised when the text cleaning pipeline cannot process an input."""


# --------------------------------------------------------------------------
# Precompiled regex patterns (compiled once at import time for performance,
# since this pipeline runs on every training row and every live prediction).
# --------------------------------------------------------------------------
_HTML_RE = re.compile(r"<.*?>")
_URL_RE = re.compile(r"http\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NUMBER_RE = re.compile(r"\d+")
_SPECIAL_CHARS_RE = re.compile(r"[^a-z\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _ensure_nltk_data() -> None:
    """Download required NLTK corpora if they are not already present.

    Safe to call repeatedly -- nltk.download() is a no-op when data already
    exists locally, so this does not add latency after the first run.
    """
    required_packages = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, package in required_packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading missing NLTK package: %s", package)
            nltk.download(package, quiet=True)


@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset:
    _ensure_nltk_data()
    return frozenset(nltk_stopwords.words("english"))


@lru_cache(maxsize=1)
def _get_lemmatizer() -> WordNetLemmatizer:
    _ensure_nltk_data()
    return WordNetLemmatizer()


@dataclass
class TextPreprocessor:
    """Configurable text-cleaning pipeline.

    Parameters
    ----------
    min_token_length:
        Tokens shorter than this (after cleaning) are dropped. Defaults to 2,
        which removes single stray letters left over from punctuation
        stripping without discarding short meaningful words like "us" or "ai".
    remove_stopwords:
        Whether to drop English stopwords. Disable for tasks where stopword
        patterns themselves carry signal (rare for this project, but kept
        configurable rather than hard-coded).
    lemmatize:
        Whether to lemmatize tokens after cleaning. Disable for a faster,
        lighter pipeline if a downstream model does its own tokenization.

    Notes
    -----
    This class is intentionally stateless aside from configuration -- all
    heavy resources (stopword sets, lemmatizer) are module-level, cached
    singletons so creating many `TextPreprocessor` instances is cheap.
    """

    min_token_length: int = 2
    remove_stopwords: bool = True
    lemmatize: bool = True

    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline on a single string.

        Parameters
        ----------
        text:
            Raw article text (or title, or any free text field).

        Returns
        -------
        str
            Cleaned, lowercased, lemmatized text with a single space between
            tokens. Returns an empty string for null/empty/whitespace-only
            input rather than raising, since this is a common and expected
            case in real-world scraped news data.

        Raises
        ------
        PreprocessingError
            If an unexpected error occurs during cleaning (e.g. a corrupt
            unicode string that regex cannot process). Callers in a batch
            pipeline should catch this per-row rather than letting one bad
            record kill an entire training run.
        """
        if text is None or not isinstance(text, str) or not text.strip():
            return ""

        try:
            cleaned = text.lower()
            cleaned = _HTML_RE.sub(" ", cleaned)
            cleaned = _URL_RE.sub(" ", cleaned)
            cleaned = _EMAIL_RE.sub(" ", cleaned)

            try:
                cleaned = contractions.fix(cleaned)
            except Exception:  # noqa: BLE001 - contractions can choke on odd unicode
                logger.debug("contractions.fix failed on input; continuing without expansion")

            cleaned = _NUMBER_RE.sub(" ", cleaned)
            cleaned = _SPECIAL_CHARS_RE.sub(" ", cleaned)

            tokens = cleaned.split()
            tokens = self._filter_and_normalize_tokens(tokens)

            cleaned = " ".join(tokens)
            cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
            return cleaned

        except Exception as exc:  # noqa: BLE001
            raise PreprocessingError(f"Failed to clean text: {exc}") from exc

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a list of texts, skipping (as empty strings) any row that
        fails rather than aborting the whole batch.
        """
        results: List[str] = []
        for i, text in enumerate(texts):
            try:
                results.append(self.clean(text))
            except PreprocessingError as exc:
                logger.warning("Skipping row %d due to preprocessing error: %s", i, exc)
                results.append("")
        return results

    def _filter_and_normalize_tokens(self, tokens: List[str]) -> List[str]:
        stop_set = _get_stopwords() if self.remove_stopwords else frozenset()
        lemmatizer = _get_lemmatizer() if self.lemmatize else None

        filtered = [
            tok for tok in tokens
            if tok not in stop_set and len(tok) >= self.min_token_length
        ]
        if lemmatizer is not None:
            filtered = [lemmatizer.lemmatize(tok) for tok in filtered]
        return filtered


# --------------------------------------------------------------------------
# Module-level convenience function for simple one-off calls, so callers who
# don't need custom configuration can do `from utils.text_preprocessing
# import clean_text` without instantiating a class themselves.
# --------------------------------------------------------------------------
_default_preprocessor = TextPreprocessor()


def clean_text(text: str) -> str:
    """Clean a single string using the default `TextPreprocessor` configuration.

    Equivalent to `TextPreprocessor().clean(text)`. Provided as a
    module-level function because it is the most common call site (both the
    notebook and the Streamlit app use this exact signature).
    """
    return _default_preprocessor.clean(text)


if __name__ == "__main__":
    # Quick manual smoke test: `python -m utils.text_preprocessing`
    logging.basicConfig(level=logging.INFO)
    samples = [
        "BREAKING NEWS!!! Visit http://fake-site.com or email tips@fake.com now!!",
        "Scientists don't believe the claim; it's been debunked 100 times.",
        None,
        "   ",
        "<p>Reuters reports that the Senate passed the bill on Tuesday.</p>",
    ]
    pre = TextPreprocessor()
    for s in samples:
        print(f"RAW:   {s!r}")
        print(f"CLEAN: {pre.clean(s)!r}")
        print("-" * 60)
