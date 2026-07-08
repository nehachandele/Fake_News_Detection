"""
feature_engineering.py
=======================
Hand-crafted statistical text features for the AI-Powered Fake News
Intelligence Platform.

Two use cases are covered:

1. Batch feature engineering over a DataFrame (mirrors Phase 1 notebook,
   Section 5) -- used for EDA and could be re-used for future model
   experiments that blend these features with TF-IDF.
2. Single-text feature extraction for the Streamlit "News Statistics" and
   "Clickbait Detection" panels, where a user pastes one article and expects
   an instant, interpretable breakdown.

None of the numbers here are estimates or placeholders -- every value is
computed directly from the input text at call time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Precompiled patterns
# --------------------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_PUNCTUATION_CHARS = set(".,;:!?\"'")
_AVG_ADULT_READING_WPM = 200  # words per minute, standard adult silent-reading estimate

# --------------------------------------------------------------------------
# Clickbait detection
# --------------------------------------------------------------------------
# Each phrase is matched case-insensitively as a whole-word/phrase pattern.
# Weights reflect how strongly a phrase signals clickbait framing rather than
# neutral reporting; these are editorial judgment calls, not learned
# parameters, and are documented here for transparency and easy tuning.
CLICKBAIT_PHRASES: Dict[str, float] = {
    "breaking": 1.0,
    "shocking": 1.5,
    "you won't believe": 2.0,
    "you wont believe": 2.0,
    "secret": 1.5,
    "exclusive": 1.0,
    "urgent": 1.0,
    "this is why": 1.0,
    "what happens next": 1.5,
    "won't believe what": 2.0,
    "goes viral": 1.0,
    "number will shock you": 2.0,
    "doctors hate": 1.5,
    "one weird trick": 2.0,
    "click here": 1.5,
    "must see": 1.0,
    "gone wrong": 1.0,
    "unbelievable": 1.5,
}
_MAX_POSSIBLE_CLICKBAIT_SCORE = sum(CLICKBAIT_PHRASES.values())


@dataclass
class ClickbaitResult:
    """Result of running clickbait phrase detection on a piece of text."""

    score: float
    normalized_score: float  # 0-100 scale, for display as a progress bar / gauge
    matched_phrases: List[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        """Human-readable risk bucket for the UI."""
        if self.normalized_score >= 50:
            return "High"
        if self.normalized_score >= 20:
            return "Medium"
        return "Low"


def detect_clickbait(text: str) -> ClickbaitResult:
    """Scan text for known clickbait phrases and produce a clickbait score.

    Parameters
    ----------
    text:
        Raw (uncleaned) article text or title. Matching is case-insensitive
        and intentionally run on raw text rather than the lemmatized/
        stopword-stripped pipeline output, since clickbait phrases are exact
        idiomatic expressions that lemmatization would otherwise mangle.

    Returns
    -------
    ClickbaitResult
        `score` is the raw weighted sum of matched phrases, `normalized_score`
        rescales that to 0-100 against the maximum possible score (all
        phrases present), and `matched_phrases` lists which phrases fired.
    """
    if not text:
        return ClickbaitResult(score=0.0, normalized_score=0.0, matched_phrases=[])

    lowered = text.lower()
    matched = []
    score = 0.0
    for phrase, weight in CLICKBAIT_PHRASES.items():
        if phrase in lowered:
            matched.append(phrase)
            score += weight

    normalized = round(min(score / _MAX_POSSIBLE_CLICKBAIT_SCORE, 1.0) * 100, 1)
    return ClickbaitResult(score=round(score, 2), normalized_score=normalized, matched_phrases=matched)


# --------------------------------------------------------------------------
# Core text statistics
# --------------------------------------------------------------------------
def sentence_count(text: str) -> int:
    """Count sentences via terminal punctuation (. ! ?), minimum of 1."""
    if not text:
        return 0
    return max(len(_SENTENCE_SPLIT_RE.split(text)) - 1, 1)


def type_token_ratio(text: str) -> float:
    """Vocabulary richness: ratio of unique words to total words.

    A low ratio (lots of word repetition) is a mild signal of low-effort or
    templated writing; a high ratio suggests varied, deliberate vocabulary.
    Returns 0.0 for empty text rather than raising a division error.
    """
    words = text.split()
    if not words:
        return 0.0
    return round(len(set(w.lower() for w in words)) / len(words), 4)


def flesch_reading_ease(text: str) -> float:
    """Approximate Flesch Reading Ease score (0-100, higher = easier to read).

    Uses a simplified syllable-counting heuristic (vowel-group counting)
    rather than a dictionary lookup, which is standard practice for
    lightweight, dependency-free readability scoring and is accurate enough
    for relative comparisons between articles.
    """
    words = re.findall(r"[a-zA-Z']+", text)
    if not words:
        return 0.0

    sentences = max(sentence_count(text), 1)
    syllable_total = sum(_count_syllables(w) for w in words)

    words_per_sentence = len(words) / sentences
    syllables_per_word = syllable_total / len(words)

    score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    return round(max(min(score, 100.0), 0.0), 1)


def _count_syllables(word: str) -> int:
    """Heuristic syllable count via vowel-group counting."""
    word = word.lower()
    vowel_groups = re.findall(r"[aeiouy]+", word)
    count = len(vowel_groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


@dataclass
class TextStats:
    """Complete statistical profile of a single piece of text, as shown in
    the Streamlit 'News Statistics' panel."""

    char_count: int
    word_count: int
    sentence_count: int
    avg_word_length: float
    reading_time_minutes: float
    uppercase_ratio: float
    punctuation_count: int
    question_marks: int
    exclamation_marks: int
    vocabulary_richness: float
    reading_ease_score: float

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def compute_text_stats(text: str) -> TextStats:
    """Compute the full statistical profile for a single article/title.

    Parameters
    ----------
    text:
        Raw, uncleaned text as typed/pasted by the user.

    Returns
    -------
    TextStats
        All fields are computed live from `text`; there are no default or
        placeholder values substituted in.
    """
    text = text or ""
    words = text.split()

    return TextStats(
        char_count=len(text),
        word_count=len(words),
        sentence_count=sentence_count(text),
        avg_word_length=round(float(np.mean([len(w) for w in words])), 2) if words else 0.0,
        reading_time_minutes=round(len(words) / _AVG_ADULT_READING_WPM, 2),
        uppercase_ratio=round(sum(1 for c in text if c.isupper()) / max(len(text), 1), 4),
        punctuation_count=sum(1 for c in text if c in _PUNCTUATION_CHARS),
        question_marks=text.count("?"),
        exclamation_marks=text.count("!"),
        vocabulary_richness=type_token_ratio(text),
        reading_ease_score=flesch_reading_ease(text),
    )


# --------------------------------------------------------------------------
# Batch version (DataFrame), matching Phase 1 notebook Section 5 exactly
# --------------------------------------------------------------------------
def engineer_features(frame: pd.DataFrame, text_col: str = "text", title_col: str = "title") -> pd.DataFrame:
    """Add all engineered feature columns to a DataFrame in place-safe fashion.

    Parameters
    ----------
    frame:
        DataFrame containing at least `text_col` (and optionally `title_col`).
    text_col:
        Name of the column containing the main article body.
    title_col:
        Name of the column containing the headline/title, used only for
        `capital_letter_ratio`. If absent, that column is filled with 0.0.

    Returns
    -------
    pd.DataFrame
        A copy of `frame` with engineered feature columns appended. The
        original DataFrame is not mutated.
    """
    frame = frame.copy()
    texts = frame[text_col].fillna("")

    frame["word_count"] = texts.str.split().apply(len)
    frame["char_count"] = texts.str.len()
    frame["sentence_count"] = texts.apply(sentence_count)
    frame["avg_word_length"] = texts.apply(
        lambda t: float(np.mean([len(w) for w in t.split()])) if t.split() else 0.0
    )
    frame["reading_time_minutes"] = (frame["word_count"] / _AVG_ADULT_READING_WPM).round(2)
    frame["uppercase_ratio"] = texts.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )
    frame["punctuation_count"] = texts.apply(
        lambda t: sum(1 for c in t if c in _PUNCTUATION_CHARS)
    )
    frame["question_marks"] = texts.str.count(r"\?")
    frame["exclamation_marks"] = texts.str.count("!")

    if title_col in frame.columns:
        titles = frame[title_col].fillna("")
        frame["capital_letter_ratio"] = titles.apply(
            lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
        )
    else:
        frame["capital_letter_ratio"] = 0.0

    return frame


if __name__ == "__main__":
    # Quick manual smoke test: `python -m utils.feature_engineering`
    sample = (
        "BREAKING: You won't believe what scientists found in this SHOCKING "
        "new study!! Click here to find out more about the secret discovery."
    )
    stats = compute_text_stats(sample)
    clickbait = detect_clickbait(sample)

    print("Text stats:", stats.to_dict())
    print("Clickbait score:", clickbait.score, "| level:", clickbait.level, "| matched:", clickbait.matched_phrases)
