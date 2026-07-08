"""
report_generator.py
=====================
Generates downloadable prediction reports (CSV and PDF) for both the
single-article and bulk-analysis pages.

Kept separate from the Streamlit pages so report formatting logic is
testable and reusable outside the app (e.g. a future scheduled batch job).
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a results DataFrame to CSV bytes for st.download_button."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def build_single_prediction_pdf(
    article_excerpt: str,
    prediction: str,
    confidence: float,
    probability_fake: float,
    risk_level: str,
    top_words: list[tuple[str, float]],
    model_name: str,
) -> bytes:
    """Build a one-article PDF prediction report.

    Returns
    -------
    bytes
        The PDF file content, ready to hand to `st.download_button`.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=18, spaceAfter=6)
    muted_style = ParagraphStyle("Muted", parent=styles["Normal"], textColor=colors.grey, fontSize=9)

    story = [
        Paragraph("Fake News Intelligence Platform", title_style),
        Paragraph("Single Article Prediction Report", styles["Heading2"]),
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &middot; Model: {model_name}",
            muted_style,
        ),
        Spacer(1, 16),
        Paragraph("Article Excerpt", styles["Heading3"]),
        Paragraph(article_excerpt[:800].replace("\n", " ") + ("..." if len(article_excerpt) > 800 else ""), styles["Normal"]),
        Spacer(1, 14),
        Paragraph("Verdict", styles["Heading3"]),
    ]

    verdict_color = colors.HexColor("#D6455A") if prediction.lower() == "fake" else colors.HexColor("#2FB88A")
    verdict_style = ParagraphStyle("Verdict", parent=styles["Normal"], textColor=verdict_color, fontSize=16, spaceAfter=8)
    story.append(Paragraph(f"<b>{prediction.upper()}</b>", verdict_style))

    result_table_data = [
        ["Confidence", f"{confidence * 100:.1f}%"],
        ["Probability (Fake)", f"{probability_fake * 100:.1f}%"],
        ["Risk Level", risk_level],
    ]
    result_table = Table(result_table_data, colWidths=[2.2 * inch, 2.2 * inch])
    result_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Top Contributing Words", styles["Heading3"]))
    word_rows = [["Word", "Contribution", "Direction"]]
    for word, contribution in top_words:
        direction = "→ Fake" if contribution > 0 else "→ Real"
        word_rows.append([word, f"{contribution:+.4f}", direction])
    word_table = Table(word_rows, colWidths=[2 * inch, 1.5 * inch, 1.2 * inch])
    word_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2A313D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(word_table)

    doc.build(story)
    return buffer.getvalue()


def build_bulk_prediction_pdf(summary: Dict[str, Any], results_df: pd.DataFrame, model_name: str) -> bytes:
    """Build a summary PDF report for a bulk CSV prediction run.

    Parameters
    ----------
    summary:
        Dict with keys like total_rows, fake_count, real_count, failed_count,
        avg_confidence.
    results_df:
        The full results DataFrame (only the first 40 rows are tabulated in
        the PDF to keep the file a reasonable size; the full data belongs in
        the CSV export).
    model_name:
        Name of the deployed model, for the report header.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=18, spaceAfter=6)
    muted_style = ParagraphStyle("Muted", parent=styles["Normal"], textColor=colors.grey, fontSize=9)

    story = [
        Paragraph("Fake News Intelligence Platform", title_style),
        Paragraph("Bulk Prediction Summary Report", styles["Heading2"]),
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &middot; Model: {model_name}",
            muted_style,
        ),
        Spacer(1, 16),
        Paragraph("Summary", styles["Heading3"]),
    ]

    summary_rows = [
        ["Total Rows Processed", f"{summary.get('total_rows', 0):,}"],
        ["Predicted Fake", f"{summary.get('fake_count', 0):,}"],
        ["Predicted Real", f"{summary.get('real_count', 0):,}"],
        ["Failed / Skipped Rows", f"{summary.get('failed_count', 0):,}"],
        ["Average Confidence", f"{summary.get('avg_confidence', 0) * 100:.1f}%"],
    ]
    summary_table = Table(summary_rows, colWidths=[2.6 * inch, 2 * inch])
    summary_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph(f"Sample of Results (first {min(40, len(results_df))} rows)", styles["Heading3"]))
    display_df = results_df.head(40).copy()
    table_data = [list(display_df.columns)] + display_df.astype(str).values.tolist()
    result_table = Table(table_data, repeatRows=1)
    result_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2A313D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(result_table)

    doc.build(story)
    return buffer.getvalue()
