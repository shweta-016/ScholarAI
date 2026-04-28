"""
evaluation.py
Simple evaluation metrics for the generated summary:
- Word count
- Sentence count
- Compression ratio (summary vs original)
- Key section coverage check
"""

from typing import Dict


def evaluate_summary(original_text: str, summary: str) -> Dict[str, str]:
    """
    Returns a dictionary of evaluation metrics for the summary.
    """
    original_words = len(original_text.split())
    summary_words = len(summary.split())
    original_sentences = original_text.count('.') + original_text.count('!') + original_text.count('?')
    summary_sentences = summary.count('.') + summary.count('!') + summary.count('?')

    compression = round((1 - summary_words / max(original_words, 1)) * 100, 1)

    # Check if key sections are present in summary
    key_sections = ["problem", "method", "result", "conclusion", "finding", "approach", "contribution"]
    summary_lower = summary.lower()
    covered = [s for s in key_sections if s in summary_lower]
    coverage_score = f"{len(covered)}/{len(key_sections)}"

    return {
        "📄 Original Word Count": f"{original_words:,} words",
        "📝 Summary Word Count": f"{summary_words:,} words",
        "📉 Compression Ratio": f"{compression}% shorter",
        "🔢 Summary Sentences": str(summary_sentences),
        "✅ Key Section Coverage": coverage_score,
    }
