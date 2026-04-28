"""
plagiarism.py
Plagiarism detection using:
1. TF-IDF cosine similarity on sentence level (highlight suspicious sentences)
2. Cross-paper comparison against all papers stored in the SQLite DB
"""

import re
import sqlite3
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DB_PATH = "research_history.db"

# Sentences with cosine similarity above this threshold are flagged
SENTENCE_THRESHOLD = 0.72
# Papers with overall similarity above this are flagged as matching
PAPER_THRESHOLD = 0.20


def split_into_sentences(text: str) -> List[str]:
    """Split text into individual sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Keep only meaningful sentences (more than 6 words)
    return [s.strip() for s in sentences if len(s.split()) > 6]


def check_plagiarism_sentences(text: str) -> Tuple[List[Tuple[str, bool]], int]:
    """
    Detect suspicious sentences using TF-IDF self-similarity clustering.
    Sentences that are highly similar to multiple other sentences in the
    same document are flagged (detects paraphrasing / repetition patterns).

    Returns:
        - List of (sentence, is_suspicious) tuples
        - Overall plagiarism percentage (0-100)
    """
    sentences = split_into_sentences(text)

    if len(sentences) < 3:
        return [(s, False) for s in sentences], 0

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words="english",
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        suspicious_flags = []
        for i, sentence in enumerate(sentences):
            # Get similarities with all OTHER sentences
            sims = [similarity_matrix[i][j] for j in range(len(sentences)) if j != i]
            max_sim = max(sims) if sims else 0
            # Flag if very similar to at least one other sentence
            is_suspicious = max_sim >= SENTENCE_THRESHOLD
            suspicious_flags.append((sentence, is_suspicious))

        suspicious_count = sum(1 for _, flag in suspicious_flags if flag)
        plag_percent = round((suspicious_count / len(sentences)) * 100)

        return suspicious_flags, plag_percent

    except Exception:
        return [(s, False) for s in sentences], 0


def compare_with_db(current_text: str) -> List[Dict]:
    """
    Compare the current paper's text against all previously stored papers in the DB.
    Returns a list of matches with filename, similarity %, and upload date.
    """
    matches = []

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT filename, summary, uploaded_at FROM papers ORDER BY uploaded_at DESC")
        stored_papers = cursor.fetchall()
        conn.close()
    except Exception:
        return []

    if not stored_papers:
        return []

    # Use summary text for comparison (faster than full text)
    texts_to_compare = []
    meta = []
    for filename, summary, uploaded_at in stored_papers:
        if summary and len(summary.strip()) > 50:
            texts_to_compare.append(summary)
            meta.append((filename, uploaded_at))

    if not texts_to_compare:
        return []

    try:
        # Truncate current text to first 3000 chars for speed
        current_sample = current_text[:3000]
        all_texts = [current_sample] + texts_to_compare

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Compare current (index 0) against all stored papers
        current_vec = tfidf_matrix[0]
        for i, (filename, uploaded_at) in enumerate(meta):
            stored_vec = tfidf_matrix[i + 1]
            sim = cosine_similarity(current_vec, stored_vec)[0][0]
            sim_percent = round(sim * 100)
            if sim >= PAPER_THRESHOLD:
                matches.append({
                    "filename": filename,
                    "similarity": sim_percent,
                    "date": uploaded_at
                })

        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)

    except Exception:
        return []

    return matches
