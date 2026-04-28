"""
database.py
SQLite database for storing:
- Uploaded paper metadata and summaries
- Q&A history per paper session
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Tuple, Optional

DB_PATH = "research_history.db"


def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table: papers — stores each uploaded paper and its summary
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            summary TEXT,
            uploaded_at TEXT NOT NULL
        )
    """)

    # Table: qa_history — stores each Q&A exchange linked to a paper
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            asked_at TEXT NOT NULL,
            FOREIGN KEY (paper_id) REFERENCES papers(id)
        )
    """)

    conn.commit()
    conn.close()


def save_paper(filename: str, summary: str) -> int:
    """
    Save a paper record to the DB.
    Returns the new paper's ID.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO papers (filename, summary, uploaded_at) VALUES (?, ?, ?)",
        (filename, summary, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    paper_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return paper_id


def save_qa(paper_id: int, question: str, answer: str):
    """Save a single Q&A exchange to the DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO qa_history (paper_id, question, answer, asked_at) VALUES (?, ?, ?, ?)",
        (paper_id, question, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()


def get_all_papers() -> List[Tuple]:
    """Return all papers: (id, filename, summary, uploaded_at)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, summary, uploaded_at FROM papers ORDER BY uploaded_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_qa_history(paper_id: int) -> List[Tuple]:
    """Return all Q&A for a given paper: (question, answer, asked_at)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT question, answer, asked_at FROM qa_history WHERE paper_id = ? ORDER BY asked_at ASC",
        (paper_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_paper(paper_id: int):
    """Delete a paper and all its Q&A history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM qa_history WHERE paper_id = ?", (paper_id,))
    cursor.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
    conn.commit()
    conn.close()
