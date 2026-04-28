"""
pdf_processor.py
Handles PDF upload, text extraction, and text chunking.
"""

import PyPDF2
import io
from typing import List


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from an uploaded Streamlit PDF file object.
    Returns the full text as a single string.
    """
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    chunk_size    : number of characters per chunk
    chunk_overlap : number of characters to overlap between chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks
