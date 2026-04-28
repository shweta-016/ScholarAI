"""
embeddings_faiss.py
Uses sentence-transformers (free, runs locally, no API needed)
for embeddings instead of OpenAI.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# Load once and cache
@staticmethod
def _load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def build_faiss_index(chunks: List[str]) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Embed all chunks locally using sentence-transformers and build a FAISS index.
    No API key needed.
    """
    model = get_model()
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, chunks


def retrieve_relevant_chunks(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: List[str],
    top_k: int = 4
) -> List[str]:
    """
    Embed the query locally and return top_k most similar chunks.
    """
    model = get_model()
    query_emb = model.encode([query])
    query_vec = np.array(query_emb, dtype="float32")
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]