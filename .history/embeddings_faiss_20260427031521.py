"""
embeddings_faiss.py
Builds a FAISS vector index from text chunks using OpenAI embeddings,
and retrieves the most relevant chunks for a given query.
"""

import faiss
import numpy as np
from openai import OpenAI
from typing import List, Tuple


def get_embedding(client: OpenAI, text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Return the embedding vector for a single text string."""
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def build_faiss_index(client: OpenAI, chunks: List[str]) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Embed all chunks and build a FAISS L2 index.
    Returns (index, chunks) so chunks stay aligned with index positions.
    """
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(client, chunk)
        embeddings.append(emb)

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    vectors = np.array(embeddings, dtype="float32")
    index.add(vectors)
    return index, chunks


def retrieve_relevant_chunks(
    client: OpenAI,
    query: str,
    index: faiss.IndexFlatL2,
    chunks: List[str],
    top_k: int = 4
) -> List[str]:
    """
    Embed the query and return the top_k most similar chunks.
    """
    query_emb = get_embedding(client, query)
    query_vec = np.array([query_emb], dtype="float32")
    distances, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
