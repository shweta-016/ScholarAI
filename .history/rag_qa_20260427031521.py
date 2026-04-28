"""
rag_qa.py
Retrieval-Augmented Generation (RAG) — answers user questions
by retrieving relevant chunks from FAISS and passing them to GPT.
"""

import faiss
from openai import OpenAI
from typing import List
from embeddings_faiss import retrieve_relevant_chunks


QA_PROMPT = """You are an expert research assistant. Use ONLY the context below to answer the question.
If the answer is not found in the context, say "I could not find the answer in this paper."

Context:
{context}

Question: {question}

Answer:"""


def answer_question(
    client: OpenAI,
    question: str,
    index: faiss.IndexFlatL2,
    chunks: List[str],
    model: str = "gpt-3.5-turbo"
) -> str:
    """
    Retrieve relevant chunks for the question, then ask GPT to answer.
    """
    relevant_chunks = retrieve_relevant_chunks(client, question, index, chunks, top_k=4)
    context = "\n\n---\n\n".join(relevant_chunks)
    prompt = QA_PROMPT.format(context=context, question=question)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful research assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()
