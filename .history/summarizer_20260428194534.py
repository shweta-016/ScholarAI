"""
summarizer.py
Uses Groq API (free) with llama3 for summarization.
"""

from groq import Groq

SUMMARY_PROMPT = """You are an expert research assistant. Read the following research paper text and provide a clear, structured summary.

Your summary must include:
1. **Title / Topic** – What is the paper about?
2. **Problem Statement** – What problem does it solve?
3. **Methodology** – What approach or methods are used?
4. **Key Findings** – What are the main results or contributions?
5. **Conclusion** – What is the overall takeaway?

Keep the summary concise, informative, and easy to understand.

Research Paper Text:
{text}

Structured Summary:"""


def summarize_paper(client: Groq, text: str) -> str:
    truncated = text[:12000] if len(text) > 12000 else text
    prompt = SUMMARY_PROMPT.format(text=truncated)

    response = client.chat.completions.create(
        
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful research paper summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()