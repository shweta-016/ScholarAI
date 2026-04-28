# 🔬 Research Paper Summarizer using Generative AI

An AI-powered web app that summarizes research papers and answers questions about them using **RAG (Retrieval-Augmented Generation)**, **OpenAI GPT**, and **FAISS** vector search.

---

## ✨ Features

- 📄 Upload any research paper PDF
- 🤖 AI-generated structured summary (Problem, Method, Findings, Conclusion)
- 💬 Ask questions about the paper — answered using RAG + FAISS
- 🗃️ Full Q&A history saved to a local SQLite database
- 📊 Summary quality evaluation metrics
- ⬇️ Download the generated summary as a text file

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT-3.5-turbo |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Search | FAISS |
| PDF Parsing | PyPDF2 |
| Database | SQLite |
| Frontend | Streamlit |

---

## 📁 Project Structure

```
Research-Paper-Summarizer-using-Generative-AI/
├── app.py                # Main Streamlit app
├── pdf_processor.py      # PDF extraction and text chunking
├── embeddings_faiss.py   # FAISS index creation and retrieval
├── summarizer.py         # GPT-based summarization
├── rag_qa.py             # RAG-based question answering
├── database.py           # SQLite DB for history storage
├── evaluation.py         # Summary quality metrics
├── requirements.txt      # Dependencies
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/shweta-016/Research-Paper-Summarizer-using-Generative-AI.git
cd Research-Paper-Summarizer-using-Generative-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
The app will open at `http://localhost:8501`

---

## 🔑 API Key

You will need an **OpenAI API key** to use this app.
- Get one at: https://platform.openai.com/api-keys
- Enter it directly in the app sidebar (it is never stored)

---

## 📬 Contact

**Shweta Jadhav**
- 📧 jadhavshweta477@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/in/shweta-jadhav-87a240276)
- 🐙 [GitHub](https://github.com/shweta-016)
