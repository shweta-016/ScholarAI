"""
app.py
Research Paper Summarizer using Generative AI
Redesigned UI — Dark Navy/Purple Theme
Features: Summarization + RAG Q&A + Plagiarism Detection + DB History
"""

import streamlit as st
from groq import Groq

from pdf_processor import extract_text_from_pdf, chunk_text
from embeddings_faiss import build_faiss_index
from summarizer import summarize_paper
from rag_qa import answer_question
from database import init_db, save_paper, save_qa, get_all_papers, get_qa_history, delete_paper
from evaluation import evaluate_summary
from plagiarism import check_plagiarism_sentences, compare_with_db

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchAI — Paper Summarizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif !important; }
.stApp { background: #050d1a; color: #e8f0fe; }
section[data-testid="stSidebar"] { background: #0a1628 !important; border-right: 1px solid rgba(0,212,255,0.15); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; }

.sidebar-logo { font-size: 1.4rem; font-weight: 700; background: linear-gradient(90deg, #00d4ff, #7b61ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.2rem; }
.sidebar-sub { font-size: 0.72rem; color: #8899aa; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 1rem; }

.page-title { font-size: 2.6rem; font-weight: 700; background: linear-gradient(90deg, #00d4ff, #7b61ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.2; margin-bottom: 0.4rem; }
.page-sub { color: #8899aa; font-size: 0.95rem; margin-bottom: 2rem; }

.upload-zone { background: linear-gradient(135deg, #0d1b2e, #111827); border: 2px dashed rgba(0,212,255,0.35); border-radius: 16px; padding: 2rem; text-align: center; margin-bottom: 1.5rem; }
.upload-icon { font-size: 2.5rem; margin-bottom: 0.6rem; }
.upload-title { font-size: 1rem; font-weight: 600; color: #e8f0fe; margin-bottom: 0.2rem; }
.upload-hint { font-size: 0.78rem; color: #8899aa; }

.feature-card { background: #0d1b2e; border: 1px solid rgba(0,212,255,0.12); border-radius: 12px; padding: 1.4rem; text-align: center; }
.feature-icon { font-size: 2rem; margin-bottom: 0.6rem; }
.feature-title { font-weight: 600; font-size: 0.95rem; color: #e8f0fe; margin-bottom: 0.3rem; }
.feature-desc { font-size: 0.78rem; color: #8899aa; }

.stat-pill { display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.2); border-radius: 20px; padding: 0.3rem 0.9rem; font-size: 0.75rem; color: #00d4ff; margin-right: 0.4rem; margin-bottom: 1rem; }

.section-header { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1.2rem; margin-top: 0.5rem; }
.section-header-icon { width: 34px; height: 34px; background: rgba(123,97,255,0.15); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1rem; }
.section-header-title { font-size: 1.1rem; font-weight: 700; color: #e8f0fe; }

.summary-box { background: linear-gradient(135deg, #0d1b2e, #0f1f3d); border: 1px solid rgba(0,212,255,0.2); border-radius: 12px; padding: 1.8rem; line-height: 1.9; font-size: 0.9rem; color: #e8f0fe; margin-bottom: 1rem; }

.q-bubble { background: rgba(123,97,255,0.12); border: 1px solid rgba(123,97,255,0.25); border-radius: 12px 12px 12px 0; padding: 0.9rem 1.2rem; margin-bottom: 0.5rem; font-size: 0.88rem; color: #c4b5fd; }
.a-bubble { background: rgba(0,212,255,0.07); border: 1px solid rgba(0,212,255,0.18); border-radius: 12px 12px 0 12px; padding: 0.9rem 1.2rem; margin-bottom: 1.2rem; font-size: 0.88rem; color: #e8f0fe; line-height: 1.8; }

.plag-score-high { background: rgba(255,59,59,0.1); border: 1px solid rgba(255,59,59,0.3); border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.plag-score-medium { background: rgba(255,165,0,0.1); border: 1px solid rgba(255,165,0,0.3); border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.plag-score-low { background: rgba(0,255,136,0.08); border: 1px solid rgba(0,255,136,0.25); border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.plag-num { font-size: 3rem; font-weight: 700; font-family: 'Space Mono', monospace; line-height: 1; }
.suspicious-sentence { background: rgba(255,59,59,0.12); border-left: 3px solid #ff3b3b; border-radius: 0 6px 6px 0; padding: 0.5rem 0.8rem; margin-bottom: 0.4rem; font-size: 0.82rem; color: #fca5a5; }
.clean-sentence { background: rgba(0,255,136,0.06); border-left: 3px solid #00ff88; border-radius: 0 6px 6px 0; padding: 0.5rem 0.8rem; margin-bottom: 0.4rem; font-size: 0.82rem; color: #86efac; }
.db-match { background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.25); border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem; font-size: 0.82rem; }

.metric-card { background: #0d1b2e; border: 1px solid rgba(0,212,255,0.12); border-radius: 10px; padding: 1rem 1.2rem; text-align: center; }
.metric-val { font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #00d4ff; display: block; }
.metric-label { font-size: 0.7rem; color: #8899aa; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.3rem; }

.history-card { background: #0d1b2e; border: 1px solid rgba(123,97,255,0.2); border-radius: 10px; padding: 0.9rem 1rem; margin-bottom: 0.6rem; font-size: 0.8rem; }
.history-title { font-weight: 600; color: #c4b5fd; margin-bottom: 0.2rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.history-meta { color: #8899aa; font-size: 0.7rem; }

.custom-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent); margin: 1.5rem 0; }

.warn-banner { background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.3); border-radius: 10px; padding: 0.9rem 1.2rem; font-size: 0.85rem; color: #fbbf24; margin-bottom: 1rem; }
.success-banner { background: rgba(0,255,136,0.07); border: 1px solid rgba(0,255,136,0.25); border-radius: 10px; padding: 0.9rem 1.2rem; font-size: 0.85rem; color: #00ff88; margin-bottom: 1rem; }

.stTabs [data-baseweb="tab-list"] { background: #0a1628; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid rgba(0,212,255,0.1); }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #8899aa; font-size: 0.82rem; font-weight: 500; padding: 0.5rem 1.2rem; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(123,97,255,0.15)) !important; color: #00d4ff !important; border: 1px solid rgba(0,212,255,0.2) !important; }

.stButton > button { background: linear-gradient(135deg, #00d4ff, #7b61ff); color: #050d1a; font-weight: 700; font-size: 0.85rem; border: none; border-radius: 8px; padding: 0.6rem 1.8rem; transition: all 0.2s; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,212,255,0.25); }
</style>
""", unsafe_allow_html=True)

# ─── Init DB ──────────────────────────────────────────────────────────────────
init_db()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🔬 ResearchAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Paper Summarizer & Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

   

    # Load from Streamlit secrets (cloud) or .env (local)
    groq_key = st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
    "🔑 Groq API Key (Free)", 
    value=groq_key,
    type="password", placeholder="gsk_...", help="Get free key from https://console.groq.com")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("**📚 History**")

    papers = get_all_papers()
    if papers:
        for paper in papers:
            pid, fname, summary, uploaded_at = paper
            qa_hist = get_qa_history(pid)
            st.markdown(f'<div class="history-card"><div class="history-title">📄 {fname[:28]}{"..." if len(fname)>28 else ""}</div><div class="history-meta">🕐 {uploaded_at} &nbsp;|&nbsp; 💬 {len(qa_hist)} Q&As</div></div>', unsafe_allow_html=True)
            if st.button("🗑️ Delete", key=f"del_{pid}"):
                delete_paper(pid)
                st.rerun()
    else:
        st.markdown('<div class="warn-banner">No papers yet. Upload one!</div>', unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#8899aa;text-align:center;">Built with Groq LLaMA3 · FAISS · RAG · SQLite</div>', unsafe_allow_html=True)

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Research Paper Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Upload any research paper · Get AI summary · Ask questions · Check plagiarism</div>', unsafe_allow_html=True)

# Upload is ALWAYS visible — no API key needed to upload
st.markdown('<div class="upload-zone"><div class="upload-icon">📂</div><div class="upload-title">Upload your research paper</div><div class="upload-hint">Supports PDF files · Text-based PDFs only (not scanned images)</div></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"], label_visibility="collapsed")

# Show feature cards on landing (no file uploaded yet)
if uploaded_file is None:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    features = [("📝","AI Summary","Structured summary with problem, method & findings"),("💬","RAG Q&A","Ask questions answered from the paper using FAISS"),("🔍","Plagiarism","Detect copied sentences & compare with your DB"),("🗃️","History","All papers & Q&A saved to local database")]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f'<div class="feature-card"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-desc">{desc}</div></div>', unsafe_allow_html=True)
    st.stop()

# API key only required after file is uploaded
if not api_key:
    st.markdown('<div class="warn-banner">✅ PDF uploaded! Now enter your Groq API key in the sidebar to process it.</div>', unsafe_allow_html=True)
    st.stop()

client = Groq(api_key=api_key)

filename = uploaded_file.name

with st.spinner("📖 Extracting text from PDF..."):
    raw_text = extract_text_from_pdf(uploaded_file)

if not raw_text or len(raw_text) < 100:
    st.markdown('<div class="warn-banner">❌ Could not extract text. PDF may be scanned/image-based.</div>', unsafe_allow_html=True)
    st.stop()

chunks = chunk_text(raw_text)
st.markdown(f'<div style="margin-bottom:1rem;"><span class="stat-pill">📄 {filename}</span><span class="stat-pill">📊 {len(raw_text):,} chars</span><span class="stat-pill">🧩 {len(chunks)} chunks</span><span class="stat-pill">📝 ~{len(raw_text.split()):,} words</span></div>', unsafe_allow_html=True)

if "faiss_index" not in st.session_state or st.session_state.get("current_file") != filename:
    with st.spinner("🔍 Building semantic index..."):
        index, stored_chunks = build_faiss_index(chunks)
        st.session_state.update({"faiss_index": index, "chunks": stored_chunks, "current_file": filename, "raw_text": raw_text, "paper_id": None, "qa_log": [], "summary": None, "summary_file": None})
else:
    index = st.session_state["faiss_index"]
    stored_chunks = st.session_state["chunks"]
    raw_text = st.session_state["raw_text"]

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📝  Summary", "💬  Q&A", "🔍  Plagiarism", "📊  Metrics"])

# ── TAB 1: SUMMARY ────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header"><div class="section-header-icon">📝</div><div class="section-header-title">AI-Generated Summary</div></div>', unsafe_allow_html=True)
    already = st.session_state.get("summary") and st.session_state.get("summary_file") == filename
    if not already:
        if st.button("✨ Generate Summary"):
            with st.spinner("🤖 GPT is analyzing your paper..."):
                summary = summarize_paper(client, raw_text)
                st.session_state["summary"] = summary
                st.session_state["summary_file"] = filename
                paper_id = save_paper(filename, summary)
                st.session_state["paper_id"] = paper_id
            st.rerun()
    else:
        summary = st.session_state["summary"]
        st.markdown(f'<div class="summary-box">{summary.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 5])
        with col1:
            st.download_button("⬇️ Download", data=summary, file_name=f"{filename.replace('.pdf','')}_summary.txt", mime="text/plain")
        with col2:
            if st.button("🔄 Regenerate"):
                st.session_state["summary"] = None
                st.rerun()

# ── TAB 2: Q&A ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header"><div class="section-header-icon">💬</div><div class="section-header-title">Ask Questions About This Paper</div></div>', unsafe_allow_html=True)
    question = st.text_input("Your question", placeholder="e.g. What dataset was used? What is the main contribution?", label_visibility="collapsed")
    if st.button("🔍 Ask") and question.strip():
        with st.spinner("🤖 Searching paper..."):
            answer = answer_question(client, question, index, stored_chunks)
        if st.session_state.get("paper_id"):
            save_qa(st.session_state["paper_id"], question, answer)
        if "qa_log" not in st.session_state:
            st.session_state["qa_log"] = []
        st.session_state["qa_log"].insert(0, (question, answer))

    if st.session_state.get("qa_log"):
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        for q, a in st.session_state["qa_log"]:
            st.markdown(f'<div class="q-bubble">❓ {q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="a-bubble">💡 {a}</div>', unsafe_allow_html=True)

    if st.session_state.get("paper_id"):
        past = get_qa_history(st.session_state["paper_id"])
        if past:
            with st.expander(f"📂 All saved Q&A ({len(past)} total)"):
                for q, a, t in past:
                    st.markdown(f'<div class="q-bubble">❓ {q} <span style="float:right;font-size:0.7rem;color:#8899aa">{t}</span></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="a-bubble">💡 {a}</div>', unsafe_allow_html=True)

# ── TAB 3: PLAGIARISM ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header"><div class="section-header-icon">🔍</div><div class="section-header-title">Plagiarism Detection</div></div>', unsafe_allow_html=True)

    if st.button("🔍 Run Plagiarism Check"):
        with st.spinner("🔍 Analyzing text for plagiarism..."):
            sentence_results, plag_percent = check_plagiarism_sentences(raw_text)
            db_matches = compare_with_db(raw_text)
            st.session_state["plag_results"] = (sentence_results, plag_percent, db_matches)

    if "plag_results" in st.session_state:
        sentence_results, plag_percent, db_matches = st.session_state["plag_results"]

        if plag_percent >= 40:
            box_class, color, verdict = "plag-score-high", "#ff3b3b", "⚠️ High Plagiarism Detected"
        elif plag_percent >= 15:
            box_class, color, verdict = "plag-score-medium", "#f97316", "⚠️ Moderate Similarity Found"
        else:
            box_class, color, verdict = "plag-score-low", "#00ff88", "✅ Mostly Original Content"

        suspicious = sum(1 for _, is_s in sentence_results if is_s)
        total = len(sentence_results)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.markdown(f'<div class="{box_class}"><div class="plag-num" style="color:{color}">{plag_percent}%</div><div style="font-size:0.78rem;color:#8899aa;margin-top:0.3rem;">Similarity Score</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card" style="margin-top:0"><span class="metric-val" style="color:{color}">{suspicious}/{total}</span><div class="metric-label">Suspicious Sentences</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card" style="margin-top:0;text-align:left;padding:1rem 1.4rem"><div style="font-size:1rem;font-weight:600;color:{color};margin-bottom:0.4rem">{verdict}</div><div style="font-size:0.78rem;color:#8899aa;line-height:1.7">Checks sentence-level similarity using TF-IDF cosine similarity and compares against previously uploaded papers in your local database.</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("**📄 Sentence-Level Analysis** &nbsp; <span style='font-size:0.75rem;color:#8899aa'>🔴 suspicious &nbsp;|&nbsp; 🟢 original</span>", unsafe_allow_html=True)

        for sentence, is_suspicious in sentence_results[:40]:
            if sentence.strip():
                css = "suspicious-sentence" if is_suspicious else "clean-sentence"
                icon = "🔴" if is_suspicious else "🟢"
                st.markdown(f'<div class="{css}">{icon} {sentence}</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("**🗃️ Comparison with Previously Uploaded Papers**")
        if db_matches:
            for match in db_matches:
                st.markdown(f'<div class="db-match"><span style="color:#fbbf24;font-weight:600">📄 {match["filename"]}</span> &nbsp;|&nbsp; <span style="color:#f97316">Similarity: {match["similarity"]}%</span> &nbsp;|&nbsp; <span style="color:#8899aa;font-size:0.75rem">{match["date"]}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-banner">✅ No matching papers found in your database.</div>', unsafe_allow_html=True)

# ── TAB 4: METRICS ────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header"><div class="section-header-icon">📊</div><div class="section-header-title">Summary Quality Metrics</div></div>', unsafe_allow_html=True)
    if st.session_state.get("summary") and st.session_state.get("summary_file") == filename:
        metrics = evaluate_summary(raw_text, st.session_state["summary"])
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics.items()):
            with col:
                st.markdown(f'<div class="metric-card"><span class="metric-val">{value.split()[0]}</span><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("**📄 Extracted Text Preview**")
        st.text_area("", raw_text[:2000], height=220, label_visibility="collapsed")
    else:
        st.markdown('<div class="warn-banner">💡 Generate a summary first (Tab 1) to see metrics.</div>', unsafe_allow_html=True)
