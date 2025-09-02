import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient
import io
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# ====
# Hugging Face Client
# ===

import os


# Try Streamlit secrets first, then environment variable
HF_TOKEN = None
if st.secrets is not None:
    HF_TOKEN = st.secrets.get("huggingface", {}).get("token")

HF_TOKEN = HF_TOKEN or os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error(" Hugging Face token not found. Please add it in Streamlit Secrets.")
    st.stop()

hf_client = InferenceClient(token=HF_TOKEN)


# =====
# PDF Processing
# =====
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf_document:
        text += page.get_text("text")
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# ===
# FAISS Embeddings
# ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks

def retrieve_context(query, index, chunks, top_k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    return [chunks[i] for i in indices[0]]

# ---
# LLM Answer Generation
# =============
def generate_answer(query, context):
    prompt_context = "\n\n".join(context)
    messages = [
        {"role": "system", "content": "You are StudyMate, an AI academic assistant. Answer only based on context."},
        {"role": "user", "content": f"Context:\n{prompt_context}\n\nQuestion: {query}"}
    ]
    try:
        response = hf_client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            max_tokens=800
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"❌ Error generating answer: {e}"

#
# PDF Export
#
def export_to_pdf(question, answer):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, height-1*inch, "📚 StudyMate - Answer")
    c.line(1*inch, height-1.1*inch, width-1*inch, height-1.1*inch)

    # Question
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1*inch, height-1.5*inch, " Question:")
    text_obj = c.beginText(1*inch, height-1.8*inch)
    text_obj.setFont("Helvetica", 12)
    for line in question.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)

    # Answer
    y_position = text_obj.getY()-0.3*inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1*inch, y_position, " Answer:")
    text_obj = c.beginText(1*inch, y_position-0.3*inch)
    text_obj.setFont("Helvetica", 12)
    max_width = width - 2*inch
    for paragraph in answer.split("\n"):
        line = ""
        for word in paragraph.split():
            if c.stringWidth(line + " " + word, "Helvetica", 12) < max_width:
                line += " " + word
            else:
                text_obj.textLine(line.strip())
                line = word
        if line:
            text_obj.textLine(line.strip())
        text_obj.textLine("")
    c.drawText(text_obj)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="StudyMate - PDF Q&A", page_icon="📚", layout="wide")
st.markdown("""
<style>
.stButton>button {border-radius: 12px; padding: 0.6rem 1rem; background-color: #4f46e5; color: white; border: none; font-weight: bold;}
.stButton>button:hover {background-color: #4338ca; transform: scale(1.05);}
.answer-box {background: white; padding: 1rem; border-radius: 12px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); margin-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("📚 StudyMate: Your AI-Powered PDF Q&A Assistant")

# Chat History
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.title("📜 Chat History")
if st.session_state.history:
    for i, chat in enumerate(st.session_state.history[::-1]):
        with st.sidebar.expander(f"Q{i+1}: {chat['q'][:30]}..."):
            st.write("*Q:*", chat["q"])
            st.write("*A:*", chat["a"])
else:
    st.sidebar.info("No previous Q&A yet.")

# PDF Upload
uploaded_file = st.file_uploader(" Drag & Drop your PDF here", type=["pdf"])
if uploaded_file:
    st.success(" PDF uploaded successfully!")
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    index, chunks = build_faiss_index(chunks)

    query = st.text_input(" Ask a Question", placeholder="e.g., What are the key points in this paper?")
    if query:
        with st.spinner(" StudyMate is analyzing your PDF..."):
            time.sleep(0.5)
            context = retrieve_context(query, index, chunks)
            answer = generate_answer(query, context)

        # Typing effect
        st.markdown("###  Answer:")
        answer_placeholder = st.empty()
        typed_text = ""
        for char in answer:
            typed_text += char
            answer_placeholder.markdown(f"<div class='answer-box'>{typed_text}</div>", unsafe_allow_html=True)
            time.sleep(0.005)

        # Save history
        st.session_state.history.append({"q": query, "a": answer})

        # Download buttons (TXT + PDF only)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇ Download TXT", data=answer, file_name="studymate_answer.txt", mime="text/plain")
        with col2:
            pdf_file = export_to_pdf(query, answer)
            st.download_button("📑 Download PDF", data=pdf_file, file_name="studymate_answer.pdf", mime="application/pdf")
