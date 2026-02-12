import streamlit as st
import PyPDF2
import numpy as np
import faiss
import requests
import os

# =========================
# 🔐 API SETUP
# =========================
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    st.error("OPENROUTER_API_KEY not found. Please set it as an environment variable.")
    st.stop()

CHAT_API_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBED_API_URL = "https://openrouter.ai/api/v1/embeddings"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# =========================
# 📄 SAFE PDF TEXT EXTRACTION (FIXED)
# =========================
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""

    for page in reader.pages:
        try:
            page_text = page.extract_text()
            if page_text:
                # Remove problematic unicode characters
                page_text = page_text.encode("utf-8", errors="ignore").decode("utf-8")
                text += page_text + "\n"
        except Exception:
            continue

    return text

# =========================
# 🧩 CHUNKING
# =========================
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

# =========================
# 🔢 EMBEDDINGS (API)
# =========================
def get_embeddings(texts):
    response = requests.post(
        EMBED_API_URL,
        headers=HEADERS,
        json={
            "model": "text-embedding-3-small",
            "input": texts
        }
    )

    data = response.json()
    embeddings = [item["embedding"] for item in data["data"]]
    return np.array(embeddings).astype("float32")

# =========================
# 📦 FAISS INDEX
# =========================
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# =========================
# 🔍 RETRIEVAL
# =========================
def retrieve_chunks(query, index, chunks, k=4):
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

# =========================
# 🧠 GENERATIVE RAG RESPONSE
# =========================
def generate_answer(context, user_request):
    prompt = f"""
You are a smart study companion.

Use ONLY the study material below as your knowledge source.

You are allowed to:
- Select relevant questions from the material
- Rephrase or simplify questions
- Organize them clearly

STRICT FORMATTING RULES:
- Each question must be on a NEW LINE
- Use numbered list format only
- Do NOT combine multiple questions on one line
- Do NOT add explanations or paragraphs

OUTPUT FORMAT:
1. Question text
2. Question text
3. Question text
...

Do NOT use outside knowledge.
Do NOT introduce topics not present in the material.

Study Material:
{context}

User Request:
{user_request}
"""

    response = requests.post(
        CHAT_API_URL,
        headers=HEADERS,
        json={
            "model": "openai/gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    return response.json()["choices"][0]["message"]["content"]

# =========================
# 🎨 STREAMLIT UI
# =========================
st.set_page_config(page_title="Smart Study Companion (RAG)", page_icon="📘")

st.title("📘 Smart Study Companion (RAG)")
st.caption("Generates answers and questions strictly from your study material")

uploaded_files = st.file_uploader(
    "Upload Study Material PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

user_request = st.text_input(
    "Ask a question or request generation (e.g., 'Generate 10 easy-level questions')"
)

# =========================
# ▶ MAIN LOGIC
# =========================
if st.button("Generate Response"):
    if not uploaded_files or not user_request:
        st.warning("Please upload PDFs and enter a request.")
    else:
        with st.spinner("Processing study material..."):

            # 1️⃣ Extract text
            full_text = ""
            for pdf in uploaded_files:
                full_text += extract_text_from_pdf(pdf)

            if not full_text.strip():
                st.error("No readable text found in the uploaded PDFs.")
                st.stop()

            # 2️⃣ Chunking
            chunks = chunk_text(full_text)

            # 3️⃣ Embeddings
            embeddings = get_embeddings(chunks)

            # 4️⃣ FAISS index
            index = create_faiss_index(embeddings)

            # 5️⃣ Retrieve relevant chunks
            relevant_chunks = retrieve_chunks(user_request, index, chunks, k=4)
            context = "\n\n".join(relevant_chunks)

            # 6️⃣ Generate grounded response
            output = generate_answer(context, user_request)

            # 7️⃣ Force clean new-line formatting
            formatted_output = output.replace(". ", ".\n")

            st.markdown("### 📖 Output")
            st.text(formatted_output)
