import os
from typing import List

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import openai

# --- Configurations ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or set directly for testing

# --- Sample Documents ---
DOCUMENTS = [
    "COVID-19 is caused by a virus called SARS-CoV-2.",
    "Vaccines help protect you from severe illness.",
    "Wearing masks can reduce the spread of the virus.",
    "Wash your hands frequently with soap and water.",
    "Maintain social distancing to avoid close contact."
]

# --- Load Embedding Model and Build Index ---
@st.cache_resource
def load_embedder_and_index():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    doc_embeddings = embedder.encode(DOCUMENTS, convert_to_tensor=False)
    index = faiss.IndexFlatL2(doc_embeddings[0].shape[0])
    index.add(doc_embeddings)
    return embedder, index

embedder, index = load_embedder_and_index()

# --- Retrieval Function ---
def retrieve_documents(query: str, top_k: int = 2) -> List[str]:
    query_embedding = embedder.encode([query])[0]
    _, indices = index.search([query_embedding], top_k)
    return [DOCUMENTS[i] for i in indices[0]]

# --- Generation Function ---
def generate_answer(query: str, retrieved_docs: List[str]) -> str:
    context = "\n".join(retrieved_docs)
    prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content'].strip()

# --- Streamlit UI ---
st.set_page_config(page_title="RAG FAQ Bot", layout="centered")
st.title("🤖 RAG-based FAQ Bot")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Retrieving relevant information..."):
        retrieved = retrieve_documents(query)
        answer = generate_answer(query, retrieved)

    st.subheader("📄 Retrieved Context")
    for i, doc in enumerate(retrieved, 1):
        st.markdown(f"**{i}.** {doc}")

    st.subheader("💬 Answer")
    st.success(answer)

st.markdown("---")
st.caption("Built with FAISS, SentenceTransformers, and OpenAI")
