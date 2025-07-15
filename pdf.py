import streamlit as st
from reader import extract_text_from_pdf
from chunker import chunk_text, embed_chunks
from qa_answer import generate_answer_tf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Page config and custom CSS ---
st.set_page_config(page_title="📚 AI PDF Chatbot", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
    .main {background-color: #f7f7fa;}
    .stButton>button {background-color: #4F8BF9; color: white; border-radius: 8px;}
    .stTextInput>div>div>input {border-radius: 8px;}
    .stFileUploader>div>div {border-radius: 8px;}
    .stExpander {background: #f0f4fa; border-radius: 8px;}
    .stAlert {border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/color/96/000000/pdf.png", width=80)
st.sidebar.title("AI PDF Chatbot")
st.sidebar.markdown(
    """
    Upload a PDF and ask questions about its content.\n
    - Uses AI to extract, chunk, and embed text
    - Finds the most relevant answer to your question
    """
)
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit, Transformers, and Sentence Transformers.")

# --- Main area ---
st.title("🤖 Chat with your PDF using AI")
st.markdown(
    """
    <div style='font-size:18px; color:#555;'>
    <b>Step 1:</b> Upload a PDF file.<br>
    <b>Step 2:</b> Ask any question about its content.<br>
    <b>Step 3:</b> Get instant, AI-powered answers!
    </div>
    """,
    unsafe_allow_html=True,
)

pdf_file = st.file_uploader("📤 <b>Upload your PDF</b>", type=["pdf"], label_visibility="visible")

if pdf_file:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(pdf_file)
        st.success("✅ Text extracted from PDF!")

    with st.spinner("Chunking and embedding text..."):
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        st.success(f"✅ Created {len(chunks)} chunks!")

    st.markdown("---")
    st.subheader("💬 Ask a question from your PDF")
    col1, col2 = st.columns([4,1])
    with col1:
        query = st.text_input("Type your question here:", key="query_input")
    with col2:
        ask_btn = st.button("Ask", use_container_width=True)

    if query and ask_btn:
        with st.spinner("Finding the best answer..."):
            query_embed = embed_chunks([query])
            sims = cosine_similarity(query_embed, embeddings)[0]
            top_idx = np.argmax(sims)
            top_chunk = chunks[top_idx]
            answer = generate_answer_tf(query, top_chunk)

        st.markdown("#### 🤖 <span style='color:#4F8BF9'>AI Answer:</span>", unsafe_allow_html=True)
        st.success(answer)

        with st.expander("🧠 <b>Matched Context</b>", expanded=False):
            st.info(top_chunk)
else:
    st.info("Please upload a PDF to get started.")