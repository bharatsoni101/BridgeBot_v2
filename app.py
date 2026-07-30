import json
import os

import pandas as pd
import streamlit as st

from ingest import ingest_pdf
from query import ask

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

PDF_FOLDER = "pdfs"
REGISTRY_FILE = "document_registry.json"

os.makedirs(PDF_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="PDF Chat using Groq",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------

def load_uploaded_documents():

    if not os.path.exists(REGISTRY_FILE):
        return pd.DataFrame()

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data)


def knowledge_base_ready():

    df = load_uploaded_documents()

    return not df.empty


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.title("📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        pdf_path = os.path.join(
            PDF_FOLDER,
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("PDF uploaded successfully.")

        if st.button("🚀 Prepare Knowledge Base"):

            with st.spinner("Creating Knowledge Base..."):

                ingest_pdf(pdf_path)

            st.success("Knowledge Base Ready!")

            st.rerun()

    st.divider()

    st.subheader("📚 Uploaded Documents")

    df = load_uploaded_documents()

    if not df.empty:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.success(f"Knowledge Base Ready ({len(df)} document(s))")

    else:

        st.warning("No documents uploaded.")

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------
# Main Screen
# ---------------------------------------------------

st.title("📄 Chat with PDF")

st.caption(
    "Groq + LangChain + ChromaDB + HuggingFace Embeddings"
)

# ---------------------------------------------------
# Display Chat History
# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------
# Chat Input
# ---------------------------------------------------
use_web_search = st.checkbox(
    "🌐 Enable Web Search",
    value=False
)

question = st.chat_input(
    "Ask anything about your documents...",
    disabled=not knowledge_base_ready()
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching Knowledge Base..."):

            result = ask(question, use_web_search=use_web_search)

        st.markdown(result["answer"])

        with st.expander("📌 Sources Used", expanded=False):

            st.write(f"**LLM :** {result['llm']}")

            st.write(f"**Vector Database :** {result['vector_db']}")

            if result["pages"]:

                pages = ", ".join(
                    str(page + 1)
                    for page in result["pages"]
                )

                st.write(f"**Pages :** {pages}")

            else:

                st.write("**Pages :** N/A")

            if result["web_used"]:

                st.write("**Web Search :** Used")

                for url in result["web_sources"]:

                    st.write(url)

            else:

                st.write("**Web Search :** Not Used")

            # Cross encoder - Reranker score
            st.write(f"**Reranker:** {result['reranker']}")

            st.write("**Reranker Scores:**")

            for score in result["rerank_scores"]:
                st.write(f"{score:.4f}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"]
        }
    )


# ---------------------------------------------------
# Empty Knowledge Base Message
# ---------------------------------------------------

if not knowledge_base_ready():

    st.info(
        """
### 📄 No Knowledge Base Found

Please follow these steps:

1. Upload one or more PDF documents.
2. Click **🚀 Prepare Knowledge Base**.
3. Wait until indexing is complete.
4. Start asking questions.

"""
    )