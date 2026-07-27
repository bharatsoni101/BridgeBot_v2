import os

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# ------------------------------------
# Configuration
# ------------------------------------

CHROMA_DB_PATH = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"

TOP_K = 3

# ------------------------------------
# Embeddings
# ------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

# ------------------------------------
# Load Vector Database
# ------------------------------------

vector_db = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings
)

# ------------------------------------
# Groq LLM
# ------------------------------------

llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0
)

# ------------------------------------
# DuckDuckGo Search
# ------------------------------------

search = DuckDuckGoSearchRun()

# ------------------------------------
# Ask Question
# ------------------------------------

def ask(question, use_web_search=False):

    try:

        docs = vector_db.similarity_search(
            question,
            k=TOP_K
        )

        context = ""

        pages = set()

        documents = set()

        # -----------------------------
        # Vector DB Context
        # -----------------------------

        for doc in docs:

            context += doc.page_content + "\n\n"

            pages.add(
                doc.metadata.get("page", "Unknown")
            )

            source = doc.metadata.get("source", "")

            if source:
                documents.add(os.path.basename(source))

        # -----------------------------
        # Optional Web Search
        # -----------------------------

        web_context = ""
        web_sources = []

        if use_web_search:

            try:

                web_context = search.run(question)

                if web_context:
                    context += "\n\n========== WEB SEARCH ==========\n\n"
                    context += web_context

                    web_sources.append("DuckDuckGo")

            except Exception as e:

                print(f"Web Search Error : {e}")

        # -----------------------------
        # No Context Found
        # -----------------------------

        if context.strip() == "":

            return {
                "answer": "No relevant information found.",
                "llm": LLM_MODEL,
                "vector_db": "ChromaDB",
                "documents": [],
                "pages": [],
                "chunks": 0,
                "web_used": use_web_search,
                "web_sources": web_sources
            }

        # -----------------------------
        # Prompt
        # -----------------------------

        prompt = f"""
You are an expert AI assistant.

Use the context below to answer the user's question.

Priority:
1. Use uploaded PDF content first.
2. If web search content exists, use it only when required.
3. If the answer isn't available anywhere, say so.

========================
Context
========================

{context}

========================
Question
========================

{question}

========================
Answer
========================
"""

        response = llm.invoke(prompt)

        return {

            "answer": response.content,

            "llm": LLM_MODEL,

            "vector_db": "ChromaDB",

            "documents": sorted(documents),

            "pages": sorted(pages),

            "chunks": len(docs),

            "web_used": use_web_search,

            "web_sources": web_sources

        }

    except Exception as ex:

        return {

            "answer": str(ex),

            "llm": LLM_MODEL,

            "vector_db": "ChromaDB",

            "documents": [],

            "pages": [],

            "chunks": 0,

            "web_used": use_web_search,

            "web_sources": []

        }


# ------------------------------------
# Test
# ------------------------------------

if __name__ == "__main__":

    result = ask(
        "What is Spring Boot?",
        use_web_search=True
    )

    print(result["answer"])
    print(result["documents"])
    print(result["pages"])
    print(result["web_used"])
    print(result["web_sources"])