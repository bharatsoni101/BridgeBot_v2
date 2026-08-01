import os

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi
import joblib
from reranker import rerank
import time
from utils.logger import (rag_logger, performance_logger, error_logger, log_performance)

load_dotenv()


# ------------------------------------
# Configuration
# ------------------------------------

CHROMA_DB_PATH = "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "llama-3.3-70b-versatile"

TOP_K = 3

BM25_INDEX = "bm25/bm25_index.pkl"

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

    request_start = time.perf_counter()

    rag_logger.info("=" * 80)
    rag_logger.info("New Query")
    rag_logger.info("Question : %s", question)
    rag_logger.info("Web Search : %s", use_web_search)

    try:

        # ---------------------------------------------------
        # Hybrid Search
        # ---------------------------------------------------

        start = time.perf_counter()

        docs = hybrid_search(question)

        performance_logger.info("Hybrid Search returned %d chunks in %.3f sec", len(docs), time.perf_counter() - start)

        # ---------------------------------------------------
        # Cross Encoder
        # ---------------------------------------------------

        start = time.perf_counter()

        docs, rerank_scores = rerank(
            question,
            docs,
            top_k=3
        )

        performance_logger.info("Cross Encoder selected %d chunks in %.3f sec", len(docs), time.perf_counter() - start)

        context = ""

        pages = set()

        documents = set()

        for doc in docs:

            context += doc.page_content + "\n\n"

            pages.add(
                doc.metadata.get("page", "Unknown")
            )

            source = doc.metadata.get("source", "")

            if source:
                documents.add(
                    os.path.basename(source)
                )

        rag_logger.info("Documents : %s", ", ".join(sorted(documents)))

        rag_logger.info("Pages : %s", sorted(pages))

        # ---------------------------------------------------
        # Optional Web Search
        # ---------------------------------------------------

        web_context = ""

        web_sources = []

        if use_web_search:

            try:

                start = time.perf_counter()

                rag_logger.info("DuckDuckGo Search Started")

                web_context = search.run(question)

                performance_logger.info("Web Search completed in %.3f sec", time.perf_counter() - start)

                if web_context:

                    context += "\n\n========== WEB SEARCH ==========\n\n"

                    context += web_context

                    web_sources.append("DuckDuckGo")

                    rag_logger.info("Web Context Size : %d chars", len(web_context))

            except Exception as e:

                error_logger.exception(f"Web Search Failed: {e}")

        if context.strip() == "":

            logger.warning("No Context Found")

            return {

                "answer": "No relevant information found.",

                "llm": LLM_MODEL,

                "vector_db": "ChromaDB",

                "documents": [],

                "pages": [],

                "chunks": 0,

                "web_used": use_web_search,

                "web_sources": web_sources,

                "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",

                "rerank_scores": []

            }

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

        # ---------------------------------------------------
        # LLM
        # ---------------------------------------------------

        start = time.perf_counter()

        rag_logger.info("Sending Prompt To Groq")

        response = llm.invoke(prompt)

        llm_time = time.perf_counter() - start

        performance_logger.info("Groq Response Time : %.3f sec", llm_time)

        rag_logger.info("Answer Length : %d characters", len(response.content))

        total_time = time.perf_counter() - request_start

        performance_logger.info("Total Query Time : %.3f sec", total_time)

        rag_logger.info("=" * 80)

        return {

            "answer": response.content,

            "llm": LLM_MODEL,

            "vector_db": "ChromaDB",

            "documents": sorted(documents),

            "pages": sorted(pages),

            "chunks": len(docs),

            "web_used": use_web_search,

            "web_sources": web_sources,

            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",

            "rerank_scores": rerank_scores

        }

    except Exception as ex:

        error_logger.exception("Query Failed")

        return {

            "answer": str(ex),

            "llm": LLM_MODEL,

            "vector_db": "ChromaDB",

            "documents": [],

            "pages": [],

            "chunks": 0,

            "web_used": use_web_search,

            "web_sources": [],

            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",

            "rerank_scores": []

        }

def bm25_search(question, k=5):

    rag_logger.info("BM25 Search Started")

    if not os.path.exists(BM25_INDEX):
        return []

    index = joblib.load(BM25_INDEX)

    bm25 = index["bm25"]

    docs = index["documents"]

    query = question.lower().split()

    scores = bm25.get_scores(query)

    ranked = sorted(zip(scores, docs), reverse=True, key=lambda x: x[0])

    rag_logger.info("BM25 returned %d chunks", min(k, len(ranked)))

    return [doc for score, doc in ranked[:k]]


def hybrid_search(question):

    rag_logger.info("Hybrid Search Started")

    dense_docs = vector_db.similarity_search(question, k=5)

    rag_logger.info("Dense Search : %d chunks", len(dense_docs))

    sparse_docs = bm25_search(question, k=5)

    rag_logger.info("BM25 Search : %d chunks", len(sparse_docs))

    results = []

    seen = set()

    for doc in dense_docs + sparse_docs:

        text = doc.page_content

        if text not in seen:

            results.append(doc)

            seen.add(text)

    rag_logger.info("Hybrid Search Final : %d chunks", len(results))

    return results


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