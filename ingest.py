import json
import os
from datetime import datetime
from rank_bm25 import BM25Okapi
import joblib
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

from utils.logger import (rag_logger, performance_logger, error_logger, log_performance)

load_dotenv()

# -----------------------------------
# Configuration
# -----------------------------------

CHROMA_DB_PATH = "chroma_db"
REGISTRY_FILE = "document_registry.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

BM25_INDEX = "bm25/bm25_index.pkl"

os.makedirs("bm25", exist_ok=True)

# -----------------------------------
# Registry Functions
# -----------------------------------

def load_registry():

    if not os.path.exists(REGISTRY_FILE):
        return []

    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(data):

    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# -----------------------------------
# Ingest PDF
# -----------------------------------

def ingest_pdf(pdf_path):

    request_start = time.perf_counter()

    document_name = os.path.basename(pdf_path)

    rag_logger.info("=" * 80)
    rag_logger.info("Knowledge Base Creation Started")
    rag_logger.info("Document : %s", document_name)

    try:

        registry = load_registry()

        # --------------------------------------------------------
        # Duplicate Check
        # --------------------------------------------------------

        if any(doc["name"] == document_name for doc in registry):

            logger.warning("Duplicate document detected : %s", document_name)

            return False

        # --------------------------------------------------------
        # Load PDF
        # --------------------------------------------------------

        start = time.perf_counter()

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        pdf_load_time = time.perf_counter() - start

        total_pages = len(documents)

        rag_logger.info("Pages : %d", total_pages)

        performance_logger.info("PDF Loaded in %.3f sec", pdf_load_time)

        # --------------------------------------------------------
        # Chunking
        # --------------------------------------------------------

        start = time.perf_counter()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)


        # --------------------------------------------------------
        # Add Metadata to Every Chunk
        # --------------------------------------------------------

        document_id = os.path.splitext(document_name)[0]

        document_without_ext = os.path.splitext(document_name)[0]
        category = document_without_ext.split("_")[0]

        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_size_mb = round(os.path.getsize(pdf_path) / (1024 * 1024), 2)

        for index, chunk in enumerate(chunks):

            chunk.metadata["document"] = document_name

            chunk.metadata["document_id"] = document_id

            chunk.metadata["category"] = category

            chunk.metadata["uploaded_on"] = upload_time

            chunk.metadata["chunk_id"] = index + 1

            chunk.metadata["total_pages"] = total_pages

            chunk.metadata["file_size_mb"] = file_size_mb

            chunk.metadata["embedding_model"] = EMBEDDING_MODEL

        rag_logger.info("Metadata Added To %d Chunks", len(chunks))

        chunk_time = time.perf_counter() - start

        rag_logger.info("Chunks Created : %d", len(chunks))

        performance_logger.info("Chunking and Metadata Time : %.3f sec", chunk_time)

        # --------------------------------------------------------
        # BM25 Index
        # --------------------------------------------------------

        start = time.perf_counter()

        tokenized_corpus = [
            chunk.page_content.lower().split()
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        joblib.dump(
            {
                "bm25": bm25,
                "documents": chunks
            },
            BM25_INDEX
        )

        bm25_time = time.perf_counter() - start

        rag_logger.info("BM25 Index Created")

        performance_logger.info("BM25 Time : %.3f sec", bm25_time)

        # --------------------------------------------------------
        # ChromaDB
        # --------------------------------------------------------

        start = time.perf_counter()

        if os.path.exists(CHROMA_DB_PATH):

            vector_db = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings
            )

            vector_db.add_documents(chunks)

            vector_db.persist()

            rag_logger.info("Documents Added To Existing ChromaDB")

        else:

            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DB_PATH
            )

            rag_logger.info("New ChromaDB Created")

        chroma_time = time.perf_counter() - start

        performance_logger.info("ChromaDB Time : %.3f sec", chroma_time)

        # --------------------------------------------------------
        # Registry
        # --------------------------------------------------------

        registry.append(
            {
                "name": document_name,
                "document_id": document_id,
                "category": category,
                "uploaded_on": upload_time,
                "pages": total_pages,
                "chunks": len(chunks),
                "file_size_mb": file_size_mb,
                "embedding_model": EMBEDDING_MODEL,
                "vector_db": "ChromaDB",
                "status": "Indexed"
            }
        )

        save_registry(registry)

        rag_logger.info("Document Registry Updated")
        rag_logger.info("Document ID : %s", document_id)
        rag_logger.info("Category : %s", category)
        rag_logger.info("File Size : %.2f MB", file_size_mb)

        total_time = time.perf_counter() - request_start

        rag_logger.info("=" * 80)
        rag_logger.info("Knowledge Base Created Successfully")
        performance_logger.info("Total Time : %.3f sec", total_time)
        rag_logger.info("=" * 80)

        return True

    except Exception as e:

        error_logger.exception(f"Knowledge Base Creation Failed {e}")

        return False

# -----------------------------------
# Run Individually
# -----------------------------------

if __name__ == "__main__":

    ingest_pdf("pdfs/sample.pdf")