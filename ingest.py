import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    document_name = os.path.basename(pdf_path)

    registry = load_registry()

    # -----------------------------------
    # Skip duplicate documents
    # -----------------------------------

    if any(doc["name"] == document_name for doc in registry):

        print(f"{document_name} already exists.")

        return False

    print(f"Loading {document_name}")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    total_pages = len(documents)

    print(f"Pages : {total_pages}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunks : {len(chunks)}")

    # -----------------------------------
    # Store in Chroma
    # -----------------------------------

    if os.path.exists(CHROMA_DB_PATH):

        vector_db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )

        vector_db.add_documents(chunks)

        vector_db.persist()

    else:

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )

    # -----------------------------------
    # Register document
    # -----------------------------------

    registry.append(
        {
            "name": document_name,
            "uploaded_on": datetime.now().strftime("%d-%b-%Y %H:%M"),
            "pages": total_pages,
            "chunks": len(chunks),
            "embedding_model": EMBEDDING_MODEL,
            "vector_db": "ChromaDB",
            "status": "Indexed"
        }
    )

    save_registry(registry)

    print("Knowledge Base Updated.")

    return True


# -----------------------------------
# Run Individually
# -----------------------------------

if __name__ == "__main__":

    ingest_pdf("pdfs/sample.pdf")