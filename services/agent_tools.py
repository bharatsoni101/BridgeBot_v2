# services/agent_tools.py

import time
from langchain_core.tools import tool
from query import ask
from utils.logger import (rag_logger, performance_logger, error_logger)


# --------------------------------------------------------
# Knowledge Base Search Tool
# --------------------------------------------------------

@tool
def knowledge_base_search(question: str):
    """
    Search the BridgeBot internal knowledge base.

    Use this tool when the user question can potentially
    be answered from documents uploaded to BridgeBot.

    The tool uses the existing:
    - ChromaDB
    - BM25
    - Hybrid Search
    - Cross-Encoder Reranker
    - Metadata filtering
    """

    start_time = time.perf_counter()

    rag_logger.info("=" * 70)
    rag_logger.info("AGENT TOOL : Knowledge Base Search")
    rag_logger.info("Question : %s", question)

    try:

        result = ask(question, use_web_search=False)

        elapsed = time.perf_counter() - start_time

        performance_logger.info("Agent KB Search Time : %.3f sec", elapsed)

        rag_logger.info("Agent KB Search Completed")

        return result

    except Exception as ex:

        error_logger.exception("Agent KB Search Failed : %s", ex)

        return {
            "answer": "Knowledge base search failed.",
            "documents": [],
            "pages": [],
            "chunks": 0,
            "web_used": False,
            "web_sources": [],
            "error": str(ex)
        }