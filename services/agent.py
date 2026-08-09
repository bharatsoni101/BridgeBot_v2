# services/agent.py

import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from services.agent_tools import knowledge_base_search
from utils.logger import (rag_logger, performance_logger, error_logger)

load_dotenv()


# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

LLM_MODEL = "llama-3.3-70b-versatile"
AGENT_TEMPERATURE = 0
MAX_AGENT_ITERATIONS = 1


# --------------------------------------------------------
# Groq Agent LLM
# --------------------------------------------------------

rag_logger.info("Initializing Agent LLM : %s", LLM_MODEL)

agent_llm = ChatGroq(model=LLM_MODEL, temperature=AGENT_TEMPERATURE)

rag_logger.info("Agent LLM Initialized Successfully")


# --------------------------------------------------------
# Agent Decision
# --------------------------------------------------------

def decide_source(question):
    """
    Decide whether the user's question should use
    the internal Knowledge Base.

    Returns:

        KB
        NOT_KB
    """

    start_time = time.perf_counter()

    rag_logger.info("-" * 70)
    rag_logger.info("AGENT DECISION STARTED")
    rag_logger.info("Question : %s", question)

    try:

        prompt = f"""
You are the routing agent for an enterprise RAG application
called BridgeBot.

BridgeBot contains an internal Knowledge Base consisting
of documents uploaded by authorized users.

Your task is to decide whether the user's question should
be answered using the internal Knowledge Base.

Use KB when:
- The question asks about information that may exist
  in uploaded company/project documents.
- The question refers to BridgeBot's internal knowledge.
- The question asks about a document, policy, process,
  architecture, project, specification, or other internal
  information.
- The question could reasonably be answered from the
  uploaded documents.

Use NOT_KB when:
- The question is casual conversation.
- The question is unrelated to the internal documents.
- The question is a general conversational request that
  does not require the Knowledge Base.

Return ONLY one of these values:

KB

NOT_KB

User Question:
{question}
"""

        response = agent_llm.invoke(prompt)

        decision = response.content.strip().upper()

        # ------------------------------------------------
        # Normalize LLM Response
        # ------------------------------------------------

        if decision == "KB":
            decision = "KB"
        elif decision == "NOT_KB":
            decision = "NOT_KB"
        else:
            rag_logger.warning(
                "Unexpected agent decision: %s",
                decision
            )
            decision = "KB"

        elapsed = time.perf_counter() - start_time

        performance_logger.info("Agent Decision Time : %.3f sec", elapsed)

        rag_logger.info("Agent Decision : %s", decision)

        rag_logger.info("AGENT DECISION COMPLETED")

        rag_logger.info("-" * 70)

        return decision

    except Exception as ex:

        error_logger.exception("Agent Decision Failed : %s", ex)

        # Fail safely to KB because this is your
        # internal RAG application.

        return "KB"


# --------------------------------------------------------
# Agentic RAG
# --------------------------------------------------------

def agentic_rag(
        question,
        selected_documents=None,
        selected_category=None,
        owner=None,
        department=None,
        team=None,
        visibility=None
):
    """
    Phase 1 Agentic RAG.

    Flow:

        Question
            ↓
        Agent Decision
            ↓
        KB
            ↓
        Existing RAG Pipeline

    """

    request_start = time.perf_counter()

    rag_logger.info("=" * 80)
    rag_logger.info("AGENTIC RAG STARTED")
    rag_logger.info("Question : %s", question)

    try:

        # ------------------------------------------------
        # Step 1: Agent decides source
        # ------------------------------------------------

        decision = decide_source(question)

        # ------------------------------------------------
        # Step 2: Knowledge Base
        # ------------------------------------------------

        if decision == "KB":

            rag_logger.info("Agent selected Knowledge Base")

            result = knowledge_base_search.invoke({"question": question})

            # ------------------------------------------------
            # Preserve Existing RAG Response Metadata
            # ------------------------------------------------

            result.setdefault("llm", LLM_MODEL)

            result.setdefault("vector_db", "ChromaDB")

            result.setdefault("reranker", "CrossEncoder")

            result.setdefault("rerank_scores", [])

            result.setdefault("documents", [])

            result.setdefault("pages", [])

            result.setdefault("chunks", 0)

            result.setdefault("web_used", False)

            result.setdefault("web_sources", [])

            # ------------------------------------------------
            # Agent Metadata
            # ------------------------------------------------

            result["agent_used"] = True
            result["agent_decision"] = "KB"

            total_time = time.perf_counter() - request_start

            performance_logger.info("Total Agentic RAG Time : %.3f sec", total_time)

            rag_logger.info("AGENTIC RAG COMPLETED")

            rag_logger.info("=" * 80)

            return result

        # ------------------------------------------------
        # Step 3: NOT KB
        # ------------------------------------------------

        rag_logger.info("Agent decided Knowledge Base is not required")

        total_time = time.perf_counter() - request_start

        performance_logger.info("Total Agentic RAG Time : %.3f sec", total_time)

        return {
            "answer": (
                "This question does not require "
                "the internal Knowledge Base."
            ),

            "agent_used": True,
            "agent_decision": "NOT_KB",

            "llm": LLM_MODEL,
            "vector_db": "ChromaDB",
            "reranker": "CrossEncoder",
            "rerank_scores": [],

            "documents": [],
            "pages": [],
            "chunks": 0,

            "web_used": False,
            "web_sources": []
        }

    except Exception as ex:

        error_logger.exception(
            "Agentic RAG Failed : %s",
            ex
        )

        return {
            "answer": "Agentic RAG processing failed.",
            "agent_used": True,
            "agent_decision": "ERROR",
            "documents": [],
            "pages": [],
            "chunks": 0,
            "web_used": False,
            "web_sources": [],
            "error": str(ex)
        }


# --------------------------------------------------------
# Test
# --------------------------------------------------------

if __name__ == "__main__":

    question = input("Enter your question: ")

    result = agentic_rag(question)

    print("\nAgent Decision:")
    print(result.get("agent_decision"))

    print("\nAnswer:")
    print(result.get("answer"))