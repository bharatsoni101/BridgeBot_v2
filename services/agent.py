import time

from langchain_groq import ChatGroq
from langchain.agents import create_agent

from .agent_tools import (knowledge_base_search, web_search_tool, knowledge_base_and_web_search)

from utils.logger import (
    rag_logger,
    performance_logger,
    error_logger
)


# ============================================================
# Configuration
# ============================================================

LLM_MODEL = "llama-3.3-70b-versatile"


# ============================================================
# Groq LLM
# ============================================================

rag_logger.info(
    "Initializing Agent LLM : %s",
    LLM_MODEL
)

llm = ChatGroq(
    model=LLM_MODEL,
    temperature=0
)

rag_logger.info(
    "Agent LLM Initialized Successfully"
)


# ============================================================
# Agent Tools
# ============================================================

tools = [
    knowledge_base_search,
    web_search_tool,
    knowledge_base_and_web_search
]


# ============================================================
# Agent Instructions
# ============================================================

SYSTEM_PROMPT = """
You are BridgeBot, an enterprise Agentic RAG assistant.

You have access to three search options:

1. KNOWLEDGE BASE (KB)
   Use this when the question can be answered from
   the organization's uploaded documents.

2. WEB SEARCH (WEB)
   Use this when the question requires current,
   public, or external internet information and
   the internal Knowledge Base is not required.

3. KNOWLEDGE BASE + WEB (KB_WEB)
   Use this when the question requires both:
   - information from internal uploaded documents
   - external/current web information

IMPORTANT ROUTING RULES:

- Prefer KB when the question is about internal documents,
  company information, uploaded PDFs, policies, procedures,
  technical documents, or internal knowledge.

- Prefer WEB when the question requires current external
  information, latest information, public information,
  news, websites, or information not expected to exist
  in the internal Knowledge Base.

- Use KB_WEB when the question requires comparing,
  combining, or validating internal information with
  external information.

- Do not use WEB unnecessarily.

- Do not use KB unnecessarily for purely public/current
  questions.

- If the question clearly requires both internal and
  external information, use KB_WEB.

After selecting the appropriate tool, use that tool to
answer the user's question.

Do not explain the routing decision unless asked.
"""


# ============================================================
# Create Agent
# ============================================================

rag_logger.info(
    "Creating Agent"
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)

rag_logger.info(
    "Agent Created Successfully"
)


# ============================================================
# Ask Agent
# ============================================================

def ask_agent(question,
              selected_documents=None,
              selected_category=None,
              owner=None,
              department=None,
              team=None,
              visibility="Private"):

    request_start = time.perf_counter()

    rag_logger.info("=" * 80)
    rag_logger.info("Agentic RAG Request Started")
    rag_logger.info("Question : %s", question)

    try:

        if not question or not question.strip():

            return {
                "answer": "Please enter a question.",
                "source": None,
                "agent_decision": None,
                "documents": [],
                "pages": [],
                "chunks": 0,
                "web_used": False,
                "web_sources": [],
                "reranker": "Not Used",
                "rerank_scores": [],
                "llm": LLM_MODEL,
                "vector_db": "Not Used"
            }

        # ----------------------------------------------------
        # Invoke Agent
        # ----------------------------------------------------

        start = time.perf_counter()

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            }
        )

        performance_logger.info(
            "Agent Execution Time : %.3f sec",
            time.perf_counter() - start
        )

        # ----------------------------------------------------
        # Detect Tool Used
        # ----------------------------------------------------

        agent_decision = None
        tool_result = None

        messages = result.get("messages", [])

        for message in messages:

            # ToolMessage contains the result returned
            # by agent_tools.py

            message_type = message.__class__.__name__

            if message_type == "ToolMessage":

                tool_name = getattr(
                    message,
                    "name",
                    ""
                )

                rag_logger.info(
                    "Agent Tool Executed : %s",
                    tool_name
                )

                if tool_name == "knowledge_base_search":

                    agent_decision = "KB"

                elif tool_name == "web_search_tool":

                    agent_decision = "WEB"

                elif tool_name == "knowledge_base_and_web_search":

                    agent_decision = "KB_WEB"

                # --------------------------------------------
                # Extract Tool Result
                # --------------------------------------------

                tool_result = message.content

        # ----------------------------------------------------
        # Final Agent Answer
        # ----------------------------------------------------

        final_message = messages[-1] if messages else None

        answer = getattr(
            final_message,
            "content",
            "No response generated."
        )

        # ----------------------------------------------------
        # If Tool Returned Complete RAG Result
        # ----------------------------------------------------

        if isinstance(tool_result, dict):

            rag_result = tool_result.copy()

            rag_result["answer"] = answer

            rag_result["source"] = agent_decision

            rag_result["agent_decision"] = agent_decision

        else:

            # ------------------------------------------------
            # Fallback
            # ------------------------------------------------

            rag_result = {

                "answer": answer,

                "source": agent_decision,

                "agent_decision": agent_decision,

                "documents": [],

                "pages": [],

                "chunks": 0,

                "web_used": (
                        agent_decision in ("WEB", "KB_WEB")
                ),

                "web_sources": [],

                "reranker": (
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    if agent_decision in ("KB", "KB_WEB")
                    else "Not Used"
                ),

                "rerank_scores": [],

                "llm": LLM_MODEL,

                "vector_db": (
                    "ChromaDB"
                    if agent_decision in ("KB", "KB_WEB")
                    else "Not Used"
                )
            }

        # ----------------------------------------------------
        # Guarantee Required Fields
        # ----------------------------------------------------

        rag_result.setdefault(
            "documents",
            []
        )

        rag_result.setdefault(
            "pages",
            []
        )

        rag_result.setdefault(
            "chunks",
            0
        )

        rag_result.setdefault(
            "web_used",
            agent_decision in ("WEB", "KB_WEB")
        )

        rag_result.setdefault(
            "web_sources",
            []
        )

        rag_result.setdefault(
            "reranker",
            "Not Used"
        )

        rag_result.setdefault(
            "rerank_scores",
            []
        )

        rag_result.setdefault(
            "llm",
            LLM_MODEL
        )

        rag_result.setdefault(
            "vector_db",
            (
                "ChromaDB"
                if agent_decision in ("KB", "KB_WEB")
                else "Not Used"
            )
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        rag_logger.info(
            "Agent Decision : %s",
            agent_decision
        )

        rag_logger.info(
            "Retrieved Documents : %s",
            rag_result["documents"]
        )

        rag_logger.info(
            "Retrieved Pages : %s",
            rag_result["pages"]
        )

        rag_logger.info(
            "Chunks : %s",
            rag_result["chunks"]
        )

        total_time = time.perf_counter() - request_start

        performance_logger.info(
            "Total Agentic RAG Time : %.3f sec",
            total_time
        )

        rag_logger.info(
            "Agentic RAG Request Completed"
        )

        rag_logger.info("=" * 80)

        return rag_result

    except Exception as e:

        error_logger.exception(
            "Agentic RAG Failed : %s",
            e
        )

        return {
            "answer": f"Agentic RAG failed: {str(e)}",
            "source": None,
            "agent_decision": None,
            "documents": [],
            "pages": [],
            "chunks": 0,
            "web_used": False,
            "web_sources": [],
            "reranker": "Not Used",
            "rerank_scores": [],
            "llm": LLM_MODEL,
            "vector_db": "Not Used"
        }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )

    result = ask_agent(question)

    print("\n===================================")
    print("Agent Decision :", result["agent_decision"])
    print("===================================")

    print("\nAnswer:")
    print(result["answer"])