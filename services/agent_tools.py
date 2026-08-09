from langchain_core.tools import tool

from query import ask

from utils.logger import (rag_logger, performance_logger, error_logger)


# ========================================================
# Knowledge Base Tool
# ========================================================

@tool
def knowledge_base_search(question: str):
    """
    Search the internal BridgeBot Knowledge Base.

    Metadata filters such as owner, department and team
    are controlled by the authenticated user/session and
    are not supplied by the LLM.
    """

    rag_logger.info("Agent Tool Selected : KB")

    try:

        result = ask(question=question, source="KB")

        return result

    except Exception as e:

        error_logger.exception("KB Agent Tool Failed : %s", e)

        return {"answer": f"Knowledge Base search failed: {str(e)}", "source": "KB", "agent_decision": "KB"}


# ========================================================
# Web Search Tool
# ========================================================
@tool
def web_search_tool(question: str):
    """
    Search the public web.
    """

    rag_logger.info("Agent Tool Selected : WEB")

    try:

        result = ask(question=question, source="WEB")

        return result

    except Exception as e:

        error_logger.exception("Web Search Agent Tool Failed : %s", e)

        return {
            "answer": f"Web search failed: {str(e)}",
            "source": "WEB",
            "agent_decision": "WEB"
        }


# ========================================================
# KB + Web Tool
# ========================================================

@tool
def knowledge_base_and_web_search(question: str):
    """
    Search the internal Knowledge Base and public web.
    """

    rag_logger.info("Agent Tool Selected : KB_WEB")

    try:

        result = ask(question=question, source="KB_WEB")

        return result

    except Exception as e:

        error_logger.exception("KB + Web Agent Tool Failed : %s", e)

        return {"answer": f"KB + Web search failed: {str(e)}", "source": "KB_WEB", "agent_decision": "KB_WEB"}