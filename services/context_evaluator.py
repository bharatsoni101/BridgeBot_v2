import json
import time

from langchain_groq import ChatGroq

from utils.logger import (rag_logger, performance_logger, error_logger)


# ============================================================
# Configuration
# ============================================================

LLM_MODEL = "llama-3.3-70b-versatile"


# ============================================================
# Context Evaluator LLM
# ============================================================

evaluator_llm = ChatGroq(model=LLM_MODEL, temperature=0)


# ============================================================
# Evaluate Context
# ============================================================

def evaluate_context(question, context):
    """
    Evaluate whether the retrieved context is sufficient
    to answer the user's question.

    Returns:

        {
            "decision": "SUFFICIENT" | "INSUFFICIENT",
            "reason": "...",
            "confidence": 0.0 - 1.0
        }
    """

    request_start = time.perf_counter()

    rag_logger.info("-" * 60)
    rag_logger.info("Context Evaluation Started")
    rag_logger.info("Question : %s", question)

    try:

        if not context or not context.strip():

            rag_logger.warning(
                "No context available for evaluation"
            )

            return {
                "decision": "INSUFFICIENT",
                "reason": "No relevant context was retrieved.",
                "confidence": 1.0
            }

        prompt = f"""
You are a Context Evaluator for an enterprise Agentic RAG system.

Your job is ONLY to determine whether the supplied context
contains enough reliable information to answer the user's question.

Do NOT answer the question.

Return ONLY valid JSON.

Required format:

{{
    "decision": "SUFFICIENT" or "INSUFFICIENT",
    "reason": "short explanation",
    "confidence": 0.0
}}

Rules:

1. Return SUFFICIENT only when the context contains enough
   relevant information to directly answer the question.

2. Return INSUFFICIENT when:
   - important information is missing
   - the context is unrelated
   - the context only partially answers the question
   - the question requires current/external information
     that is not present in the context
   - the context contains conflicting or ambiguous information

3. Do not use your own knowledge.

4. Evaluate ONLY the supplied context.

5. confidence must be a number between 0.0 and 1.0.

-------------------------
QUESTION
-------------------------

{question}

-------------------------
CONTEXT
-------------------------

{context}

-------------------------
EVALUATION
-------------------------
"""

        start = time.perf_counter()

        response = evaluator_llm.invoke(prompt)

        evaluation_time = time.perf_counter() - start

        performance_logger.info("Context Evaluation LLM Time : %.3f sec", evaluation_time)

        raw_response = response.content.strip()

        rag_logger.info("Context Evaluator Response : %s", raw_response)

        # ----------------------------------------------------
        # Remove Markdown JSON fences if present
        # ----------------------------------------------------

        raw_response = raw_response.replace("```json", "").replace("```", "").strip()

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        evaluation = json.loads(raw_response)

        decision = evaluation.get("decision", "INSUFFICIENT").upper()

        reason = evaluation.get("reason", "")

        confidence = float(evaluation.get("confidence", 0.0))

        # ----------------------------------------------------
        # Validate Decision
        # ----------------------------------------------------

        if decision not in ("SUFFICIENT", "INSUFFICIENT"):

            rag_logger.warning("Invalid evaluator decision : %s", decision)

            decision = "INSUFFICIENT"

        confidence = max(0.0, min(1.0, confidence))

        total_time = time.perf_counter() - request_start

        performance_logger.info("Total Context Evaluation Time : %.3f sec", total_time)

        rag_logger.info("Context Decision : %s", decision)

        rag_logger.info("Context Confidence : %.3f", confidence)

        rag_logger.info("Context Evaluation Completed")

        rag_logger.info("-" * 60)

        return {"decision": decision, "reason": reason, "confidence": confidence}

    except Exception as e:

        error_logger.exception("Context Evaluation Failed : %s", e)

        # Fail closed.
        # If evaluation fails, assume context is insufficient.

        return {"decision": "INSUFFICIENT", "reason": f"Context evaluation failed: {str(e)}", "confidence": 0.0}