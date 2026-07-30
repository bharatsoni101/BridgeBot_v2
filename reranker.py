from sentence_transformers import CrossEncoder

# Production-quality reranker
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(MODEL_NAME)


def rerank(question, documents, top_k=3):

    if len(documents) == 0:
        return [], []

    pairs = [
        (question, doc.page_content)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, documents),
        key=lambda x: x[0],
        reverse=True
    )

    reranked_docs = [
        doc for score, doc in ranked[:top_k]
    ]

    reranked_scores = [
        float(score)
        for score, doc in ranked[:top_k]
    ]

    return reranked_docs, reranked_scores