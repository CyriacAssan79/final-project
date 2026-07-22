from typing import Any

from config.setting import RERANKER_MODEL_NAME


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME, model: Any | None = None):
        if model is not None:
            self.model = model
            self.model_name = model_name
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'sentence-transformers'. Activate the project "
                "environment before running reranking."
            ) from exc

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, passages: list[dict], top_k: int = 5) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not passages:
            return []

        pairs = [(query, passage["text"]) for passage in passages]
        scores = self.model.predict(pairs)

        reranked_results = []
        for passage, score in zip(passages, scores):
            result = passage.copy()
            result["rerank_score"] = float(score)
            reranked_results.append(result)

        reranked_results.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked_results[:top_k]
