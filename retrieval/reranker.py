from sentence_transformers import CrossEncoder


class Reranker:
    """
    Réévalue la pertinence des passages
    par rapport à une requête.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):

        self.model = CrossEncoder(model_name)

        print("Reranker chargé.")

    def rerank(self,query: str,passages: list[dict],top_k: int = 5) -> list[dict]:
        """
        Réordonne les passages selon leur pertinence
        par rapport à la requête.
        """

        if not passages:
            return []

        pairs = [(query,passage["text"]) for passage in passages]
        scores = self.model.predict(pairs)
        reranked_results = []

        for passage, score in zip(passages,scores):

            result = passage.copy()
            result["rerank_score"] = float(score)
            reranked_results.append(result)

        reranked_results.sort(key=lambda x: x["rerank_score"],reverse=True)
        return reranked_results[:top_k]
    

if __name__ == "__main__":

    from retrieval.document_retriever import (DocumentRetriever)
    from retrieval.passage_retriever import (PassageRetriever)

    document_retriever = DocumentRetriever()
    passage_retriever = PassageRetriever(document_retriever)
    reranker = Reranker()
    query = "Who was the first president of the United States?"

    passages = passage_retriever.retrieve(query=query,top_k=50)
    results = reranker.rerank(query=query,passages=passages,top_k=5)

    for result in results:
        print("=" * 80)
        print(f"Title: {result['title']}")
        print(f"FAISS score: {result['score']}")
        print(f"Rerank score: {result['rerank_score']}")
        print(f"Text: {result['text']}")