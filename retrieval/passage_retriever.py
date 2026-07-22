from retrieval.document_retriever import DocumentRetriever


class PassageRetriever:
    def __init__(self, document_retriever: DocumentRetriever):
        self.document_retriever = document_retriever

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        max_chunks_per_document: int = 3,
    ) -> list[dict]:
        if max_chunks_per_document <= 0:
            raise ValueError("max_chunks_per_document must be greater than zero.")

        results = self.document_retriever.search(query=query, top_k=top_k)
        results = self.limit_chunks_per_document(results, max_chunks_per_document)
        return self.format_results(results)

    def limit_chunks_per_document(
        self,
        results: list[dict],
        max_chunks_per_document: int = 3,
    ) -> list[dict]:
        document_counts: dict[str, int] = {}
        filtered_results = []

        for result in results:
            doc_id = result["doc_id"]
            current_count = document_counts.get(doc_id, 0)

            if current_count >= max_chunks_per_document:
                continue

            filtered_results.append(result)
            document_counts[doc_id] = current_count + 1

        return filtered_results

    def format_result(self, result: dict) -> dict:
        return {
            "chunk_id": result["chunk_id"],
            "doc_id": result["doc_id"],
            "title": result["title"],
            "source": result["source"],
            "start_sentence": result["start_sentence"],
            "end_sentence": result["end_sentence"],
            "sentence_ids": result["sentence_ids"],
            "num_sentences": result["num_sentences"],
            "text_length": result["text_length"],
            "text": result["text"],
            "score": result["score"],
        }

    def format_results(self, results: list[dict]) -> list[dict]:
        return [self.format_result(result) for result in results]
