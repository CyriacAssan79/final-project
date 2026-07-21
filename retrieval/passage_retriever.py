from typing import List

from retrieval.document_retriever import (DocumentRetriever)

class PassageRetriever:
    """
    Récupère et prépare les passages
    les plus pertinents pour une requête.
    """

    def __init__(self, document_retriever: DocumentRetriever):

        self.document_retriever = (document_retriever) #L'idée est que PassageRetriever réutilise le DocumentRetriever déjà chargé.

    def retrieve(self,query: str,top_k: int = 50,max_chunks_per_document: int = 3) -> list[dict]:

        results = (self.document_retriever.search(query=query,top_k=top_k))
        results = (self.limit_chunks_per_document(results,max_chunks_per_document))
        results = (self.format_results(results))
        return results
    

    # def deduplicate_by_document(self,results: list[dict]) -> list[dict]:
    #     """
    #     Supprime les résultats provenant du même document.
    #     """

    #     seen_documents = set()
    #     unique_results = []

    #     for result in results:
    #         doc_id = result["doc_id"]
    #         if doc_id in seen_documents:
    #             continue
    #         seen_documents.add(doc_id)
    #         unique_results.append(result)

    #     return unique_results
    
    def limit_chunks_per_document(self,results: list[dict],max_chunks_per_document: int = 3) -> list[dict]:
        """
        Limite le nombre de chunks provenantd'un même document.
        """

        document_counts = {}
        filtered_results = []

        for result in results:
            doc_id = result["doc_id"]
            current_count = (document_counts.get(doc_id,0))

            if (current_count>= max_chunks_per_document):
                continue

            filtered_results.append(result)
            document_counts[doc_id] = (current_count + 1)

        return filtered_results
    
    def format_result(self,result: dict) -> dict:
        """
        Normalise un résultat pour les étapes suivantes du pipeline.
        """

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
        """
        Normalise tous les résultats.
        """

        return [self.format_result(result) for result in results ]
    
retriever = DocumentRetriever()

passage_retriever = PassageRetriever(retriever)

results = passage_retriever.retrieve(query="Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",top_k=50)

for result in results:
    print("=" * 80)
    print(result["title"])
    print(result["score"])
    print(result["text"])