from dataclasses import dataclass

from config.setting import (
    DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RETRIEVAL_TOP_K,
)
from retrieval.document_retriever import DocumentRetriever
from retrieval.passage_retriever import PassageRetriever
from retrieval.reranker import Reranker
from verification.verifier import Verifier


@dataclass(frozen=True)
class PipelineSettings:
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K
    rerank_top_k: int = DEFAULT_RERANK_TOP_K
    max_chunks_per_document: int = DEFAULT_MAX_CHUNKS_PER_DOCUMENT


class FactCheckerPipeline:
    def __init__(
        self,
        document_retriever: DocumentRetriever | None = None,
        passage_retriever: PassageRetriever | None = None,
        reranker: Reranker | None = None,
        verifier: Verifier | None = None,
        settings: PipelineSettings | None = None,
    ):
        self.settings = settings or PipelineSettings()
        self.document_retriever = document_retriever or DocumentRetriever()
        self.passage_retriever = passage_retriever or PassageRetriever(
            self.document_retriever
        )
        self.reranker = reranker or Reranker()
        self.verifier = verifier or Verifier()

    def verify_claim(self, claim: str) -> dict:
        claim = claim.strip()
        if not claim:
            raise ValueError("Claim must not be empty.")

        passages = self.passage_retriever.retrieve(
            query=claim,
            top_k=self.settings.retrieval_top_k,
            max_chunks_per_document=self.settings.max_chunks_per_document,
        )
        reranked_passages = self.reranker.rerank(
            query=claim,
            passages=passages,
            top_k=self.settings.rerank_top_k,
        )
        verification = self.verifier.verify(
            claim=claim,
            evidence=reranked_passages,
        )

        return {
            "claim": claim,
            "label": verification["label"],
            "confidence": verification["confidence"],
            "label_scores": verification["label_scores"],
            "best_evidence": verification["best_evidence"],
            "evidence": verification["evidence"],
            "retrieved_passages": passages,
            "reranked_passages": reranked_passages,
            "settings": {
                "retrieval_top_k": self.settings.retrieval_top_k,
                "rerank_top_k": self.settings.rerank_top_k,
                "max_chunks_per_document": self.settings.max_chunks_per_document,
            },
        }

    def close(self) -> None:
        self.document_retriever.close()
