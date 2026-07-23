from typing import Any

import torch

from config.setting import NLI_MODEL_NAME


class Verifier:
    def __init__(
        self,
        model_name: str = NLI_MODEL_NAME,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ):
        self.model_name = model_name

        if tokenizer is not None and model is not None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.eval()
        self.id2label = dict(self.model.config.id2label)

    def predict(self, claim: str, evidence: str) -> dict:
        return self.predict_batch(claim, [evidence])[0]

    def predict_batch(self, claim: str, evidence_texts: list[str]) -> list[dict]:
        claim = claim.strip()
        if not claim:
            raise ValueError("Claim must not be empty.")

        evidence_texts = [text.strip() for text in evidence_texts]
        if any(not text for text in evidence_texts):
            raise ValueError("Evidence text must not be empty.")
        if not evidence_texts:
            return []

        inputs = self.tokenizer(
            evidence_texts,
            [claim] * len(evidence_texts),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_ids = torch.argmax(probabilities, dim=1)

        results = []
        for row_probabilities, predicted_id_tensor in zip(probabilities, predicted_ids):
            predicted_id = predicted_id_tensor.item()
            results.append(
                {
                    "label": self.id2label[predicted_id],
                    "confidence": row_probabilities[predicted_id].item(),
                    "probabilities": row_probabilities.tolist(),
                }
            )
        return results

    def convert_to_fever_label(self, nli_label: str) -> str:
        label = nli_label.lower()

        if "entail" in label:
            return "SUPPORTS"
        if "contradiction" in label or "contradict" in label:
            return "REFUTES"
        return "NOT ENOUGH INFO"

    def analyze_passage(self, claim: str, passage: dict) -> dict:
        return self.analyze_passages(claim=claim, passages=[passage])[0]

    def analyze_passages(self, claim: str, passages: list[dict]) -> list[dict]:
        if not passages:
            return []

        results = self.predict_batch(
            claim, [passage["text"] for passage in passages]
        )

        analyzed = []
        for passage, result in zip(passages, results):
            fever_label = self.convert_to_fever_label(result["label"])
            analyzed.append(
                {
                    "chunk_id": passage.get("chunk_id"),
                    "doc_id": passage["doc_id"],
                    "title": passage.get("title", passage["doc_id"].replace("_", " ")),
                    "text": passage["text"],
                    "sentence_ids": passage.get("sentence_ids", []),
                    "retrieval_score": passage.get("score"),
                    "rerank_score": passage.get("rerank_score"),
                    "nli_label": result["label"],
                    "fever_label": fever_label,
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                }
            )
        return analyzed

    def aggregate_predictions(self, predictions: list[dict]) -> dict:
        label_scores = {
            "SUPPORTS": 0.0,
            "REFUTES": 0.0,
            "NOT ENOUGH INFO": 0.0,
        }

        for prediction in predictions:
            label_scores[prediction["fever_label"]] += prediction["confidence"]

        best_label = max(label_scores, key=label_scores.get)
        total_score = sum(label_scores.values())
        confidence = label_scores[best_label] / total_score if total_score else 0.0
        matching_predictions = [
            prediction
            for prediction in predictions
            if prediction["fever_label"] == best_label
        ]
        best_evidence = max(
            matching_predictions or predictions,
            key=lambda item: item["confidence"],
        )

        return {
            "label": best_label,
            "confidence": confidence,
            "label_scores": label_scores,
            "best_evidence": best_evidence,
        }

    def verify(self, claim: str, evidence: list[dict]) -> dict:
        if not evidence:
            return {
                "label": "NOT ENOUGH INFO",
                "confidence": 0.0,
                "label_scores": {
                    "SUPPORTS": 0.0,
                    "REFUTES": 0.0,
                    "NOT ENOUGH INFO": 1.0,
                },
                "best_evidence": None,
                "evidence": [],
            }

        predictions = self.analyze_passages(claim=claim, passages=evidence)
        aggregated_result = self.aggregate_predictions(predictions)

        return {
            "label": aggregated_result["label"],
            "confidence": aggregated_result["confidence"],
            "label_scores": aggregated_result["label_scores"],
            "best_evidence": aggregated_result["best_evidence"],
            "evidence": predictions,
        }
