from typing import List

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)


class Verifier:

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/nli-deberta-v3-base"
        )
    ):

        print(
            f"Chargement du modèle NLI : "
            f"{model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_name)
        )

        self.model.eval()

        self.id2label = (
            self.model.config.id2label
        )

    # =========================================================================
    # PRÉDICTION NLI D'UN PASSAGE
    # =========================================================================

    def predict(
        self,
        claim: str,
        evidence: str
    ) -> dict:

        inputs = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

        predicted_id = (
            torch.argmax(
                probabilities
            ).item()
        )

        nli_label = (
            self.id2label[
                predicted_id
            ]
        )

        confidence = (
            probabilities[
                predicted_id
            ].item()
        )

        return {

            "label": nli_label,

            "confidence": confidence,

            "probabilities": (
                probabilities.tolist()
            )
        }

    # =========================================================================
    # CONVERSION NLI → FEVER
    # =========================================================================

    def convert_to_fever_label(
        self,
        nli_label: str
    ) -> str:

        label = (
            nli_label.lower()
        )

        if "entail" in label:

            return "SUPPORTS"

        if (
            "contradiction" in label
            or "contradict" in label
        ):

            return "REFUTES"

        return "NOT ENOUGH INFO"

    # =========================================================================
    # ANALYSE D'UN PASSAGE
    # =========================================================================

    def analyze_passage(
        self,
        claim: str,
        passage: dict
    ) -> dict:

        result = self.predict(
            claim=claim,
            evidence=passage["text"]
        )

        fever_label = (
            self.convert_to_fever_label(
                result["label"]
            )
        )

        return {

            "doc_id": passage["doc_id"],

            "title": passage["title"],

            "text": passage["text"],

            "sentence_ids": (
                passage["sentence_ids"]
            ),

            "nli_label": result["label"],

            "fever_label": fever_label,

            "confidence": (
                result["confidence"]
            ),

            "probabilities": (
                result["probabilities"]
            )
        }

    # =========================================================================
    # VÉRIFICATION DU CLAIM
    # =========================================================================
    def aggregate_predictions(
    self,
    predictions: List[dict]
    ) -> dict:

        label_scores = {

            "SUPPORTS": 0.0,

            "REFUTES": 0.0,

            "NOT ENOUGH INFO": 0.0

        }

        for prediction in predictions:

            label = (
                prediction["fever_label"]
            )

            confidence = (
                prediction["confidence"]
            )

            label_scores[label] += (
                confidence
            )

        best_label = max(
            label_scores,
            key=label_scores.get
        )

        return {

            "label": best_label,

            "confidence": (
                label_scores[best_label]
            ),

            "label_scores": label_scores

        }
    
    
    def verify(
        self,
        claim: str,
        evidence: List[dict]
    ) -> dict:

        predictions = []

        # ---------------------------------------------------------------------
        # ANALYSE DE CHAQUE PASSAGE
        # ---------------------------------------------------------------------

        for passage in evidence:

            prediction = (
                self.analyze_passage(
                    claim=claim,
                    passage=passage
                )
            )

            predictions.append(
                prediction
            )

        # ---------------------------------------------------------------------
        # AUCUNE PREUVE
        # ---------------------------------------------------------------------

        if not predictions:

            return {

                "label": (
                    "NOT ENOUGH INFO"
                ),

                "confidence": 0.0,

                "best_evidence": None,

                "evidence": []
            }

        # ---------------------------------------------------------------------
        # MEILLEURE PRÉDICTION INDIVIDUELLE
        # ---------------------------------------------------------------------

        best_prediction = max(
            predictions,
            key=lambda item: (
                item["confidence"]
            )
        )

        aggregated_result = (
            self.aggregate_predictions(
                predictions
            )
        )

        # ---------------------------------------------------------------------
        # RÉSULTAT FINAL
        # ---------------------------------------------------------------------

        return {

            "label": (
                aggregated_result[
                    "fever_label"
                ]
            ),

            "confidence": (
                aggregated_result[
                    "confidence"
                ]
            ),

            "best_evidence": (
                aggregated_result
            ),

            "evidence": predictions
        }
    

