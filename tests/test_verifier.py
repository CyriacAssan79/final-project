from verification.verifier import Verifier


# =============================================================================
# TEST DU VÉRIFICATEUR
# =============================================================================

if __name__ == "__main__":

    claim = "Andrew Kevin Walker is only Chinese."
    evidence = [
        {
            "doc_id": "Andrew_Kevin_Walker",
            "title":  "Andrew Kevin Walker",
            "text": "Andrew Kevin Walker is an American BAFTA-nominated screenwriter."  
        }
    ]

    # -------------------------------------------------------------------------
    # CHARGEMENT DU VÉRIFICATEUR
    # -------------------------------------------------------------------------

    verifier = Verifier()

    # -------------------------------------------------------------------------
    # VÉRIFICATION
    # -------------------------------------------------------------------------

    result = verifier.verify(claim=claim,evidence=evidence)

    # -------------------------------------------------------------------------
    # AFFICHAGE
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("RÉSULTAT DE LA VÉRIFICATION")
    print("=" * 100)

    print()
    print(f"CLAIM : {claim}")

    print()
    print(f"LABEL : {result['label']}")

    print()
    print(f"CONFIANCE : {result['confidence']:.4f}")

    print()
    print("=" * 100)
    print("DÉTAIL DES PRÉDICTIONS")
    print("=" * 100)

    for prediction in result["evidence"]:

        print()
        print(f"DOCUMENT : {prediction['doc_id']}")
        print(f"LABEL NLI : {prediction['nli_label']}")
        print(f"LABEL FEVER : {prediction['fever_label']}")
        print(f"CONFIANCE : {prediction['confidence']:.4f}")