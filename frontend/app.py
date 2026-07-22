import json
import sqlite3
import sys
from html import escape
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.pipeline import FactCheckerPipeline, PipelineSettings
from config.setting import (
    CHUNKS_DATABASE_FILE,
    DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RETRIEVAL_TOP_K,
    EMBEDDINGS_METADATA_FILE,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_FILE,
    NLI_MODEL_NAME,
    PROCESSED_CHUNKS_FILE,
    PROTOTYPE_METRICS,
    RERANKER_MODEL_NAME,
)


st.set_page_config(
    page_title="Fact-Checker",
    page_icon="FC",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.4rem;
        max-width: 1240px;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9e1e8;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .verdict {
        border: 1px solid #cfd8e3;
        border-left: 6px solid #2f6fed;
        border-radius: 8px;
        padding: 18px 20px;
        background: #ffffff;
        margin: 10px 0 18px 0;
    }
    .verdict.supports { border-left-color: #14804a; }
    .verdict.refutes { border-left-color: #c43b2f; }
    .verdict.nei { border-left-color: #a46a00; }
    .badge {
        display: inline-block;
        border-radius: 999px;
        padding: 5px 11px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0;
    }
    .badge.supports { color: #0d5f38; background: #dff5e8; }
    .badge.refutes { color: #8a2219; background: #fde4df; }
    .badge.nei { color: #785000; background: #fff0c2; }
    .evidence-card {
        border: 1px solid #d9e1e8;
        border-radius: 8px;
        padding: 15px 16px;
        background: #ffffff;
        margin-bottom: 12px;
    }
    .muted {
        color: #52606d;
        font-size: 0.92rem;
    }
    .step {
        border: 1px solid #d9e1e8;
        border-radius: 8px;
        padding: 13px 14px;
        background: #ffffff;
        min-height: 82px;
    }
</style>
"""


def label_class(label: str) -> str:
    if label == "SUPPORTS":
        return "supports"
    if label == "REFUTES":
        return "refutes"
    return "nei"


@st.cache_data(show_spinner=False)
def load_artifact_status() -> dict:
    status = {
        "faiss_index": FAISS_INDEX_FILE.exists(),
        "sqlite_database": CHUNKS_DATABASE_FILE.exists(),
        "processed_chunks": PROCESSED_CHUNKS_FILE.exists(),
        "embeddings_metadata": EMBEDDINGS_METADATA_FILE.exists(),
        "chunk_count": None,
        "document_count": None,
        "embedding_dimension": None,
        "embedding_batches": None,
        "embedding_chunks": None,
    }

    if CHUNKS_DATABASE_FILE.exists():
        connection = sqlite3.connect(CHUNKS_DATABASE_FILE)
        try:
            row = connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT doc_id) FROM chunks"
            ).fetchone()
            status["chunk_count"] = row[0]
            status["document_count"] = row[1]
        finally:
            connection.close()

    if EMBEDDINGS_METADATA_FILE.exists():
        with EMBEDDINGS_METADATA_FILE.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        status["embedding_dimension"] = metadata.get("embedding_dimension")
        status["embedding_batches"] = metadata.get("total_batches")
        status["embedding_chunks"] = metadata.get("total_chunks")

    return status


@st.cache_resource(show_spinner="Loading retrieval, reranker and NLI models...")
def load_pipeline(
    retrieval_top_k: int,
    rerank_top_k: int,
    max_chunks_per_document: int,
) -> FactCheckerPipeline:
    settings = PipelineSettings(
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        max_chunks_per_document=max_chunks_per_document,
    )
    return FactCheckerPipeline(settings=settings)


def render_sidebar() -> tuple[int, int, int]:
    status = load_artifact_status()

    with st.sidebar:
        st.header("System")
        st.caption("Evidence-based FEVER pipeline")

        st.subheader("Models")
        st.write(f"Embedding: `{EMBEDDING_MODEL_NAME}`")
        st.write(f"Reranker: `{RERANKER_MODEL_NAME}`")
        st.write(f"NLI: `{NLI_MODEL_NAME}`")

        st.subheader("Runtime")
        retrieval_top_k = st.slider("Retrieval top-k", 5, 100, DEFAULT_RETRIEVAL_TOP_K, 5)
        rerank_top_k = st.slider("Rerank top-k", 1, 20, DEFAULT_RERANK_TOP_K, 1)
        max_chunks = st.slider(
            "Max chunks / document",
            1,
            10,
            DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
            1,
        )

        st.subheader("Artifacts")
        st.write("FAISS index", "OK" if status["faiss_index"] else "Missing")
        st.write("SQLite database", "OK" if status["sqlite_database"] else "Missing")
        st.write("Processed chunks", "OK" if status["processed_chunks"] else "Missing")
        st.write("Embeddings metadata", "OK" if status["embeddings_metadata"] else "Missing")

        if status["chunk_count"] is not None:
            st.metric("Chunks", f"{status['chunk_count']:,}")
            st.metric("Documents", f"{status['document_count']:,}")
        if status["embedding_dimension"] is not None:
            st.metric("Embedding dim.", status["embedding_dimension"])
            st.metric("Embedding batches", status["embedding_batches"])

    return retrieval_top_k, rerank_top_k, max_chunks


def render_verdict(result: dict) -> None:
    label = result["label"]
    css_class = label_class(label)
    confidence = result["confidence"]

    st.markdown(
        f"""
        <div class="verdict {css_class}">
            <div class="muted">Final verdict</div>
            <h2 style="margin: 0.2rem 0 0.45rem 0;">{label}</h2>
            <span class="badge {css_class}">Confidence {confidence:.1%}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns(3)
    scores = result["label_scores"]
    col_a.metric("SUPPORTS score", f"{scores['SUPPORTS']:.3f}")
    col_b.metric("REFUTES score", f"{scores['REFUTES']:.3f}")
    col_c.metric("NEI score", f"{scores['NOT ENOUGH INFO']:.3f}")


def render_evidence_card(evidence: dict, rank: int, primary: bool = False) -> None:
    title = escape(str(evidence.get("title") or evidence.get("doc_id") or "Unknown document"))
    text = escape(str(evidence.get("text", "")))
    label = evidence.get("fever_label", "NOT ENOUGH INFO")
    css_class = label_class(label)
    retrieval_score = evidence.get("retrieval_score")
    rerank_score = evidence.get("rerank_score")

    st.markdown(
        f"""
        <div class="evidence-card">
            <div class="muted">{'Primary evidence' if primary else f'Evidence #{rank}'}</div>
            <h4 style="margin: 0.15rem 0 0.35rem 0;">{title}</h4>
            <span class="badge {css_class}">{label}</span>
            <p style="margin-top: 0.8rem;">{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    cols[0].metric("NLI confidence", f"{evidence.get('confidence', 0):.3f}")
    cols[1].metric(
        "FAISS score",
        "n/a" if retrieval_score is None else f"{retrieval_score:.3f}",
    )
    cols[2].metric(
        "Rerank score",
        "n/a" if rerank_score is None else f"{rerank_score:.3f}",
    )
    cols[3].metric("Sentences", ", ".join(map(str, evidence.get("sentence_ids", []))) or "n/a")
    st.caption(f"Document source: {evidence.get('doc_id', 'n/a')}")


def page_fact_checking() -> None:
    st.title("Fact-Checker")
    st.caption("Automatic claim verification grounded in retrieved Wikipedia evidence.")

    retrieval_top_k, rerank_top_k, max_chunks = render_sidebar()

    default_claim = "Andrew Kevin Walker is only Chinese."
    claim = st.text_area(
        "Claim",
        value=default_claim,
        height=110,
        placeholder="Enter a factual claim to verify.",
    )

    submitted = st.button("Verify claim", type="primary", use_container_width=False)

    if submitted:
        if not claim.strip():
            st.warning("Enter a non-empty claim before launching verification.")
            return

        try:
            pipeline = load_pipeline(retrieval_top_k, rerank_top_k, max_chunks)
            result = pipeline.verify_claim(claim)
        except Exception as exc:
            st.error("The pipeline could not be loaded or executed.")
            st.exception(exc)
            return

        st.subheader("Analyzed Claim")
        st.write(result["claim"])
        render_verdict(result)

        best_evidence = result["best_evidence"]
        if best_evidence:
            st.subheader("Primary Evidence")
            render_evidence_card(best_evidence, rank=1, primary=True)
        else:
            st.info("No evidence was selected for this claim.")

        with st.expander("All NLI evidence", expanded=False):
            for rank, evidence in enumerate(result["evidence"], start=1):
                render_evidence_card(evidence, rank=rank)

        with st.expander("Technical details", expanded=False):
            st.json(
                {
                    "settings": result["settings"],
                    "retrieved_count": len(result["retrieved_passages"]),
                    "reranked_count": len(result["reranked_passages"]),
                    "label_scores": result["label_scores"],
                }
            )


def page_pipeline() -> None:
    render_sidebar()
    st.title("Pipeline")
    st.caption("Execution flow and model responsibilities.")

    st.graphviz_chart(
        """
        digraph {
            graph [rankdir=TB]
            node [shape=box, style="rounded,filled", color="#cfd8e3", fillcolor="#ffffff", fontname="Arial"]
            Claim -> "FAISS Retrieval" -> "Passage Retrieval" -> Reranking -> "NLI Verification" -> "Final Verdict"
        }
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="step"><b>Retrieval</b><br><span class="muted">Dense search over normalized MiniLM embeddings with FAISS inner product.</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="step"><b>Reranking</b><br><span class="muted">Cross-Encoder scores claim and passage pairs before final evidence selection.</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="step"><b>NLI</b><br><span class="muted">DeBERTa NLI labels are converted to FEVER labels.</span></div>', unsafe_allow_html=True)

    st.subheader("Model Configuration")
    st.table(
        [
            {"Stage": "Embeddings", "Model": EMBEDDING_MODEL_NAME},
            {"Stage": "Reranking", "Model": RERANKER_MODEL_NAME},
            {"Stage": "NLI", "Model": NLI_MODEL_NAME},
        ]
    )


def page_evaluation() -> None:
    render_sidebar()
    st.title("Evaluation")
    st.caption("Prototype metrics measured on a subset of the FEVER development data.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{PROTOTYPE_METRICS['accuracy']:.0%}")
    col2.metric("Evidence Recall", f"{PROTOTYPE_METRICS['evidence_recall']:.0%}")
    col3.metric("MRR", f"{PROTOTYPE_METRICS['mrr']:.2f}")
    col4.metric("FEVER Score", f"{PROTOTYPE_METRICS['fever_score']:.0%}")

    st.subheader("Metric Definition")
    st.write(
        "The FEVER Score used in this project counts an example only when the "
        "predicted label is correct and at least one gold evidence sentence is retrieved."
    )


def run() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    if hasattr(st, "navigation") and hasattr(st, "Page"):
        navigation = st.navigation(
            [
                st.Page(page_fact_checking, title="Fact-Checking"),
                st.Page(page_pipeline, title="Pipeline"),
                st.Page(page_evaluation, title="Evaluation"),
            ]
        )
        navigation.run()
        return

    page_name = st.sidebar.radio(
        "Navigation",
        ["Fact-Checking", "Pipeline", "Evaluation"],
    )
    if page_name == "Pipeline":
        page_pipeline()
    elif page_name == "Evaluation":
        page_evaluation()
    else:
        page_fact_checking()


if __name__ == "__main__":
    run()
