import json
import random
import sqlite3
import sys
from collections import Counter
from datetime import datetime
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
from retrieval.document_retriever import DocumentRetriever
from retrieval.passage_retriever import PassageRetriever
from retrieval.reranker import Reranker
from verification.verifier import Verifier


st.set_page_config(
    page_title="Fact-Checker AI",
    page_icon="FC",
    layout="wide",
    initial_sidebar_state="expanded",
)


EXAMPLE_CLAIMS = [
    "The Earth is flat.",
    "Albert Einstein won the Nobel Prize in Physics.",
    "The Eiffel Tower is located in London.",
]

PAGES = [
    "Check a Claim",
    "Fact-Check Result",
    "Fact-check history",
    "About the AI",
    "Evaluation",
]


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@600;700;800&display=swap');

:root {
    --fc-bg: #fbf8fa;
    --fc-surface: #f5f3f4;
    --fc-card: #ffffff;
    --fc-border: #c5c6cd;
    --fc-muted-surface: #eae7e9;
    --fc-ink: #091426;
    --fc-ink-soft: #1b1b1d;
    --fc-text: #45474c;
    --fc-muted: #8590a6;
    --fc-green: #6cf8bb;
    --fc-green-dark: #006c49;
    --fc-green-text: #00714d;
    --fc-red-soft: #ffdad6;
    --fc-red: #ba1a1a;
    --fc-red-text: #93000a;
    --fc-neutral: #e4e2e3;
    --fc-neutral-text: #75777d;
    --fc-dark-panel: #1e293b;
    --fc-radius-sm: 4px;
    --fc-radius-md: 8px;
    --fc-radius-lg: 12px;
}

@media (prefers-color-scheme: dark) {
    :root {
        --fc-bg: #0f1115;
        --fc-surface: #171a21;
        --fc-card: #1b1f27;
        --fc-border: #2b303b;
        --fc-muted-surface: #232733;
        --fc-ink: #f5f6f8;
        --fc-ink-soft: #e2e4e9;
        --fc-text: #b7bcc7;
        --fc-neutral: #2b303b;
        --fc-neutral-text: #b7bcc7;
    }
}

html, body, [class*="css"], .stApp {
    background: var(--fc-bg);
    color: var(--fc-ink);
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.block-container {
    max-width: 1280px;
    padding: 0 48px 48px;
}

section[data-testid="stSidebar"] {
    background: var(--fc-bg);
    border-right: 1px solid var(--fc-border);
}

section[data-testid="stSidebar"] > div {
    padding: 32px 16px;
}

h1, h2, h3, .fc-title, .fc-page-title {
    font-family: "Plus Jakarta Sans", Inter, sans-serif;
    color: var(--fc-ink);
    letter-spacing: 0;
}

.fc-brand {
    padding: 0 8px 24px;
}

.fc-brand-title {
    font-family: "Plus Jakarta Sans", Inter, sans-serif;
    font-size: 24px;
    line-height: 32px;
    font-weight: 800;
    color: var(--fc-ink);
}

.fc-brand-subtitle {
    color: var(--fc-text);
    font-size: 14px;
    line-height: 20px;
    opacity: .7;
}

.fc-nav-active {
    background: var(--fc-muted-surface);
    border-right: 4px solid var(--fc-ink);
    border-radius: var(--fc-radius-md);
    color: var(--fc-ink);
    font-size: 14px;
    line-height: 16px;
    font-weight: 800;
    letter-spacing: .14px;
    padding: 14px 16px;
    margin: 4px 0;
}

.fc-topbar {
    height: 64px;
    border-bottom: 1px solid var(--fc-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -48px 0;
    padding: 0 48px;
    background: var(--fc-bg);
}

.fc-page-title {
    font-size: 24px;
    line-height: 32px;
    font-weight: 800;
}

.fc-main {
    padding-top: 48px;
}

.fc-hero {
    max-width: 928px;
    margin: 72px auto 32px;
    text-align: center;
}

.fc-badge {
    display: inline-flex;
    gap: 8px;
    align-items: center;
    background: var(--fc-green);
    color: var(--fc-green-text);
    border-radius: 9999px;
    padding: 4px 12px;
    font-size: 10px;
    line-height: 15px;
    letter-spacing: .5px;
    text-transform: uppercase;
}

.fc-badge-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--fc-green-text);
}

.fc-hero h1 {
    margin: 26px 0 14px;
    font-size: 40px;
    line-height: 48px;
    font-weight: 800;
}

.fc-hero p {
    max-width: 672px;
    margin: 0 auto;
    color: var(--fc-text);
    font-size: 18px;
    line-height: 28px;
}

.fc-panel,
.fc-card,
.fc-evidence-card,
.fc-history-table,
.fc-bento-card {
    background: var(--fc-card);
    border: 1px solid var(--fc-border);
    border-radius: var(--fc-radius-lg);
}

.fc-claim-panel {
    max-width: 896px;
    margin: 0 auto;
    padding: 9px;
    box-shadow: 0 8px 15px rgba(0, 0, 0, .04);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid var(--fc-border);
    border-radius: var(--fc-radius-lg);
    background: var(--fc-card);
    box-shadow: 0 8px 15px rgba(0, 0, 0, .04);
    padding: 9px;
}

.stTextArea textarea {
    min-height: 96px !important;
    border: 0 !important;
    border-radius: var(--fc-radius-lg) !important;
    padding: 24px !important;
    color: var(--fc-ink) !important;
    font-size: 20px !important;
    line-height: 30px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

.stTextArea textarea::placeholder {
    color: rgba(197, 198, 205, .95) !important;
}

.stButton > button {
    border-radius: var(--fc-radius-md);
    border: 1px solid transparent;
    min-height: 44px;
    padding: 9px 16px;
    font-family: Inter, sans-serif;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: .14px;
    color: var(--fc-text);
    background: transparent;
}

.stButton > button:hover {
    border-color: var(--fc-border);
    background: var(--fc-surface);
    color: var(--fc-ink);
}

.stButton > button[kind="primary"] {
    background: var(--fc-ink);
    color: #ffffff;
    border-color: var(--fc-ink);
    padding-left: 32px;
    padding-right: 32px;
    box-shadow: 0 10px 15px -3px rgba(9, 20, 38, .10),
        0 4px 6px -4px rgba(9, 20, 38, .10);
}

.fc-chip-row {
    max-width: 896px;
    margin: 32px auto 0;
}

.fc-section-grid {
    max-width: 928px;
    margin: 64px auto 0;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 24px;
}

.fc-feature-card {
    background: var(--fc-surface);
    border: 1px solid var(--fc-border);
    border-radius: var(--fc-radius-lg);
    padding: 25px;
    min-height: 221px;
}

.fc-feature-icon {
    width: 40px;
    height: 40px;
    border-radius: var(--fc-radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 16px;
}

.fc-feature-card h3 {
    margin: 0 0 8px;
    font-size: 14px;
    line-height: 16px;
    font-weight: 700;
}

.fc-feature-card p {
    margin: 0;
    color: var(--fc-text);
    font-size: 14px;
    line-height: 22.75px;
}

.fc-result-grid {
    display: grid;
    grid-template-columns: minmax(0, 8fr) minmax(280px, 4fr);
    gap: 24px;
}

.fc-card {
    padding: 33px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, .05);
}

.fc-card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
}

.fc-label {
    color: var(--fc-text);
    font-size: 12px;
    line-height: 14px;
    font-weight: 700;
    letter-spacing: .6px;
    text-transform: uppercase;
}

.fc-claim-title {
    margin-top: 8px;
    font-family: "Plus Jakarta Sans", Inter, sans-serif;
    font-size: 24px;
    line-height: 32px;
    font-weight: 700;
    color: var(--fc-ink);
}

.fc-verdict-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border-radius: 9999px;
    padding: 8px 16px;
    font-size: 14px;
    line-height: 16px;
    font-weight: 800;
    letter-spacing: .14px;
}

.fc-verdict-pill.supports { background: var(--fc-green); color: var(--fc-green-text); }
.fc-verdict-pill.refutes { background: var(--fc-red-soft); color: var(--fc-red-text); }
.fc-verdict-pill.nei { background: var(--fc-neutral); color: var(--fc-text); }

.fc-score-row {
    border-top: 1px solid var(--fc-border);
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 32px;
    margin-top: 32px;
    padding-top: 17px;
}

.fc-score {
    display: flex;
    align-items: baseline;
    gap: 8px;
}

.fc-score strong {
    font-family: "Plus Jakarta Sans", Inter, sans-serif;
    font-size: 32px;
    line-height: 40px;
    letter-spacing: 0;
}

.fc-progress-track {
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    background: var(--fc-muted-surface);
    margin-top: 8px;
}

.fc-progress-fill {
    height: 100%;
    border-radius: inherit;
    background: var(--fc-ink);
}

.fc-progress-fill.supports { background: var(--fc-green-dark); }
.fc-progress-fill.refutes { background: var(--fc-red); }
.fc-progress-fill.nei { background: var(--fc-neutral-text); }

.fc-summary-copy {
    display: flex;
    gap: 12px;
    color: var(--fc-text);
    font-size: 14px;
    line-height: 20px;
}

.fc-section-heading {
    margin: 20px 0 16px;
    font-family: "Plus Jakarta Sans", Inter, sans-serif;
    font-size: 24px;
    line-height: 32px;
    font-weight: 700;
}

.fc-evidence-card {
    padding: 17px;
    margin-bottom: 16px;
}

.fc-evidence-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
}

.fc-evidence-title {
    display: flex;
    gap: 12px;
    align-items: center;
    color: var(--fc-ink);
    font-size: 14px;
    line-height: 16px;
    font-weight: 800;
}

.fc-source-icon {
    width: 38px;
    height: 34px;
    border-radius: 6px;
    background: #d8e3fb;
    color: var(--fc-ink);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 800;
}

.fc-relevance {
    text-align: right;
    color: var(--fc-text);
    font-size: 12px;
    line-height: 14px;
    font-weight: 700;
    letter-spacing: .36px;
}

.fc-relevance strong {
    display: block;
    color: var(--fc-green-dark);
    font-size: 14px;
    line-height: 16px;
}

.fc-quote {
    margin: 12px 0 0;
    padding-left: 16px;
    border-left: 4px solid var(--fc-muted-surface);
    color: var(--fc-ink-soft);
    font-size: 14px;
    line-height: 22.75px;
    font-style: italic;
}

.fc-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-top: 12px;
}

.fc-tag {
    border-radius: var(--fc-radius-sm);
    background: var(--fc-muted-surface);
    color: var(--fc-text);
    padding: 4px 8px;
    font-size: 10px;
    line-height: 15px;
    font-weight: 800;
    text-transform: uppercase;
}

.fc-tag-soft {
    color: var(--fc-text);
    font-size: 12px;
    line-height: 14px;
    font-weight: 700;
    letter-spacing: .36px;
}

.fc-ai-panel {
    background: var(--fc-dark-panel);
    color: #ffffff;
    border: 1px solid var(--fc-ink);
    border-radius: var(--fc-radius-lg);
    padding: 33px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, .10),
        0 4px 6px -4px rgba(0, 0, 0, .10);
}

.fc-ai-panel h3 {
    margin: 0 0 16px;
    color: #ffffff;
    font-size: 24px;
    line-height: 32px;
}

.fc-ai-panel p {
    color: var(--fc-muted);
    font-size: 16px;
    line-height: 26px;
}

.fc-consensus {
    margin-top: 24px;
    background: var(--fc-bg);
    border: 1px solid var(--fc-border);
    border-radius: var(--fc-radius-lg);
    padding: 16px;
}

.fc-consensus-title {
    font-size: 14px;
    line-height: 16px;
    font-weight: 800;
    margin-bottom: 14px;
}

.fc-consensus-circle {
    width: 160px;
    height: 160px;
    border: 2px dashed var(--fc-border);
    border-radius: 999px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.fc-consensus-circle strong {
    font-family: "Plus Jakarta Sans", Inter, sans-serif;
    font-size: 32px;
    line-height: 40px;
}

.fc-consensus-circle span {
    color: var(--fc-text);
    font-size: 10px;
    line-height: 15px;
    font-weight: 800;
    text-transform: uppercase;
}

.fc-steps {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    margin: 20px 0;
}

.fc-step {
    background: var(--fc-card);
    border: 1px solid var(--fc-border);
    border-radius: var(--fc-radius-md);
    padding: 13px 14px;
    min-height: 82px;
}

.fc-step.done { border-color: rgba(0, 108, 73, .35); background: rgba(108, 248, 187, .16); }
.fc-step.active { border-color: var(--fc-ink); box-shadow: 0 0 0 3px rgba(9, 20, 38, .06); }

.fc-step-title {
    color: var(--fc-ink);
    font-size: 14px;
    font-weight: 800;
    line-height: 16px;
}

.fc-step-body {
    margin-top: 8px;
    color: var(--fc-text);
    font-size: 12px;
    line-height: 18px;
}

.fc-history-table {
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(0, 0, 0, .05);
}

.fc-history-row,
.fc-history-head {
    display: grid;
    grid-template-columns: 2fr .9fr 1fr .8fr .8fr;
    align-items: center;
    gap: 16px;
    padding: 20px 24px;
    border-bottom: 1px solid var(--fc-border);
}

.fc-history-head {
    background: var(--fc-surface);
    color: var(--fc-text);
    font-size: 12px;
    line-height: 14px;
    font-weight: 800;
    letter-spacing: .6px;
    text-transform: uppercase;
}

.fc-history-row {
    color: var(--fc-ink-soft);
    font-size: 15px;
    line-height: 22px;
}

.fc-footer {
    border-top: 1px solid var(--fc-border);
    margin: 48px -48px -48px;
    padding: 17px 48px 16px;
    background: var(--fc-surface);
    display: flex;
    justify-content: space-between;
    gap: 24px;
    color: var(--fc-text);
    font-size: 14px;
}

.fc-footer strong { color: var(--fc-ink-soft); }

.fc-workflow {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 16px;
    margin: 40px 0;
}

.fc-workflow-step {
    text-align: center;
    color: var(--fc-ink);
    font-size: 14px;
    font-weight: 700;
}

.fc-workflow-dot {
    width: 64px;
    height: 64px;
    margin: 0 auto 8px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-weight: 800;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, .10);
}

.fc-bento-grid {
    display: grid;
    grid-template-columns: repeat(12, minmax(0, 1fr));
    gap: 24px;
}

.fc-bento-card {
    background: rgba(255, 255, 255, .7);
    border-color: rgba(226, 232, 240, .8);
    padding: 33px;
}

.fc-bento-card h3 {
    margin: 12px 0 8px;
    font-size: 24px;
    line-height: 32px;
}

.fc-bento-card p {
    color: var(--fc-text);
    font-size: 15px;
    line-height: 24px;
}

.span-8 { grid-column: span 8; }
.span-4 { grid-column: span 4; }
.span-12 { grid-column: span 12; }

div[data-testid="stMetric"] {
    background: var(--fc-card);
    border: 1px solid var(--fc-border);
    border-radius: var(--fc-radius-md);
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
}

@media (max-width: 980px) {
    .block-container { padding: 0 20px 32px; }
    .fc-topbar, .fc-footer { margin-left: -20px; margin-right: -20px; padding-left: 20px; padding-right: 20px; }
    .fc-result-grid, .fc-section-grid, .fc-score-row, .fc-workflow, .fc-steps { grid-template-columns: 1fr; }
    .fc-bento-grid { grid-template-columns: 1fr; }
    .span-4, .span-8, .span-12 { grid-column: span 1; }
    .fc-hero { margin-top: 40px; }
    .fc-hero h1 { font-size: 32px; line-height: 40px; }
    .fc-card-header, .fc-footer { flex-direction: column; }
    .fc-history-row, .fc-history-head { grid-template-columns: 1fr; }
}
</style>
"""


def initialize_state() -> None:
    st.session_state.setdefault("page", "Check a Claim")
    st.session_state.setdefault("claim_text", "")
    st.session_state.setdefault("current_result", None)
    st.session_state.setdefault("history", [])


def set_page(page: str) -> None:
    st.session_state.page = page


def set_claim(claim: str) -> None:
    st.session_state.claim_text = claim


def random_claim() -> None:
    st.session_state.claim_text = random.choice(EXAMPLE_CLAIMS)


def clear_claim() -> None:
    st.session_state.claim_text = ""


def label_class(label: str) -> str:
    if label == "SUPPORTS":
        return "supports"
    if label == "REFUTES":
        return "refutes"
    return "nei"


def label_display(label: str) -> str:
    if label == "NOT ENOUGH INFO":
        return "NEI"
    return label


def relation_label(label: str) -> str:
    if label == "SUPPORTS":
        return "SUPPORTS"
    if label == "REFUTES":
        return "CONTRADICTS"
    return "INSUFFICIENT"


def percent(value: float | None) -> str:
    return f"{(value or 0.0) * 100:.0f}%"


def compact_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def label_progress_class(label: str) -> str:
    return label_class(label)


def truncate_text(text: str, length: int = 130) -> str:
    text = " ".join(str(text).split())
    if len(text) <= length:
        return text
    return f"{text[: length - 1].rstrip()}..."


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
def load_model_bundle() -> tuple[DocumentRetriever, PassageRetriever, Reranker, Verifier]:
    document_retriever = DocumentRetriever()
    passage_retriever = PassageRetriever(document_retriever)
    reranker = Reranker()
    verifier = Verifier()
    return document_retriever, passage_retriever, reranker, verifier


def load_pipeline(
    retrieval_top_k: int,
    rerank_top_k: int,
    max_chunks_per_document: int,
) -> FactCheckerPipeline:
    document_retriever, passage_retriever, reranker, verifier = load_model_bundle()
    settings = PipelineSettings(
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        max_chunks_per_document=max_chunks_per_document,
    )
    return FactCheckerPipeline(
        document_retriever=document_retriever,
        passage_retriever=passage_retriever,
        reranker=reranker,
        verifier=verifier,
        settings=settings,
    )


def render_sidebar() -> tuple[int, int, int]:
    status = load_artifact_status()

    with st.sidebar:
        st.markdown(
            """
            <div class="fc-brand">
                <div class="fc-brand-title">Fact-Checker AI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "+ New Fact-Check",
            key="nav_new",
            type="primary",
            use_container_width=True,
            on_click=set_page,
            args=("Check a Claim",),
        )

        visible_pages = [
            page
            for page in PAGES
            if page != "Fact-Check Result" or st.session_state.current_result
        ]
        for page in visible_pages:
            if page == st.session_state.page:
                st.markdown(
                    f'<div class="fc-nav-active">{escape(page)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.button(
                    page,
                    key=f"nav_{page}",
                    use_container_width=True,
                    on_click=set_page,
                    args=(page,),
                )

        st.divider()
        with st.expander("Runtime settings", expanded=False):
            retrieval_top_k = st.slider(
                "Retrieval top-k",
                5,
                100,
                DEFAULT_RETRIEVAL_TOP_K,
                5,
            )
            rerank_top_k = st.slider("Rerank top-k", 1, 20, DEFAULT_RERANK_TOP_K, 1)
            max_chunks = st.slider(
                "Max chunks / document",
                1,
                10,
                DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
                1,
            )

        with st.expander("Artifacts", expanded=False):
            st.write("FAISS index", "OK" if status["faiss_index"] else "Missing")
            st.write("SQLite database", "OK" if status["sqlite_database"] else "Missing")
            st.write("Processed chunks", "OK" if status["processed_chunks"] else "Missing")
            st.write(
                "Embeddings metadata",
                "OK" if status["embeddings_metadata"] else "Missing",
            )
            if status["chunk_count"] is not None:
                st.metric("Chunks", f"{status['chunk_count']:,}")
                st.metric("Documents", f"{status['document_count']:,}")
            if status["embedding_dimension"] is not None:
                st.metric("Embedding dim.", status["embedding_dimension"])

    return retrieval_top_k, rerank_top_k, max_chunks


def render_topbar(title: str) -> None:
    st.markdown(
        f"""
        <div class="fc-topbar">
            <div class="fc-page-title">{escape(title)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="fc-footer">
            <div><strong>Fact-Checker AI</strong> © 2026. All rights reserved.</div>
            <div>See methodology on the &ldquo;About the AI&rdquo; page.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="fc-hero">
            <span class="fc-badge"><span class="fc-badge-dot"></span>Powered by neural verification</span>
            <h1>Verify any claim with AI</h1>
            <p>Retrieval-augmented fact-checking over an indexed Wikipedia corpus: dense
            search, cross-encoder reranking, and NLI verification produce a
            SUPPORTS / REFUTES / NOT ENOUGH INFO verdict with inspectable evidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_grid() -> None:
    st.markdown(
        """
        <div class="fc-section-grid">
            <div class="fc-feature-card">
                <div class="fc-feature-icon" style="background:#1e293b;color:#fff;">CR</div>
                <h3>Cross-Reference</h3>
                <p>The pipeline compares claims against indexed Wikipedia evidence, then keeps the strongest matching passages.</p>
            </div>
            <div class="fc-feature-card">
                <div class="fc-feature-icon" style="background:#6cf8bb;color:#00714d;">AI</div>
                <h3>Unbiased Analysis</h3>
                <p>Dense retrieval, reranking, and NLI are separated so each stage can be inspected and tuned.</p>
            </div>
            <div class="fc-feature-card">
                <div class="fc-feature-icon" style="background:#ffb2b7;color:#93000a;">SR</div>
                <h3>Source Reliability</h3>
                <p>Every verdict displays the source document, retrieved text, relevance scores, and NLI confidence.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_claim_input() -> bool:
    with st.container(border=True):
        st.text_area(
            "Claim",
            key="claim_text",
            height=120,
            label_visibility="collapsed",
            placeholder="Enter a claim to fact-check...",
        )

        left, spacer, right_a, right_b = st.columns([2.2, .15, 1, 1.25])
        with left:
            action_cols = st.columns(3)
            action_cols[0].button(
                "Import text",
                disabled=True,
                help="Use the text area for now; file import is not wired to the pipeline.",
                use_container_width=True,
            )
            action_cols[1].button(
                "Paste claim",
                disabled=True,
                help="Browsers do not expose clipboard reads to Streamlit without a custom component.",
                use_container_width=True,
            )
            action_cols[2].button("Clear", on_click=clear_claim, use_container_width=True)
        with right_a:
            st.button("Random example", on_click=random_claim, use_container_width=True)
        with right_b:
            submitted = st.button("Check Claim", type="primary", use_container_width=True)

    st.markdown('<div class="fc-chip-row">', unsafe_allow_html=True)
    chip_cols = st.columns([.7, 1.25, 2.8, 2.5])
    chip_cols[0].markdown(
        '<div class="fc-label" style="padding-top:14px;">Try these:</div>',
        unsafe_allow_html=True,
    )
    for index, claim in enumerate(EXAMPLE_CLAIMS, start=1):
        chip_cols[index].button(
            f'"{claim.rstrip(".")}"',
            key=f"example_{index}",
            on_click=set_claim,
            args=(claim,),
            use_container_width=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    return submitted


def render_processing_steps(active_index: int, details: dict[str, str] | None = None) -> None:
    details = details or {}
    steps = [
        ("Load", "Models and indexes"),
        ("Retrieval", details.get("Retrieval", "Dense search over FAISS")),
        ("Reranking", details.get("Reranking", "Cross-encoder scoring")),
        ("Verification", details.get("Verification", "NLI over evidence")),
        ("Result", details.get("Result", "Aggregate verdict")),
    ]
    html = ['<div class="fc-steps">']
    for index, (title, body) in enumerate(steps):
        if index < active_index:
            state = "done"
        elif index == active_index:
            state = "active"
        else:
            state = ""
        html.append(
            f'<div class="fc-step {state}">'
            f'<div class="fc-step-title">{escape(title)}</div>'
            f'<div class="fc-step-body">{escape(body)}</div>'
            f"</div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def run_fact_check_with_steps(
    claim: str,
    retrieval_top_k: int,
    rerank_top_k: int,
    max_chunks: int,
) -> dict:
    claim = claim.strip()
    if not claim:
        raise ValueError("Claim must not be empty.")

    steps_box = st.empty()
    progress_bar = st.progress(0)
    details: dict[str, str] = {}

    with steps_box.container():
        render_processing_steps(0, details)
    pipeline = load_pipeline(retrieval_top_k, rerank_top_k, max_chunks)
    progress_bar.progress(20)

    with steps_box.container():
        render_processing_steps(1, details)
    passages = pipeline.passage_retriever.retrieve(
        query=claim,
        top_k=pipeline.settings.retrieval_top_k,
        max_chunks_per_document=pipeline.settings.max_chunks_per_document,
    )
    details["Retrieval"] = f"{len(passages)} passages retrieved"
    progress_bar.progress(45)

    with steps_box.container():
        render_processing_steps(2, details)
    reranked_passages = pipeline.reranker.rerank(
        query=claim,
        passages=passages,
        top_k=pipeline.settings.rerank_top_k,
    )
    details["Reranking"] = f"{len(reranked_passages)} passages selected"
    progress_bar.progress(68)

    with steps_box.container():
        render_processing_steps(3, details)
    verification = pipeline.verifier.verify(
        claim=claim,
        evidence=reranked_passages,
    )
    details["Verification"] = f"{len(verification['evidence'])} NLI decisions"
    progress_bar.progress(90)

    result = {
        "claim": claim,
        "label": verification["label"],
        "confidence": verification["confidence"],
        "label_scores": verification["label_scores"],
        "best_evidence": verification["best_evidence"],
        "evidence": verification["evidence"],
        "retrieved_passages": passages,
        "reranked_passages": reranked_passages,
        "settings": {
            "retrieval_top_k": pipeline.settings.retrieval_top_k,
            "rerank_top_k": pipeline.settings.rerank_top_k,
            "max_chunks_per_document": pipeline.settings.max_chunks_per_document,
        },
    }
    details["Result"] = f"{label_display(result['label'])} at {percent(result['confidence'])}"

    with steps_box.container():
        render_processing_steps(4, details)
    progress_bar.progress(100)
    return result


def add_history_entry(result: dict) -> None:
    entry = {
        "claim": result["claim"],
        "label": result["label"],
        "confidence": result["confidence"],
        "evidence_count": len(result["evidence"]),
        "retrieved_count": len(result["retrieved_passages"]),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "result": result,
    }
    st.session_state.history.insert(0, entry)


def build_explanation(result: dict) -> list[str]:
    evidence = result["evidence"]
    counts = Counter(item["fever_label"] for item in evidence)
    scores = result["label_scores"]
    best = result["best_evidence"]
    passages = len(result["retrieved_passages"])
    reranked = len(result["reranked_passages"])

    explanation = [
        (
            f"The pipeline retrieved {passages} candidate passages, kept "
            f"{reranked} after reranking, and evaluated {len(evidence)} evidence "
            "blocks with the NLI verifier."
        ),
        (
            "Aggregated label scores are "
            f"SUPPORTS {scores['SUPPORTS']:.3f}, "
            f"REFUTES {scores['REFUTES']:.3f}, and "
            f"NOT ENOUGH INFO {scores['NOT ENOUGH INFO']:.3f}."
        ),
        (
            f"The final verdict is {label_display(result['label'])} because that "
            f"label received the strongest aggregate confidence across the selected "
            f"evidence. Evidence labels: {dict(counts)}."
        ),
    ]
    if best:
        explanation.append(
            "The strongest matching evidence came from "
            f"{best.get('title') or best.get('doc_id')} with NLI confidence "
            f"{best.get('confidence', 0):.1%}."
        )
    return explanation


def render_result_summary(result: dict) -> None:
    label = result["label"]
    css_class = label_class(label)
    confidence = result["confidence"]
    best = result["best_evidence"]
    summary = (
        "No evidence was selected for this verdict."
        if not best
        else (
            f"Best evidence: {best.get('title') or best.get('doc_id')} "
            f"({best.get('fever_label')}, NLI {best.get('confidence', 0):.1%})."
        )
    )

    st.markdown(
        f"""
        <div class="fc-card">
            <div class="fc-card-header">
                <div>
                    <div class="fc-label">Original claim</div>
                    <div class="fc-claim-title">&ldquo;{escape(result["claim"])}&rdquo;</div>
                </div>
                <div class="fc-verdict-pill {css_class}">{escape(label_display(label))}</div>
            </div>
            <div class="fc-score-row">
                <div>
                    <div class="fc-score">
                        <strong>{percent(confidence)}</strong>
                        <span class="fc-label" style="text-transform:none;letter-spacing:.14px;">Confidence Score</span>
                    </div>
                    <div class="fc-progress-track">
                        <div class="fc-progress-fill" style="width:{max(0, min(confidence, 1)) * 100:.1f}%;"></div>
                    </div>
                </div>
                <div class="fc-summary-copy">
                    <span class="fc-source-icon">AI</span>
                    <span>{escape(summary)}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evidence_card(evidence: dict, rank: int) -> None:
    title = evidence.get("title") or evidence.get("doc_id") or "Unknown document"
    text = evidence.get("text", "")
    relevance = evidence.get("rerank_score")
    relation = relation_label(evidence.get("fever_label", "NOT ENOUGH INFO"))
    reliability = "Indexed source"
    if evidence.get("retrieval_score") is not None:
        reliability = f"FAISS {compact_score(evidence.get('retrieval_score'))}"

    st.markdown(
        f"""
        <div class="fc-evidence-card">
            <div class="fc-evidence-top">
                <div class="fc-evidence-title">
                    <span class="fc-source-icon">E{rank}</span>
                    <span>{escape(str(title))}</span>
                </div>
                <div class="fc-relevance">
                    NLI Confidence
                    <strong>{percent(evidence.get("confidence"))}</strong>
                </div>
            </div>
            <div class="fc-quote">&ldquo;{escape(truncate_text(text, 420))}&rdquo;</div>
            <div class="fc-tags">
                <span class="fc-tag">Relation: {escape(relation)}</span>
                <span class="fc-tag-soft">Source Reliability: {escape(reliability)}</span>
                <span class="fc-tag-soft">Rerank: {escape(compact_score(relevance))}</span>
                <span class="fc-tag-soft">Sentences: {escape(", ".join(map(str, evidence.get("sentence_ids", []))) or "n/a")}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ai_explanation(result: dict) -> None:
    paragraphs = "".join(
        f"<p>{escape(paragraph)}</p>" for paragraph in build_explanation(result)
    )
    total_points = len(result["retrieved_passages"])
    st.markdown(
        f"""
        <div class="fc-ai-panel">
            <h3>AI Explanation</h3>
            {paragraphs}
        </div>
        <div class="fc-consensus">
            <div class="fc-consensus-title">Consensus Mapping</div>
            <div class="fc-consensus-circle">
                <strong>{total_points}</strong>
                <span>Retrieved passages</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result: dict) -> None:
    left, right = st.columns([8, 4], gap="large")

    with left:
        render_result_summary(result)
        st.markdown(
            '<div class="fc-section-heading">Evidence used for this verdict</div>',
            unsafe_allow_html=True,
        )
        if result["evidence"]:
            for rank, evidence in enumerate(result["evidence"], start=1):
                render_evidence_card(evidence, rank)
        else:
            st.info("No evidence reached the NLI verification step.")

        with st.expander("Retrieved and reranked passages", expanded=False):
            st.write(
                {
                    "retrieved_passages": len(result["retrieved_passages"]),
                    "reranked_passages": len(result["reranked_passages"]),
                    "settings": result["settings"],
                }
            )
            for rank, passage in enumerate(result["retrieved_passages"], start=1):
                st.markdown(
                    f"**Retrieved #{rank}:** {passage.get('title') or passage.get('doc_id')}"
                )
                st.caption(
                    f"FAISS score: {compact_score(passage.get('score'))} | "
                    f"Chunk: {passage.get('chunk_id', 'n/a')}"
                )
                st.write(passage.get("text", ""))

    with right:
        render_ai_explanation(result)


def page_fact_checking(
    retrieval_top_k: int,
    rerank_top_k: int,
    max_chunks: int,
) -> None:
    render_topbar("Check a Claim")
    if not st.session_state.current_result:
        render_hero()
    submitted = render_claim_input()

    if not submitted:
        if st.session_state.current_result:
            st.markdown(
                '<div class="fc-section-heading" style="max-width:928px;margin:48px auto 16px;">Latest result</div>',
                unsafe_allow_html=True,
            )
            render_result(st.session_state.current_result)
        else:
            render_feature_grid()
        render_footer()
        return

    claim = st.session_state.claim_text.strip()
    if not claim:
        st.warning("Enter a non-empty claim before launching verification.")
        if not st.session_state.current_result:
            render_feature_grid()
        render_footer()
        return

    st.markdown(
        '<div class="fc-section-heading" style="max-width:928px;margin:48px auto 16px;">Processing claim</div>',
        unsafe_allow_html=True,
    )
    try:
        result = run_fact_check_with_steps(
            claim=claim,
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
            max_chunks=max_chunks,
        )
    except Exception as exc:
        st.error("The pipeline could not be loaded or executed.")
        st.exception(exc)
        render_footer()
        return

    st.session_state.current_result = result
    add_history_entry(result)
    st.markdown(
        '<div class="fc-section-heading" style="max-width:928px;margin:48px auto 16px;">Fact-check result</div>',
        unsafe_allow_html=True,
    )
    render_result(result)
    render_footer()


def page_result() -> None:
    render_topbar("Fact-Check Result")
    result = st.session_state.current_result
    if not result:
        st.info("No result yet. Run a claim verification first.")
        render_footer()
        return
    render_result(result)
    render_footer()


def render_history_row(entry: dict) -> str:
    label = entry["label"]
    width = max(0, min(entry["confidence"], 1)) * 100
    return (
        f'<div class="fc-history-row">'
        f'<div>&ldquo;{escape(truncate_text(entry["claim"], 86))}&rdquo;</div>'
        f'<div><span class="fc-verdict-pill {label_class(label)}">{escape(label_display(label))}</span></div>'
        f"<div>"
        f'<div class="fc-progress-track">'
        f'<div class="fc-progress-fill {label_progress_class(label)}" style="width:{width:.1f}%;"></div>'
        f"</div>"
        f"<strong>{percent(entry['confidence'])}</strong>"
        f"</div>"
        f"<div>{entry['evidence_count']} verified / {entry['retrieved_count']} retrieved</div>"
        f"<div>{escape(entry['created_at'])}</div>"
        f"</div>"
    )


def page_history() -> None:
    render_topbar("Fact-check history")

    history = st.session_state.history
    if not history:
        st.info("No fact-check history yet. Only real checks from this session appear here.")
        render_footer()
        return

    filter_col, search_col = st.columns([1, 2], gap="large")
    with filter_col:
        verdict_filter = st.selectbox(
            "Verdict",
            ["All Verdicts", "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
            label_visibility="collapsed",
        )
    with search_col:
        search_query = st.text_input(
            "Search claims, sources, or keywords",
            placeholder="Search claims, sources, or keywords...",
            label_visibility="collapsed",
        ).strip().lower()

    filtered_history = []
    for entry in history:
        label_matches = verdict_filter == "All Verdicts" or entry["label"] == verdict_filter
        text_matches = not search_query or search_query in entry["claim"].lower()
        if label_matches and text_matches:
            filtered_history.append(entry)

    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:32px;">
            <div>
                <span class="fc-verdict-pill nei">{escape(verdict_filter)}</span>
                <span class="fc-verdict-pill nei" style="margin-left:12px;">This Session</span>
            </div>
            <div class="fc-label" style="text-transform:none;letter-spacing:.36px;">
                Showing {len(filtered_history)} of {len(history)} real result{"s" if len(history) != 1 else ""}
            </div>
        </div>
        <div class="fc-history-table">
            <div class="fc-history-head">
                <div>Claim</div><div>Verdict</div><div>Confidence</div><div>Evidence</div><div>Date</div>
            </div>
            {"".join(render_history_row(entry) for entry in filtered_history)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_footer()


def page_about() -> None:
    render_topbar("About the AI")
    status = load_artifact_status()

    st.markdown(
        """
        <div class="fc-hero" style="margin-top:48px;">
            <h1>The Clinical Intelligence Behind Truth</h1>
            <p>Our pipeline combines semantic retrieval, reranking, and NLI verification to provide inspectable evidence for every claim.</p>
        </div>
        <div class="fc-workflow">
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#1e293b;">01</div>Claim</div>
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#6cf8bb;color:#00714d;">02</div>Retrieval</div>
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#e4e2e3;color:#45474c;">03</div>Reranking</div>
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#ffdadb;color:#93000a;">04</div>Evidence</div>
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#091426;">05</div>Verification</div>
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#006c49;">06</div>Verdict</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fc-bento-grid">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="fc-bento-card span-8">
            <div class="fc-source-icon">SE</div>
            <h3>Semantic Search Engine</h3>
            <p>The retrieval stage embeds the claim with <strong>{escape(EMBEDDING_MODEL_NAME)}</strong> and searches the FAISS index for semantically related passages.</p>
            <span class="fc-tag">Dense Retrieval</span>
            <span class="fc-tag">FAISS</span>
        </div>
        <div class="fc-bento-card span-4">
            <div class="fc-source-icon" style="background:#6cf8bb;color:#00714d;">DB</div>
            <h3>Vector Database</h3>
            <p>{status["chunk_count"] or "n/a"} chunks from {status["document_count"] or "n/a"} documents are available in the local evidence store.</p>
        </div>
        <div class="fc-bento-card span-4">
            <div class="fc-source-icon" style="background:#ffdadb;color:#93000a;">ER</div>
            <h3>Evidence Retrieval</h3>
            <p>The passage retriever limits duplicate chunks per source document, then forwards candidates to the reranker.</p>
        </div>
        <div class="fc-bento-card span-8">
            <div class="fc-source-icon" style="background:#e4e2e3;color:#45474c;">RR</div>
            <h3>Cross-Encoder Reranking</h3>
            <p>The reranker uses <strong>{escape(RERANKER_MODEL_NAME)}</strong> to score each claim-passage pair and select the strongest candidates.</p>
        </div>
        <div class="fc-bento-card span-12" style="border-left:4px solid #091426;">
            <h3>Final Verification Logic</h3>
            <p>The NLI stage uses <strong>{escape(NLI_MODEL_NAME)}</strong>. Each evidence block receives a FEVER-style label, then the app aggregates confidence scores into the final verdict.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    render_footer()


def page_evaluation() -> None:
    render_topbar("Evaluation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{PROTOTYPE_METRICS['accuracy']:.0%}")
    col2.metric("Evidence Recall", f"{PROTOTYPE_METRICS['evidence_recall']:.0%}")
    col3.metric("MRR", f"{PROTOTYPE_METRICS['mrr']:.2f}")
    col4.metric("FEVER Score", f"{PROTOTYPE_METRICS['fever_score']:.0%}")
    st.markdown(
        """
        <div class="fc-card" style="margin-top:24px;">
            <div class="fc-section-heading" style="margin-top:0;">Metric Definition</div>
            <p style="color:#45474c;line-height:26px;">
                The FEVER Score counts an example only when the predicted label is correct
                and at least one gold evidence sentence is retrieved. These are prototype
                metrics supplied by the project configuration, not simulated UI values.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_footer()


def run() -> None:
    initialize_state()
    st.markdown(CSS, unsafe_allow_html=True)

    retrieval_top_k, rerank_top_k, max_chunks = render_sidebar()
    page = st.session_state.page

    if page == "Fact-Check Result":
        page_result()
    elif page == "Fact-check history":
        page_history()
    elif page == "About the AI":
        page_about()
    elif page == "Evaluation":
        page_evaluation()
    else:
        page_fact_checking(retrieval_top_k, rerank_top_k, max_chunks)


if __name__ == "__main__":
    run()
