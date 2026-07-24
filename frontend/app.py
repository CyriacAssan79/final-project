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
    FAISS_INDEX_FILE,
    PROCESSED_CHUNKS_FILE,
    PROTOTYPE_METRICS,
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

PAGE_CHECK = "Vérifier une affirmation"
PAGE_RESULT = "Résultat"
PAGE_HISTORY = "Historique"
PAGE_ABOUT = "Comment ça marche"
PAGE_EVALUATION = "Fiabilité du système"

PAGES = [PAGE_CHECK, PAGE_RESULT, PAGE_HISTORY, PAGE_ABOUT, PAGE_EVALUATION]

VERDICT_FILTER_OPTIONS = [
    ("Tous les verdicts", None),
    ("Confirme", "SUPPORTS"),
    ("Contredit", "REFUTES"),
    ("Preuves insuffisantes", "NOT ENOUGH INFO"),
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
    st.session_state.setdefault("page", PAGE_CHECK)
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
    if label == "SUPPORTS":
        return "CONFIRME"
    if label == "REFUTES":
        return "CONTREDIT"
    return "PREUVES INSUFFISANTES"


def percent(value: float | None) -> str:
    return f"{(value or 0.0) * 100:.0f}%"


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

    return status


@st.cache_resource(show_spinner="Chargement des modèles de recherche, de tri et d'analyse...")
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
    system_ready = all(
        [
            status["faiss_index"],
            status["sqlite_database"],
            status["processed_chunks"],
            status["embeddings_metadata"],
        ]
    )

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
            "+ Nouvelle vérification",
            key="nav_new",
            type="primary",
            use_container_width=True,
            on_click=set_page,
            args=(PAGE_CHECK,),
        )

        visible_pages = [
            page
            for page in PAGES
            if page != PAGE_RESULT or st.session_state.current_result
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
        with st.expander("Réglages avancés", expanded=False):
            retrieval_top_k = st.slider(
                "Nombre de sources explorées",
                5,
                100,
                DEFAULT_RETRIEVAL_TOP_K,
                5,
                help=(
                    "Combien de passages le moteur de recherche récupère avant de les "
                    "trier. Plus élevé = recherche plus large mais plus lente."
                ),
            )
            rerank_top_k = st.slider(
                "Nombre de preuves retenues",
                1,
                20,
                DEFAULT_RERANK_TOP_K,
                1,
                help=(
                    "Combien des meilleurs passages sont gardés pour l'analyse finale. "
                    "Plus élevé = plus de preuves examinées mais plus lent."
                ),
            )
            max_chunks = st.slider(
                "Répétitions max. par source",
                1,
                10,
                DEFAULT_MAX_CHUNKS_PER_DOCUMENT,
                1,
                help="Combien de passages différents peuvent venir du même document source.",
            )

        with st.expander("État du système", expanded=False):
            if system_ready:
                st.success("Système prêt.")
            else:
                st.warning(
                    "Certains composants sont manquants. Relancez l'indexation avant "
                    "de vérifier une affirmation."
                )

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
        f"""
        <div class="fc-footer">
            <div><strong>Fact-Checker AI</strong> © 2026. Tous droits réservés.</div>
            <div>Voir la méthodologie sur la page &ldquo;{escape(PAGE_ABOUT)}&rdquo;.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="fc-hero">
            <span class="fc-badge"><span class="fc-badge-dot"></span>Analyse automatique par intelligence artificielle</span>
            <h1>Vérifiez n'importe quelle affirmation</h1>
            <p>Ce système recherche des preuves dans un corpus Wikipedia indexé, compare
            les sources les plus pertinentes, puis vérifie si elles confirment ou
            contredisent l'affirmation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_grid() -> None:
    st.markdown(
        """
        <div class="fc-section-grid">
            <div class="fc-feature-card">
                <div class="fc-feature-icon" style="background:#1e293b;color:#fff;">RC</div>
                <h3>Recherche croisée</h3>
                <p>L'affirmation est comparée aux preuves indexées de Wikipedia, et seuls les passages les plus pertinents sont conservés.</p>
            </div>
            <div class="fc-feature-card">
                <div class="fc-feature-icon" style="background:#6cf8bb;color:#00714d;">IA</div>
                <h3>Analyse impartiale</h3>
                <p>La recherche, la sélection des meilleures preuves et l'analyse finale sont trois étapes séparées, ce qui permet de vérifier chacune indépendamment.</p>
            </div>
            <div class="fc-feature-card">
                <div class="fc-feature-icon" style="background:#ffb2b7;color:#93000a;">PV</div>
                <h3>Preuves vérifiables</h3>
                <p>Chaque verdict indique le document source, le passage utilisé et le niveau de confiance de l'analyse.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_claim_input() -> bool:
    with st.container(border=True):
        st.text_area(
            "Affirmation",
            key="claim_text",
            height=120,
            label_visibility="collapsed",
            placeholder="Entrez une affirmation à vérifier...",
        )
        st.caption(
            "Écrivez de préférence en anglais : le corpus de vérification (Wikipedia) "
            "et les modèles d'analyse ne comprennent que l'anglais."
        )

        left, spacer, right_a, right_b = st.columns([2.2, .15, 1, 1.25])
        with left:
            action_cols = st.columns(3)
            action_cols[0].button(
                "Importer un texte",
                disabled=True,
                help="Utilisez le champ de texte pour l'instant ; l'import de fichier n'est pas encore connecté au pipeline.",
                use_container_width=True,
            )
            action_cols[1].button(
                "Coller l'affirmation",
                disabled=True,
                help="Les navigateurs ne permettent pas à Streamlit de lire le presse-papiers sans composant personnalisé.",
                use_container_width=True,
            )
            action_cols[2].button("Effacer", on_click=clear_claim, use_container_width=True)
        with right_a:
            st.button("Exemple aléatoire", on_click=random_claim, use_container_width=True)
        with right_b:
            submitted = st.button("Vérifier", type="primary", use_container_width=True)

    st.markdown('<div class="fc-chip-row">', unsafe_allow_html=True)
    chip_cols = st.columns([.7, 1.25, 2.8, 2.5])
    chip_cols[0].markdown(
        '<div class="fc-label" style="padding-top:14px;">Exemples :</div>',
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
        ("Chargement", "Modèles et index"),
        ("Recherche", details.get("Retrieval", "Recherche de sources pertinentes")),
        ("Sélection", details.get("Reranking", "Sélection des meilleures preuves")),
        ("Analyse", details.get("Verification", "Comparaison avec l'affirmation")),
        ("Résultat", details.get("Result", "Verdict final")),
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
    details["Retrieval"] = f"{len(passages)} passages trouvés"
    progress_bar.progress(45)

    with steps_box.container():
        render_processing_steps(2, details)
    reranked_passages = pipeline.reranker.rerank(
        query=claim,
        passages=passages,
        top_k=pipeline.settings.rerank_top_k,
    )
    details["Reranking"] = f"{len(reranked_passages)} preuves retenues"
    progress_bar.progress(68)

    with steps_box.container():
        render_processing_steps(3, details)
    verification = pipeline.verifier.verify(
        claim=claim,
        evidence=reranked_passages,
    )
    details["Verification"] = f"{len(verification['evidence'])} preuves analysées"
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
    details["Result"] = f"{label_display(result['label'])} à {percent(result['confidence'])}"

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
    total_score = sum(scores.values()) or 1.0

    confirms = counts.get("SUPPORTS", 0)
    contradicts = counts.get("REFUTES", 0)
    insufficient = counts.get("NOT ENOUGH INFO", 0)

    explanation = [
        (
            f"{passages} passages ont été trouvés, {reranked} ont été retenus comme "
            f"les plus pertinents, puis chacun a été comparé à l'affirmation."
        ),
        (
            f"Résultat par preuve : {confirms} confirment l'affirmation, "
            f"{contradicts} la contredisent, {insufficient} sont jugées insuffisantes."
        ),
        (
            f"Le verdict final est &laquo; {label_display(result['label'])} &raquo; "
            "car c'est la conclusion la plus soutenue par l'ensemble des preuves "
            f"({scores['SUPPORTS'] / total_score:.0%} confirme, "
            f"{scores['REFUTES'] / total_score:.0%} contredit, "
            f"{scores['NOT ENOUGH INFO'] / total_score:.0%} insuffisant)."
        ),
    ]
    if best:
        explanation.append(
            "La preuve la plus déterminante provient de "
            f"{best.get('title') or best.get('doc_id')}, avec une confiance de "
            f"{best.get('confidence', 0):.0%}."
        )
    return explanation


def render_result_summary(result: dict) -> None:
    label = result["label"]
    css_class = label_class(label)
    confidence = result["confidence"]
    best = result["best_evidence"]
    summary = (
        "Aucune preuve n'a permis d'établir ce verdict."
        if not best
        else (
            f"Preuve principale : {best.get('title') or best.get('doc_id')} "
            f"({label_display(best.get('fever_label', 'NOT ENOUGH INFO'))}, "
            f"confiance {best.get('confidence', 0):.0%})."
        )
    )

    st.markdown(
        f"""
        <div class="fc-card">
            <div class="fc-card-header">
                <div>
                    <div class="fc-label">Affirmation</div>
                    <div class="fc-claim-title">&ldquo;{escape(result["claim"])}&rdquo;</div>
                </div>
                <div class="fc-verdict-pill {css_class}">{escape(label_display(label))}</div>
            </div>
            <div class="fc-score-row">
                <div>
                    <div class="fc-score">
                        <strong>{percent(confidence)}</strong>
                        <span class="fc-label" style="text-transform:none;letter-spacing:.14px;">Score de confiance</span>
                    </div>
                    <div class="fc-progress-track">
                        <div class="fc-progress-fill" style="width:{max(0, min(confidence, 1)) * 100:.1f}%;"></div>
                    </div>
                </div>
                <div class="fc-summary-copy">
                    <span class="fc-source-icon">IA</span>
                    <span>{escape(summary)}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Ce système est un prototype et peut se tromper (environ "
        f"{PROTOTYPE_METRICS['accuracy']:.0%} d'exactitude mesurée) — vérifiez "
        f"toujours auprès d'autres sources. Détails sur la page « {PAGE_EVALUATION} » "
        "(menu de gauche)."
    )


def render_evidence_card(evidence: dict, rank: int) -> None:
    title = evidence.get("title") or evidence.get("doc_id") or "Document inconnu"
    text = evidence.get("text", "")
    relation = label_display(evidence.get("fever_label", "NOT ENOUGH INFO"))

    st.markdown(
        f"""
        <div class="fc-evidence-card">
            <div class="fc-evidence-top">
                <div class="fc-evidence-title">
                    <span class="fc-source-icon">P{rank}</span>
                    <span>{escape(str(title))}</span>
                </div>
                <div class="fc-relevance">
                    Confiance
                    <strong>{percent(evidence.get("confidence"))}</strong>
                </div>
            </div>
            <div class="fc-quote">&ldquo;{escape(truncate_text(text, 420))}&rdquo;</div>
            <div class="fc-tags">
                <span class="fc-tag">{escape(relation)}</span>
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
            <h3>Explication</h3>
            {paragraphs}
        </div>
        <div class="fc-consensus">
            <div class="fc-consensus-title">Sources examinées</div>
            <div class="fc-consensus-circle">
                <strong>{total_points}</strong>
                <span>Passages trouvés</span>
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
            '<div class="fc-section-heading">Preuves utilisées pour ce verdict</div>',
            unsafe_allow_html=True,
        )
        if result["evidence"]:
            for rank, evidence in enumerate(result["evidence"], start=1):
                render_evidence_card(evidence, rank)
        else:
            st.info("Aucune preuve n'a atteint l'étape d'analyse.")

    with right:
        render_ai_explanation(result)


def page_fact_checking(
    retrieval_top_k: int,
    rerank_top_k: int,
    max_chunks: int,
) -> None:
    render_topbar(PAGE_CHECK)
    if not st.session_state.current_result:
        render_hero()
    submitted = render_claim_input()

    if not submitted:
        if st.session_state.current_result:
            st.markdown(
                '<div class="fc-section-heading" style="max-width:928px;margin:48px auto 16px;">Dernier résultat</div>',
                unsafe_allow_html=True,
            )
            render_result(st.session_state.current_result)
        else:
            render_feature_grid()
        render_footer()
        return

    claim = st.session_state.claim_text.strip()
    if not claim:
        st.warning("Entrez une affirmation avant de lancer la vérification.")
        if not st.session_state.current_result:
            render_feature_grid()
        render_footer()
        return

    st.markdown(
        '<div class="fc-section-heading" style="max-width:928px;margin:48px auto 16px;">Vérification en cours</div>',
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
        st.error("Une erreur s'est produite pendant la vérification.")
        with st.expander("Détails techniques de l'erreur", expanded=False):
            st.exception(exc)
        render_footer()
        return

    st.session_state.current_result = result
    add_history_entry(result)
    st.markdown(
        '<div class="fc-section-heading" style="max-width:928px;margin:48px auto 16px;">Résultat de la vérification</div>',
        unsafe_allow_html=True,
    )
    render_result(result)
    render_footer()


def page_result() -> None:
    render_topbar(PAGE_RESULT)
    result = st.session_state.current_result
    if not result:
        st.info("Aucun résultat pour l'instant. Vérifiez d'abord une affirmation.")
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
        f"<div>{entry['evidence_count']} analysées / {entry['retrieved_count']} trouvées</div>"
        f"<div>{escape(entry['created_at'])}</div>"
        f"</div>"
    )


def page_history() -> None:
    render_topbar(PAGE_HISTORY)

    history = st.session_state.history
    if not history:
        st.info("Aucun historique pour l'instant. Seules les vérifications réelles de cette session apparaissent ici.")
        render_footer()
        return

    filter_col, search_col = st.columns([1, 2], gap="large")
    with filter_col:
        selected_label = st.selectbox(
            "Filtrer par verdict",
            [label for label, _ in VERDICT_FILTER_OPTIONS],
            label_visibility="collapsed",
        )
    verdict_filter = dict(VERDICT_FILTER_OPTIONS)[selected_label]
    with search_col:
        search_query = st.text_input(
            "Rechercher une affirmation, une source ou un mot-clé",
            placeholder="Rechercher une affirmation, une source ou un mot-clé...",
            label_visibility="collapsed",
        ).strip().lower()

    filtered_history = []
    for entry in history:
        label_matches = verdict_filter is None or entry["label"] == verdict_filter
        text_matches = not search_query or search_query in entry["claim"].lower()
        if label_matches and text_matches:
            filtered_history.append(entry)

    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:32px;">
            <div>
                <span class="fc-verdict-pill nei">{escape(selected_label)}</span>
                <span class="fc-verdict-pill nei" style="margin-left:12px;">Cette session</span>
            </div>
            <div class="fc-label" style="text-transform:none;letter-spacing:.36px;">
                {len(filtered_history)} sur {len(history)} résultat{"s" if len(history) != 1 else ""} affiché{"s" if len(filtered_history) != 1 else ""}
            </div>
        </div>
        <div class="fc-history-table">
            <div class="fc-history-head">
                <div>Affirmation</div><div>Verdict</div><div>Confiance</div><div>Preuves</div><div>Date</div>
            </div>
            {"".join(render_history_row(entry) for entry in filtered_history)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_footer()


def page_about() -> None:
    render_topbar(PAGE_ABOUT)
    status = load_artifact_status()

    st.markdown(
        """
        <div class="fc-hero" style="margin-top:48px;">
            <h1>Comment fonctionne la vérification</h1>
            <p>Le système combine recherche sémantique, tri des meilleures preuves et
            analyse automatique pour fournir, pour chaque affirmation, des preuves
            vérifiables.</p>
        </div>
        <div class="fc-workflow">
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#1e293b;">01</div>Affirmation</div>
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#6cf8bb;color:#00714d;">02</div>Recherche</div>
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#e4e2e3;color:#45474c;">03</div>Tri</div>
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#ffdadb;color:#93000a;">04</div>Preuves</div>
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#091426;">05</div>Analyse</div>
            <div class="fc-workflow-step"><div class="fc-workflow-dot" style="background:#006c49;">06</div>Verdict</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fc-bento-grid">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="fc-bento-card span-8">
            <div class="fc-source-icon">RS</div>
            <h3>Recherche sémantique</h3>
            <p>L'affirmation est transformée en représentation numérique puis comparée à l'ensemble des passages indexés pour trouver les plus proches en sens.</p>
        </div>
        <div class="fc-bento-card span-4">
            <div class="fc-source-icon" style="background:#6cf8bb;color:#00714d;">BC</div>
            <h3>Base de connaissances</h3>
            <p>{status["chunk_count"] or "n/a"} passages issus de {status["document_count"] or "n/a"} documents Wikipedia sont disponibles pour la recherche.</p>
        </div>
        <div class="fc-bento-card span-4">
            <div class="fc-source-icon" style="background:#ffdadb;color:#93000a;">SP</div>
            <h3>Sélection des preuves</h3>
            <p>Le nombre de passages provenant d'un même document est limité, pour éviter qu'une seule source domine les résultats.</p>
        </div>
        <div class="fc-bento-card span-8">
            <div class="fc-source-icon" style="background:#e4e2e3;color:#45474c;">TR</div>
            <h3>Tri des meilleures preuves</h3>
            <p>Chaque passage est comparé individuellement à l'affirmation pour ne garder que les plus pertinents avant l'analyse finale.</p>
        </div>
        <div class="fc-bento-card span-12" style="border-left:4px solid #091426;">
            <h3>Analyse finale</h3>
            <p>Chaque preuve reçoit une étiquette — confirme, contredit, ou insuffisant — puis ces résultats sont combinés pour produire le verdict final et son score de confiance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    render_footer()


def page_evaluation() -> None:
    render_topbar(PAGE_EVALUATION)
    st.warning(
        "Ces métriques sont mesurées sur un sous-ensemble de test, pas sur l'ensemble "
        "du corpus. Ce prototype se trompe encore sur une part notable des "
        "affirmations testées : utilisez-le comme aide à la vérification, jamais "
        "comme unique source de vérité."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Exactitude",
        f"{PROTOTYPE_METRICS['accuracy']:.0%}",
        help="Sur l'ensemble des affirmations testées, la part pour laquelle le verdict final était correct.",
    )
    col2.metric(
        "Preuves retrouvées",
        f"{PROTOTYPE_METRICS['evidence_recall']:.0%}",
        help="À quelle fréquence le système retrouve au moins une preuve correcte parmi ses résultats de recherche.",
    )
    col3.metric(
        "Classement des preuves",
        f"{PROTOTYPE_METRICS['mrr']:.2f}",
        help="À quel point la bonne preuve apparaît tôt dans les résultats de recherche (proche de 1 = très bien classée). Nom technique : MRR.",
    )
    col4.metric(
        "Score global",
        f"{PROTOTYPE_METRICS['fever_score']:.0%}",
        help="Ne compte une affirmation comme réussie que si le verdict ET la preuve trouvée sont corrects — la mesure la plus stricte. Nom technique : FEVER Score.",
    )
    st.markdown(
        """
        <div class="fc-card" style="margin-top:24px;">
            <div class="fc-section-heading" style="margin-top:0;">Comment lire ces chiffres</div>
            <p style="color:var(--fc-text);line-height:26px;">
                Le score global ne compte une affirmation comme réussie que si le verdict
                final est correct et qu'au moins une preuve pertinente a été retrouvée en
                même temps. Ce sont des métriques de prototype, mesurées sur un
                sous-ensemble de test — pas un score final garanti sur toutes les
                affirmations possibles.
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

    if page == PAGE_RESULT:
        page_result()
    elif page == PAGE_HISTORY:
        page_history()
    elif page == PAGE_ABOUT:
        page_about()
    elif page == PAGE_EVALUATION:
        page_evaluation()
    else:
        page_fact_checking(retrieval_top_k, rerank_top_k, max_chunks)


if __name__ == "__main__":
    run()
