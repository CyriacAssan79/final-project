from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
FEVER_DIR = DATA_DIR / "fever"
WIKI_DIR = DATA_DIR / "wikipedia"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
EMBEDDINGS_DIR = INDEX_DIR / "embeddings"

PROCESSED_CHUNKS_FILE = PROCESSED_DIR / "processed_chunks.jsonl"
CHUNKS_DATABASE_FILE = INDEX_DIR / "chunks.db"
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
EMBEDDINGS_METADATA_FILE = EMBEDDINGS_DIR / "metadata.json"

TRAIN_FILE = FEVER_DIR / "train.jsonl"
DEV_FILE = FEVER_DIR / "shared_task_dev.jsonl"
TEST_FILE = FEVER_DIR / "shared_task_test.jsonl"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

DEFAULT_RETRIEVAL_TOP_K = 50
DEFAULT_RERANK_TOP_K = 5
DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 3

PROTOTYPE_METRICS = {
    "accuracy": 0.62,
    "evidence_recall": 0.56,
    "mrr": 0.52,
    "fever_score": 0.38,
}
