"""Persistent multilingual vector index and reranker used by hybrid retrieval."""
import json
import logging
import os
import threading
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer
from app.metadata import METADATA_PATH, infer_page_metadata

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "knowledge.jsonl"
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(ROOT / "data" / "chroma")))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
COLLECTION_NAME = "darwasalt_chunks"
VECTOR_INDEX_MAX_CHUNKS = int(os.getenv("VECTOR_INDEX_MAX_CHUNKS", "0") or "0")
_LOCK = threading.RLock()
_EMBEDDER = None
_RERANKER = None
_CLIENT = None
_COLLECTION = None
_INDEX_FINGERPRINT = None
logger = logging.getLogger(__name__)


def _corpus_fingerprint():
    stat = CORPUS.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _metadata_fingerprint():
    if not METADATA_PATH.exists():
        return "missing"
    stat = METADATA_PATH.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def _client():
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            import chromadb
            CHROMA_PATH.mkdir(parents=True, exist_ok=True)
            _CLIENT = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _CLIENT


def _embedder():
    global _EMBEDDER
    with _LOCK:
        if _EMBEDDER is None:
            logger.info("Loading embedding model %s", EMBEDDING_MODEL)
            _EMBEDDER = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _EMBEDDER


def _reranker():
    global _RERANKER
    with _LOCK:
        if _RERANKER is None:
            logger.info("Loading cross-encoder reranker %s", RERANKER_MODEL)
            _RERANKER = CrossEncoder(RERANKER_MODEL, device="cpu")
    return _RERANKER


def ensure_vector_index():
    """Build/update Chroma only when the prepared JSONL corpus changes."""
    global _COLLECTION, _INDEX_FINGERPRINT
    if not CORPUS.exists():
        return None
    fingerprint = _corpus_fingerprint()
    with _LOCK:
        if fingerprint == _INDEX_FINGERPRINT and _COLLECTION is not None:
            return _COLLECTION
        client = _client()
        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            collection = None

        records = []
        with CORPUS.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("text") and item.get("url"):
                    records.append(item)

        expected_count = len(records)
        target_count = min(expected_count, VECTOR_INDEX_MAX_CHUNKS) if VECTOR_INDEX_MAX_CHUNKS > 0 else expected_count
        metadata_fingerprint = _metadata_fingerprint()
        metadata_by_url = {}
        if METADATA_PATH.exists():
            try:
                metadata_by_url = json.loads(METADATA_PATH.read_text(encoding="utf-8")).get("pages", {})
            except (OSError, ValueError, TypeError):
                logger.exception("Could not load page metadata; using inferred defaults")
        collection_is_current = (
            collection is not None
            and (collection.metadata or {}).get("corpus_fingerprint") == fingerprint
            and (collection.metadata or {}).get("embedding_model") == EMBEDDING_MODEL
            and collection.count() <= expected_count
        )
        if not collection_is_current:
            if collection is not None:
                client.delete_collection(COLLECTION_NAME)
            collection = client.create_collection(
                COLLECTION_NAME,
                metadata={"hnsw:space": "cosine", "corpus_fingerprint": fingerprint,
                          "embedding_model": EMBEDDING_MODEL},
            )
        model = _embedder()
        batch_size = 256
        indexed_count = collection.count()
        for start in range(indexed_count, target_count, batch_size):
                batch = records[start:start + batch_size]
                passages = [f"passage: {item.get('title', '')}. {item['text']}" for item in batch]
                vectors = model.encode(
                    passages, batch_size=128, convert_to_numpy=True,
                    normalize_embeddings=True, show_progress_bar=False,
                )
                # SQLite's first inserted document receives rowid 1; use the same
                # stable IDs so dense and BM25 candidates fuse on the same chunk.
                ids = [str(i + 1) for i in range(start, start + len(batch))]
                metadatas = [{
                    "url": str(item["url"]),
                    "title": str(item.get("title", "")),
                    "source": str(item.get("source", "")),
                    "scraped_at": str(item.get("scraped_at", "")),
                    "chunk_index": start + offset,
                    **metadata_by_url.get(item["url"], infer_page_metadata(item)),
                    "page_metadata_fingerprint": metadata_fingerprint,
                } for offset, item in enumerate(batch)]
                collection.add(
                    ids=ids, embeddings=vectors.tolist(), documents=[item["text"] for item in batch],
                    metadatas=metadatas,
                )
                logger.info("Embedded %s/%s corpus chunks (total corpus: %s)", min(start + len(batch), target_count), target_count, expected_count)
        # Updating only metadata leaves the existing embeddings untouched.
        stored_metadata_fingerprint = None
        if collection.count():
            sample = collection.get(limit=1, include=["metadatas"])
            sample_metadatas = sample.get("metadatas") or []
            if sample_metadatas:
                stored_metadata_fingerprint = sample_metadatas[0].get("page_metadata_fingerprint")
        if collection.count() and stored_metadata_fingerprint != metadata_fingerprint:
            for start in range(0, min(collection.count(), target_count), batch_size):
                batch = records[start:start + batch_size]
                ids = [str(i + 1) for i in range(start, start + len(batch))]
                metadatas = [{
                    "url": str(item["url"]), "title": str(item.get("title", "")),
                    "source": str(item.get("source", "")),
                    "scraped_at": str(item.get("scraped_at", "")),
                    "chunk_index": start + offset,
                    **metadata_by_url.get(item["url"], infer_page_metadata(item)),
                    "page_metadata_fingerprint": metadata_fingerprint,
                } for offset, item in enumerate(batch)]
                collection.update(ids=ids, metadatas=metadatas)
            logger.info("Updated page metadata for %s existing vector records without re-embedding", min(collection.count(), target_count))
        logger.info("Chroma index contains %s/%s chunks", collection.count(), expected_count)

        _COLLECTION = collection
        _INDEX_FINGERPRINT = fingerprint
        return _COLLECTION


def dense_candidates(query, limit=60, where=None):
    collection = ensure_vector_index()
    if collection is None or collection.count() == 0:
        return []
    vector = _embedder().encode(
        [f"query: {query}"], convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )[0]
    result = collection.query(
        query_embeddings=[vector.tolist()], n_results=min(limit, collection.count()),
        where=where, include=["documents", "metadatas", "distances"],
    )
    candidates = []
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    for chunk_id, text, metadata in zip(ids, documents, metadatas):
        candidates.append({**metadata, "text": text, "_id": chunk_id})
    return candidates


def rerank_candidates(query, candidates, limit=5):
    if not candidates:
        return []
    model = _reranker()
    pairs = [(query, f"{item.get('title', '')}. {item.get('text', '')[:1600]}") for item in candidates]
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda row: float(row[1]), reverse=True)
    return [item for item, _score in ranked[:limit]]


def index_status():
    try:
        if not CHROMA_PATH.exists():
            return {"ready": False, "documents": 0}
        collection = _client().get_collection(COLLECTION_NAME)
        with CORPUS.open("r", encoding="utf-8") as source:
            total = sum(1 for line in source if line.strip())
        indexed = collection.count()
        current = collection.metadata.get("corpus_fingerprint") == _corpus_fingerprint()
        return {"ready": current and indexed >= total, "documents": indexed,
                "total": total, "limited": current and indexed < total}
    except Exception:
        return {"ready": False, "documents": 0, "total": 0, "limited": False}


def build_index():
    collection = ensure_vector_index()
    return collection.count() if collection is not None else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Indexed {build_index()} corpus chunks in {CHROMA_PATH}")
