import os
import re
import json
import sqlite3
import threading
import unicodedata
import hashlib
import logging
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from app.corpus import prepare_question
from app.metadata import classify_query, metadata_for_url, page_metadata, rerank_with_metadata

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "knowledge.jsonl"
SEARCH_DB = ROOT / "data" / "search.sqlite3"
_INDEX_LOCK = threading.Lock()
_INDEXED_FINGERPRINT = None
_VECTOR_DISABLED = False
VECTOR_SEARCH_ENABLED = os.getenv("VECTOR_SEARCH_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
logger = logging.getLogger(__name__)
app = FastAPI(title="DarGlobal + Wasalt Assistant")

class Question(BaseModel):
    message: str

ARABIC_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ؤ": "و", "ئ": "ي", "ـ": ""})
ARABIC_MARKS = re.compile("[\\u064b-\\u065f\\u0670]")
STOPWORDS = {
    "what", "is", "the", "and", "where", "it", "a", "an", "of", "in", "on", "for", "to", "does", "do", "tell", "me", "about", "how", "which", "who", "when", "with", "from", "are", "was", "were", "did", "can", "i", "you", "your", "please", "list", "show", "give", "summarize", "include", "includes", "including", "their", "source", "sources", "link", "links", "url", "urls", "page", "pages", "say", "says", "said", "mention", "mentions", "mentioned", "describe", "describes", "described", "website", "article", "its",
    "ما", "ماذا", "هل", "في", "من", "عن", "على", "و", "هو", "هي", "ماهي", "ماهو", "كم", "أين", "اين", "كيف", "لـ", "الذي", "التي",
}

def normalize_text(value):
    value = unicodedata.normalize("NFKC", value or "").translate(ARABIC_FOLD)
    return ARABIC_MARKS.sub("", value).lower()

def ensure_search_index():
    """Build/update a persistent FTS5 index only when the corpus changes."""
    global _INDEXED_FINGERPRINT
    if not DATA.exists():
        return
    stat = DATA.stat()
    fingerprint = (stat.st_size, stat.st_mtime_ns)
    if fingerprint == _INDEXED_FINGERPRINT:
        return
    with _INDEX_LOCK:
        if fingerprint == _INDEXED_FINGERPRINT:
            return
        import json
        SEARCH_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(SEARCH_DB, timeout=60) as db:
            db.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, url TEXT NOT NULL, title TEXT NOT NULL, source TEXT NOT NULL, scraped_at TEXT, text TEXT NOT NULL)")
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(title, source, body, tokenize='unicode61 remove_diacritics 2')")
            db.execute("CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            cached = db.execute("SELECT value FROM index_meta WHERE key='corpus_fingerprint'").fetchone()
            if cached and cached[0] == f"{fingerprint[0]}:{fingerprint[1]}":
                _INDEXED_FINGERPRINT = fingerprint
                return
            db.execute("DELETE FROM search_fts")
            db.execute("DELETE FROM documents")
            for line in DATA.open("r", encoding="utf-8"):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = str(item.get("title", ""))
                source = str(item.get("source", ""))
                body = str(item.get("text", ""))
                cur = db.execute("INSERT INTO documents(url,title,source,scraped_at,text) VALUES(?,?,?,?,?)", (str(item.get("url", "")), title, source, str(item.get("scraped_at", "")), body))
                db.execute("INSERT INTO search_fts(rowid,title,source,body) VALUES(?,?,?,?)", (cur.lastrowid, normalize_text(title), normalize_text(source), normalize_text(body)))
            db.execute("INSERT OR REPLACE INTO index_meta(key,value) VALUES('corpus_fingerprint',?)", (f"{fingerprint[0]}:{fingerprint[1]}",))
        _INDEXED_FINGERPRINT = fingerprint

def public_sources(docs):
    """Return the retrieved pages as clickable source links."""
    return [{"url": doc["url"], "title": doc.get("title", "Source")} for doc in docs]

def lexical_candidates(query, limit=60):
    ensure_search_index()
    search_query = prepare_question(query)
    terms = [t for t in re.findall(r"[\w]+", normalize_text(search_query)) if len(t) > 1 and t not in STOPWORDS]
    if not terms:
        terms = re.findall(r"[\w]+", normalize_text(search_query))
    if not terms:
        return []
    terms = list(dict.fromkeys(terms))
    ensure_search_index()
    with sqlite3.connect(SEARCH_DB, timeout=30) as db:
        quoted = {term: '"' + term.replace('"', '""') + '"' for term in terms}
        frequencies = {}
        for term, expression in quoted.items():
            frequencies[term] = db.execute("SELECT count(*) FROM search_fts WHERE search_fts MATCH ?", (expression,)).fetchone()[0]
        present = [term for term in terms if frequencies[term] > 0]
        if not present:
            return []
        # Anchor the search on its rarest query term (often a project or city
        # name), then use the other terms to rank candidates with BM25.
        match = " OR ".join(quoted[term] for term in present)
        for anchor in sorted(present, key=lambda term: frequencies[term]):
            others = [quoted[term] for term in present if term != anchor]
            candidate = quoted[anchor] if not others else f"{quoted[anchor]} AND (" + " OR ".join(others) + ")"
            if db.execute("SELECT 1 FROM search_fts WHERE search_fts MATCH ? LIMIT 1", (candidate,)).fetchone():
                match = candidate
                break
        rows = db.execute("""
            SELECT d.id,d.url,d.title,d.source,d.scraped_at,d.text,
                   bm25(search_fts, 8.0, 3.0, 1.0) AS score
            FROM search_fts JOIN documents d ON d.id=search_fts.rowid
            WHERE search_fts MATCH ?
            ORDER BY score ASC LIMIT ?
        """, (match, max(limit * 5, limit))).fetchall()
    # Repeated navigation and canonical/encoded URL variants can fill the
    # lexical candidate window with identical chunks. Remove those copies
    # before RRF so distinct relevant chunks still reach dense fusion/reranking.
    candidates, seen_content = [], set()
    for row in rows:
        content_key = hashlib.sha256(normalize_text(row[5]).encode("utf-8")).hexdigest()
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        candidates.append({"_id": str(row[0]), "chunk_index": row[0] - 1,
                           "url": row[1], "title": row[2], "source": row[3],
                           "scraped_at": row[4], "text": row[5],
                           "page_metadata": metadata_for_url(row[1])})
        if len(candidates) >= limit:
            break
    return candidates

def retrieve(query, limit=5):
    """Hybrid BM25+dense search, RRF fusion, and cross-encoder reranking."""
    global _VECTOR_DISABLED
    search_query = prepare_question(query)
    intent = classify_query(query)
    lexical = lexical_candidates(search_query, limit=80)
    dense = []
    if VECTOR_SEARCH_ENABLED and not _VECTOR_DISABLED:
        try:
            from app.vector_store import dense_candidates
            dense = dense_candidates(search_query, limit=80)
            metadata_filter = None
            if intent["intent"] == "location_projects":
                if intent.get("city"):
                    metadata_filter = {"project_city": intent["city"]}
                elif intent.get("country"):
                    metadata_filter = {"project_country": intent["country"]}
            elif intent["intent"] == "named_project" and intent.get("project_name"):
                metadata_filter = {"project_name": intent["project_name"]}
            elif intent["intent"] == "property_listing":
                metadata_filter = {"page_type": "property_listing"}
            if metadata_filter:
                filtered_dense = dense_candidates(search_query, limit=60, where=metadata_filter)
                if intent["intent"] == "named_project" and filtered_dense:
                    filtered_lexical = [item for item in lexical
                                        if item.get("page_metadata", {}).get("project_name") == intent["project_name"]]
                    if filtered_lexical:
                        lexical = filtered_lexical
                    dense = filtered_dense
                else:
                    dense.extend(filtered_dense)
        except Exception as e:
            _VECTOR_DISABLED = True
            logger.exception("Dense retrieval unavailable; continuing with BM25: %s", type(e).__name__)

    # Reciprocal Rank Fusion combines lexical and semantic candidate ranks.
    fused, rows_by_id = {}, {}
    for ranked_list in (lexical, dense):
        for rank, item in enumerate(ranked_list, 1):
            key = item["_id"]
            rows_by_id[key] = item
            fused[key] = fused.get(key, 0.0) + 1.0 / (60 + rank)
    ordered = [rows_by_id[key] for key in sorted(fused, key=fused.get, reverse=True)]

    # Do not feed copied site navigation chunks to the reranker or LLM twice.
    unique, seen_content = [], set()
    for item in ordered:
        content_key = hashlib.sha256(normalize_text(item["text"]).encode("utf-8")).hexdigest()
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        item["page_metadata"] = metadata_for_url(item.get("url", "")) or item.get("page_metadata", {})
        unique.append(item)

    reranked = unique
    if unique and VECTOR_SEARCH_ENABLED:
        try:
            from app.vector_store import rerank_candidates
            reranked = rerank_candidates(search_query, unique[:40], limit=min(len(unique[:40]), max(limit * 8, 40)))
        except Exception as e:
            logger.exception("Cross-encoder unavailable; using RRF order: %s", type(e).__name__)
    reranked = rerank_with_metadata(search_query, reranked)
    # A location question needs evidence of the project's location, not a page
    # that merely mentions that city (such as a launch event held there).
    if intent["intent"] == "location_projects":
        if intent.get("city"):
            location_matches = [item for item in reranked
                                if (item.get("page_metadata") or {}).get("project_city", "").casefold()
                                == intent["city"].casefold()]
        else:
            location_matches = [item for item in reranked
                                if (item.get("page_metadata") or {}).get("project_country", "").casefold()
                                == intent.get("country", "").casefold()]
        if location_matches:
            reranked = location_matches

    # Merge only neighboring chunks for a page; unrelated distant chunks can
    # contain recommendations or other listings from the same scraped URL.
    candidate_by_position = {
        (item.get("url"), item.get("chunk_index")): item
        for item in reranked if item.get("chunk_index") is not None
    }
    results, by_url = [], {}
    for item in reranked:
        url, title, source = item["url"], item.get("title", ""), item.get("source", "")
        scraped_at, text = item.get("scraped_at", ""), item.get("text", "")
        if url in by_url:
            doc = results[by_url[url]]
            if len(doc["text"]) < 4000 and text not in doc["text"]:
                doc["text"] += " " + text[:max(0, 4000 - len(doc["text"]))]
            continue
        if len(results) >= limit:
            break
        by_url[url] = len(results)
        chunk_index = item.get("chunk_index")
        if chunk_index is not None:
            neighbors = [candidate_by_position.get((url, chunk_index + delta)) for delta in (-1, 1)]
            for neighbor in neighbors:
                if neighbor and neighbor.get("text") and neighbor["text"] not in text:
                    text = (neighbor["text"] + " " + text) if neighbor["chunk_index"] < chunk_index else (text + " " + neighbor["text"])
        results.append({"url": url, "title": title, "source": source,
                        "scraped_at": scraped_at, "text": text[:2600],
                        "page_metadata": item.get("page_metadata", {})})
    return results

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "app" / "index.html").read_text(encoding="utf-8")

@app.get("/health")
def health():
    ensure_search_index()
    with sqlite3.connect(SEARCH_DB, timeout=30) as db:
        count = db.execute("SELECT count(*) FROM documents").fetchone()[0]
    try:
        if not VECTOR_SEARCH_ENABLED:
            vector = {"ready": False, "documents": 0, "enabled": False}
        else:
            from app.vector_store import index_status
            vector = index_status()
    except Exception:
        vector = {"ready": False, "documents": 0}
    return {"status": "ok", "documents": count, "vector_index": vector,
            "llm_configured": bool(os.getenv("OPENROUTER_API_KEY")),
            "metadata_pages": len(page_metadata()),
            "project_city_pages": sum(bool(v.get("project_city")) for v in page_metadata().values())}

@app.post("/chat")
def chat(q: Question):
    request_started = time.perf_counter()
    retrieval_started = time.perf_counter()
    docs = [{**doc, "citation": i} for i, doc in enumerate(retrieve(q.message), 1)]
    retrieval_seconds = time.perf_counter() - retrieval_started
    logger.info("Chat retrieval completed in %.2fs; sources=%d", retrieval_seconds, len(docs))
    if not docs:
        return {"answer": "I couldn't find a relevant collected DarGlobal or Wasalt page for that question. Try asking about a named project, property, location, or service.", "sources": []}
    context = "\n\n".join(
        f"[{d['citation']}] {d.get('title', 'Source')}\n"
        f"Page metadata: {json.dumps(d.get('page_metadata', {}), ensure_ascii=False)}\n"
        f"Excerpt: {d['text']}\nSource URL: {d['url']}\n"
        f"Collected at: {d.get('scraped_at', 'unknown')}"
        for d in docs
    )
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        return JSONResponse(status_code=503, content={
            "answer": "Answer generation is unavailable because the OpenRouter API key is not configured.",
            "sources": public_sources(docs), "mode": "error",
        })

    model = os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free").strip()
    query_intent = classify_query(q.message)
    location_rule = ""
    if query_intent["intent"] == "location_projects" and (query_intent["city"] or query_intent["country"]):
        target_location = query_intent["city"] or query_intent["country"]
        location_field = "project_city" if query_intent["city"] else "project_country"
        location_rule = (
            f"The question asks for projects located in {target_location}. "
            f"Use a source as evidence of project location only when its page metadata has {location_field}="
            f"{target_location}. A city recorded as event_city is only an event venue and does not count. "
        )
    request_body = {
        "model": model,
        "temperature": 0,
        # Pinning a model with configurable reasoning lets us disable hidden
        # reasoning so it cannot consume the visible-answer token budget.
        "reasoning": {"effort": "none"},
        "messages": [{
            "role": "system",
            "content": (
                "Answer the user's question concisely using the supplied DarGlobal and Wasalt sources. "
                "Use 1–3 sentences and cite factual statements with the source numbers, such as [1]. "
                "Every factual sentence must end with citations like [1] or [2]. Cite only a numbered source "
                "that supports the answer. If the sources do not contain the answer, say what is missing. "
                "Do not invent details. Prefer project and portfolio pages over general editorial pages when the "
                "question asks about company projects. Treat event_city and project_city as different facts. "
                + location_rule + "\n\nSOURCES:\n" + context
            ),
        }, {"role": "user", "content": q.message}],
        "max_tokens": 800,
    }
    try:
        provider_started = time.perf_counter()
        timeout_seconds = max(5, int(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "35")))
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("APP_PUBLIC_URL", "http://localhost:8000"),
                "X-Title": os.getenv("APP_NAME", "DarGlobal + Wasalt Assistant"),
            },
            json=request_body,
            timeout=timeout_seconds,
        )
        provider_seconds = time.perf_counter() - provider_started
        logger.info("OpenRouter responded in %.2fs with HTTP %s", provider_seconds, resp.status_code)
        resp.raise_for_status()
        payload = resp.json()
        choices = payload.get("choices") or []
        message = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
        answer = message.get("content") if isinstance(message, dict) else None
        if isinstance(answer, list):
            answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
        if not isinstance(answer, str) or not answer.strip():
            choice = choices[0] if choices and isinstance(choices[0], dict) else {}
            usage = payload.get("usage") or {}
            token_details = usage.get("completion_tokens_details") or {}
            routed_model = payload.get("model", model)
            finish_reason = choice.get("finish_reason", "unknown")
            logger.warning(
                "OpenRouter returned empty visible content: model=%s finish_reason=%s "
                "completion_tokens=%s reasoning_tokens=%s message_fields=%s",
                routed_model, finish_reason, usage.get("completion_tokens", "unknown"),
                token_details.get("reasoning_tokens", "unknown"),
                sorted(message.keys()) if isinstance(message, dict) else [],
            )
            empty_message = (
                "The selected model used its output budget before producing a visible answer. "
                "Please retry."
                if finish_reason == "length" else
                "The selected OpenRouter model returned an empty answer. Please retry."
            )
            return JSONResponse(status_code=502, content={
                "answer": empty_message,
                "sources": public_sources(docs), "mode": "error",
            })
        answer = answer.strip()
        used_model = payload.get("model", model)
        logger.info("OpenRouter generated an answer with model %s", used_model)
    except requests.HTTPError as e:
        provider_seconds = time.perf_counter() - provider_started
        status = e.response.status_code if e.response is not None else "unknown"
        provider_code, provider_message = "", ""
        if e.response is not None:
            try:
                error_payload = e.response.json().get("error", {})
                if isinstance(error_payload, dict):
                    provider_code = str(error_payload.get("code", ""))[:80]
                    provider_message = str(error_payload.get("message", ""))[:240]
            except (ValueError, AttributeError):
                pass
        logger.warning(
            "OpenRouter HTTP error after %.2fs: status=%s code=%s message=%s",
            provider_seconds, status, provider_code, provider_message,
        )
        if status == 429:
            message = "The configured free model is temporarily rate-limited upstream. Please wait a little and retry."
        else:
            message = f"OpenRouter could not generate an answer (HTTP {status}). Please retry."
        return JSONResponse(status_code=502, content={
            "answer": message, "sources": public_sources(docs), "mode": "error",
        })
    except requests.RequestException as e:
        provider_seconds = time.perf_counter() - provider_started
        logger.warning("OpenRouter request failed after %.2fs (%s)", provider_seconds, type(e).__name__)
        return JSONResponse(status_code=502, content={
            "answer": "OpenRouter could not be reached. Check network access and retry.",
            "sources": public_sources(docs), "mode": "error",
        })
    except Exception as e:
        logger.exception("Answer generation failed: %s", type(e).__name__)
        return JSONResponse(status_code=500, content={
            "answer": "Answer generation failed. Please retry.",
            "sources": public_sources(docs), "mode": "error",
        })
    logger.info("Chat request completed in %.2fs", time.perf_counter() - request_started)
    return {"answer": answer, "sources": public_sources(docs), "mode": "llm", "model": used_model}

