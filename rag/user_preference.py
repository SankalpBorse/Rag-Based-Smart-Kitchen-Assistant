"""
user_preference.py – ChromaDB storage for soft (behavioural) user preferences.

Each document is a natural-language statement describing the user's observed
habits, tendencies, or context-specific preferences.  These are written by the
LLM after every session and retrieved semantically during cooking to personalise
responses.

Examples
--------
"User prefers very spicy food and always asks to increase chilli levels."
"User tends to skip deep-frying steps and opts for air-frying instead."
"User is lactose-intolerant and replaces dairy with coconut milk."
"""

import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent
CHROMA_PREF_PATH = str(_HERE / "chroma_user_prefs")
COLLECTION_NAME  = "user_preferences"
EMBED_MODEL      = "all-MiniLM-L6-v2"

_client = chromadb.PersistentClient(path=CHROMA_PREF_PATH)
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

try:
    collection = _client.get_collection(COLLECTION_NAME, embedding_function=_embed_fn)
    logger.info(f"Loaded user-preference collection ({collection.count()} docs)")
except Exception:
    collection = _client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Created new user-preference collection")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_id(text: str) -> str:
    ts   = str(int(time.time()))
    h    = hashlib.md5(f"{ts}_{text[:60]}".encode()).hexdigest()[:8]
    return f"pref_{ts}_{h}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def add_preference(
    text: str,
    pref_type: str = "behavioral",
    temporal: Optional[str] = None,
    mood_trigger: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Insert a new behavioural preference statement.

    Parameters
    ----------
    text        : natural-language description of the preference
    pref_type   : category tag (e.g. "flavor", "technique", "behavioral")
    temporal    : when relevant (e.g. "weekday", "summer")
    mood_trigger: emotional state if applicable
    timestamp   : ISO string; defaults to current UTC time

    Returns
    -------
    Unique ID of the inserted document.
    """
    ts = timestamp or _now_iso()
    pref_id = _make_id(text)
    meta: Dict[str, Any] = {"pref_type": pref_type, "timestamp": ts}
    if temporal:
        meta["temporal"] = temporal
    if mood_trigger:
        meta["mood_trigger"] = mood_trigger
    try:
        collection.add(documents=[text], metadatas=[meta], ids=[pref_id])
        logger.info(f"Added preference {pref_id}: {text[:80]!r}")
        return pref_id
    except Exception as exc:
        logger.error(f"add_preference failed: {exc}")
        raise


def update_preference(pref_id: str, **kwargs) -> bool:
    """Update an existing preference.  Allowed kwargs: text, pref_type, temporal, mood_trigger."""
    try:
        existing = collection.get(ids=[pref_id], include=["documents", "metadatas"])
        if not existing["ids"]:
            logger.warning(f"Preference {pref_id!r} not found.")
            return False
        doc  = kwargs.get("text", existing["documents"][0])
        meta = existing["metadatas"][0].copy()
        for field in ("pref_type", "temporal", "mood_trigger", "timestamp"):
            if field in kwargs:
                meta[field] = kwargs[field]
        collection.delete(ids=[pref_id])
        collection.add(documents=[doc], metadatas=[meta], ids=[pref_id])
        return True
    except Exception as exc:
        logger.error(f"update_preference failed: {exc}")
        return False


def delete_preference(pref_id: str) -> bool:
    try:
        collection.delete(ids=[pref_id])
        return True
    except Exception as exc:
        logger.error(f"delete_preference failed: {exc}")
        return False


def get_preference(pref_id: str) -> Optional[Dict[str, Any]]:
    try:
        r = collection.get(ids=[pref_id], include=["documents", "metadatas"])
        if r["ids"]:
            return {"id": r["ids"][0], "text": r["documents"][0], "metadata": r["metadatas"][0]}
        return None
    except Exception as exc:
        logger.error(f"get_preference failed: {exc}")
        return None


def get_all_preferences() -> List[Dict[str, Any]]:
    try:
        r = collection.get(include=["documents", "metadatas"])
        return [
            {"id": pid, "text": doc, "metadata": meta}
            for pid, doc, meta in zip(r["ids"], r["documents"], r["metadatas"])
        ]
    except Exception as exc:
        logger.error(f"get_all_preferences failed: {exc}")
        return []


def reset_all_preferences() -> bool:
    global collection
    try:
        _client.delete_collection(COLLECTION_NAME)
        collection = _client.create_collection(
            name=COLLECTION_NAME, embedding_function=_embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        return True
    except Exception as exc:
        logger.error(f"reset_all_preferences failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------------------------

def retrieve_similar_preferences(
    query: str,
    top_k: int = 5,
    filter_metadata: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Semantically retrieve preferences most relevant to *query*.

    Returns list of dicts with keys: id, text, metadata, relevance_score (0-1).
    """
    if collection.count() == 0:
        return []
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where=filter_metadata or None,
            include=["documents", "metadatas", "distances"],
        )
        prefs = []
        for pid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            prefs.append({
                "id": pid,
                "text": doc,
                "metadata": meta,
                "relevance_score": round(1.0 - dist, 4),
            })
        return prefs
    except Exception as exc:
        logger.error(f"retrieve_similar_preferences failed: {exc}")
        return []


def delete_preferences_older_than(days: int) -> int:
    """Prune preferences whose timestamp is older than *days* days."""
    cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
    deleted = 0
    for pref in get_all_preferences():
        ts_str = pref["metadata"].get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str).timestamp()
                if ts < cutoff:
                    delete_preference(pref["id"])
                    deleted += 1
            except Exception:
                pass
    logger.info(f"Pruned {deleted} preferences older than {days} days.")
    return deleted


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    add_preference("User loves very spicy North Indian food.", pref_type="flavor")
    add_preference("User is vegetarian and avoids eggs.", pref_type="dietary")
    for p in get_all_preferences():
        print(p)
    print("\nRelevant to 'spicy curry':")
    for r in retrieve_similar_preferences("spicy curry", top_k=2):
        print(r)
