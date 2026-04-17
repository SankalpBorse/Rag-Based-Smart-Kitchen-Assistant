"""
rag.py – Recipe retrieval from ChromaDB.
All knowledge, technique, and substitution queries are routed to the LLM directly.
"""

import logging
from typing import Dict, Optional

import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EMBED_MODEL  = "all-MiniLM-L6-v2"
RECIPE_DB_PATH = "./rag/chroma_db_recipes"

# Shared embedding function (loaded once)
_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

# ChromaDB client – creates the directory if it does not yet exist
_recipe_client = chromadb.PersistentClient(path=RECIPE_DB_PATH)
recipe_collection = _recipe_client.get_or_create_collection(
    name="indian_recipes",
    embedding_function=_embedding_fn,
    metadata={"hnsw:space": "cosine"},
)
logger.info(f"Recipe collection ready: {recipe_collection.count()} documents")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_recipe(dish_name: str, filters: Optional[Dict] = None, top_k: int = 1) -> Dict:
    """
    Retrieve the closest matching recipe for *dish_name*.

    Returns
    -------
    {
        "status": "success" | "empty" | "error",
        "results": [
            {
                "content":    str,   # raw recipe text
                "metadata":   dict,
                "similarity": float  # 0-1, higher is better
            },
            ...
        ]
    }
    """
    count = recipe_collection.count()
    if count == 0:
        logger.warning("Recipe collection is empty – no retrieval possible.")
        return {"status": "empty", "results": []}

    try:
        results = recipe_collection.query(
            query_texts=[dish_name],
            n_results=min(top_k, count),
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )
        docs   = results.get("documents",  [[]])[0]
        metas  = results.get("metadatas",  [[]])[0]
        dists  = results.get("distances",  [[]])[0]

        output = [
            {
                "content":    doc,
                "metadata":   meta,
                "similarity": round(1.0 - dist, 4),
            }
            for doc, meta, dist in zip(docs, metas, dists)
        ]
        logger.info(f"Retrieved {len(output)} recipe(s) for '{dish_name}'")
        return {"status": "success", "results": output}

    except Exception as exc:
        logger.error(f"Recipe retrieval failed: {exc}")
        return {"status": "error", "results": [], "error": str(exc)}


def add_recipe(doc_id: str, content: str, metadata: Optional[Dict] = None) -> bool:
    """Add a single recipe document to the collection."""
    try:
        recipe_collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata or {}],
        )
        return True
    except Exception as exc:
        logger.error(f"Failed to add recipe '{doc_id}': {exc}")
        return False


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    result = retrieve_recipe("vada pav", top_k=1)
    print(json.dumps(result, indent=2, ensure_ascii=False))
