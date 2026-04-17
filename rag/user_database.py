"""
user_database.py – SQLite storage for hard preferences and pantry inventory.
updated_at is now returned in all pantry read functions.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_HERE   = Path(__file__).parent
DB_PATH = _HERE / "user_prefs.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pantry (
            ingredient  TEXT PRIMARY KEY,
            quantity    REAL,
            unit        TEXT,
            expiry_date TEXT,
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


# ─── Preferences ──────────────────────────────────────────────────────────────

def insert_preference(key: str, value: Any) -> bool:
    try:
        conn = _get_connection()
        conn.execute("INSERT INTO preferences (key, value) VALUES (?, ?)",
                     (key, json.dumps(value, ensure_ascii=False)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        logger.error(f"Key {key!r} exists — use update_preference().")
        return False
    except Exception as exc:
        logger.error(f"insert_preference: {exc}")
        return False
    finally:
        conn.close()

def update_preference(key: str, value: Any) -> bool:
    try:
        conn = _get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (key, json.dumps(value, ensure_ascii=False)))
        conn.commit()
        return True
    except Exception as exc:
        logger.error(f"update_preference: {exc}")
        return False
    finally:
        conn.close()

def get_preference(key: str) -> Optional[Any]:
    try:
        conn = _get_connection()
        row = conn.execute("SELECT value FROM preferences WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
    except Exception as exc:
        logger.error(f"get_preference: {exc}")
        return None
    finally:
        conn.close()

def get_all_preferences() -> Dict[str, Any]:
    try:
        conn = _get_connection()
        rows = conn.execute("SELECT key, value FROM preferences").fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}
    except Exception as exc:
        logger.error(f"get_all_preferences: {exc}")
        return {}
    finally:
        conn.close()

def delete_preference(key: str) -> bool:
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM preferences WHERE key=?", (key,))
        conn.commit()
        return True
    except Exception as exc:
        logger.error(f"delete_preference: {exc}")
        return False
    finally:
        conn.close()

def reset_all_preferences() -> bool:
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM preferences")
        conn.commit()
        return True
    except Exception as exc:
        logger.error(f"reset_all_preferences: {exc}")
        return False
    finally:
        conn.close()


# ─── Pantry ───────────────────────────────────────────────────────────────────

def add_pantry_item(ingredient: str, quantity: float, unit: str,
                    expiry_date: Optional[str] = None) -> bool:
    try:
        conn = _get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO pantry (ingredient, quantity, unit, expiry_date, updated_at) "
            "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (ingredient.lower().strip(), quantity, unit, expiry_date),
        )
        conn.commit()
        return True
    except Exception as exc:
        logger.error(f"add_pantry_item: {exc}")
        return False
    finally:
        conn.close()

def update_pantry_quantity(ingredient: str, new_quantity: float) -> bool:
    try:
        conn = _get_connection()
        conn.execute(
            "UPDATE pantry SET quantity=?, updated_at=CURRENT_TIMESTAMP WHERE ingredient=?",
            (new_quantity, ingredient.lower().strip()),
        )
        conn.commit()
        return conn.total_changes > 0
    except Exception as exc:
        logger.error(f"update_pantry_quantity: {exc}")
        return False
    finally:
        conn.close()

def get_pantry_item(ingredient: str) -> Optional[Dict]:
    """Returns item dict including updated_at."""
    try:
        conn = _get_connection()
        row = conn.execute(
            "SELECT ingredient, quantity, unit, expiry_date, updated_at FROM pantry WHERE ingredient=?",
            (ingredient.lower().strip(),),
        ).fetchone()
        if row:
            return {"ingredient": row[0], "quantity": row[1], "unit": row[2],
                    "expiry_date": row[3], "updated_at": row[4]}
        return None
    except Exception as exc:
        logger.error(f"get_pantry_item: {exc}")
        return None
    finally:
        conn.close()

def get_all_pantry_items() -> List[Dict]:
    """Returns all pantry items including updated_at."""
    try:
        conn = _get_connection()
        rows = conn.execute(
            "SELECT ingredient, quantity, unit, expiry_date, updated_at FROM pantry ORDER BY ingredient"
        ).fetchall()
        return [{"ingredient": r[0], "quantity": r[1], "unit": r[2],
                 "expiry_date": r[3], "updated_at": r[4]} for r in rows]
    except Exception as exc:
        logger.error(f"get_all_pantry_items: {exc}")
        return []
    finally:
        conn.close()

def delete_pantry_item(ingredient: str) -> bool:
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM pantry WHERE ingredient=?", (ingredient.lower().strip(),))
        conn.commit()
        return True
    except Exception as exc:
        logger.error(f"delete_pantry_item: {exc}")
        return False
    finally:
        conn.close()

def reset_pantry() -> bool:
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM pantry")
        conn.commit()
        return True
    except Exception as exc:
        logger.error(f"reset_pantry: {exc}")
        return False
    finally:
        conn.close()

def generate_pantry_summary() -> str:
    items = get_all_pantry_items()
    if not items:
        return "Pantry is empty."
    return "\n".join(f"  - {i['ingredient']}: {i['quantity']} {i['unit']}" for i in items)

def get_grocery_suggestions(required_ingredients: List[str]) -> List[str]:
    pantry_names = {i["ingredient"].lower() for i in get_all_pantry_items()}
    missing = []
    for ing in required_ingredients:
        ing_lower = ing.lower().strip()
        if len(ing_lower) < 2:
            continue
        if not any(p in ing_lower or ing_lower in p for p in pantry_names):
            missing.append(ing)
    return missing
