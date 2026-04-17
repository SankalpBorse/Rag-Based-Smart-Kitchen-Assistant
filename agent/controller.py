"""
controller.py – Kitchen Assistant state machine.

Modes:   IDLE | INGREDIENT_CONFIRM | COOKING

LLM calls per turn:
  START_COOKING                 2  (Groq intent + Gemini personalise)
  Step advance (ENTER/blank)    0
  Step question / substitution  1  (Groq)
  Pantry add/remove/stock       1  (Groq extraction)
  Everything else               1  (Groq)
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import llm_client
from .prompts import (
    AFTER_RECIPE_PROMPT,
    CONFIRM_INTENT_PROMPT,
    CONFIRM_YES_PROMPT,
    GENERAL_CHAT_PROMPT,
    INTENT_CLASSIFY_PROMPT,
    PANTRY_EXTRACT_PROMPT,
    PANTRY_SUGGEST_STOCK_PROMPT,
    RECIPE_PERSONALIZE_PROMPT,
    SAVE_PREFERENCE_PROMPT,
    STEP_QUESTION_PROMPT,
    SUBSTITUTION_PROMPT,
    SUGGEST_FROM_PANTRY_PROMPT,
)
from rag.rag import retrieve_recipe
from rag.user_database import (
    add_pantry_item,
    delete_pantry_item,
    generate_pantry_summary,
    get_all_pantry_items,
    get_all_preferences,
    get_grocery_suggestions,
    reset_pantry,
    update_preference as save_hard_pref,
)
from rag.user_preference import (
    add_preference as save_soft_pref,
    retrieve_similar_preferences,
)

logger = logging.getLogger(__name__)

# ─── Quantity sanity guard ────────────────────────────────────────────────────
_UNIT_MAX: Dict[str, float] = {
    "kg": 50, "g": 10_000, "l": 20, "ml": 20_000,
    "cup": 100, "tbsp": 200, "tsp": 500, "pcs": 500,
}
_UNIT_MIN: Dict[str, float] = {"g": 0.1, "ml": 1.0}


def _sane_quantity(qty: float, unit: str) -> Tuple[bool, str]:
    unit = unit.lower()
    mx = _UNIT_MAX.get(unit, 10_000)
    mn = _UNIT_MIN.get(unit, 0.01)
    if qty > mx:
        return False, f"quantity {qty} {unit} seems unrealistically large"
    if qty < mn:
        return False, f"quantity {qty} {unit} seems unrealistically small"
    return True, ""


# ─── Normalise ingredient names ───────────────────────────────────────────────
_STRIP_QTY = re.compile(
    r"^\s*[\d./½¼¾⅓⅔]+\s*"            # leading number
    r"(?:kg|g|ml|l|cup|tbsp|tsp|pcs|teaspoon|tablespoon|bunch|sprig|clove|piece|inch|can|packet|handful|pinch)s?\s*",
    re.IGNORECASE,
)
_PARENTHETICAL = re.compile(r"\(.*?\)")
_TRAILING_PREP  = re.compile(
    r"\s*[-–,]\s*(finely\s+chopped|chopped|sliced|diced|mashed|boiled|grated|minced|optional|to taste).*$",
    re.IGNORECASE,
)


def _normalise_ingredient(raw: str) -> str:
    """Strip quantity text, parentheticals and prep notes; return lowercase name."""
    s = _PARENTHETICAL.sub("", raw)
    s = _STRIP_QTY.sub("", s)
    s = _TRAILING_PREP.sub("", s)
    # Drop numeric-only words that slipped through
    s = re.sub(r"\b\d+\b", "", s)
    return s.strip().lower().rstrip("s")  # naive de-plural (onion not onions)


class KitchenController:
    def __init__(self):
        self.mode:    str           = "IDLE"
        self.active_recipe: Optional[Dict[str, Any]] = None
        self.step_idx: int          = 0
        self.hard_prefs: Dict       = {}
        self.soft_prefs: List[str]  = []
        self.history:   List[Dict]  = []
        self._missing_for_confirm: List[str] = []
        self._last_recipe_name: str = ""     # for post-recipe context
        self._load_user_data()

    # ─── Bootstrap ────────────────────────────────────────────────────────────

    def _load_user_data(self):
        self.hard_prefs = get_all_preferences()
        soft = retrieve_similar_preferences("cooking preferences habits diet", top_k=5)
        self.soft_prefs = [p["text"] for p in soft]
        logger.info(f"User data loaded: {len(self.hard_prefs)} hard, {len(self.soft_prefs)} soft prefs")

    # ─── Public API ───────────────────────────────────────────────────────────

    def process(self, user_input: str) -> str:
        """Route user text to the appropriate handler based on mode."""
        self._add_history("user", user_input)

        if self.mode == "INGREDIENT_CONFIRM":
            response = self._handle_confirm(user_input)
        elif self.mode == "COOKING":
            response = self._handle_cooking_query(user_input)
        else:
            response = self._handle_idle(user_input)

        self._add_history("assistant", response)
        return response

    def advance_step(self) -> str:
        """Called when user presses ENTER in COOKING mode. 0 LLM calls."""
        if not self.active_recipe:
            return "No active recipe."

        steps    = self.active_recipe["steps"]
        next_idx = self.step_idx + 1

        if next_idx >= len(steps):
            name = self.active_recipe["name"]
            self._last_recipe_name = name
            self._reset_session()
            msg = llm_client.generate(
                AFTER_RECIPE_PROMPT.format(recipe_name=name),
                tier="fast", temperature=0.8,
            )
            self._add_history("assistant", msg)
            return msg

        self.step_idx = next_idx
        return "SHOW_STEP"

    def current_step_info(self) -> Tuple[str, int, int, str]:
        """Returns (step_text, step_num, total_steps, recipe_name)."""
        if not self.active_recipe:
            return ("No active recipe.", 0, 0, "")
        steps = self.active_recipe["steps"]
        idx   = self.step_idx
        text  = steps[idx] if idx < len(steps) else "All steps complete."
        return (text, idx + 1, len(steps), self.active_recipe["name"])

    def stop_recipe(self) -> str:
        name = self.active_recipe["name"] if self.active_recipe else "recipe"
        self._reset_session()
        return f"Recipe '{name}' stopped.  What would you like to do next?"

    # ─── IDLE handler ─────────────────────────────────────────────────────────

    def _handle_idle(self, user_input: str) -> str:
        intent_data = llm_client.generate_json(
            INTENT_CLASSIFY_PROMPT.format(
                profile=self._profile_text(),
                history=self._history_text(4),
                message=user_input,
            ),
            tier="fast",
        )

        if not intent_data:
            # JSON parse failed — fall back to general chat
            return self._handle_general(user_input)

        intent    = intent_data.get("intent", "GENERAL")
        dish_name = intent_data.get("dish_name") or ""

        dispatch = {
            "START_COOKING":        lambda: self._start_cooking(dish_name or user_input),
            "PANTRY_ADD":           lambda: self._handle_pantry_add(user_input),
            "PANTRY_REMOVE":        lambda: self._handle_pantry_remove(user_input),
            "CLEAR_PANTRY":         lambda: self._handle_clear_pantry(),
            "PANTRY_VIEW":          lambda: self._handle_pantry_view(),
            "PANTRY_SUGGEST_STOCK": lambda: self._handle_pantry_suggest_stock(),
            "SUGGEST_RECIPE":       lambda: self._handle_suggest(),
            "CHECK_PREFERENCES":    lambda: self._handle_check_preferences(),
            "SAVE_PREFERENCE":      lambda: self._handle_save_preference(user_input),
            "NEXT_STEP":            lambda: "You're not in a cooking session right now. Say the dish you'd like to make!",
            "STOP_COOKING":         lambda: "You're not currently cooking anything.",
            "REPEAT_STEP":          lambda: "You're not in a cooking session right now.",
        }
        handler = dispatch.get(intent, lambda: self._handle_general(user_input))
        return handler()

    # ─── INGREDIENT_CONFIRM handler ───────────────────────────────────────────

    def _handle_confirm(self, user_input: str) -> str:
        intent_data = llm_client.generate_json(
            CONFIRM_INTENT_PROMPT.format(
                recipe_name=self.active_recipe["name"],
                ingredients=", ".join(self.active_recipe.get("ingredients", [])),
                missing=", ".join(self._missing_for_confirm) or "none",
                message=user_input,
            ),
            tier="fast",
        )

        if not intent_data:
            return (
                "Sorry, I didn't catch that. Are you ready to start, "
                "or is something missing?"
            )

        intent  = intent_data.get("intent", "UNCLEAR")
        missing = intent_data.get("missing_ingredient")

        if intent == "CONFIRM_YES":
            self.mode     = "COOKING"
            self.step_idx = 0
            confirm_msg = llm_client.generate(
                CONFIRM_YES_PROMPT.format(recipe_name=self.active_recipe["name"]),
                tier="fast", temperature=0.8,
            )
            return confirm_msg

        if intent == "CONFIRM_MISSING" and missing:
            return self._suggest_substitution(
                missing_ingredient=missing,
                context="pre-cook, checking ingredients before starting",
                current_step="",
            )

        return (
            "Are you ready to start? Say 'yes' when you have everything, "
            "or tell me what you're missing."
        )

    # ─── COOKING handler ──────────────────────────────────────────────────────

    def _handle_cooking_query(self, user_input: str) -> str:
        """Handle text typed during step-by-step cooking. 1 LLM call max."""
        lower = user_input.lower()

        # In COOKING mode we honour stop/repeat here for robustness
        if any(w in lower for w in ["stop", "quit recipe", "cancel", "abort"]):
            return self.stop_recipe()

        if any(w in lower for w in ["repeat", "again", "say that again", "what was the step"]):
            text, num, total, name = self.current_step_info()
            return f"Step {num} of {total}: {text}"

        # Substitution triggers
        sub_triggers = ["don't have", "dont have", "missing", "out of", "no more", "ran out", "can i use", "replace"]
        if any(t in lower for t in sub_triggers):
            text, _, _, _ = self.current_step_info()
            return self._suggest_substitution(
                missing_ingredient=user_input,
                context="mid-cook",
                current_step=text,
            )

        # General cooking question
        text, num, total, _ = self.current_step_info()
        return llm_client.generate(
            STEP_QUESTION_PROMPT.format(
                skill=self.hard_prefs.get("skill_level", "intermediate"),
                recipe_name=self.active_recipe["name"],
                current_step=text,
                step_num=num,
                total_steps=total,
                question=user_input,
            ),
            tier="fast", temperature=0.6,
        )

    # ─── Feature handlers ─────────────────────────────────────────────────────

    def _start_cooking(self, dish: str) -> str:
        """RAG → personalise → pantry diff → confirm."""

        # ── 1. RAG retrieval ─────────────────────────────────────────────
        rag_res         = retrieve_recipe(dish, top_k=1)
        raw_recipe_text = ""
        if rag_res.get("status") == "success" and rag_res.get("results"):
            raw_recipe_text = rag_res["results"][0]["content"]
            logger.info(f"RAG hit '{dish}' sim={rag_res['results'][0]['similarity']:.2f}")
        else:
            logger.info(f"RAG miss '{dish}' — LLM will generate from knowledge")

        # ── 2. Personalise (Gemini quality tier) ─────────────────────────
        recipe_data = llm_client.generate_json(
            RECIPE_PERSONALIZE_PROMPT.format(
                dish=dish,
                raw_recipe=raw_recipe_text or "Not found in database — generate from culinary knowledge.",
                diet=self.hard_prefs.get("diet", "no restriction"),
                spice_level=self.hard_prefs.get("spice_level", 3),
                skill=self.hard_prefs.get("skill_level", "intermediate"),
                soft_prefs="; ".join(self.soft_prefs[:3]) or "none",
                pantry=generate_pantry_summary(),
            ),
            tier="quality",
            temperature=0.3,
        )

        if not recipe_data or not recipe_data.get("steps"):
            return (
                f"Sorry, I couldn't build a recipe for '{dish}'. "
                "Could you try a different dish name?"
            )

        steps = recipe_data.get("steps") or []
        if len(steps) < 2:
            return f"The recipe for '{dish}' came back incomplete. Please try again."

        self.active_recipe = {
            "name":        recipe_data.get("name", dish),
            "steps":       steps,
            "ingredients": recipe_data.get("ingredients") or [],
        }
        notes = recipe_data.get("notes", "")

        # ── 3. Pantry diff using normalised names ─────────────────────────
        normalised_ing = [
            _normalise_ingredient(i) for i in self.active_recipe["ingredients"]
        ]
        # Filter out empty strings
        normalised_ing = [n for n in normalised_ing if len(n) > 1]
        self._missing_for_confirm = get_grocery_suggestions(normalised_ing)

        # ── 4. Build user-facing message ──────────────────────────────────
        ing_lines = "\n".join(f"  • {i}" for i in self.active_recipe["ingredients"])
        if notes:
            ing_lines += f"\n\n  📝 {notes}"

        if self._missing_for_confirm:
            missing_lines = "\n".join(f"  • {m}" for m in self._missing_for_confirm)
            response = (
                f"Let's make **{self.active_recipe['name']}**!  Here's what you'll need:\n\n"
                f"{ing_lines}\n\n"
                f"⚠️  Checking your pantry — you may be missing:\n{missing_lines}\n\n"
                "Do you have these, or would you like a substitute for anything?"
            )
        else:
            response = (
                f"Let's make **{self.active_recipe['name']}**!  Here's what you'll need:\n\n"
                f"{ing_lines}\n\n"
                "✅  Your pantry looks good for this recipe.  Ready to start? (say 'yes')"
            )

        self.mode = "INGREDIENT_CONFIRM"
        return response

    def _suggest_substitution(
        self, missing_ingredient: str, context: str, current_step: str
    ) -> str:
        return llm_client.generate(
            SUBSTITUTION_PROMPT.format(
                recipe_name=self.active_recipe["name"] if self.active_recipe else "the dish",
                missing_ingredient=missing_ingredient,
                context=context,
                current_step=current_step,
                skill=self.hard_prefs.get("skill_level", "intermediate"),
                pantry=generate_pantry_summary(),
            ),
            tier="fast", temperature=0.5,
        )

    def _handle_pantry_add(self, user_input: str) -> str:
        data  = llm_client.generate_json(
            PANTRY_EXTRACT_PROMPT.format(action="add", message=user_input),
            tier="fast",
        )
        items = data.get("items") or [] if data else []

        if not items:
            return (
                "I couldn't identify what to add. Try: "
                "'Add 2 onions, 500g chicken, and a can of tomatoes.'"
            )

        added, skipped = [], []
        for item in items:
            name = str(item.get("ingredient") or "").strip().lower()
            qty  = float(item.get("quantity") or 1)
            unit = str(item.get("unit") or "pcs").lower()
            if not name or name == "all":
                continue

            ok, reason = _sane_quantity(qty, unit)
            if not ok:
                skipped.append(f"{name} ({reason})")
                continue

            # Merge with existing
            from rag.user_database import get_pantry_item
            existing = get_pantry_item(name)
            if existing:
                new_qty = existing["quantity"] + qty
                from rag.user_database import update_pantry_quantity
                update_pantry_quantity(name, new_qty)
                added.append(f"{name} → now {new_qty} {unit}")
            else:
                add_pantry_item(name, qty, unit)
                added.append(f"{qty} {unit} {name}")

        parts = []
        if added:
            parts.append(f"Added/updated: {', '.join(added)}.")
        if skipped:
            parts.append(f"Skipped (unusual quantities): {', '.join(skipped)}.")
        if not parts:
            return "Nothing was added to your pantry."
        return " ".join(parts)

    def _handle_pantry_remove(self, user_input: str) -> str:
        data  = llm_client.generate_json(
            PANTRY_EXTRACT_PROMPT.format(action="remove", message=user_input),
            tier="fast",
        )
        items = data.get("items") or [] if data else []

        if not items:
            return (
                "I couldn't figure out what to remove. "
                "Try: 'Remove garlic, butter and oil from my pantry.'"
            )

        # Check for ALL sentinel
        if any(str(i.get("ingredient", "")).strip().upper() == "ALL" for i in items):
            return self._handle_clear_pantry()

        removed, not_found = [], []
        for item in items:
            name = str(item.get("ingredient") or "").strip().lower()
            if not name:
                continue
            from rag.user_database import get_pantry_item
            if get_pantry_item(name):
                delete_pantry_item(name)
                removed.append(name)
            else:
                not_found.append(name)

        parts = []
        if removed:
            parts.append(f"Removed {len(removed)} item(s): {', '.join(removed)}.")
        if not_found:
            parts.append(f"Not found in pantry: {', '.join(not_found)}.")
        if not parts:
            return "Nothing was removed."
        return " ".join(parts)

    def _handle_clear_pantry(self) -> str:
        items = get_all_pantry_items()
        count = len(items)
        if count == 0:
            return "Your pantry is already empty."
        reset_pantry()
        return f"Cleared pantry — {count} item(s) removed."

    def _handle_pantry_view(self) -> str:
        items = get_all_pantry_items()
        if not items:
            return (
                "Your pantry is empty.  Add items by saying something like "
                "'Add 2 onions, 1 kg potatoes, and garlic to my pantry.'"
            )
        lines = [f"  • {i['ingredient']}: {i['quantity']} {i['unit']}" for i in items]
        return f"Your pantry ({len(items)} items):\n\n" + "\n".join(lines)

    def _handle_pantry_suggest_stock(self) -> str:
        return llm_client.generate(
            PANTRY_SUGGEST_STOCK_PROMPT.format(
                pantry=generate_pantry_summary(),
                profile=self._profile_text(),
            ),
            tier="fast", temperature=0.7,
        )

    def _handle_suggest(self) -> str:
        pantry  = generate_pantry_summary()
        is_empty = "empty" in pantry.lower()
        fallback_note = (
            "The pantry is empty. Suggest beginner-friendly popular Indian dishes "
            "that need minimal ingredients (mention the ~5 things they'd need to buy)."
            if is_empty else ""
        )
        return llm_client.generate(
            SUGGEST_FROM_PANTRY_PROMPT.format(
                pantry=pantry,
                profile=self._profile_text(),
                fallback_note=fallback_note,
            ),
            tier="quality", temperature=0.8,
        )

    def _handle_check_preferences(self) -> str:
        hard = self.hard_prefs
        soft = self.soft_prefs
        if not hard and not soft:
            return (
                "You haven't saved any preferences yet.  "
                "You can say things like 'I am vegetarian', 'I prefer less spice', "
                "or 'I am a beginner cook' and I'll remember them."
            )
        lines = []
        if hard:
            for k, v in hard.items():
                lines.append(f"  • {k.replace('_', ' ').title()}: {v}")
        if soft:
            lines.append("\n  Observed habits:")
            for s in soft:
                lines.append(f"  – {s}")
        return "Your saved preferences:\n\n" + "\n".join(lines)

    def _handle_save_preference(self, user_input: str) -> str:
        data = llm_client.generate_json(
            SAVE_PREFERENCE_PROMPT.format(message=user_input),
            tier="fast",
        )
        if not data:
            return "Got it — I'll keep that in mind!"

        hard = data.get("hard_prefs") or []
        soft = data.get("soft_preferences") or []

        saved = []
        for hp in hard:
            k = str(hp.get("key") or "").strip()
            v = hp.get("value", "")
            if k:
                save_hard_pref(k, v)
                self.hard_prefs[k] = v
                saved.append(f"{k.replace('_', ' ')}: {v}")

        for sp in soft:
            text = str(sp).strip()
            if text and text not in self.soft_prefs:
                save_soft_pref(text, pref_type="behavioral")
                self.soft_prefs.append(text)
                saved.append(text[:60])

        if not saved:
            return "Got it — I'll keep that in mind!"
        return f"Saved!  {'; '.join(saved)}."

    def _handle_general(self, user_input: str) -> str:
        return llm_client.generate(
            GENERAL_CHAT_PROMPT.format(
                profile=self._profile_text(),
                pantry_summary=generate_pantry_summary(),
                history=self._history_text(4),
                message=user_input,
            ),
            tier="fast", temperature=0.8,
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _reset_session(self):
        self.mode              = "IDLE"
        self.active_recipe     = None
        self.step_idx          = 0
        self._missing_for_confirm = []

    def _add_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > 14:
            self.history = self.history[-14:]

    def _history_text(self, n: int = 4) -> str:
        recent = self.history[-n * 2:]
        return "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m in recent) or "(none)"

    def _profile_text(self) -> str:
        parts = []
        if self.hard_prefs:
            parts.append(f"Preferences: {self.hard_prefs}")
        if self.soft_prefs:
            parts.append(f"Habits: {'; '.join(self.soft_prefs[:2])}")
        return "\n".join(parts) if parts else "No preferences saved yet."
