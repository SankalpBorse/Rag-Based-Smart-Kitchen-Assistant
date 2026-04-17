"""
main.py – FastAPI backend for the Kitchen Assistant.
Run:  uvicorn main:app --reload
"""

import logging
import re
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    ChatRequest, ChatResponse, PantryItem, PantrySearchResponse,
    PantryUpdate, StepResponse, VoiceChatResponse,
)
from agent.controller import KitchenController
from rag.user_database import (
    add_pantry_item, delete_pantry_item, get_all_pantry_items,
    get_all_preferences, get_pantry_item, reset_pantry, update_pantry_quantity,
)
from backend.voice_agent import VoiceAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── App init ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Kitchen Assistant API", version="2.1")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

controller  = KitchenController()
voice_agent = VoiceAgent(default_stt="groq", default_tts="edge")

# Phrase patterns that trigger a step advance in voice mode
_NEXT_STEP_PATTERNS = [
    r"\bgo\s+to\s+next\s+step\b",
    r"\bnext\s+step\b",
    r"\bstep\s+done\b",
    r"\bmark\s+done\b",
]
_NEXT_STEP_RE = re.compile("|".join(_NEXT_STEP_PATTERNS), re.IGNORECASE)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _step_payload(c: KitchenController):
    if c.mode == "COOKING":
        text, num, total, name = c.current_step_info()
        return StepResponse(text=text, step_number=num, total_steps=total, recipe_name=name)
    return None

def _chat_response(text: str) -> ChatResponse:
    return ChatResponse(response=text, mode=controller.mode, step=_step_payload(controller))


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "mode": controller.mode}


# ─── Chat ─────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return _chat_response(controller.process(req.message))
    except Exception as exc:
        logger.error(f"/chat error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ─── Cooking ──────────────────────────────────────────────────────────────────
@app.post("/next-step", response_model=ChatResponse)
def next_step():
    if controller.mode != "COOKING":
        raise HTTPException(status_code=400, detail="Not currently in cooking mode.")
    result = controller.advance_step()
    if result == "SHOW_STEP":
        text, num, total, name = controller.current_step_info()
        return ChatResponse(
            response="", mode="COOKING",
            step=StepResponse(text=text, step_number=num, total_steps=total, recipe_name=name),
        )
    return ChatResponse(response=result, mode=controller.mode, step=None)

@app.get("/cooking/state")
def cooking_state():
    if controller.mode != "COOKING":
        return {"active": False}
    text, num, total, name = controller.current_step_info()
    return {"active": True, "recipe_name": name, "step_number": num,
            "total_steps": total, "step_text": text}


# ─── Pantry ───────────────────────────────────────────────────────────────────
@app.get("/pantry")
def get_pantry():
    """Returns all pantry items including updated_at timestamp."""
    return get_all_pantry_items()   # updated below to include updated_at

@app.delete("/pantry/clear")
def clear_pantry():
    reset_pantry()
    return {"message": "Pantry cleared."}

@app.get("/pantry/search", response_model=PantrySearchResponse)
def search_pantry(item: str):
    data = get_pantry_item(item.lower().strip())
    if data:
        return PantrySearchResponse(found=True, item=PantryItem(**{
            k: data[k] for k in ('ingredient','quantity','unit')}))
    return PantrySearchResponse(found=False)

@app.post("/pantry/add")
def add_pantry(data: dict):
    try:
        if "items" in data:
            added = []
            for it in data["items"]:
                add_pantry_item(
                    str(it["ingredient"]), float(it.get("quantity", 1)),
                    str(it.get("unit", "pcs")), it.get("expiry_date"),
                )
                added.append(it["ingredient"])
            return {"message": f"Added: {', '.join(added)}", "count": len(added)}

        name   = str(data.get("ingredient", "")).strip().lower()
        if not name:
            raise HTTPException(status_code=400, detail="ingredient is required")
        qty    = float(data.get("quantity", 1))
        unit   = str(data.get("unit", "pcs"))
        expiry = data.get("expiry_date") or None

        existing = get_pantry_item(name)
        if existing:
            new_qty = existing["quantity"] + qty
            update_pantry_quantity(name, new_qty)
            return {"message": f"Updated {name} to {new_qty} {unit}"}

        add_pantry_item(name, qty, unit, expiry)
        return {"message": f"Added {qty} {unit} {name}"}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.put("/pantry/update")
def update_pantry(data: PantryUpdate):
    item = get_pantry_item(data.ingredient.lower().strip())
    if not item:
        raise HTTPException(status_code=404, detail=f"'{data.ingredient}' not found.")
    if data.quantity is not None:
        update_pantry_quantity(data.ingredient, data.quantity)
        return {"message": f"Set {data.ingredient} to {data.quantity}"}
    if data.change is not None:
        new_qty = max(0.0, item["quantity"] + data.change)
        update_pantry_quantity(data.ingredient, new_qty)
        return {"message": f"Updated {data.ingredient} to {new_qty}"}
    raise HTTPException(status_code=400, detail="Provide quantity or change.")

@app.delete("/pantry/delete/{item}")
def delete_pantry(item: str):
    existing = get_pantry_item(item.lower().strip())
    if not existing:
        raise HTTPException(status_code=404, detail=f"'{item}' not found.")
    delete_pantry_item(item)
    return {"message": f"Removed {item}."}


# ─── Preferences ──────────────────────────────────────────────────────────────
@app.get("/preferences")
def get_preferences():
    return get_all_preferences()


# ─── Voice ────────────────────────────────────────────────────────────────────
@app.post("/chat/voice", response_model=VoiceChatResponse)
async def chat_voice(audio: UploadFile = File(...)):
    logger.info(f"Voice request: {audio.filename}")
    try:
        audio_bytes = await audio.read()
        user_text   = await voice_agent.speech_to_text(audio_bytes)
        logger.info(f"Transcribed: {user_text!r}")

        if not user_text.strip():
            user_text = "I didn't catch that, could you repeat?"

        # ── "Go to next step" voice shortcut ─────────────────────────────
        if controller.mode == "COOKING" and _NEXT_STEP_RE.search(user_text):
            result = controller.advance_step()
            if result == "SHOW_STEP":
                text, num, total, name = controller.current_step_info()
                response_text = f"Step {num} of {total}: {text}"
                step_info     = StepResponse(text=text, step_number=num,
                                             total_steps=total, recipe_name=name)
            else:
                response_text = result
                step_info     = None
        else:
            response_text = controller.process(user_text)
            step_info     = _step_payload(controller)

        audio_b64 = await voice_agent.text_to_speech(response_text)
        return VoiceChatResponse(
            user_transcription=user_text,
            response=response_text,
            audio_base64=audio_b64,
            mode=controller.mode,
            step=step_info,
        )

    except Exception as exc:
        logger.error(f"/chat/voice error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
