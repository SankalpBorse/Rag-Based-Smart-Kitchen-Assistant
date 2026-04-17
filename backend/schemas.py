from pydantic import BaseModel, Field
from typing import List, Optional


# ─── Shared ──────────────────────────────────────────────────────────────────

class StepResponse(BaseModel):
    text: str
    step_number: int
    total_steps: int
    recipe_name: str


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str
    mode: str
    step: Optional[StepResponse] = None


# ─── Voice ────────────────────────────────────────────────────────────────────

class VoiceChatResponse(BaseModel):
    user_transcription: str
    response: str
    audio_base64: str
    mode: str
    step: Optional[StepResponse] = None


# ─── Pantry ───────────────────────────────────────────────────────────────────

class PantryItem(BaseModel):
    ingredient: str
    quantity: float = 1.0
    unit: str = "pcs"


class PantryBulkAdd(BaseModel):
    items: List[PantryItem]


class PantryUpdate(BaseModel):
    ingredient: str
    quantity: Optional[float] = None   # absolute set
    change:   Optional[float] = None   # delta +/-


class PantrySearchResponse(BaseModel):
    found: bool
    item: Optional[PantryItem] = None
