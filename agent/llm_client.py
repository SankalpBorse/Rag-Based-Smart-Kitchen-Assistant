"""
llm_client.py – Unified LLM client.

Tier routing
------------
"fast"    → Groq   (low latency, intent classification, substitutions, pantry extraction)
"quality" → Gemini (longer reasoning, recipe personalization, pantry suggestions)

Both tiers fall back within themselves, then cross-fall back to the other tier.
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Union

import groq
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS: Dict[str, List[Dict[str, str]]] = {
    "fast": [
        {"name": "openai/gpt-oss-120b", "provider": "groq"},
        {"name": "llama-3.3-70b-versatile",    "provider": "groq"},
        {"name": "llama-3.1-8b-instant",    "provider": "groq"}
    ],
    "quality": [
        {"name": "gemini-3.1-pro-preview",  "provider": "gemini"},
        {"name": "gemini-2.5-flash",  "provider": "gemini"},
        {"name": "gemini-2.5-flash-lite",  "provider": "gemini"},
    ],
}


class LLMClient:
    def __init__(self):
        self._groq   = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # ------------------------------------------------------------------
    # Provider-level call wrappers
    # ------------------------------------------------------------------

    def _call_groq(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        json_mode: bool,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  2048,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._groq.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def _call_gemini(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        json_mode: bool,
    ) -> str:
        # Separate system messages from conversation
        system_parts: List[str] = []
        contents:     List[types.Content] = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(
                    types.Content(role=role, parts=[types.Part(text=msg["content"])])
                )

        cfg_kwargs: Dict[str, Any] = {"temperature": temperature}
        if system_parts:
            cfg_kwargs["system_instruction"] = "\n".join(system_parts)
        if json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"

        resp = self._gemini.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        return resp.text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        tier: str = "fast",
        temperature: float = 0.7,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a text response.

        Parameters
        ----------
        prompt_or_messages : plain string or OpenAI-style message list
        tier               : "fast" (Groq) or "quality" (Gemini)
        temperature        : sampling temperature
        json_mode          : ask the provider to enforce JSON output
        """
        messages: List[Dict[str, str]] = (
            prompt_or_messages
            if isinstance(prompt_or_messages, list)
            else [{"role": "user", "content": prompt_or_messages}]
        )

        # Try primary tier, then cross-tier fallback
        tiers_to_try = [tier, "quality" if tier == "fast" else "fast"]
        last_error   = None

        for t in tiers_to_try:
            for model_info in MODELS.get(t, []):
                name     = model_info["name"]
                provider = model_info["provider"]
                try:
                    logger.info(f"LLM [{t}] → {name}")
                    if provider == "groq":
                        return self._call_groq(name, messages, temperature, json_mode)
                    else:
                        return self._call_gemini(name, messages, temperature, json_mode)
                except Exception as exc:
                    logger.warning(f"  {name} failed: {exc!r}")
                    last_error = exc
                    time.sleep(0.6)

        raise RuntimeError(f"All models exhausted. Last error: {last_error}")

    def generate_json(
        self,
        prompt: str,
        tier: str = "fast",
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Generate and parse a JSON response.
        Retries once with a stricter instruction if parsing fails.
        Returns {} on total failure (never raises).
        """
        base = (
            prompt
            + "\n\nIMPORTANT: Respond with ONLY valid JSON. "
            "No markdown fences, no explanation, no text outside the JSON object."
        )

        for attempt in range(2):
            extra = (
                "" if attempt == 0
                else "\nPREVIOUS RESPONSE FAILED JSON PARSING. "
                     "Output raw JSON only — start with { and end with }."
            )
            try:
                raw = self.generate(base + extra, tier=tier,
                                    temperature=temperature, json_mode=True)
                return self._parse_json(raw)
            except Exception as exc:
                logger.warning(f"generate_json attempt {attempt + 1} failed: {exc!r}")

        logger.error("generate_json: all attempts failed, returning {}")
        return {}

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Strip markdown fences and extract the first JSON object."""
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*",          "", text)
        text = text.strip()
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
        return json.loads(text)


# Singleton
llm_client = LLMClient()
