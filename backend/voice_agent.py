"""
voice_agent.py – Unified STT and TTS with ElevenLabs integration.
"""

import asyncio
import base64
import io
import logging
import os

import requests
from groq import Groq

logger = logging.getLogger(__name__)

# ── Optional heavy dependencies ───────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
    HAS_LOCAL_WHISPER = True
except ImportError:
    HAS_LOCAL_WHISPER = False

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

class VoiceAgent:
    def __init__(
        self,
        default_stt: str = "local",
        default_tts: str = "edge",
    ):
        self.default_stt = default_stt
        self.default_tts = default_tts

        self._groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._local_whisper = None

        if default_stt == "local":
            if HAS_LOCAL_WHISPER:
                logger.info("Loading local Whisper model…")
                self._local_whisper = WhisperModel("small", device="cpu", compute_type="int8")
            else:
                logger.warning("faster-whisper not installed — falling back to Groq STT.")
                self.default_stt = "groq"

    # ── STT ───────────────────────────────────────────────────────────────────

    async def speech_to_text(self, audio_bytes: bytes, provider: str | None = None) -> str:
        provider = provider or self.default_stt
        try:
            if provider == "groq":
                return await asyncio.to_thread(self._stt_groq, audio_bytes)
            elif provider == "local" and self._local_whisper:
                return await asyncio.to_thread(self._stt_local, audio_bytes)
            else:
                return await asyncio.to_thread(self._stt_groq, audio_bytes)
        except Exception as exc:
            logger.error(f"STT ({provider}) failed: {exc} — retrying with Groq.")
            return await asyncio.to_thread(self._stt_groq, audio_bytes)

    def _stt_groq(self, audio_bytes: bytes) -> str:
        result = self._groq.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model="whisper-large-v3",
        )
        return result.text.strip()

    def _stt_local(self, audio_bytes: bytes) -> str:
        segments, _ = self._local_whisper.transcribe(io.BytesIO(audio_bytes), beam_size=5)
        return "".join(s.text for s in segments).strip()

    # ── TTS ───────────────────────────────────────────────────────────────────

    async def text_to_speech(self, text: str, provider: str | None = None) -> str:
        """Returns Base64-encoded audio string."""
        provider = provider or self.default_tts
        try:
            if provider == "elevenlabs":
                return await self._tts_elevenlabs(text)
            elif provider == "bulbul":
                return await self._tts_bulbul(text)
            elif provider == "edge" and HAS_EDGE_TTS:
                return await self._tts_edge(text)
            else:
                logger.warning(f"TTS provider {provider!r} unavailable — trying Edge TTS.")
                return await self._tts_edge(text)
        except Exception as exc:
            logger.error(f"TTS ({provider}) failed: {exc} — trying Edge TTS fallback.")
            try:
                return await self._tts_edge(text)
            except Exception as exc2:
                logger.error(f"Edge TTS fallback also failed: {exc2}")
                return ""

    async def _tts_elevenlabs(self, text: str) -> str:
        """ElevenLabs TTS implementation using REST API."""
        api_key = os.getenv("ELEVENLABS_API_KEY")
        # You can find Voice IDs in the ElevenLabs Voice Library. 
        # This one is "Aria" (a versatile, popular voice).
        voice_id = "oO7sLA3dWfQXsKeSAjpA" 
        
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY missing from environment variables")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        resp = await asyncio.to_thread(
            requests.post, url, json=data, headers=headers
        )
        
        if resp.status_code != 200:
            logger.error(f"ElevenLabs Error: {resp.text}")
            resp.raise_for_status()

        # Convert raw binary audio to Base64 to keep it consistent with your other methods
        return base64.b64encode(resp.content).decode("utf-8")

    async def _tts_bulbul(self, text: str) -> str:
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise ValueError("SARVAM_API_KEY missing")
        payload = {
            "text": [text],
            "target_language_code": "en-IN",
            "speaker": "arya",
            "model": "bulbul:v2",
            "pace": 1,
            "speech_sample_rate": 22050,
            "pitch": 0,
            "loudness": 1,
            "output_audio_codec": "mp3",
            "enable_preprocessing": True
        }
        headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
        resp = await asyncio.to_thread(
            requests.post, "https://api.sarvam.ai/text-to-speech",
            json=payload, headers=headers
        )
        resp.raise_for_status()
        return resp.json()["audios"][0]

    async def _tts_edge(self, text: str) -> str:
        communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural")
        chunks = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks += chunk["data"]
        return base64.b64encode(chunks).decode()