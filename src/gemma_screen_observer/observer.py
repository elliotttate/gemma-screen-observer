"""Gemma 4 vision observer - analyzes screen frames and detects changes."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

import httpx

from .capture import Frame
from .config import ModelConfig

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """\
Analyze this game screenshot. Output ONLY a JSON object:
{"scene":"<menu|gameplay|combat|loading|cutscene|settings|other>","description":"<1 sentence>","ui":{"health":null,"score":null,"position":null,"lap":null,"time":null,"menu_items":[]},"text_on_screen":["<readable text>"]}"""

CHANGE_DETECTION_PROMPT = """\
Compare these two game screenshots. Output ONLY a JSON object:
{"has_changes":true,"changes":[{"category":"<ui|scene|text|player|environment>","element":"<what>","from":"<old>","to":"<new>","significance":"<low|medium|high|critical>"}],"summary":"<1 sentence>"}
If nothing meaningful changed, output: {"has_changes":false,"changes":[],"summary":"no change"}"""

QUERY_PROMPT_TEMPLATE = """\
You are a game screen analyzer. Look at this screenshot and answer the following question precisely.

Question: {question}

Respond with a clear, factual answer based only on what you can see in the screenshot."""


class VisionBackend(ABC):
    """Abstract base for vision model backends."""

    @abstractmethod
    async def analyze_frame(self, frame: Frame, prompt: str) -> str:
        """Send a frame with a prompt to the vision model and return the response."""

    @abstractmethod
    async def analyze_two_frames(self, prev: Frame, current: Frame, prompt: str) -> str:
        """Send two frames with a prompt for comparison."""

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""


class OllamaBackend(VisionBackend):
    """Ollama local inference backend."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.endpoint,
            timeout=httpx.Timeout(config.timeout, connect=10.0),
        )

    async def analyze_frame(self, frame: Frame, prompt: str) -> str:
        response = await self.client.post(
            "/api/chat",
            json={
                "model": self.config.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [frame.base64_jpeg],
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    async def analyze_two_frames(self, prev: Frame, current: Frame, prompt: str) -> str:
        response = await self.client.post(
            "/api/chat",
            json={
                "model": self.config.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [prev.base64_jpeg, current.base64_jpeg],
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]

    async def close(self) -> None:
        await self.client.aclose()


class GoogleAIBackend(VisionBackend):
    """Google AI (Gemini API) backend for Gemma 4 cloud inference."""

    def __init__(self, config: ModelConfig):
        self.config = config
        api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google AI API key required. Set model.api_key in config or GOOGLE_API_KEY env var."
            )
        self.client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            timeout=httpx.Timeout(config.timeout, connect=10.0),
        )
        self.api_key = api_key

    async def _generate(self, parts: list[dict]) -> str:
        response = await self.client.post(
            f"/models/{self.config.model_name}:generateContent",
            params={"key": self.api_key},
            json={
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": self.config.temperature,
                    "maxOutputTokens": self.config.max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    async def analyze_frame(self, frame: Frame, prompt: str) -> str:
        parts = [
            {"text": prompt},
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": frame.base64_jpeg,
                }
            },
        ]
        return await self._generate(parts)

    async def analyze_two_frames(self, prev: Frame, current: Frame, prompt: str) -> str:
        parts = [
            {"text": prompt},
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": prev.base64_jpeg,
                }
            },
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": current.base64_jpeg,
                }
            },
        ]
        return await self._generate(parts)

    async def close(self) -> None:
        await self.client.aclose()


class TransformersBackend(VisionBackend):
    """Direct transformers/Unsloth backend for maximum GPU utilization."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.model_name
        self.max_tokens = config.max_tokens

    async def analyze_frame(self, frame: Frame, prompt: str) -> str:
        from .unsloth_backend import analyze_frame
        return await analyze_frame(
            frame, prompt,
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            load_in_4bit=True,
        )

    async def analyze_two_frames(self, prev: Frame, current: Frame, prompt: str) -> str:
        from .unsloth_backend import analyze_two_frames
        return await analyze_two_frames(
            prev, current, prompt,
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            load_in_4bit=True,
        )

    async def close(self) -> None:
        from .unsloth_backend import unload_model
        unload_model()


def create_backend(config: ModelConfig) -> VisionBackend:
    """Factory to create the appropriate vision backend."""
    if config.backend == "ollama":
        return OllamaBackend(config)
    elif config.backend == "google_ai":
        return GoogleAIBackend(config)
    elif config.backend == "transformers":
        return TransformersBackend(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")


def _extract_json(text: str) -> dict:
    """Extract JSON from a model response that may contain markdown fences or extra text."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    # Try to repair truncated JSON — E2B often gets cut off before closing braces
    if start is not None and depth > 0:
        truncated = text[start:]
        # Track nesting to know exactly what needs closing
        in_string = False
        escaped = False
        stack = []  # track open brackets/braces
        for ch in truncated:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in ("}", "]") and stack:
                stack.pop()
        if in_string:
            truncated += '"'
        # Close all open brackets/braces in reverse order
        truncated += "".join(reversed(stack))
        try:
            result = json.loads(truncated)
            logger.debug("Recovered truncated JSON response")
            return result
        except json.JSONDecodeError:
            pass

    # Last resort: extract key fields with regex
    import re
    scene = "unknown"
    description = text[:200]
    text_on_screen = []
    m = re.search(r'"scene"\s*:\s*"([^"]+)"', text)
    if m:
        scene = m.group(1)
    m = re.search(r'"description"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text)
    if m:
        description = m.group(1).replace('\\"', '"').replace("\\n", " ")
    for m in re.finditer(r'"text_on_screen"\s*:\s*\[([^\]]*)', text):
        for item in re.findall(r'"([^"]+)"', m.group(1)):
            text_on_screen.append(item)

    logger.debug("Extracted fields via regex fallback: scene=%s", scene)
    return {"scene": scene, "description": description, "elements": {}, "text_on_screen": text_on_screen}


class ScreenObserver:
    """Analyzes screen frames using Gemma 4 vision."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.backend = create_backend(config)

    async def analyze(self, frame: Frame) -> dict:
        """Analyze a single frame and return structured game state."""
        raw = await self.backend.analyze_frame(frame, ANALYSIS_SYSTEM_PROMPT)
        return _extract_json(raw)

    async def detect_changes(self, previous: Frame, current: Frame) -> dict:
        """Compare two frames and return detected changes."""
        raw = await self.backend.analyze_two_frames(
            previous, current, CHANGE_DETECTION_PROMPT
        )
        return _extract_json(raw)

    async def query(self, frame: Frame, question: str) -> str:
        """Ask a free-form question about the current screen."""
        prompt = QUERY_PROMPT_TEMPLATE.format(question=question)
        return await self.backend.analyze_frame(frame, prompt)

    async def close(self) -> None:
        await self.backend.close()
