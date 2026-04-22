"""Google Gemini API model runner using the google-genai SDK."""

import json
import logging
import os
import time
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBaseModel
from PIL import Image

from sgmriqa.models.base import BaseModelRunner, InferenceResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema for Gemini localization (normalized 0-1 coords)
# ---------------------------------------------------------------------------

class BBoxNormalized(PydanticBaseModel):
    """A single normalized bounding box returned by Gemini."""
    frame: int
    label: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float


class LocalizationResponse(PydanticBaseModel):
    """Structured output for localization tasks."""
    reasoning: str
    answer: str
    bboxes: List[BBoxNormalized]


# Localization prompt marker (set in prompt_builder.py for Gemini)
_LOCALIZATION_MARKER = "normalized coordinates"


class GeminiRunner(BaseModelRunner):
    """Runner for Google Gemini vision models using google.genai Client.

    Supports multiple API keys for quota rotation. Keys are loaded from
    environment variables: GEMINI_API_KEY, GOOGLE_API_KEY,
    GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, etc.
    On 429 (quota exceeded), automatically rotates to the next key.
    """

    def __init__(self, model_id: str = "gemini-2.0-flash", **kwargs):
        super().__init__(model_id, **kwargs)
        self._clients = []
        self._api_keys: List[str] = []
        self._current_key_idx: int = 0

    def load_model(self):
        from google import genai

        load_dotenv()

        # Collect all available API keys
        self._api_keys = []

        # Primary key (check both GEMINI_API_KEY and GOOGLE_API_KEY)
        for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            key = os.getenv(var)
            if key and key not in self._api_keys:
                self._api_keys.append(key)

        # Additional keys: GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, ...
        for i in range(2, 20):
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if key and key not in self._api_keys:
                self._api_keys.append(key)

        if not self._api_keys:
            raise ValueError(
                "No Google API keys found. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                "(and optionally GOOGLE_API_KEY_2, ...) in your .env file."
            )

        logger.info(f"Loaded {len(self._api_keys)} Google API key(s)")
        self._current_key_idx = 0

        # Create a genai.Client per key
        self._clients = [genai.Client(api_key=k) for k in self._api_keys]
        self._loaded = True

    @property
    def _client(self):
        return self._clients[self._current_key_idx]

    def _rotate_key(self):
        """Switch to the next API key."""
        self._current_key_idx = (self._current_key_idx + 1) % len(self._api_keys)
        logger.info(
            f"Rotated to API key {self._current_key_idx + 1}/{len(self._api_keys)}"
        )

    def _is_localization(self, user_prompt: str) -> bool:
        """Detect whether this is a localization task from the prompt."""
        return _LOCALIZATION_MARKER in user_prompt

    def _run_inference_impl(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> InferenceResult:
        from google.genai import types

        is_localization = self._is_localization(user_prompt)

        # Build content: images first, then text prompt
        content = list(images) + [user_prompt]

        max_retries = len(self._api_keys) * 2  # Try each key up to 2 times
        last_error = None

        for attempt in range(max_retries):
            try:
                if is_localization:
                    # Structured JSON output — Gemini returns normalized [0,1] coords
                    response = self._client.models.generate_content(
                        model=self.model_id,
                        contents=content,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            max_output_tokens=self.max_new_tokens,
                            temperature=0.0,
                            response_mime_type="application/json",
                            response_schema=LocalizationResponse,
                        ),
                    )

                    # Parse and clamp coords to [0,1]
                    raw_text = response.text if response.text else "{}"
                    parsed = LocalizationResponse.model_validate_json(raw_text)

                    # Crop all bbox values to [0, 1]
                    clamped_bboxes = []
                    for b in parsed.bboxes:
                        clamped_bboxes.append({
                            "frame": b.frame,
                            "label": b.label,
                            "min_x": min(max(b.min_x, 0.0), 1.0),
                            "min_y": min(max(b.min_y, 0.0), 1.0),
                            "max_x": min(max(b.max_x, 0.0), 1.0),
                            "max_y": min(max(b.max_y, 0.0), 1.0),
                        })

                    # Save as JSON with normalized coords — evaluation will convert
                    output = json.dumps({
                        "reasoning": parsed.reasoning,
                        "answer": parsed.answer,
                        "bboxes": clamped_bboxes,
                    })
                else:
                    # Standard free-text response for all other tasks
                    response = self._client.models.generate_content(
                        model=self.model_id,
                        contents=content,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            max_output_tokens=self.max_new_tokens,
                            temperature=0.0,
                        ),
                    )
                    output = response.text if response.text else ""

                # Extract token counts
                prompt_tokens = 0
                completion_tokens = 0
                if hasattr(response, "usage_metadata"):
                    prompt_tokens = getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    ) or 0
                    completion_tokens = getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    ) or 0

                return InferenceResult(
                    model_output=output,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=0.0,
                )

            except Exception as e:
                error_str = str(e)
                last_error = e

                # Check for quota/rate-limit errors (429)
                is_quota = (
                    "429" in error_str
                    or "quota" in error_str.lower()
                    or "rate" in error_str.lower()
                )

                if is_quota and len(self._api_keys) > 1:
                    logger.warning(
                        f"Quota error on key {self._current_key_idx + 1}, "
                        f"rotating (attempt {attempt + 1}/{max_retries})"
                    )
                    self._rotate_key()
                    time.sleep(1.0)
                    continue
                elif is_quota and len(self._api_keys) == 1:
                    if attempt == 0:
                        logger.warning("Quota error, waiting 60s before retry...")
                        time.sleep(60)
                        continue
                    raise
                else:
                    raise

        # All retries exhausted
        raise last_error

    def unload_model(self):
        self._clients = []
        self._loaded = False

    @property
    def supports_multiple_images(self) -> bool:
        return True
