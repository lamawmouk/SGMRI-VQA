"""Abstract base class for model runners."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL import Image


@dataclass
class InferenceResult:
    """Result of a single model inference call."""

    model_output: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    duration: float = 0.0
    metadata: Dict = field(default_factory=dict)


class BaseModelRunner(ABC):
    """Abstract base class for all model runners."""

    def __init__(self, model_id: str, max_new_tokens: int = 1024, **kwargs):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._loaded = False

    @abstractmethod
    def load_model(self):
        """Load the model into memory / initialize the API client."""
        pass

    @abstractmethod
    def _run_inference_impl(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> InferenceResult:
        """Run inference on a list of images with system and user prompts.

        Args:
            images: List of PIL Images (1 for image-level, N for volume-level).
            system_prompt: System-level prompt (expert context + output contract).
            user_prompt: User-level prompt (question + task instruction).

        Returns:
            InferenceResult with model output and usage stats.
        """
        pass

    def run_inference(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> InferenceResult:
        """Run inference with timing. Calls _run_inference_impl internally."""
        if not self._loaded:
            self.load_model()
            self._loaded = True

        start = time.time()
        result = self._run_inference_impl(images, system_prompt, user_prompt)
        result.duration = time.time() - start
        return result

    @abstractmethod
    def unload_model(self):
        """Release model from memory."""
        pass

    @property
    def supports_multiple_images(self) -> bool:
        """Whether the model natively supports multiple images."""
        return False
