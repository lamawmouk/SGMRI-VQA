"""OpenAI API model runner (GPT-4o, GPT-4o-mini, o1, o3, o4-mini)."""

import base64
import io
import os
from typing import List

from dotenv import load_dotenv
from PIL import Image

from sgmriqa.models.base import BaseModelRunner, InferenceResult

PRICING = {
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
}


class OpenAIRunner(BaseModelRunner):
    """Runner for OpenAI vision models."""

    def __init__(self, model_id: str = "gpt-4o-mini", **kwargs):
        super().__init__(model_id, **kwargs)
        self.client = None

    def load_model(self):
        import openai

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = openai.OpenAI(api_key=api_key)
        self._loaded = True

    def _image_to_base64(self, img: Image.Image) -> str:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    def _run_inference_impl(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> InferenceResult:
        content = []

        # Add images
        for img in images:
            b64 = self._image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            })

        # Add text prompt
        content.append({"type": "text", "text": user_prompt})

        # Models that require max_completion_tokens instead of max_tokens
        is_reasoning = self.model_id in ("o1", "o3", "o4-mini", "gpt-5", "gpt-5-mini")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        kwargs = {"model": self.model_id, "messages": messages}
        if is_reasoning:
            kwargs["max_completion_tokens"] = self.max_new_tokens
        else:
            kwargs["max_tokens"] = self.max_new_tokens
            kwargs["temperature"] = 0.0

        response = self.client.chat.completions.create(**kwargs)

        output = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        # Calculate cost
        pricing = PRICING.get(self.model_id, {"prompt": 0, "completion": 0})
        cost = (
            prompt_tokens * pricing["prompt"] / 1_000_000
            + completion_tokens * pricing["completion"] / 1_000_000
        )

        return InferenceResult(
            model_output=output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
        )

    def unload_model(self):
        self.client = None
        self._loaded = False

    @property
    def supports_multiple_images(self) -> bool:
        return True
