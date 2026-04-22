"""HuggingFace MedGemma model runner (AutoModelForImageTextToText)."""

from typing import List

import torch
from PIL import Image

from sgmriqa.models.base import BaseModelRunner, InferenceResult


class MedGemmaRunner(BaseModelRunner):
    """Runner for MedGemma 1.5 4B (instruction-tuned) via HuggingFace."""

    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it", **kwargs):
        super().__init__(model_id, **kwargs)
        self.processor = None
        self.model_obj = None
        self.device = None

    def load_model(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model_obj = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model_obj = self.model_obj.eval()
        self._loaded = True

    def _run_inference_impl(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> InferenceResult:
        # Combine system + user prompt
        prompt = system_prompt + "\n\n" + user_prompt if system_prompt else user_prompt

        # Build chat messages
        content = []
        if images:
            content.append({"type": "image", "image": images[0]})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model_obj.device, dtype=dtype)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output_ids = self.model_obj.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        output_text = self.processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()

        return InferenceResult(
            model_output=output_text,
            prompt_tokens=input_len,
            completion_tokens=len(output_ids[0]) - input_len,
        )

    def unload_model(self):
        del self.model_obj
        del self.processor
        self.model_obj = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
