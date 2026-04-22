"""HuggingFace LLaVA-Med v1.5 (Mistral) model runner.

Uses the HF-native conversion: chaoyinshe/llava-med-v1.5-mistral-7b-hf
which works with LlavaForConditionalGeneration from transformers.
"""

from typing import List

import torch
from PIL import Image

from sgmriqa.models.base import BaseModelRunner, InferenceResult


class LLaVAMedRunner(BaseModelRunner):
    """Runner for LLaVA-Med v1.5 (Mistral-7B) via HF transformers."""

    def __init__(self, model_id: str = "chaoyinshe/llava-med-v1.5-mistral-7b-hf", **kwargs):
        super().__init__(model_id, **kwargs)
        self.processor = None
        self.model_obj = None

    def load_model(self):
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model_obj = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
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
        image = images[0] if images else None

        prompt = system_prompt + "\n\n" + user_prompt if system_prompt else user_prompt

        # Mistral instruct format: [INST] <image>\n{prompt} [/INST]
        text_prompt = f"[INST] <image>\n{prompt} [/INST]"

        inputs = self.processor(
            images=[image] if image else None,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.model_obj.device, torch.bfloat16)

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
