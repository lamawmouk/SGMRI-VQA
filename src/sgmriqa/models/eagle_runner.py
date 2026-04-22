"""Eagle 2.5 model runner using HuggingFace transformers with trust_remote_code.

Requires:
    - torch, transformers, PIL
    - flash-attn (recommended, falls back to sdpa)
    - Model weights: nvidia/Eagle-2.5-8B (auto-downloaded from HuggingFace)

Eagle 2.5 uses SigLIP vision encoder + Qwen2 LLM backbone.
Processor is Qwen2-VL-compatible (process_vision_info, apply_chat_template).
Uses trust_remote_code=True for custom model/processor code from HuggingFace.

Follows the official inference pattern from Eagle2_5/document/5.inference.md:
    - AutoModel.from_pretrained(..., trust_remote_code=True)
    - AutoProcessor.from_pretrained(..., trust_remote_code=True, use_fast=True)
    - processor.apply_chat_template() for message formatting
    - processor.process_vision_info() for vision data extraction
    - processor(text, images, videos) for combined input processing
    - model.generate() for inference
"""

import gc
from typing import List

import torch
from PIL import Image

from sgmriqa.models.base import BaseModelRunner, InferenceResult


class EagleRunner(BaseModelRunner):
    """Runner for Eagle 2.5 models using HuggingFace transformers."""

    def __init__(
        self,
        model_id: str = "nvidia/Eagle-2.5-8B",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.torch_dtype = torch_dtype
        self.model_obj = None
        self.processor = None

    def load_model(self):
        from transformers import AutoConfig, AutoModel, AutoProcessor

        dtype = torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float16

        # Load config first and force sdpa attention (Eagle config hardcodes
        # flash_attention_2 which fails without flash_attn installed)
        config = AutoConfig.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        config._attn_implementation = "sdpa"
        # Also override on sub-configs if present
        if hasattr(config, "llm_config"):
            config.llm_config._attn_implementation = "sdpa"
        if hasattr(config, "vision_config"):
            config.vision_config._attn_implementation = "sdpa"

        self.model_obj = AutoModel.from_pretrained(
            self.model_id,
            config=config,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).eval()

        if torch.cuda.is_available():
            self.model_obj = self.model_obj.cuda()

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=True,
        )
        self.processor.tokenizer.padding_side = "left"

        self._loaded = True

    def _build_messages(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> list:
        """Build OpenAI-style messages with PIL images.

        Args:
            images: List of PIL Images.
            system_prompt: System-level prompt.
            user_prompt: User-level prompt.

        Returns:
            List of message dicts for apply_chat_template.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user_prompt})

        messages.append({"role": "user", "content": content})
        return messages

    def _run_inference_impl(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> InferenceResult:
        messages = self._build_messages(images, system_prompt, user_prompt)

        # Apply chat template to get text prompt
        text_list = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        ]

        # Extract vision data from messages
        image_inputs, video_inputs = self.processor.process_vision_info(messages)

        # Process text + vision into model inputs
        # Limit dynamic tiles to 1 per image for multi-frame MRI input
        # (small 256x256/320x320 images don't benefit from tiling)
        proc_kwargs = {}
        if len(images) > 1:
            proc_kwargs["images_kwargs"] = {"max_dynamic_tiles": 1}

        inputs = self.processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            **proc_kwargs,
        )

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        # Generate — let the model's generation_config handle token IDs
        with torch.inference_mode():
            generated_ids = self.model_obj.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        # Eagle passes inputs_embeds internally (like Qwen2-VL), so
        # generate() returns only new tokens — no trimming needed.
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        response = output_text[0].strip()
        return InferenceResult(model_output=response)

    def unload_model(self):
        if self.model_obj is not None:
            del self.model_obj
            self.model_obj = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    @property
    def supports_multiple_images(self) -> bool:
        return True
