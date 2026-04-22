"""LLaVA-Video / LLaVA-NeXT model runner using the local LLaVA-NeXT codebase.

Requires:
    - The LLaVA-NeXT repo at ../../LLaVA-NeXT (relative to this file)
    - torch, transformers, PIL
    - flash-attn (recommended, falls back to sdpa)

Follows the exact inference pattern from the official video_demo.py:
    - load_pretrained_model() from llava.model.builder
    - image_processor.preprocess() for frame tensors
    - conv_templates for conversation formatting
    - model.generate() with modalities="video" for spatial pooling

LLaVA-Video-7B uses Qwen2 backbone with SigLip vision encoder.
Frames are treated as video input (spatial pooling reduces tokens per frame).
"""

import copy
import gc
import os
import sys
from typing import List

import numpy as np
import torch
from PIL import Image

from sgmriqa.models.base import BaseModelRunner, InferenceResult


class LLaVAVideoRunner(BaseModelRunner):
    """Runner for LLaVA-Video / LLaVA-NeXT models using native LLaVA-NeXT codebase."""

    def __init__(
        self,
        model_id: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
        conv_mode: str = "qwen_1_5",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.conv_mode = conv_mode
        self.torch_dtype = torch_dtype
        self.model_obj = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None

    def load_model(self):
        _ensure_llava_next_package()

        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_name = get_model_name_from_path(self.model_id)

        # Try flash_attention_2, fall back to sdpa
        try:
            self.tokenizer, self.model_obj, self.image_processor, self.context_len = (
                load_pretrained_model(
                    self.model_id,
                    None,
                    model_name,
                    torch_dtype=self.torch_dtype,
                    attn_implementation="flash_attention_2",
                )
            )
        except (ImportError, ValueError):
            self.tokenizer, self.model_obj, self.image_processor, self.context_len = (
                load_pretrained_model(
                    self.model_id,
                    None,
                    model_name,
                    torch_dtype=self.torch_dtype,
                    attn_implementation="sdpa",
                )
            )

        self.model_obj.eval()

        # Set pad_token_id for qwen models
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                self.tokenizer.pad_token_id = 151643

        self._loaded = True

    def _preprocess_frames(self, images: List[Image.Image]) -> torch.Tensor:
        """Convert PIL images to a video tensor via image_processor.

        Args:
            images: List of PIL Images (MRI slices).

        Returns:
            Tensor of shape (num_frames, 3, H, W) on CUDA.
        """
        # Convert PIL images to numpy array (T, H, W, 3) — what image_processor expects
        frames = np.stack([np.array(img.convert("RGB")) for img in images])

        # Preprocess to tensor
        dtype = torch.bfloat16 if self.torch_dtype == "bfloat16" else torch.float16
        video = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        video = video.to(dtype=dtype)
        if torch.cuda.is_available():
            video = video.cuda()

        return video

    def _run_inference_impl(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> InferenceResult:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

        # Preprocess frames as video tensor
        video = self._preprocess_frames(images)
        video = [video]  # Wrap in list for model.generate()

        # Build prompt with <image> token
        qs = DEFAULT_IMAGE_TOKEN + "\n" + user_prompt

        # Set up conversation template
        conv = copy.deepcopy(conv_templates[self.conv_mode])

        # Inject system prompt if provided
        if system_prompt:
            conv.system = f"<|im_start|>system\n{system_prompt}"

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize with image placeholder tokens
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0)

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        if torch.cuda.is_available():
            attention_mask = attention_mask.cuda()

        # Stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        # Generate
        with torch.inference_mode():
            output_ids = self.model_obj.generate(
                inputs=input_ids,
                images=video,
                attention_mask=attention_mask,
                modalities="video",
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                num_beams=1,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        # LLaVA's generate() passes inputs_embeds (not input_ids) to
        # super().generate(), so output contains ONLY new tokens — no trimming needed.
        response = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()

        # Strip stop string if present at end
        if response.endswith(stop_str):
            response = response[: -len(stop_str)].strip()

        return InferenceResult(model_output=response)

    def unload_model(self):
        if self.model_obj is not None:
            del self.model_obj
            self.model_obj = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.image_processor is not None:
            del self.image_processor
            self.image_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    @property
    def supports_multiple_images(self) -> bool:
        return True


def _ensure_llava_next_package():
    """Make the llava package importable from the LLaVA-NeXT repo."""
    try:
        from llava.model.builder import load_pretrained_model  # noqa: F401
        return
    except ImportError:
        pass

    # Try adding LLaVA-NeXT to sys.path
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "LLaVA-NeXT"),
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)
            try:
                from llava.model.builder import load_pretrained_model  # noqa: F401
                return
            except ImportError:
                sys.path.remove(abs_path)

    raise ImportError(
        "llava package not found. Ensure LLaVA-NeXT/ is at the project root "
        "or install the llava package: cd LLaVA-NeXT && pip install -e ."
    )
