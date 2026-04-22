"""Qwen3-VL model runner using vLLM with proper chat template and vision processing.

Requires:
    - vllm
    - transformers (AutoProcessor)
    - qwen_vl_utils (from Qwen3-VL repo or pip install qwen-vl-utils)

Supports single-image, multi-image (volume frames), and system prompts natively.
Follows the exact inference pattern from the official Qwen3-VL evaluation scripts.
"""

import gc
import os
import sys
from typing import List

import torch
from PIL import Image

from sgmriqa.models.base import BaseModelRunner, InferenceResult

# MRI images are small (256x256 brain, 320x320 knee).
# These pixel budgets match Qwen3-VL's RealWorldQA defaults (768*28*28, 5120*28*28).
_DEFAULT_MIN_PIXELS = 768 * 28 * 28      # 602,112
_DEFAULT_MAX_PIXELS = 5120 * 28 * 28     # 4,014,080


class Qwen3VLRunner(BaseModelRunner):
    """Runner for Qwen3-VL models via vLLM with native multi-image and system prompt support."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_model_len: int = 131072,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = None,
        limit_mm_per_prompt: int = 200,
        min_pixels: int = _DEFAULT_MIN_PIXELS,
        max_pixels: int = _DEFAULT_MAX_PIXELS,
        processor_id: str = None,
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.processor_id = processor_id
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.limit_mm_per_prompt = limit_mm_per_prompt
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.llm = None
        self.processor = None

    def load_model(self):
        from vllm import LLM
        from transformers import AutoProcessor

        # Required by vLLM for multi-process workers
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        # Ensure qwen_vl_utils is importable (may be in Qwen3-VL subdir)
        _ensure_qwen_vl_utils()

        processor_id = self.processor_id or self.model_id
        self.processor = AutoProcessor.from_pretrained(
            processor_id, trust_remote_code=True
        )

        tp_size = self.tensor_parallel_size
        if tp_size is None:
            tp_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

        self.llm = LLM(
            model=self.model_id,
            tokenizer=processor_id,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp_size,
            limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
            seed=42,
        )
        self._loaded = True

    def _build_messages(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> list:
        """Build Qwen3-VL conversation messages with images and pixel constraints."""
        messages = []

        # System message (text-only, optional)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # User message: images + text
        content = []
        for img in images:
            content.append({
                "type": "image",
                "image": img,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            })
        content.append({"type": "text", "text": user_prompt})

        messages.append({"role": "user", "content": content})
        return messages

    def _run_inference_impl(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> InferenceResult:
        from vllm import SamplingParams
        from qwen_vl_utils import process_vision_info

        # Build conversation messages
        messages = self._build_messages(images, system_prompt, user_prompt)

        # Apply chat template to get prompt text (must be tokenize=False for vLLM)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info — pass patch_size from processor as Qwen3-VL eval does
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        # Build multi-modal data dict
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        vllm_input = {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        # temperature=0 for deterministic eval (Qwen3-VL defaults use 0.7
        # but we want reproducible evaluation results)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
            stop_token_ids=[],
        )

        outputs = self.llm.generate([vllm_input], sampling_params=sampling_params)

        response = outputs[0].outputs[0].text

        return InferenceResult(model_output=response)

    def unload_model(self):
        if self.llm is not None:
            del self.llm
            self.llm = None
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


def _ensure_qwen_vl_utils():
    """Make qwen_vl_utils importable if it's in the Qwen3-VL subdir."""
    try:
        import qwen_vl_utils  # noqa: F401
        return
    except ImportError:
        pass

    # Try adding the local Qwen3-VL qwen-vl-utils to path
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "Qwen3-VL", "qwen-vl-utils", "src"),
        os.path.join(os.path.dirname(__file__), "..", "..", "Qwen3-VL", "qwen-vl-utils"),
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)
            try:
                import qwen_vl_utils  # noqa: F401
                return
            except ImportError:
                sys.path.remove(abs_path)

    raise ImportError(
        "qwen_vl_utils not found. Install it with: "
        "pip install qwen-vl-utils, or ensure Qwen3-VL/qwen-vl-utils/src is accessible."
    )
