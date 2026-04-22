"""InternVL 2.5 model runner using HuggingFace AutoModel with trust_remote_code.

Requires:
    - torch, torchvision, transformers, PIL
    - Model weights: OpenGVLab/InternVL2_5-8B (auto-downloaded from HuggingFace)

Follows the official inference pattern from the HuggingFace model card:
    - AutoModel.from_pretrained(..., trust_remote_code=True)
    - AutoTokenizer.from_pretrained(..., trust_remote_code=True)
    - build_transform() for image preprocessing (resize to 448, ImageNet normalize)
    - model.chat() with pixel_values tensor and num_patches_list
    - System prompt injected via model.system_message override

Note: language_model.generate() is monkey-patched with a manual greedy decode
loop to bypass transformers 4.57+ DynamicCache incompatibilities with InternLM2.
"""

import gc
from typing import List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from sgmriqa.models.base import BaseModelRunner, InferenceResult

# ImageNet normalization constants (same as InternVL uses)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    """Build the InternVL inference transform: resize + normalize."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class InternVLRunner(BaseModelRunner):
    """Runner for InternVL 2.5 models using HuggingFace AutoModel."""

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2_5-8B",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        dynamic: bool = False,
        max_num: int = 6,
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.dynamic = dynamic
        self.max_num = max_num
        self.model_obj = None
        self.tokenizer = None
        self.transform = None
        self.image_size = 448  # updated from model config in load_model

    def load_model(self):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, use_fast=False
        )

        self.model_obj = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
        ).eval()

        # Move to GPU if single GPU and not quantized
        if (
            not self.load_in_8bit
            and not self.load_in_4bit
            and torch.cuda.is_available()
        ):
            self.model_obj = self.model_obj.cuda()

        # Monkey-patch language_model.generate with manual greedy decode.
        # transformers 4.57+ GenerationMixin creates DynamicCache objects that
        # break InternLM2's legacy past_key_values handling (empty DynamicCache
        # is truthy, causing prepare_inputs_for_generation to truncate the first
        # forward pass to 1 token and skip inputs_embeds). Manual decode avoids
        # all DynamicCache/GenerationMixin interactions.
        lm = self.model_obj.language_model

        @torch.no_grad()
        def _greedy_generate(
            inputs_embeds=None, attention_mask=None,
            max_new_tokens=512, eos_token_id=None, **kwargs
        ):
            past_key_values = None
            generated_ids = []

            if isinstance(eos_token_id, (list, tuple)):
                stop_ids = set(eos_token_id)
            elif eos_token_id is not None:
                stop_ids = {eos_token_id}
            else:
                stop_ids = set()

            for step in range(max_new_tokens):
                if step == 0:
                    outputs = lm(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        past_key_values=None,
                        use_cache=True,
                        return_dict=True,
                    )
                else:
                    outputs = lm(
                        input_ids=next_token_id,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

                past_key_values = outputs.past_key_values
                next_token_id = outputs.logits[:, -1, :].argmax(
                    dim=-1, keepdim=True
                )
                token_id = next_token_id.item()
                generated_ids.append(token_id)

                attention_mask = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((1, 1)),
                ], dim=-1)

                if token_id in stop_ids:
                    break

            return torch.tensor(
                [generated_ids], device=inputs_embeds.device
            )

        # Patch generate on the language model — also handle cases where the
        # remote code's InternLM2 class does not inherit GenerationMixin
        lm_obj = self.model_obj.language_model
        lm_obj.generate = _greedy_generate
        # Also patch on the class itself in case chat() accesses it via type
        type(lm_obj).generate = _greedy_generate

        # Get image size from model config
        self.image_size = (
            self.model_obj.config.force_image_size
            or self.model_obj.config.vision_config.image_size
        )
        self.transform = _build_transform(self.image_size)
        self._loaded = True

    def _preprocess_images(self, images: List[Image.Image]):
        """Preprocess PIL images into a stacked pixel_values tensor.

        For small MRI images (256x256 / 320x320), each image becomes 1 patch.
        For larger images with dynamic=True, uses tiling.

        Returns:
            pixel_values: Tensor of shape (total_patches, 3, image_size, image_size)
            num_patches_list: List of int, one per input image
        """
        all_patches = []
        num_patches_list = []

        for img in images:
            # dynamic=False for small MRI images: each image = 1 patch
            tiles = [img]

            for tile in tiles:
                all_patches.append(self.transform(tile))
            num_patches_list.append(len(tiles))

        pixel_values = torch.stack(all_patches).to(torch.bfloat16)
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()

        return pixel_values, num_patches_list

    def _run_inference_impl(
        self,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
    ) -> InferenceResult:
        # Override model's system message with our system prompt
        if system_prompt:
            self.model_obj.system_message = system_prompt

        # Preprocess all images
        pixel_values, num_patches_list = self._preprocess_images(images)

        # Build question with <image> tags — match official InternVL eval format:
        # single image: "<image>\n", multiple images: "Image-{i}: <image>\n" per image
        if len(images) == 1:
            question = "<image>\n" + user_prompt
        else:
            image_tags = "".join(f"Image-{i+1}: <image>\n" for i in range(len(images)))
            question = image_tags + user_prompt

        generation_config = {
            "num_beams": 1,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }

        response = self.model_obj.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            num_patches_list=num_patches_list,
            verbose=False,
        )

        return InferenceResult(model_output=response.strip())

    def unload_model(self):
        if self.model_obj is not None:
            del self.model_obj
            self.model_obj = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.transform = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    @property
    def supports_multiple_images(self) -> bool:
        return True
