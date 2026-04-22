"""Model registry for the 10 VLMs evaluated in the paper + fine-tuned model."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ModelConfig:
    name: str
    model_id: str
    model_type: str  # "api_openai", "api_gemini", "hf", "vllm"
    runner_module: str
    runner_class: str

    max_context_window: int = 4096
    max_new_tokens: int = 1024
    max_image_tokens: int = 2048

    supports_video: bool = False
    supports_grounding: bool = False
    supports_multi_image: bool = False
    supports_thinking: bool = False

    bbox_format: str = "<bbx>[{x}, {y}, {w}, {h}]</bbx>"
    bbox_instruction: str = (
        "When localizing findings, output bounding boxes in the format "
        "<bbx>[x, y, width, height]</bbx> where coordinates are in pixels."
    )
    think_open: str = ""
    think_close: str = ""

    runner_kwargs: Dict = field(default_factory=dict)


_BBOX_PLAIN = "[{x}, {y}, {w}, {h}]"
_BBOX_INST_PLAIN = (
    "When localizing findings, output bounding boxes as "
    "[x, y, width, height] in pixel coordinates."
)

_QWEN_BBOX = "<|box_start|>({x},{y}),({x2},{y2})<|box_end|>"
_QWEN_BBOX_INST = (
    "When localizing findings, output bounding boxes using the format "
    "<|box_start|>(x1,y1),(x2,y2)<|box_end|> in pixel coordinates."
)

MODEL_REGISTRY: Dict[str, ModelConfig] = {

    # ── Proprietary API models (3) ────────────────────────────────────────

    "gpt-4o": ModelConfig(
        name="GPT-4o",
        model_id="gpt-4o",
        model_type="api_openai",
        runner_module="sgmriqa.models.api_openai",
        runner_class="OpenAIRunner",
        max_context_window=128_000,
        max_new_tokens=4096,
        max_image_tokens=1536,
        supports_video=True,
        supports_grounding=True,
        supports_multi_image=True,
    ),

    "gemini-2.5-pro": ModelConfig(
        name="Gemini 2.5 Pro",
        model_id="gemini-2.5-pro",
        model_type="api_gemini",
        runner_module="sgmriqa.models.api_gemini",
        runner_class="GeminiRunner",
        max_context_window=1_048_576,
        max_new_tokens=8192,
        max_image_tokens=258,
        supports_video=True,
        supports_grounding=True,
        supports_multi_image=True,
        supports_thinking=True,
        think_open="<think>",
        think_close="</think>",
    ),

    "gemini-2.5-flash": ModelConfig(
        name="Gemini 2.5 Flash",
        model_id="gemini-2.5-flash",
        model_type="api_gemini",
        runner_module="sgmriqa.models.api_gemini",
        runner_class="GeminiRunner",
        max_context_window=1_048_576,
        max_new_tokens=8192,
        max_image_tokens=258,
        supports_video=True,
        supports_grounding=True,
        supports_multi_image=True,
        supports_thinking=True,
        think_open="<think>",
        think_close="</think>",
    ),

    # ── Open-source general VLMs (6) ─────────────────────────────────────

    "llava-video-7b": ModelConfig(
        name="LLaVA-Video 7B Qwen2",
        model_id="lmms-lab/LLaVA-Video-7B-Qwen2",
        model_type="hf",
        runner_module="sgmriqa.models.llava_video_runner",
        runner_class="LLaVAVideoRunner",
        max_context_window=32_768,
        max_new_tokens=4096,
        max_image_tokens=196,
        supports_video=True,
        supports_grounding=True,
        supports_multi_image=True,
        bbox_format=_BBOX_PLAIN,
        bbox_instruction=_BBOX_INST_PLAIN,
    ),

    "eagle2.5-8b": ModelConfig(
        name="Eagle 2.5 8B",
        model_id="nvidia/Eagle-2.5-8B",
        model_type="hf",
        runner_module="sgmriqa.models.eagle_runner",
        runner_class="EagleRunner",
        max_context_window=32_768,
        max_new_tokens=4096,
        max_image_tokens=256,
        supports_video=True,
        supports_grounding=True,
        supports_multi_image=True,
        bbox_format=_BBOX_PLAIN,
        bbox_instruction=_BBOX_INST_PLAIN,
    ),

    "qwen3-vl-8b": ModelConfig(
        name="Qwen3-VL-8B-Instruct",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        model_type="vllm",
        runner_module="sgmriqa.models.qwen3_vl_runner",
        runner_class="Qwen3VLRunner",
        max_context_window=262_144,
        max_new_tokens=16384,
        max_image_tokens=1280,
        supports_video=True,
        supports_grounding=True,
        supports_multi_image=True,
        supports_thinking=True,
        bbox_format=_QWEN_BBOX,
        bbox_instruction=_QWEN_BBOX_INST,
        think_open="<think>",
        think_close="</think>",
    ),

    "internvl2.5-8b": ModelConfig(
        name="InternVL2.5 8B",
        model_id="OpenGVLab/InternVL2_5-8B",
        model_type="hf",
        runner_module="sgmriqa.models.internvl_runner",
        runner_class="InternVLRunner",
        max_context_window=32_768,
        max_new_tokens=4096,
        max_image_tokens=256,
        supports_video=True,
        supports_grounding=True,
        supports_multi_image=True,
        bbox_format=_BBOX_PLAIN,
        bbox_instruction=_BBOX_INST_PLAIN,
    ),

    "qwen2.5-vl-7b": ModelConfig(
        name="Qwen2.5-VL-7B-Instruct",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        model_type="vllm",
        runner_module="sgmriqa.models.qwen2_vl_runner",
        runner_class="Qwen2VLRunner",
        max_context_window=32_768,
        max_new_tokens=2048,
        max_image_tokens=480,
        supports_video=True,
        supports_grounding=True,
        supports_multi_image=True,
        supports_thinking=True,
        bbox_format=_QWEN_BBOX,
        bbox_instruction=_QWEN_BBOX_INST,
        think_open="<think>",
        think_close="</think>",
    ),

    # ── Medical-specialized VLMs (2) ─────────────────────────────────────

    "llava-med-v1.5": ModelConfig(
        name="LLaVA-Med v1.5",
        model_id="chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        model_type="hf",
        runner_module="sgmriqa.models.hf_llava_med",
        runner_class="LLaVAMedRunner",
        max_context_window=4096,
        max_new_tokens=512,
        max_image_tokens=576,
        bbox_format=_BBOX_PLAIN,
        bbox_instruction=_BBOX_INST_PLAIN,
    ),

    "medgemma-4b": ModelConfig(
        name="MedGemma 1.5 4B",
        model_id="google/medgemma-1.5-4b-it",
        model_type="hf",
        runner_module="sgmriqa.models.hf_medgemma",
        runner_class="MedGemmaRunner",
        max_context_window=8192,
        max_new_tokens=1024,
        max_image_tokens=256,
        bbox_format=_BBOX_PLAIN,
        bbox_instruction=_BBOX_INST_PLAIN,
    ),

    # ── Fine-tuned (Ours) ────────────────────────────────────────────────

    "qwen3-vl-8b-mri-vqa-full": ModelConfig(
        name="Qwen3-VL-8B-SGMRIQA-SFT",
        model_id="lamamkh/Qwen3-8B-SGMRIQA-SFT",
        model_type="vllm",
        runner_module="sgmriqa.models.qwen3_vl_runner",
        runner_class="Qwen3VLRunner",
        max_context_window=262_144,
        max_new_tokens=16384,
        max_image_tokens=1280,
        supports_video=True,
        supports_grounding=True,
        supports_multi_image=True,
        supports_thinking=True,
        bbox_format=_QWEN_BBOX,
        bbox_instruction=_QWEN_BBOX_INST,
        think_open="<think>",
        think_close="</think>",
    ),
}


def get_model_config(key: str) -> ModelConfig:
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{key}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[key]


def list_models() -> list:
    return list(MODEL_REGISTRY.keys())


def list_models_by_type(model_type: str = None) -> list:
    if model_type is None:
        return list_models()
    return [k for k, v in MODEL_REGISTRY.items() if v.model_type == model_type]


def get_max_volume_images(model_key: str) -> int:
    config = get_model_config(model_key)
    reserved = 1000 + config.max_new_tokens
    available = config.max_context_window - reserved
    if available <= 0 or config.max_image_tokens <= 0:
        return 1
    return max(1, available // config.max_image_tokens)


MIN_VIDEO_FRAMES = 40


def list_video_capable_models() -> list:
    return [
        k for k, cfg in MODEL_REGISTRY.items()
        if cfg.supports_video
        and cfg.supports_multi_image
        and get_max_volume_images(k) >= MIN_VIDEO_FRAMES
    ]
