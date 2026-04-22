"""Unified data conversion: raw QA JSON → per-model SFT training format.

Reads the 4 GPT-4o QA JSON files (brain/knee × image/volume) and outputs
training data in each model's expected conversation format.

Each training sample is structured as:
  SYSTEM:    Expert role + image/volume context + JSON output contract
  USER:      [image/video] + question + task-specific instruction
  ASSISTANT: {"reasoning": "...", "answer": "...", "bboxes": [...]}

The assistant response is the SFT training target — it contains the
chain-of-thought reasoning (with inline <bbx> tags) and the answer,
taken directly from the GPT-4o QA data.

Usage:
    python -m finetuning.convert_data --format all --data-root /path/to/data
    python -m finetuning.convert_data --format qwen --data-root /path/to/data
"""

import argparse
import glob as globmod
import json
import logging
import os
import sys
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.data.prompt_builder import (
    RESEARCH_CONTEXT,
    _EXPERT_ROLE as EXPERT_ROLE,
    _IMAGE_DIMS as IMAGE_DIMS,
)
from evaluation.schemas import format_schema_instruction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QA_DATA_DIR = os.path.join(PROJECT_ROOT, "data_generation", "qa_reasoning_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "finetuning", "output")

QA_FILES = {
    "brain_image": os.path.join(QA_DATA_DIR, "brain", "gpt4o_brain_train_image_qa_image_qa_pairs.json"),
    "brain_volume": os.path.join(QA_DATA_DIR, "brain", "gpt4o_brain_train_volume_qa_volume_qa_pairs.json"),
    "knee_image": os.path.join(QA_DATA_DIR, "knee", "gpt4o_knee_train_image_qa_image_qa_pairs.json"),
    "knee_volume": os.path.join(QA_DATA_DIR, "knee", "gpt4o_knee_train_volume_qa_volume_qa_pairs.json"),
}

ALL_FORMATS = ["qwen", "internvl", "llava", "eagle", "molmo", "plm", "keye"]


# ═══════════════════════════════════════════════════════════════════════════
# SFT Prompt Builder (uses evaluation prompt builder for consistency)
# ═══════════════════════════════════════════════════════════════════════════

def build_system_prompt(dataset: str, level: str, modality: str, num_frames: int) -> str:
    """Build the system prompt for an SFT training sample.

    Uses the same components as the evaluation prompt builder for consistency.
    For volumes, does not hardcode a specific frame count since the model's
    video processor determines the actual number of extracted frames at
    training time (e.g. Qwen uses --video_max_frames 16).
    """
    parts = [EXPERT_ROLE.get(dataset, "")]
    parts.append(RESEARCH_CONTEXT)

    img_w, img_h = IMAGE_DIMS.get(dataset, (256, 256))

    if level == "volume" and num_frames > 1:
        parts.append(
            f"You are shown {num_frames} frames (Frame 1 through Frame {num_frames}) "
            f"from a {modality} MRI volume. Each frame is {img_w}x{img_h} pixels. "
            f"When reporting bounding boxes, specify which frame each box belongs to."
        )
    else:
        parts.append(
            f"You are shown a {modality} MRI image. "
            f"The image is {img_w}x{img_h} pixels. "
            f"When reporting bounding boxes, use frame 1."
        )

    parts.append(format_schema_instruction())
    return "\n\n".join(parts)


def build_user_prompt(question: str, task: str, qa_type: str,
                      level: str = "image", dataset: str = "brain") -> str:
    """Build the user prompt for an SFT training sample.

    Contains: the question + task-specific instruction.
    """
    parts = []

    parts.append(f"Question: {question}")

    # Task-specific instruction
    if task == "detection":
        parts.append('Answer with Yes or No in the "answer" field.')
    elif task == "counting":
        parts.append('Answer with a number in the "answer" field.')
    elif task in ("classification", "diagnosis"):
        if qa_type == "multiple_choice":
            parts.append('Select all that apply in the "answer" field, e.g. (A), (C).')
        else:
            parts.append('Respond with the letter only in the "answer" field, e.g. (A).')
    elif task == "localization":
        parts.append('Report the location in "answer" and list bounding boxes in "bboxes".')
    elif task == "captioning":
        parts.append('Provide a comprehensive summary in "answer" and list all bounding boxes in "bboxes".')

    parts.append("Respond ONLY with the JSON object, no other text.")
    return "\n".join(parts)


def build_assistant_response(item: Dict[str, Any]) -> str:
    """Build the JSON assistant response — the SFT training target.

    Format: {"reasoning": "...", "answer": "...", "bboxes": [...]}

    Only localization and captioning tasks include bboxes.
    Detection, classification, diagnosis, counting → bboxes: [].
    """
    task = item.get("task", "")
    gt = item.get("ground_truth", {})

    # Only localization and captioning produce bboxes
    bboxes = []
    if task in ("localization", "captioning"):
        for bb in gt.get("bboxes", []):
            bboxes.append({
                "frame": bb.get("frame", 1),
                "label": bb.get("label", ""),
                "bbox": [bb.get("x", 0), bb.get("y", 0), bb.get("width", 0), bb.get("height", 0)],
            })

    response = {
        "reasoning": item.get("reasoning", ""),
        "answer": item.get("answer", ""),
        "bboxes": bboxes,
    }
    return json.dumps(response)


# ═══════════════════════════════════════════════════════════════════════════
# Image/volume path helpers
# ═══════════════════════════════════════════════════════════════════════════

def resolve_image_path(image_path: str, data_root: str) -> str:
    """Resolve a QA entry's image_path to an absolute path."""
    if not image_path:
        return image_path
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    # Relative paths in QA files (e.g. "../data/train_labeled_.../...")
    resolved = os.path.normpath(os.path.join(QA_DATA_DIR, image_path))
    if os.path.exists(resolved):
        return resolved
    # Remap via /data/ marker (for cluster paths)
    marker = "/data/"
    idx = image_path.find(marker)
    if idx >= 0:
        return os.path.join(data_root, image_path[idx + len(marker):])
    return os.path.join(data_root, image_path.lstrip("./"))


def get_volume_image_paths(
    volume_id: str, modality: str, dataset: str, data_root: str,
    num_slices: int = 0,
) -> List[str]:
    """Get sorted PNG paths for all slices in a volume directory."""
    if dataset == "brain":
        vol_dir = os.path.join(data_root, "train_labeled_raw_by_modality", modality, volume_id)
    elif dataset == "knee":
        vol_dir = os.path.join(data_root, "knee_train_labeled_raw", volume_id)
    else:
        return []

    if os.path.isdir(vol_dir):
        pattern = os.path.join(vol_dir, f"{volume_id}_slice_*.png")
        found = sorted(globmod.glob(pattern))
        if found:
            return found

    if num_slices > 0:
        return [
            os.path.join(vol_dir, f"{volume_id}_slice_{i:03d}.png")
            for i in range(1, num_slices + 1)
        ]
    return []


def get_mp4_path(volume_id: str, dataset: str, data_root: str) -> str:
    """Get MP4 video path for a volume (created by create_mp4_volumes.py)."""
    return os.path.join(data_root, "mp4_volumes", dataset, f"{volume_id}.mp4")


# ═══════════════════════════════════════════════════════════════════════════
# Load all training samples
# ═══════════════════════════════════════════════════════════════════════════

def load_all_samples(data_root: str) -> List[Dict[str, Any]]:
    """Load all QA entries, resolve paths, build prompts.

    Returns a list of dicts, each with:
        dataset, level, volume_id, modality, task,
        image_paths, num_frames,
        system_prompt, user_prompt, assistant_response
    """
    samples = []

    for key, path in QA_FILES.items():
        dataset, level = key.split("_")  # "brain_image" → ("brain", "image")

        if not os.path.exists(path):
            logger.warning(f"QA file not found: {path}")
            continue

        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} entries from {os.path.basename(path)}")

        for item in data:
            volume_id = item.get("volume_id", "")
            modality = item.get("modality", "")
            task = item.get("task", "")
            qa_type = item.get("type", "open_ended")

            # Resolve image/video paths
            if level == "image":
                raw_path = item.get("image_path", "")
                abs_path = resolve_image_path(raw_path, data_root)
                image_paths = [abs_path] if abs_path else []
                num_frames = 1
            else:
                # Use the actual num_slices from the QA data — no subsampling.
                # The model sees all frames in the volume.
                num_slices = item.get("num_slices", 0)
                image_paths = get_volume_image_paths(
                    volume_id, modality, dataset, data_root,
                    num_slices=num_slices,
                )
                num_frames = len(image_paths) if image_paths else num_slices

            # Build the 3 prompt components
            system_prompt = build_system_prompt(dataset, level, modality, num_frames)
            user_prompt = build_user_prompt(item.get("question", ""), task, qa_type,
                                           level=level, dataset=dataset)
            assistant_response = build_assistant_response(item)

            samples.append({
                "dataset": dataset,
                "level": level,
                "volume_id": volume_id,
                "modality": modality,
                "task": task,
                "image_paths": image_paths,
                "num_frames": num_frames,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "assistant_response": assistant_response,
            })

    logger.info(f"Total training samples: {len(samples)}")
    return samples


# ═══════════════════════════════════════════════════════════════════════════
# Per-model format converters
#
# Every converter produces training data where each sample has:
#   System:    Expert context + output schema
#   User:      [image/video token] + question + task instruction
#   Assistant: JSON {reasoning, answer, bboxes}
# ═══════════════════════════════════════════════════════════════════════════

def convert_qwen(samples: List[Dict], data_root: str):
    """Qwen format: JSON array, conversations with system/human/gpt turns.

    Image:  {"image": "/abs/path.png", "conversations": [...]}
    Volume: {"video": "/abs/path.mp4", "conversations": [...]}
    """
    out_dir = os.path.join(OUTPUT_DIR, "qwen")
    os.makedirs(out_dir, exist_ok=True)

    entries = []
    skipped = 0
    for s in samples:
        conv = [
            {"from": "system", "value": s["system_prompt"]},
            {"from": "human", "value": None},
            {"from": "gpt", "value": s["assistant_response"]},
        ]

        if s["level"] == "image":
            if not s["image_paths"]:
                skipped += 1
                continue
            conv[1]["value"] = "<image>\n" + s["user_prompt"]
            entries.append({"image": s["image_paths"][0], "conversations": conv})
        else:
            mp4 = get_mp4_path(s["volume_id"], s["dataset"], data_root)
            conv[1]["value"] = "<video>\n" + s["user_prompt"]
            entries.append({"video": mp4, "conversations": conv})

    out_path = os.path.join(out_dir, "train_qwen.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    logger.info(f"[qwen] Wrote {len(entries)} entries (skipped {skipped})")

    # Also write per-dataset files for the Qwen finetuning registry
    per_dataset = {}
    entry_idx = 0
    for s in samples:
        if s["level"] == "image" and not s["image_paths"]:
            continue
        key = f"{s['dataset']}_{s['level']}"
        per_dataset.setdefault(key, []).append(entries[entry_idx])
        entry_idx += 1
    for key, ds_entries in per_dataset.items():
        ds_path = os.path.join(out_dir, f"train_qwen_{key}.json")
        with open(ds_path, "w") as f:
            json.dump(ds_entries, f, indent=2)
        logger.info(f"[qwen] Per-dataset: {key} → {len(ds_entries)} entries")


def convert_internvl(samples: List[Dict], data_root: str):
    """InternVL format: JSONL + meta.json.

    Image:  {"image": "/path.png", "conversations": [...], "system_prompt": "..."}
    Volume: {"image": ["/s1.png",...], "conversations": [...], "system_prompt": "..."}
    """
    out_dir = os.path.join(OUTPUT_DIR, "internvl")
    os.makedirs(out_dir, exist_ok=True)

    entries = []
    skipped = 0
    for s in samples:
        if not s["image_paths"]:
            skipped += 1
            continue

        if s["level"] == "image":
            user_val = "<image>\n" + s["user_prompt"]
            image_field = s["image_paths"][0]
        else:
            n = len(s["image_paths"])
            user_val = " ".join("<image>" for _ in range(n)) + "\n" + s["user_prompt"]
            image_field = s["image_paths"]

        conv = [
            {"from": "human", "value": user_val},
            {"from": "gpt", "value": s["assistant_response"]},
        ]
        entries.append({
            "image": image_field,
            "system_prompt": s["system_prompt"],
            "conversations": conv,
        })

    out_path = os.path.join(out_dir, "train_internvl.jsonl")
    with open(out_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    meta = {
        "mri_grounding": {
            "root": "",
            "annotation": out_path,
            "data_augment": False,
            "repeat_time": 1,
            "length": len(entries),
        }
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[internvl] Wrote {len(entries)} entries (skipped {skipped})")


def convert_llava(samples: List[Dict], data_root: str):
    """LLaVA-Video format: JSON array + YAML data config.

    Image:  {"id": "...", "image": "/path.png", "conversations": [...]}
    Volume: {"id": "...", "video": "/path.mp4", "conversations": [...]}

    LLaVA doesn't support a system turn natively, so system prompt is
    prepended to the human turn.
    """
    out_dir = os.path.join(OUTPUT_DIR, "llava")
    os.makedirs(out_dir, exist_ok=True)

    entries = []
    skipped = 0
    for idx, s in enumerate(samples):
        sid = f"{s['dataset']}_{s['level']}_{s['volume_id']}_{s['task']}_{idx}"

        # LLaVA: system prompt goes into human turn
        human_text = s["system_prompt"] + "\n\n" + s["user_prompt"]

        conv = [
            {"from": "human", "value": None},
            {"from": "gpt", "value": s["assistant_response"]},
        ]

        if s["level"] == "image":
            if not s["image_paths"]:
                skipped += 1
                continue
            conv[0]["value"] = "<image>\n" + human_text
            entries.append({"id": sid, "image": s["image_paths"][0], "conversations": conv})
        else:
            mp4 = get_mp4_path(s["volume_id"], s["dataset"], data_root)
            conv[0]["value"] = "<video>\n" + human_text
            entries.append({"id": sid, "video": mp4, "conversations": conv})

    out_path = os.path.join(out_dir, "train_llava.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)

    yaml_path = os.path.join(out_dir, "data_config.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"datasets:\n  - json_path: {out_path}\n    sampling_strategy: \"all\"\n")

    logger.info(f"[llava] Wrote {len(entries)} entries (skipped {skipped})")


def convert_eagle(samples: List[Dict], data_root: str):
    """Eagle 2.5 format: JSONL with image_list + <image-N> tokens.

    Image:  {"image_list": ["/path.png"], "conversations": [...]}
    Volume: {"image_list": ["/s1.png",...], "conversations": [...]}
    """
    out_dir = os.path.join(OUTPUT_DIR, "eagle")
    os.makedirs(out_dir, exist_ok=True)

    entries = []
    skipped = 0
    for s in samples:
        if not s["image_paths"]:
            skipped += 1
            continue

        if s["level"] == "image":
            image_list = s["image_paths"][:1]
            img_tokens = "<image-1>"
        else:
            image_list = s["image_paths"]
            img_tokens = " ".join(f"<image-{i+1}>" for i in range(len(image_list)))

        conv = [
            {"from": "system", "value": s["system_prompt"]},
            {"from": "human", "value": f"{img_tokens}\n{s['user_prompt']}"},
            {"from": "gpt", "value": s["assistant_response"]},
        ]
        entries.append({"image_list": image_list, "conversations": conv})

    out_path = os.path.join(out_dir, "train_eagle.jsonl")
    with open(out_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"[eagle] Wrote {len(entries)} entries (skipped {skipped})")


def convert_molmo(samples: List[Dict], data_root: str):
    """Molmo format: JSON with HF messages + image paths.

    {"images": [...], "messages": [{role, content}, ...]}
    """
    out_dir = os.path.join(OUTPUT_DIR, "molmo")
    os.makedirs(out_dir, exist_ok=True)

    entries = []
    skipped = 0
    for s in samples:
        if not s["image_paths"]:
            skipped += 1
            continue

        entries.append({
            "images": s["image_paths"],
            "messages": [
                {"role": "system", "content": s["system_prompt"]},
                {"role": "user", "content": s["user_prompt"]},
                {"role": "assistant", "content": s["assistant_response"]},
            ],
        })

    out_path = os.path.join(out_dir, "train_molmo.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)

    logger.info(f"[molmo] Wrote {len(entries)} entries (skipped {skipped})")


def convert_plm(samples: List[Dict], data_root: str):
    """PLM format: JSONL + datasets.yaml.

    Image:  {"image": "/path.png", "conversations": [...]}
    Volume: {"image": ["/s1.png",...], "conversations": [...]}

    PLM doesn't support a system role, so system is prepended to human.
    """
    out_dir = os.path.join(OUTPUT_DIR, "plm")
    os.makedirs(out_dir, exist_ok=True)

    entries = []
    skipped = 0
    for s in samples:
        if not s["image_paths"]:
            skipped += 1
            continue

        human_text = s["system_prompt"] + "\n\n" + s["user_prompt"]

        if s["level"] == "image":
            img_token = "<image>"
            image_field = s["image_paths"][0]
        else:
            n = len(s["image_paths"])
            img_token = " ".join("<image>" for _ in range(n))
            image_field = s["image_paths"]

        conv = [
            {"from": "human", "value": f"{img_token}\n{human_text}"},
            {"from": "gpt", "value": s["assistant_response"]},
        ]
        entries.append({"image": image_field, "conversations": conv})

    out_path = os.path.join(out_dir, "train_plm.jsonl")
    with open(out_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    yaml_path = os.path.join(out_dir, "datasets.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"mri_grounding:\n  annotation: {out_path}\n  root_dir: \"\"\n")

    logger.info(f"[plm] Wrote {len(entries)} entries (skipped {skipped})")


def convert_keye(samples: List[Dict], data_root: str):
    """Keye-VL format: JSON with HF messages + image paths.

    {"images": [...], "messages": [{role, content}, ...]}
    """
    out_dir = os.path.join(OUTPUT_DIR, "keye")
    os.makedirs(out_dir, exist_ok=True)

    entries = []
    skipped = 0
    for s in samples:
        if not s["image_paths"]:
            skipped += 1
            continue

        entries.append({
            "images": s["image_paths"],
            "messages": [
                {"role": "system", "content": s["system_prompt"]},
                {"role": "user", "content": s["user_prompt"]},
                {"role": "assistant", "content": s["assistant_response"]},
            ],
        })

    out_path = os.path.join(out_dir, "train_keye.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)

    logger.info(f"[keye] Wrote {len(entries)} entries (skipped {skipped})")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

CONVERTERS = {
    "qwen": convert_qwen,
    "internvl": convert_internvl,
    "llava": convert_llava,
    "eagle": convert_eagle,
    "molmo": convert_molmo,
    "plm": convert_plm,
    "keye": convert_keye,
}


def main():
    parser = argparse.ArgumentParser(description="Convert QA data to model-specific SFT training formats")
    parser.add_argument("--format", type=str, default="all", choices=ALL_FORMATS + ["all"])
    parser.add_argument("--data-root", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--preview", action="store_true", help="Print one sample per level and exit")
    args = parser.parse_args()

    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    samples = load_all_samples(args.data_root)
    if not samples:
        logger.error("No samples loaded. Check QA file paths.")
        sys.exit(1)

    # Preview mode: show the prompt structure
    if args.preview:
        for level in ("image", "volume"):
            for s in samples:
                if s["level"] == level:
                    print(f"\n{'='*70}")
                    print(f"LEVEL: {level} | DATASET: {s['dataset']} | TASK: {s['task']}")
                    print(f"{'='*70}")
                    print(f"\n--- SYSTEM PROMPT ---\n{s['system_prompt']}")
                    print(f"\n--- USER PROMPT ---\n{s['user_prompt']}")
                    print(f"\n--- ASSISTANT RESPONSE (SFT target) ---\n{s['assistant_response'][:500]}")
                    if s["image_paths"]:
                        print(f"\n--- IMAGE PATHS ({len(s['image_paths'])}) ---")
                        print(f"  {s['image_paths'][0]}")
                        if len(s["image_paths"]) > 1:
                            print(f"  ... {s['image_paths'][-1]}")
                    break
        return

    # Convert
    formats = ALL_FORMATS if args.format == "all" else [args.format]
    for fmt in formats:
        logger.info(f"Converting to {fmt} format...")
        CONVERTERS[fmt](samples, args.data_root)

    logger.info("Done.")


if __name__ == "__main__":
    main()
