"""Generate GPT-4o QA pairs for knee MRI volumes."""

import json
import openai
import os
import base64
import time
import argparse
import logging
import random
import io
from datetime import datetime
from typing import List, Dict
from PIL import Image, ImageDraw
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv

load_dotenv()

TEMPERATURE = 0.2
MAX_TOKENS = 2048
VOLUME_MAX_TOKENS = 16384

log_filename = f"gpt4o_knee_qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(message)s")


def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def draw_bboxes(image: Image.Image, bboxes: List[Dict]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for i, bbox in enumerate(bboxes, 1):
        x0, y0 = float(bbox["x"]), float(bbox["y"])
        x1, y1 = x0 + float(bbox["width"]), y0 + float(bbox["height"])
        draw.rectangle(((x0, y0), (x1, y1)), outline="white", width=2)
        draw.text((x0, max(0, y0 - 12)), str(i), fill="white")
    return img


def format_bbox_coordinates(bboxes: List[Dict]) -> str:
    if not bboxes:
        return "No bounding boxes - image appears normal."
    lines = ["Finding Locations (numbers on image are visual guides to help you locate each finding):"]
    for i, bbox in enumerate(bboxes, 1):
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        label = bbox.get("label", "Unknown")
        lines.append(f"  '{i}' on image = {label} at <bbx>[{x}, {y}, {w}, {h}]</bbx>")
    return "\n".join(lines)


def format_context(sample: Dict) -> str:
    diagnosis = sample.get("final_diagnosis", [])
    labels = sample.get("labels", [])
    return f"""MRI Type: Knee MRI (Coronal)
Final Diagnosis: {', '.join(diagnosis) if diagnosis else 'Not specified'}
Slice Labels: {', '.join(labels) if labels else 'None'}"""


def create_image_prompt(sample: Dict) -> str:
    bbox_text = format_bbox_coordinates(sample.get("bounding_boxes", []))
    context = format_context(sample)
    labels = sample.get("labels", [])
    label_list = ', '.join(labels) if labels else "Unknown"

    return f"""You are an expert musculoskeletal radiologist. You are given:
1. The original clean knee MRI image (use this to analyze visual features)
2. The same image with numbered bounding boxes (use this as a guide for finding locations)
3. Clinical context

Generate exactly 5 Q&A pairs with reasoning.

Rules:
1. Your reasoning must be based on what you see in the original image (signal intensity, shape, morphology, tear pattern).
2. The bounding box image only shows you where to look - your analysis must come from the clean original image.
3. Never use "Region 1", "Region 2", etc. Always use actual finding/label names: {label_list}
4. If multiple findings share a label, distinguish by anatomical location.
5. Radiological convention for coronal knee MRI (mandatory):
- The medial side of the knee has a deeper concavity on the tibial plateau and a larger femoral condyle. The lateral side has a more elevated tibial plateau and a smaller, rounder femoral condyle. If none of these are visibile then use other information such as which side the fibula is on (the fibula is always lateral) to determine which side of the image is medial vs lateral. If the fibula is not visible, use the structural rules above (condyle size, tibial plateau shape) to determine medial vs lateral.
- Always report findings using medial/lateral anatomical side for spatial location (e.g., "medial meniscus", "lateral femoral condyle").
## Context
{context}

## Finding locations (numbers on image are visual guides only)

{bbox_text}

## 5 tasks (each can use different question types)

Question types: closed_ended (Yes/No or short answer), single_choice (A/B/C/D), multiple_choice (select all that apply), open_ended (free text)

**Q1 - Detection**: Is there an injury? (closed_ended: Yes/No)
**Q2 - Localization**: Where is the finding? (open-ended: location and <bbx>[x,y,w,h]</bbx>, multiple if needed). If the image is normal, answer "No findings to localize" and reasoning should explain the image appears normal.
**Q3 - Classification**: What type of injury? What is the imaging modality? (single_choice, multiple_choice, or closed_ended)
**Q4 - Diagnosis**: What is the diagnosis? (single_choice or multiple_choice: A/B/C/D)
**Q5 - Captioning**: Describe the findings. (open_ended: free text)

## Output format:
```json
{{
    "qa_pairs": [


        {{
            "task": "detection",
            "type": "closed_ended",
            "question": "Is there evidence of injury in this knee MRI image?",
            "answer": "Yes",
            "reasoning": "In the [modality] knee MRI image, there is [visual description] in the [spatial location], which is suggestive of [finding name]."
        }},
        {{
            "task": "localization",
            "type": "closed_ended",
            "question": "Where is the [finding name] located?",
            "answer": "It is located in the medial compartment at the medial meniscus at <bbx>[x,y,w,h]</bbx>.
            "reasoning": "The [finding name] can be identified [spatial location]] at <bbx>[x,y,w,h]</bbx>.
            "task": "classification",
            "type": "single_choice",
            "question": "What type of injury is present?\\n(A) [option]\\n(B) [option]\\n(C) [option]\\n(D) [option]",
            "answer": "(A)",
            "reasoning": "The injury in the [spatial location], <bbx>[x,y,w,h]</bbx>, shows [visual features], which is suggestive of [answer]. [other option] is unlikely because [reason]. [other option] is unlikely because [reason]. [other option] is unlikely because [reason]. For other options, if the structure is not fully visualized or the degree of confidence in diagnosis is less than 90% then state [diagnosis] cannot be ruled in/out based on the single image alone.""
        }},
        {{
            "task": "diagnosis",
            "type": "single_choice",
            "question": "What is the most likely finding?\\n(A) [option]\\n(B) [option]\\n(C) [option]\\n(D) [option]",
            "answer": "(A)",
            "reasoning": "The finding in the [spatial location], <bbx>[x,y,w,h]</bbx>, shows [features], which is most consistent with [answer]. [other option] can be ruled out because [reason]. [other option] is less likely because [reason]. [other option] does not fit because [reason]. For other options, if the structure is not fully visualized or the degree of confidence in diagnosis is less than 90% then state [diagnosis] cannot be ruled in/out based on the single image alone.""
        }},
        {{
            "task": "captioning",
            "type": "open_ended",
            "question": "Describe the findings in this image.",
            "answer": "[Free text clinical summary describing: what was found, where it is located, how many, what type, and clinical significance]",
            "reasoning": "In a patient with [finding name], there is [finding] with [characteristics] in the [spatial location] at <bbx>[x,y,w,h]</bbx>. "
        }}


    ]
}}
```

Respond only with valid JSON.
"""


def create_volume_prompt(volume_samples: List[Dict]) -> str:
    final_diagnosis = volume_samples[0].get("volume_final_diagnosis") or volume_samples[0].get("final_diagnosis", [])

    all_labels = []
    all_bboxes = []
    slice_info_list = []

    for sample in volume_samples:
        labels = sample.get("labels", [])
        bboxes = sample.get("bounding_boxes", [])
        slice_num = sample.get("slice_num", 0)

        if labels and labels != ["Normal"]:
            all_labels.extend(labels)
        if bboxes:
            all_bboxes.extend(bboxes)
            bbox_strs = []
            for i, bbox in enumerate(bboxes, 1):
                x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("width", 0), bbox.get("height", 0)
                label = bbox.get("label", "lesion")
                bbox_strs.append(f"'{i}' on image = {label} at <bbx>[{x},{y},{w},{h}]</bbx>")
            slice_info_list.append(f"Frame {slice_num}: {', '.join(labels)} - {'; '.join(bbox_strs)}")

    unique_labels = list(set(all_labels)) if all_labels else ["Normal"]
    num_distinct_abnormalities = len(unique_labels) if unique_labels != ["Normal"] else 0
    num_affected_slices = len([s for s in volume_samples if s.get("bounding_boxes")])
    label_list = ', '.join(unique_labels)

    return f"""You are an expert musculoskeletal radiologist interpreting a knee MRI volume.
You are shown all {len(volume_samples)} slices (frames) from this volume, in order.
For each slice with findings, you receive:
  - The original clean image (use this to analyze visual features)
  - The same image with numbered bounding boxes (use this as a guide to know where findings are located)

Generate exactly 5 Q&A pairs following this diagnostic workflow.

Rules:
1. Your reasoning must be based on what you see in the original images (signal intensity, shape, morphology, tear pattern).
2. The bounding box images only show where to look - your analysis must come from the clean original images.
3. Never use "Region 1", "Region 2", etc. Always use actual finding/label names: {label_list}
4. If multiple findings share a label, distinguish by anatomical location or frame number.
5. Radiological convention for coronal knee MRI (mandatory):
- The medial side of the knee has a deeper concavity on the tibial plateau and a larger femoral condyle. The lateral side has a more elevated tibial plateau and a smaller, rounder femoral condyle. If none of these are visibile then use other information such as which side the fibula is on (the fibula is always lateral) to determine which side of the image is medial vs lateral. If the fibula is not visible, use the structural rules above (condyle size, tibial plateau shape) to determine medial vs lateral.
- Always report findings using medial/lateral anatomical side for spatial location (e.g., "medial meniscus", "lateral femoral condyle").

## Volume information
MRI Type: Knee MRI (Coronal)
Total Slices (Frames): {len(volume_samples)}
Slices with Findings: {num_affected_slices}
Distinct Abnormalities: {num_distinct_abnormalities} ({', '.join(unique_labels) if unique_labels != ["Normal"] else 'None'})

## Annotated diagnosis
{', '.join(final_diagnosis) if final_diagnosis else 'Not specified'}

## Findings by slice (numbers on images are visual guides only)
{chr(10).join(slice_info_list) if slice_info_list else 'No abnormal findings detected'}

## Labels
{label_list}

## 5 tasks (each can use different question types)

Question types: closed_ended (Yes/No or short answer), single_choice (A/B/C/D), multiple_choice (select all that apply), open_ended (free text)

**Q1 - Detection**: Is there an injury? (closed_ended: Yes/No)
**Q2 - Localization**: Where are the findings located across the volume? List all frames and bounding boxes for each finding. (open_ended: free text). If the volume is normal, answer "No findings to localize".
**Q3 - Counting**: How many distinct types of abnormalities are in this volume? Count by diagnosis (e.g. ACL tear and Meniscus tear = 2), not by how many frames or bounding boxes they appear in. (closed_ended: a number)
**Q4 - Classification**: What types of injuries are present in this volume? Since a volume can contain multiple injuries, use multiple_choice (select all that apply).
**Q5 - Captioning**: Comprehensive summary of findings. (open_ended: free text)
  - The answer must be a free text clinical summary describing: what was found, where it is located, how many, what type, and clinical significance.
  - The reasoning must be written as a paragraph. It should cover: what the volume shows, which frames are normal, what findings appear on which frames with <bbx> coordinates and visual evidence, how findings relate across frames, and an overall diagnostic impression.


## Output format:
```json
{{
    "qa_pairs": [
        {{
            "task": "detection",
            "type": "closed_ended",
            "question": "Is there evidence of [finding name] in this volume?",
            "answer": "Yes",
            "reasoning": "In the [modality] knee MRI volume, [finding name] is detected on frames [list of all affected frames] in the [spatial location]. [If multiple findings exist, list each with their affected frames.]"
        }},
        {{"task": "counting",
            "type": "closed_ended",
            "question": "How many distinct types of injuries are present in this volume?",
            "answer": "[number]",
            "reasoning": "There are [number] distinct injuries in this volume: [injury 1] seen on frames [list], and [injury 2] seen on frames [list]. Total = [number] distinct injuries."}},
        {{
            "task": "localization",
            "type": "open_ended",
            "question": "Where are the injuries located in this volume?",
            "answer": "[injury 1] is located in the [spatial location] on frames [list]: frame N <bbx>[x,y,w,h]</bbx>, frame M <bbx>[x,y,w,h]</bbx>. [injury 2] is located in the [spatial location] on frames [list]: frame N <bbx>[x,y,w,h]</bbx>, frame M <bbx>[x,y,w,h]</bbx>.",
            "reasoning": "[injury 1] appears on frames [list] in the [spatial location], spanning [extent]. [injury 2] appears on frames [list] in the [spatial location]."
        }},
        {{
            "task": "classification",
            "type": "multiple_choice",
            "question": "What types of injuries are present in this volume? (Select all that apply)\\n(A) [option]\\n(B) [option]\\n(C) [option]\\n(D) [option]",
            "answer": "(A), (B)",
            "reasoning": "Across frames [list], [injury 1] is seen in the [spatial location], <bbx>[x,y,w,h]</bbx>, showing [features]. [injury 2] is seen in the [spatial location], <bbx>[x,y,w,h]</bbx>, showing [features]. [other option] is not present because [reason]. For other options, if the structure is not fully visualized or the degree of confidence in diagnosis is less than 90% then state [diagnosis] cannot be ruled in/out based on the single image alone."
        }},
        {{
            "task": "captioning",
            "type": "open_ended",
            "question": "Provide a comprehensive summary of the findings in this volume.",
            "answer": "[What was found, where it is located, how many, what type, and clinical significance]",
            "reasoning": "[Free text paragraph covering: what the volume shows, which frames are normal, what findings appear on which frames with <bbx> coordinates and visual evidence, how findings relate across frames, and an overall diagnostic impression. Must not be N/A.]"
        }}
    ]
}}
```

Respond only with valid JSON.
"""


@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(3))
def call_gpt4o(images: List[Image.Image], prompt: str, client: openai.OpenAI, model: str = "gpt-4o", sample_info: Dict = None, max_tokens: int = MAX_TOKENS, timeout: int = 120) -> Dict:
    sample_id = f"{sample_info.get('volume_id', 'unknown')}_{sample_info.get('slice_num', 0)}" if sample_info else "unknown"

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_to_base64(img)}", "detail": "high"}}
        for img in images
    ]
    content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "system",
            "content": "You are an expert musculoskeletal radiologist. Generate high-quality Q&A pairs for medical imaging education and evaluation."
        },
        {"role": "user", "content": content}
    ]

    try:
        start_time = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            timeout=timeout,
        )

        duration = time.time() - start_time
        result_text = response.choices[0].message.content.strip()

        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(result_text)

        log_entry = {
            "sample_id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "duration": round(duration, 2),
            "status": "success",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "qa_pairs_generated": len(parsed.get("qa_pairs", [])),
        }
        logging.info(json.dumps(log_entry))

        return {
            "success": True,
            "qa_pairs": parsed.get("qa_pairs", []),
            "tokens": {"prompt": response.usage.prompt_tokens, "completion": response.usage.completion_tokens},
        }

    except openai.RateLimitError as e:
        log_entry = {
            "sample_id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "status": "error",
            "error_type": "rate_limit",
            "error": str(e),
        }
        logging.info(json.dumps(log_entry))
        print(f"\nRate limit hit for {sample_id}. Waiting before retry...", flush=True)
        raise

    except openai.APIError as e:
        log_entry = {
            "sample_id": sample_id,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "status": "error",
            "error_type": "api_error",
            "error": str(e),
        }
        logging.info(json.dumps(log_entry))
        print(f"\nAPI error for {sample_id}: {str(e)}", flush=True)
        raise

    except Exception as e:
        logging.error(json.dumps({"error": str(e), "sample_info": sample_info}))
        return {"success": False, "error": str(e), "qa_pairs": []}


def load_samples(json_path: str, data_root: str, max_samples: int = None,
                 include_normal: bool = False, normal_sample_size: int = None) -> List[Dict]:
    """Knee uses flat JSON structure (no modality nesting)."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    diseased_samples = []
    normal_samples = []

    for volume_id, volume_data in data.items():
        diagnosis = volume_data.get("final_diagnosis", [])

        for slice_info in volume_data.get("slices", []):
            has_bboxes = bool(slice_info.get("bounding_boxes"))

            img_path = slice_info.get("image_path", "")
            if data_root:
                img_path = os.path.abspath(os.path.join(data_root, img_path))

            if not os.path.exists(img_path):
                continue

            sample = {
                "volume_id": volume_id,
                "modality": "knee",
                "slice_num": slice_info["slice"],
                "image_path": img_path,
                "labels": slice_info.get("label", []) if has_bboxes else ["Normal"],
                "bounding_boxes": slice_info.get("bounding_boxes", []),
                "final_diagnosis": diagnosis if has_bboxes else ["Normal"],
            }

            if has_bboxes:
                diseased_samples.append(sample)
            else:
                normal_samples.append(sample)

    random.shuffle(diseased_samples)
    if max_samples and normal_sample_size:
        diseased_limit = max_samples - normal_sample_size
        samples = diseased_samples[:diseased_limit]
    elif max_samples:
        samples = diseased_samples[:max_samples]
    else:
        samples = diseased_samples.copy()

    if include_normal or normal_sample_size:
        if normal_sample_size and normal_sample_size < len(normal_samples):
            sampled_normal = random.sample(normal_samples, normal_sample_size)
            print(f"Sampled {normal_sample_size} normal slices from {len(normal_samples)} available")
            samples.extend(sampled_normal)
        elif include_normal:
            samples.extend(normal_samples)

    random.shuffle(samples)

    diseased = sum(1 for s in samples if s["bounding_boxes"])
    print(f"Loaded {len(samples)} samples ({diseased} diseased, {len(samples) - diseased} normal)")
    return samples


def load_full_volumes(json_path: str, data_root: str, volume_ids: set) -> Dict[str, List[Dict]]:
    """Load all slices for specified volumes (not just sampled ones)."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    volumes = {}
    for volume_id, volume_data in data.items():
        if volume_id not in volume_ids:
            continue

        diagnosis = volume_data.get("final_diagnosis", [])
        slices = []

        for slice_info in volume_data.get("slices", []):
            img_path = slice_info.get("image_path", "")
            if data_root:
                img_path = os.path.abspath(os.path.join(data_root, img_path))

            if not os.path.exists(img_path):
                continue

            has_bboxes = bool(slice_info.get("bounding_boxes"))
            sample = {
                "volume_id": volume_id,
                "modality": "knee",
                "slice_num": slice_info["slice"],
                "image_path": img_path,
                "labels": slice_info.get("label", []) if has_bboxes else ["Normal"],
                "bounding_boxes": slice_info.get("bounding_boxes", []),
                "final_diagnosis": diagnosis if has_bboxes else ["Normal"],
                "volume_final_diagnosis": diagnosis,
            }
            slices.append(sample)

        if slices:
            volumes[volume_id] = sorted(slices, key=lambda x: x["slice_num"])

    return volumes


def main(args):
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or use --api-key")

    client = openai.OpenAI(api_key=api_key)

    samples = load_samples(
        json_path=args.input,
        data_root=args.data_root,
        max_samples=args.max_samples,
        include_normal=args.include_normal,
        normal_sample_size=args.normal_sample_size,
    )

    if not samples:
        print("No samples found!")
        return

    print(f"\nGenerating Q&A pairs using {args.model}")
    print(f"Samples: {len(samples)}")
    print("-" * 50)

    image_results = []
    slice_qa_pairs = []

    if not args.volume_only:
        print("\n--- Image-Level Q&A Generation ---")
        for sample in tqdm(samples, desc="Processing slices"):
            orig_img = Image.open(sample["image_path"]).convert("RGB")

            images = [orig_img]
            if sample["bounding_boxes"]:
                images.append(draw_bboxes(orig_img, sample["bounding_boxes"]))

            prompt = create_image_prompt(sample)

            result = call_gpt4o(
                images, prompt, client, args.model,
                sample_info={"volume_id": sample["volume_id"], "slice_num": sample["slice_num"]}
            )

            sample_result = {
                "volume_id": sample["volume_id"],
                "slice_num": sample["slice_num"],
                "modality": sample["modality"],
                "image_path": sample["image_path"],
                "labels": sample["labels"],
                "bounding_boxes": sample["bounding_boxes"],
                "final_diagnosis": sample["final_diagnosis"],
                "success": result["success"],
                "qa_pairs": result.get("qa_pairs", []),
            }
            image_results.append(sample_result)

            if result["success"]:
                for qa in result.get("qa_pairs", []):
                    qa_entry = {
                        "volume_id": sample["volume_id"],
                        "slice_num": sample["slice_num"],
                        "modality": sample["modality"],
                        "image_path": sample["image_path"],
                        "level": "slice",
                        "task": qa.get("task", "unknown"),
                        "type": qa.get("type", "unknown"),
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "reasoning": qa.get("reasoning", ""),
                        "ground_truth": {
                            "labels": sample["labels"],
                            "diagnosis": sample["final_diagnosis"],
                            "bboxes": sample["bounding_boxes"],
                        },
                    }
                    slice_qa_pairs.append(qa_entry)

            time.sleep(1)
    else:
        print("\n--- Skipping Image-Level Q&A (volume-only mode) ---")

    volume_results = []
    volume_qa_pairs = []

    if args.image_only:
        print("\n--- Skipping Volume-Level Q&A (image-only mode) ---")
    else:
        print("\n--- Volume-Level Q&A Generation ---")

    if not args.image_only:
        sampled_volume_ids = set(s["volume_id"] for s in samples)
        full_volumes = load_full_volumes(args.input, args.data_root, sampled_volume_ids)
        print(f"Loaded full volumes: {', '.join(f'{vid} ({len(slices)} slices)' for vid, slices in full_volumes.items())}")

        for volume_id, vol_samples in tqdm(full_volumes.items(), desc="Processing volumes"):

            images = []
            for s in vol_samples:
                orig_img = Image.open(s["image_path"]).convert("RGB")
                images.append(orig_img)
                if s["bounding_boxes"]:
                    images.append(draw_bboxes(orig_img, s["bounding_boxes"]))

            prompt = create_volume_prompt(vol_samples)

            result = call_gpt4o(
                images, prompt, client, args.model,
                sample_info={"volume_id": volume_id, "type": "volume"},
                max_tokens=VOLUME_MAX_TOKENS,
            )

            volume_diagnosis = vol_samples[0].get("volume_final_diagnosis") or vol_samples[0].get("final_diagnosis", [])
            vol_result = {
                "volume_id": volume_id,
                "modality": vol_samples[0]["modality"],
                "num_slices": len(vol_samples),
                "final_diagnosis": volume_diagnosis,
                "slices": [{"slice_num": s["slice_num"], "labels": s["labels"], "bounding_boxes": s["bounding_boxes"]} for s in vol_samples],
                "success": result["success"],
                "qa_pairs": result.get("qa_pairs", []),
            }
            volume_results.append(vol_result)

            if result["success"]:
                for qa in result.get("qa_pairs", []):
                    qa_entry = {
                        "volume_id": volume_id,
                        "modality": vol_samples[0]["modality"],
                        "num_slices": len(vol_samples),
                        "level": "volume",
                        "step": qa.get("step", 0),
                        "task": qa.get("task", "unknown"),
                        "type": qa.get("type", "unknown"),
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "reasoning": qa.get("reasoning", ""),
                        "ground_truth": {
                            "final_diagnosis": volume_diagnosis,
                        },
                    }
                    volume_qa_pairs.append(qa_entry)

            time.sleep(2)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    slice_output = args.output.replace(".json", "_slice_qa_pairs.json")
    with open(slice_output, 'w') as f:
        json.dump(slice_qa_pairs, f, indent=2)

    volume_output = args.output.replace(".json", "_volume_qa_pairs.json")
    with open(volume_output, 'w') as f:
        json.dump(volume_qa_pairs, f, indent=2)

    full_output = {
        "model": args.model,
        "image_level": image_results,
        "volume_level": volume_results,
    }
    with open(args.output, 'w') as f:
        json.dump(full_output, f, indent=2)

    slice_ok = sum(1 for r in image_results if r["success"])
    vol_ok = sum(1 for r in volume_results if r["success"])

    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Slices:  {slice_ok}/{len(image_results)} successful | {len(slice_qa_pairs)} Q&A pairs")
    print(f"  Volumes: {vol_ok}/{len(volume_results)} successful | {len(volume_qa_pairs)} Q&A pairs")
    print(f"\nOutput files:")
    print(f"  Slice Q&A:  {slice_output}")
    print(f"  Volume Q&A: {volume_output}")
    print(f"  Full JSON:  {args.output}")
    print(f"  API log:    {log_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Q&A pairs using GPT-4o for Knee MRI")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--data-root", required=True, help="Data root directory")
    parser.add_argument("--output", default="gpt4o_knee_qa.json", help="Output file")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--model", default="gpt-4o",
                        choices=["gpt-4o", "gpt-4o-mini"])
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--include-normal", action="store_true", default=False)
    parser.add_argument("--normal-sample-size", type=int, default=None)
    parser.add_argument("--volume-only", action="store_true", default=False,
                        help="Skip slice-level Q&A, only generate volume-level")
    parser.add_argument("--image-only", action="store_true", default=False,
                        help="Skip volume-level Q&A, only generate image/slice-level")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save checkpoint every N samples (0 to disable)")
    parser.add_argument("--api-timeout", type=int, default=120,
                        help="Timeout in seconds for each API call")

    args = parser.parse_args()
    main(args)
