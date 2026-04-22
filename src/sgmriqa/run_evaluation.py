"""Main evaluation script: computes AR/A/V scores on inference results.

Supports eval-mode-aware scoring:
    - image mode: image-level QA (A-Score, AR-Score) + image-level grounding (V-Score)
    - video mode: video-level QA (A-Score, AR-Score) + video-level grounding (V-Score)
    - all mode: runs both image and video

Usage:
    python -m sgmriqa.run_evaluation --models gpt-4o-mini --eval-mode image
    python -m sgmriqa.run_evaluation --models gpt-4o-mini --eval-mode video
    python -m sgmriqa.run_evaluation --models gpt-4o-mini gemini-2.0-flash --eval-mode all --skip-ar
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

from sgmriqa.config.paths import ensure_dirs, get_evaluation_dir, get_inference_dir
from sgmriqa.metrics.a_score import compute_a_score
from sgmriqa.metrics.ar_score import compute_ar_score
from sgmriqa.metrics.utils import (
    extract_answer_text,
    parse_bboxes_with_frames,
    parse_structured_output,
)
from sgmriqa.metrics.v_score import compute_v_score

# Image dimensions per dataset (for denormalizing Qwen 0-1000 bbox coordinates)
_IMAGE_DIMS = {
    "brain": (256, 256),
    "knee": (320, 320),
}


def _strip_bbox_text(text: str) -> str:
    """Remove bbox coordinates and frame references from text for AR-Score.

    Strips <bbx>[x,y,w,h]</bbx> tags, bare [x,y,w,h] coordinate lists,
    and frame references so GPT judge evaluates anatomical descriptions
    (e.g. 'right hemisphere', 'medial compartment') not numbers.
    """
    # Remove <bbx>[...]</bbx> tags
    text = re.sub(r"<bbx>\[[\d,\s]+\]</bbx>", "", text)
    # Remove bare bbox-like coordinate lists: [123, 45, 67, 89]
    text = re.sub(r"\[\d+,\s*\d+,\s*\d+,\s*\d+\]", "", text)
    # Remove "on Frame(s) N-M:" or "on Frame N, N, N:" with trailing bbox refs
    text = re.sub(r"\s+on\s+Frames?\s+[\d\-–,\s]+(?::\s*)?", " ", text)
    # Remove standalone "Frame N:" prefixes
    text = re.sub(r"Frame\s+\d+:\s*", "", text)
    # Remove "observed on Frame N" / "appears on Frame N-M" / "appears on Frame N, N, and N"
    text = re.sub(r"(?:observed|appears|seen)\s+on\s+Frames?\s+[\d\-–,\s]+(?:and\s+\d+)?", "", text)
    # Clean up leftover punctuation and whitespace
    text = re.sub(r"\s*,\s*(?:and\s+)?\.", ".", text)  # ", ." → "."
    text = re.sub(r",(\s*,)+", ",", text)               # repeated commas
    text = re.sub(r",\s*\.", ".", text)                  # ",." → "."
    text = re.sub(r"\s{2,}", " ", text)                  # extra spaces
    text = re.sub(r"\s+\.", ".", text)                   # space before period
    return text.strip().rstrip(",. ")


def load_inference_results(model_key: str, eval_mode: str = "image", inference_dir: str = None) -> dict:
    """Load inference results JSON for a model.

    Tries the new filename ({model}_inference.json in mode-specific dir) first,
    then falls back to the legacy filename ({model}_{mode}_inference.json).
    """
    inference_dir = inference_dir or get_inference_dir(eval_mode)

    # Primary: new layout — {model}_inference.json in mode-specific dir
    primary_path = os.path.join(inference_dir, f"{model_key}_inference.json")
    if os.path.exists(primary_path):
        with open(primary_path, "r") as f:
            return json.load(f)

    # Fallback: legacy filename with mode in name
    legacy_path = os.path.join(inference_dir, f"{model_key}_{eval_mode}_inference.json")
    if os.path.exists(legacy_path):
        with open(legacy_path, "r") as f:
            return json.load(f)

    raise FileNotFoundError(
        f"Inference results not found for {model_key} (tried {primary_path} and {legacy_path})"
    )


def _extract_video_answer(prediction: str, qa_type: str) -> str:
    """Extract the final answer from a step-by-step video QA response.

    Looks for the pattern "Therefore, the final answer is: X" and returns X.
    If the pattern is not found, returns the full prediction as fallback.

    Args:
        prediction: The model's full response text.
        qa_type: The question type (single_choice, multiple_choice, open_ended, etc.).

    Returns:
        The extracted answer string.
    """
    # Match "Therefore, the final answer is: <answer>"
    pattern = r"[Tt]herefore,?\s+the\s+final\s+answer\s+is:\s*(.+)"
    match = re.search(pattern, prediction)
    if match:
        answer = match.group(1).strip().rstrip(".")
        # Strip markdown bold markers
        answer = re.sub(r"\*{1,2}", "", answer).strip()

        # For MCQ, extract just the letter
        if qa_type in ("single_choice", "multiple_choice"):
            letter_match = re.match(r"^([A-D])\b", answer)
            if letter_match:
                return letter_match.group(1)

        # For closed-ended (yes/no), extract just the first word
        if qa_type == "closed_ended":
            first_word = answer.split()[0].strip(".,!;:") if answer.split() else answer
            if first_word.lower() in ("yes", "no"):
                return first_word

        return answer

    return prediction


def evaluate_model(
    model_key: str,
    eval_mode: str = "image",
    inference_dir: str = None,
    output_dir: str = None,
    skip_ar: bool = False,
    skip_v: bool = False,
    gpt_judge_model: str = "gpt-4o",
    semantic_model_name: str = "all-MiniLM-L6-v2",
    iou_threshold: float = 0.5,
    semantic_model=None,
):
    """Evaluate a single model's inference results.

    Args:
        model_key: Model key.
        eval_mode: Which inference results to load ("image" or "video").
        inference_dir: Directory containing inference JSONs.
        output_dir: Directory to save evaluation JSONs.
        skip_ar: Skip AR-Score computation (saves API costs).
        skip_v: Skip V-Score computation.
        gpt_judge_model: Model for GPT-based judging in AR-Score (default: gpt-4o).
        semantic_model_name: Sentence-transformers model for A-Score.
        iou_threshold: IoU threshold for V-Score.
        semantic_model: Pre-loaded semantic model (avoids reloading across modes).
    """
    inference_dir = inference_dir or get_inference_dir(eval_mode)
    output_dir = output_dir or get_evaluation_dir(eval_mode)

    # Load inference results
    data = load_inference_results(model_key, eval_mode, inference_dir)
    results = data.get("results", [])

    if not results:
        logging.warning(f"No results found for {model_key}")
        return

    logging.info(f"Evaluating {model_key}: {len(results)} samples (mode={eval_mode})")

    # Load semantic model for A-Score if not provided
    if semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            semantic_model = SentenceTransformer(semantic_model_name)
            logging.info(f"Loaded semantic model: {semantic_model_name}")
        except ImportError:
            logging.warning(
                "sentence-transformers not installed; open-ended A-Score will use word overlap"
            )

    evaluated = []
    score_sums = defaultdict(float)
    score_counts = defaultdict(int)

    # Deduplicate localization: skip "Locate the bounding box..." phrasing
    # which duplicates the "Where is..." question for the same finding.
    dedup_skipped = 0
    for item in tqdm(results, desc=f"Scoring {model_key}"):
        task = item.get("task", "")
        question = item.get("question", "")
        if task == "localization" and question.startswith("Locate the bounding box"):
            dedup_skipped += 1
            continue

        prediction = extract_answer_text(item.get("model_output", ""))
        gt_answer = item.get("gt_answer", "")
        gt_reasoning = item.get("gt_reasoning", "")
        qa_type = item.get("qa_type", "open_ended")
        labels = item.get("labels", [])
        bboxes = item.get("bboxes", [])
        sample_id = item.get("sample_id", "")

        # Determine sample category from ID tags
        is_grounding = "_ground_" in sample_id
        is_qa = "_qa_" in sample_id
        is_video = "_video_" in sample_id

        # Try structured JSON parsing first, fall back to legacy
        parsed = parse_structured_output(prediction)
        if parsed:
            answer_text = parsed["answer"]
            reasoning_text = parsed["reasoning"]
            pred_bboxes = parsed["bboxes"]
        else:
            # Legacy fallback
            answer_text = (
                _extract_video_answer(prediction, qa_type)
                if is_video
                else prediction
            )
            reasoning_text = prediction
            pred_bboxes = None

        # For localization, try to extract bboxes from free-text output
        # when structured parsing failed or returned no bboxes
        task = item.get("task", "")
        if task == "localization" and not pred_bboxes:
            dataset = item.get("dataset", "")
            img_w, img_h = _IMAGE_DIMS.get(dataset, (None, None))
            freetext_bboxes = parse_bboxes_with_frames(
                prediction, image_width=img_w, image_height=img_h,
            )
            if freetext_bboxes:
                pred_bboxes = [
                    {"frame": frame or 1, "label": "finding",
                     "bbox": bb}
                    for frame, bb in freetext_bboxes
                ]

        scores = {}

        # --- A-Score (for QA samples with deterministic answers) ---
        # Skip open-ended (captioning) and localization — captioning uses AR-Score,
        # localization uses AR-Score (text) + V-Score (bboxes).
        if is_qa and qa_type != "open_ended" and task != "localization":
            a_result = compute_a_score(
                prediction=answer_text,
                gt_answer=gt_answer,
                qa_type=qa_type,
                gt_labels=labels,
                semantic_model=semantic_model,
            )
            scores["a_score"] = a_result

        # --- AR-Score (for captioning + localization) ---
        # Uses GPT-4o judge + NLG metrics for answer + reasoning quality.
        # For localization, strip bbox coordinates so judge evaluates the
        # anatomical description ("right hemisphere") not the numbers.
        if not skip_ar and is_qa and task in ("captioning", "localization"):
            ar_pred = reasoning_text
            ar_gt_answer = gt_answer
            ar_gt_reasoning = gt_reasoning
            if task == "localization":
                ar_pred = _strip_bbox_text(ar_pred)
                ar_gt_answer = _strip_bbox_text(ar_gt_answer)
                ar_gt_reasoning = _strip_bbox_text(ar_gt_reasoning)
            ar_result = compute_ar_score(
                prediction=ar_pred,
                gt_answer=ar_gt_answer,
                gt_reasoning=ar_gt_reasoning,
                gpt_judge_model=gpt_judge_model,
            )
            scores["ar_score"] = ar_result

        # --- V-Score (mIoU for localization only) ---
        # For video localization, GT bboxes may be embedded in gt_answer text
        # (e.g. "Frame 7: <bbx>[128,160,7,8]</bbx>") rather than in a separate
        # bboxes field. Parse them on the fly when the field is empty.
        if not bboxes and task == "localization" and "<bbx>" in gt_answer:
            dataset = item.get("dataset", "")
            img_w, img_h = _IMAGE_DIMS.get(dataset, (None, None))
            parsed_gt = parse_bboxes_with_frames(gt_answer, image_width=img_w, image_height=img_h)
            bboxes = [
                {"x": bb[0], "y": bb[1], "width": bb[2], "height": bb[3],
                 **({"frame": frame} if frame is not None else {})}
                for frame, bb in parsed_gt
            ]

        if not skip_v and task == "localization" and bboxes:
            dataset = item.get("dataset", "")
            img_w, img_h = _IMAGE_DIMS.get(dataset, (None, None))

            # Denormalize predicted bboxes before V-Score comparison.
            # Note: Qwen3 0-1000 denorm + xyxy→xywh is handled in parse_bboxes_with_frames.
            # Only Gemini [0,1] from structured JSON needs handling here.
            if pred_bboxes and img_w and img_h:
                denorm_bboxes = []
                for b in pred_bboxes:
                    b_copy = dict(b)
                    bb = b_copy.get("bbox", [])
                    if b_copy.get("normalized_01") and len(bb) >= 4:
                        # Gemini: (min_x, min_y, max_x, max_y) in [0,1]
                        min_x = min(max(bb[0], 0.0), 1.0)
                        min_y = min(max(bb[1], 0.0), 1.0)
                        max_x = min(max(bb[2], 0.0), 1.0)
                        max_y = min(max(bb[3], 0.0), 1.0)
                        b_copy["bbox"] = [
                            min_x * img_w,
                            min_y * img_h,
                            (max_x - min_x) * img_w,
                            (max_y - min_y) * img_h,
                        ]
                        b_copy.pop("normalized_01", None)
                    denorm_bboxes.append(b_copy)
                pred_bboxes = denorm_bboxes

            v_result = compute_v_score(
                prediction=prediction,
                gt_bboxes_raw=bboxes,
                iou_threshold=iou_threshold,
                pred_bboxes_parsed=pred_bboxes,
                image_width=img_w,
                image_height=img_h,
            )
            scores["v_score"] = v_result

        # Accumulate for aggregation
        for metric in ["a_score", "ar_score", "v_score", "map"]:
            if metric in scores:
                val = scores[metric].get(metric)
                if val is not None:
                    score_sums[metric] += val
                    score_counts[metric] += 1
            # map lives inside v_score result dict
            elif metric == "map" and "v_score" in scores:
                val = scores["v_score"].get("map")
                if val is not None:
                    score_sums["map"] += val
                    score_counts["map"] += 1

        # Accumulate per-threshold detection metrics from v_score
        if "v_score" in scores:
            det = scores["v_score"].get("details", {}).get("detection_metrics", {})
            for k, v in det.items():
                score_sums[f"det_{k}"] += v
                score_counts[f"det_{k}"] += 1

        evaluated.append({
            "sample_id": sample_id,
            "dataset": item.get("dataset", ""),
            "level": item.get("level", ""),
            "qa_type": qa_type,
            "task": item.get("task", ""),
            "model_output": prediction,
            "gt_answer": gt_answer,
            "scores": scores,
        })

    if dedup_skipped:
        logging.info(f"Dedup: skipped {dedup_skipped} duplicate localization questions")

    # Compute aggregates
    aggregates = {}
    for metric in ["a_score", "ar_score", "v_score", "map"]:
        if score_counts[metric] > 0:
            aggregates[metric] = {
                "mean": score_sums[metric] / score_counts[metric],
                "count": score_counts[metric],
            }

    # Detection metrics at each IoU threshold
    det_agg = {}
    for k in ["P@0.1", "R@0.1", "F1@0.1", "P@0.25", "R@0.25", "F1@0.25",
              "P@0.5", "R@0.5", "F1@0.5"]:
        dk = f"det_{k}"
        if score_counts.get(dk, 0) > 0:
            det_agg[k] = score_sums[dk] / score_counts[dk]
    if det_agg:
        aggregates["detection"] = det_agg

    # Breakdown by dataset, level, qa_type, task
    breakdowns = _compute_breakdowns(evaluated)

    # Save
    output_data = {
        "model_key": model_key,
        "model_name": data.get("model_name", ""),
        "model_id": data.get("model_id", ""),
        "eval_mode": eval_mode,
        "gpt_judge_model": gpt_judge_model,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(evaluated),
        "aggregates": aggregates,
        "breakdowns": breakdowns,
        "results": evaluated,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_key}_{eval_mode}_evaluation.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logging.info(f"Saved evaluation to {output_path}")
    logging.info(f"Aggregates: {json.dumps(aggregates, indent=2)}")


def _compute_breakdowns(evaluated: list) -> dict:
    """Compute metric breakdowns by dataset, level, qa_type, task."""
    breakdowns = {}

    for group_key in ["dataset", "level", "qa_type", "task"]:
        groups = defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0}))

        for item in evaluated:
            group_val = item.get(group_key, "unknown")
            for metric in ["a_score", "ar_score", "v_score", "map"]:
                if metric in item["scores"]:
                    val = item["scores"][metric].get(metric)
                    if val is not None:
                        groups[group_val][metric]["sum"] += val
                        groups[group_val][metric]["count"] += 1
                elif metric == "map" and "v_score" in item["scores"]:
                    val = item["scores"]["v_score"].get("map")
                    if val is not None:
                        groups[group_val][metric]["sum"] += val
                        groups[group_val][metric]["count"] += 1

        breakdown = {}
        for group_val, metrics in groups.items():
            breakdown[group_val] = {}
            for metric, vals in metrics.items():
                if vals["count"] > 0:
                    breakdown[group_val][metric] = {
                        "mean": vals["sum"] / vals["count"],
                        "count": vals["count"],
                    }
        breakdowns[group_key] = breakdown

    return breakdowns


def _run_single_mode(args, eval_mode: str, semantic_model=None):
    """Evaluate all models for a single eval mode."""
    for model_key in args.models:
        try:
            evaluate_model(
                model_key=model_key,
                eval_mode=eval_mode,
                inference_dir=args.inference_dir,
                output_dir=args.output_dir,
                skip_ar=args.skip_ar,
                skip_v=args.skip_v,
                gpt_judge_model=args.gpt_judge_model,
                semantic_model_name=args.semantic_model,
                iou_threshold=args.iou_threshold,
                semantic_model=semantic_model,
            )
        except FileNotFoundError as e:
            logging.error(str(e))
        except Exception as e:
            logging.error(f"Failed to evaluate {model_key}: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM inference results")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model keys to evaluate",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["image", "video", "all"],
        default="all",
        help="Which inference results to evaluate",
    )
    parser.add_argument(
        "--inference-dir", default=None, help="Override inference directory"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Override output directory"
    )
    parser.add_argument(
        "--skip-ar",
        action="store_true",
        help="Skip AR-Score (saves API costs)",
    )
    parser.add_argument(
        "--skip-v", action="store_true", help="Skip V-Score"
    )
    parser.add_argument(
        "--gpt-judge-model",
        default="gpt-4o",
        help="GPT model for AR-Score judging (default: gpt-4o per paper)",
    )
    parser.add_argument(
        "--semantic-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for A-Score",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for V-Score",
    )

    args = parser.parse_args()
    ensure_dirs()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Pre-load semantic model once (both modes have QA samples)
    semantic_model = None
    try:
        from sentence_transformers import SentenceTransformer
        semantic_model = SentenceTransformer(args.semantic_model)
        logging.info(f"Loaded semantic model: {args.semantic_model}")
    except ImportError:
        logging.warning(
            "sentence-transformers not installed; open-ended A-Score will use word overlap"
        )

    # Dispatch by eval mode
    if args.eval_mode == "all":
        for mode in ["image", "video"]:
            _run_single_mode(args, mode, semantic_model=semantic_model)
    else:
        _run_single_mode(args, args.eval_mode, semantic_model=semantic_model)


if __name__ == "__main__":
    main()
