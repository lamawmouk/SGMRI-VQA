"""Compute AR-Score only and merge into existing evaluation files.

Avoids re-running A-Score and V-Score. Reads gt_reasoning from inference files,
computes AR-Score for captioning + localization tasks, and merges results.

Usage:
    python -m sgmriqa.run_ar_only --models gpt-4o gpt-4o-mini --eval-mode image
    python -m sgmriqa.run_ar_only --models gpt-4o --eval-mode all
    python -m sgmriqa.run_ar_only --all-models --eval-mode all
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
from sgmriqa.metrics.ar_score import compute_ar_score
from sgmriqa.run_evaluation import _strip_bbox_text, _compute_breakdowns

ALL_MODELS = [
    "gpt-4o", "gpt-4o-mini", "gemini-2.5-pro", "gemini-2.5-flash",
    "qwen3-vl-8b", "qwen3-vl-8b-mri", "qwen2.5-vl-7b",
    "llava-video-7b", "internvl2.5-8b", "eagle2.5-8b",
]


def run_ar_score(model_key: str, eval_mode: str, gpt_judge_model: str = "gpt-4o"):
    """Compute AR-Score and merge into existing evaluation file."""
    inference_dir = get_inference_dir(eval_mode)
    eval_dir = get_evaluation_dir(eval_mode)

    # Load inference (has gt_reasoning)
    inf_path = os.path.join(inference_dir, f"{model_key}_inference.json")
    if not os.path.exists(inf_path):
        logging.warning(f"No inference file for {model_key} ({eval_mode})")
        return
    inf_data = json.load(open(inf_path))
    inf_results = inf_data.get("results", [])

    # Build lookup: sample_id -> gt_reasoning
    gt_reasoning_map = {}
    for r in inf_results:
        sid = r.get("sample_id", "")
        gt_reasoning_map[sid] = r.get("gt_reasoning", "")

    # Load existing evaluation
    eval_path = os.path.join(eval_dir, f"{model_key}_{eval_mode}_evaluation.json")
    if not os.path.exists(eval_path):
        logging.warning(f"No evaluation file for {model_key} ({eval_mode})")
        return
    eval_data = json.load(open(eval_path))
    results = eval_data.get("results", [])

    # Find samples needing AR-Score
    eligible = []
    for i, r in enumerate(results):
        task = r.get("task", "")
        sample_id = r.get("sample_id", "")
        is_qa = "_qa_" in sample_id
        if is_qa and task in ("captioning", "localization"):
            if "ar_score" not in r.get("scores", {}):
                eligible.append(i)

    if not eligible:
        logging.info(f"{model_key} ({eval_mode}): All AR-Scores already computed, skipping")
        return

    logging.info(f"{model_key} ({eval_mode}): Computing AR-Score for {len(eligible)} samples")

    computed = 0
    for idx in tqdm(eligible, desc=f"AR-Score {model_key} ({eval_mode})"):
        r = results[idx]
        sample_id = r.get("sample_id", "")
        task = r.get("task", "")
        prediction = r.get("model_output", "")
        gt_answer = r.get("gt_answer", "")
        gt_reasoning = gt_reasoning_map.get(sample_id, "")

        ar_pred = prediction
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
        results[idx]["scores"]["ar_score"] = ar_result
        computed += 1

    logging.info(f"{model_key} ({eval_mode}): Computed {computed} AR-Scores")

    # Recompute aggregates and breakdowns
    score_sums = defaultdict(float)
    score_counts = defaultdict(int)
    for r in results:
        for metric in ["a_score", "ar_score", "v_score", "map"]:
            if metric in r.get("scores", {}):
                val = r["scores"][metric].get(metric)
                if val is not None:
                    score_sums[metric] += val
                    score_counts[metric] += 1
            elif metric == "map" and "v_score" in r.get("scores", {}):
                val = r["scores"]["v_score"].get("map")
                if val is not None:
                    score_sums["map"] += val
                    score_counts["map"] += 1

    aggregates = {}
    for metric in ["a_score", "ar_score", "v_score", "map"]:
        if score_counts[metric] > 0:
            aggregates[metric] = {
                "mean": score_sums[metric] / score_counts[metric],
                "count": score_counts[metric],
            }

    # Preserve detection metrics from existing aggregates
    if "detection" in eval_data.get("aggregates", {}):
        aggregates["detection"] = eval_data["aggregates"]["detection"]

    breakdowns = _compute_breakdowns(results)

    # Update and save
    eval_data["aggregates"] = aggregates
    eval_data["breakdowns"] = breakdowns
    eval_data["results"] = results
    eval_data["ar_score_timestamp"] = datetime.now().isoformat()
    eval_data["gpt_judge_model"] = gpt_judge_model

    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    ar_agg = aggregates.get("ar_score", {})
    logging.info(
        f"Saved {eval_path} — AR-Score: {ar_agg.get('mean', 'N/A'):.4f} "
        f"(n={ar_agg.get('count', 0)})"
    )


def main():
    parser = argparse.ArgumentParser(description="Compute AR-Score only (merge into existing eval)")
    parser.add_argument("--models", nargs="+", help="Model keys to evaluate")
    parser.add_argument("--all-models", action="store_true", help="Run all 10 models")
    parser.add_argument(
        "--eval-mode", choices=["image", "video", "all"], default="all",
        help="Which eval mode(s)",
    )
    parser.add_argument(
        "--gpt-judge-model", default="gpt-4o",
        help="GPT model for AR-Score judging",
    )

    args = parser.parse_args()
    ensure_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    models = ALL_MODELS if args.all_models else (args.models or [])
    if not models:
        parser.error("Specify --models or --all-models")

    modes = ["image", "video"] if args.eval_mode == "all" else [args.eval_mode]

    for mode in modes:
        for model_key in models:
            try:
                run_ar_score(model_key, mode, args.gpt_judge_model)
            except Exception as e:
                logging.error(f"Failed {model_key} ({mode}): {e}", exc_info=True)


if __name__ == "__main__":
    main()
