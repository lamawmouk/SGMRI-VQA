"""Aggregate evaluation results into leaderboard CSV and breakdown reports.

Usage:
    python -m sgmriqa.aggregate_results
    python -m sgmriqa.aggregate_results --output-dir outputs/aggregate
"""

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from glob import glob

from sgmriqa.config.paths import (
    AGGREGATE_DIR,
    IMAGE_EVALUATION_DIR,
    VIDEO_EVALUATION_DIR,
    ensure_dirs,
)


def load_all_evaluations(evaluation_dirs: list = None) -> list:
    """Load all evaluation JSON files from image_level and video_level dirs."""
    if evaluation_dirs is None:
        evaluation_dirs = [
            ("image", IMAGE_EVALUATION_DIR),
            ("video", VIDEO_EVALUATION_DIR),
        ]

    evaluations = []
    for eval_level, eval_dir in evaluation_dirs:
        pattern = os.path.join(eval_dir, "*_evaluation.json")
        files = sorted(glob(pattern))
        for f in files:
            with open(f, "r") as fp:
                data = json.load(fp)
            # Tag with eval_level so we can distinguish in leaderboard
            data["eval_level"] = eval_level
            evaluations.append(data)

    return evaluations


def build_leaderboard(evaluations: list) -> list:
    """Build overall leaderboard from evaluation results.

    Returns list of dicts with model info and aggregate scores.
    """
    rows = []
    for ev in evaluations:
        row = {
            "model_key": ev.get("model_key", ""),
            "model_name": ev.get("model_name", ""),
            "model_id": ev.get("model_id", ""),
            "eval_mode": ev.get("eval_mode", ""),
            "eval_level": ev.get("eval_level", ""),
            "total_samples": ev.get("total_samples", 0),
        }

        aggregates = ev.get("aggregates", {})
        for metric in ["a_score", "ar_score", "v_score"]:
            if metric in aggregates:
                row[f"{metric}_mean"] = round(aggregates[metric]["mean"], 4)
                row[f"{metric}_count"] = aggregates[metric]["count"]
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_count"] = 0

        rows.append(row)

    # Sort by eval_level then a_score descending
    rows.sort(key=lambda r: (r.get("eval_level", ""), -(r.get("a_score_mean") or 0)))
    return rows


def build_breakdown_tables(evaluations: list) -> dict:
    """Build breakdown tables by dataset, level, qa_type, task.

    Returns dict of {breakdown_key: list of row dicts}.
    """
    tables = {}

    for breakdown_key in ["dataset", "level", "qa_type", "task"]:
        rows = []
        for ev in evaluations:
            model_key = ev.get("model_key", "")
            eval_level = ev.get("eval_level", "")
            breakdowns = ev.get("breakdowns", {}).get(breakdown_key, {})

            for group_val, metrics in breakdowns.items():
                row = {
                    "model_key": model_key,
                    "eval_level": eval_level,
                    "group": group_val,
                }
                for metric in ["a_score", "ar_score", "v_score"]:
                    if metric in metrics:
                        row[f"{metric}_mean"] = round(metrics[metric]["mean"], 4)
                        row[f"{metric}_count"] = metrics[metric]["count"]
                    else:
                        row[f"{metric}_mean"] = None
                        row[f"{metric}_count"] = 0
                rows.append(row)

        tables[breakdown_key] = rows

    return tables


def save_csv(rows: list, path: str):
    """Save list of dicts as CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_leaderboard(rows: list):
    """Print leaderboard to console."""
    if not rows:
        print("No results to display.")
        return

    # Header
    print(
        f"\n{'Model':<25} {'Level':<7} {'A-Score':>8} {'AR-Score':>9} "
        f"{'V-Score':>8} {'Samples':>8}"
    )
    print("-" * 71)
    for row in rows:
        a = f"{row['a_score_mean']:.4f}" if row["a_score_mean"] is not None else "N/A"
        ar = f"{row['ar_score_mean']:.4f}" if row["ar_score_mean"] is not None else "N/A"
        v = f"{row['v_score_mean']:.4f}" if row["v_score_mean"] is not None else "N/A"
        lvl = row.get("eval_level", "")
        print(
            f"{row['model_key']:<25} {lvl:<7} {a:>8} {ar:>9} "
            f"{v:>8} {row['total_samples']:>8}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument(
        "--output-dir", default=None, help="Override output directory"
    )

    args = parser.parse_args()
    ensure_dirs()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = args.output_dir or AGGREGATE_DIR

    evaluations = load_all_evaluations()
    if not evaluations:
        logging.error(
            f"No evaluation files found in {IMAGE_EVALUATION_DIR} or {VIDEO_EVALUATION_DIR}"
        )
        return

    logging.info(f"Loaded {len(evaluations)} evaluation files")

    # Build and save leaderboard
    leaderboard = build_leaderboard(evaluations)
    leaderboard_path = os.path.join(output_dir, "leaderboard.csv")
    save_csv(leaderboard, leaderboard_path)
    logging.info(f"Saved leaderboard to {leaderboard_path}")
    print_leaderboard(leaderboard)

    # Build and save breakdowns
    breakdown_tables = build_breakdown_tables(evaluations)
    for key, rows in breakdown_tables.items():
        path = os.path.join(output_dir, f"breakdown_{key}.csv")
        save_csv(rows, path)
        logging.info(f"Saved {key} breakdown to {path}")

    # Save combined JSON report
    report = {
        "leaderboard": leaderboard,
        "breakdowns": breakdown_tables,
    }
    report_path = os.path.join(output_dir, "aggregate_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(f"Saved aggregate report to {report_path}")


if __name__ == "__main__":
    main()
