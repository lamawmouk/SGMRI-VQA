"""Main inference script: runs VLM models on evaluation samples.

Supports two evaluation modes organised by input level:
    - image: Run on image-level samples (QA + grounding)
    - video: Run on video-level samples (QA + grounding)
    - all: Run on both

Usage:
    # Image-level evaluation (QA + grounding) on brain data
    python -m sgmriqa.run_inference --models gpt-4o-mini --datasets brain --eval-mode image

    # Video-level evaluation (QA + grounding)
    python -m sgmriqa.run_inference --models gpt-4o-mini gemini-2.0-flash --eval-mode video

    # Full evaluation (both image and video)
    python -m sgmriqa.run_inference --models gpt-4o-mini --eval-mode all
"""

import argparse
import importlib
import json
import logging
import os
import time
from datetime import datetime

from PIL import Image
from tqdm import tqdm

from sgmriqa.config.model_configs import (
    get_model_config,
    get_max_volume_images,
    list_models,
    list_video_capable_models,
    MIN_VIDEO_FRAMES,
)
from sgmriqa.config.paths import ensure_dirs, get_inference_dir
from sgmriqa.data.loader import EvalSample, load_all_samples
from sgmriqa.data.prompt_builder import (
    build_system_prompt,
    build_user_prompt,
    select_volume_images,
    build_grid_image,
)


def load_runner(model_key: str, max_new_tokens: int = None):
    """Dynamically load and instantiate a model runner."""
    config = get_model_config(model_key)
    tokens = max_new_tokens or config.max_new_tokens
    module = importlib.import_module(config.runner_module)
    runner_cls = getattr(module, config.runner_class)
    return runner_cls(model_id=config.model_id, max_new_tokens=tokens, **config.runner_kwargs)


def load_images_for_sample(
    sample: EvalSample,
    model_key: str = None,
) -> list:
    """Load PIL images for a sample, applying token-aware volume subsampling."""
    paths = sample.image_paths

    # For volume-level samples, subsample images to fit model context
    if model_key and sample.level == "volume_level" and len(paths) > 1:
        paths = select_volume_images(paths, model_key)

    images = []
    for path in paths:
        if os.path.exists(path):
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception:
                continue
    return images


def run_inference_for_model(
    model_key: str,
    samples: list,
    output_path: str,
    max_new_tokens: int = None,
    rate_limit_delay: float = 1.0,
    resume: bool = True,
    save_every: int = 5,
    no_system_prompt: bool = False,
    minimal: bool = False,
):
    """Run inference for a single model on all samples.

    Args:
        model_key: Key from model registry.
        samples: List of EvalSample objects.
        output_path: Path to save results JSON.
        max_new_tokens: Max tokens for generation (overrides model config).
        rate_limit_delay: Delay between API calls (seconds).
        resume: Whether to resume from existing output.
        save_every: Save checkpoint every N samples.
    """
    config = get_model_config(model_key)

    # Load existing results for resume
    existing_results = {}
    if resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)
        for r in data.get("results", []):
            existing_results[r["sample_id"]] = r
        logging.info(f"Resuming: {len(existing_results)} samples already done")

    # Filter out already-completed samples
    remaining = [s for s in samples if s.sample_id not in existing_results]
    if not remaining:
        logging.info(f"All {len(samples)} samples already completed for {model_key}")
        return

    logging.info(
        f"Running {model_key} on {len(remaining)} samples "
        f"({len(existing_results)} already done)"
    )
    logging.info(
        f"Model config: context_window={config.max_context_window}, "
        f"max_new_tokens={config.max_new_tokens}, "
        f"grounding={config.supports_grounding}, "
        f"multi_image={config.supports_multi_image}"
    )

    # Load model
    runner = load_runner(model_key, max_new_tokens)
    runner.load_model()

    results = list(existing_results.values())
    total_cost = sum(r.get("cost", 0) for r in results)

    try:
        for i, sample in enumerate(tqdm(remaining, desc=model_key)):
            images = load_images_for_sample(sample, model_key)
            if not images:
                results.append({
                    "sample_id": sample.sample_id,
                    "model_output": "[ERROR: No images found]",
                    "error": "no_images",
                    **sample.to_dict(),
                })
                continue

            # Unified system/user prompt
            system_prompt = "" if no_system_prompt else build_system_prompt(sample, config, minimal=minimal)
            user_prompt = build_user_prompt(sample, config, minimal=minimal)

            try:
                result = runner.run_inference(images, system_prompt, user_prompt)
                results.append({
                    "sample_id": sample.sample_id,
                    "model_output": result.model_output,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "cost": result.cost,
                    "duration": result.duration,
                    **sample.to_dict(),
                })
                total_cost += result.cost
            except Exception as e:
                logging.error(f"Error on {sample.sample_id}: {e}")
                results.append({
                    "sample_id": sample.sample_id,
                    "model_output": f"[ERROR: {e}]",
                    "error": str(e),
                    **sample.to_dict(),
                })

            # Rate limiting for API models
            if config.model_type.startswith("api"):
                time.sleep(rate_limit_delay)

            # Periodic checkpoint save
            if (i + 1) % save_every == 0:
                _save_results(output_path, model_key, config, results, total_cost, minimal=minimal)
                logging.info(
                    f"Checkpoint saved: {len(results)} results, cost: ${total_cost:.4f}"
                )

    finally:
        runner.unload_model()

    # Final save
    _save_results(output_path, model_key, config, results, total_cost, minimal=minimal)
    logging.info(
        f"Completed {model_key}: {len(results)} results, total cost: ${total_cost:.4f}"
    )


def _save_results(output_path, model_key, config, results, total_cost, minimal=False):
    """Save results to JSON with metadata."""
    output_data = {
        "model_key": model_key,
        "model_name": config.name,
        "model_id": config.model_id,
        "model_config": {
            "max_context_window": config.max_context_window,
            "max_new_tokens": config.max_new_tokens,
            "supports_grounding": config.supports_grounding,
            "supports_multi_image": config.supports_multi_image,
            "supports_thinking": config.supports_thinking,
            "bbox_format": config.bbox_format,
        },
        "prompt_mode": "minimal" if minimal else "icl",
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(results),
        "total_cost": total_cost,
        "results": results,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def _run_single_mode(args, eval_mode: str):
    """Run inference for a single eval mode (image or video)."""
    samples = load_all_samples(
        datasets=args.datasets,
        eval_mode=eval_mode,
        split=args.split,
    )
    # Filter by task if specified
    if args.tasks:
        samples = [s for s in samples if s.task in args.tasks]

    # Limit to first N cases (unique volume_ids) per dataset
    if args.max_cases:
        from collections import OrderedDict
        # Collect cases per dataset preserving order
        cases_per_ds = {}
        for s in samples:
            ds = s.dataset
            if ds not in cases_per_ds:
                cases_per_ds[ds] = OrderedDict()
            cases_per_ds[ds][s.volume_id] = True
        # Keep first max_cases per dataset
        keep_cases = set()
        for ds, cases in cases_per_ds.items():
            for i, case_id in enumerate(cases):
                if i >= args.max_cases:
                    break
                keep_cases.add(case_id)
        samples = [s for s in samples if s.volume_id in keep_cases]

    # Limit to N questions per task per case
    if args.max_per_task:
        from collections import defaultdict
        counts = defaultdict(int)  # (volume_id, task) -> count
        filtered = []
        for s in samples:
            key = (s.volume_id, s.task)
            if counts[key] < args.max_per_task:
                filtered.append(s)
                counts[key] += 1
        samples = filtered

    # Shard by index range if specified
    if args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx or 0
        end = args.end_idx or len(samples)
        samples = samples[start:end]

    logging.info(
        f"Loaded {len(samples)} total samples "
        f"(mode={eval_mode}, datasets={args.datasets}, tasks={args.tasks or 'all'})"
    )

    if not samples:
        logging.warning(f"No samples found for mode={eval_mode}")
        return

    output_dir = args.output_dir or get_inference_dir(eval_mode)

    # For video mode, filter to models that can handle >=40 frames
    models = args.models
    if eval_mode == "video":
        video_capable = set(list_video_capable_models())
        skipped = [m for m in models if m not in video_capable]
        if skipped:
            for m in skipped:
                max_f = get_max_volume_images(m)
                logging.warning(
                    f"Skipping {m} for video mode: can only handle {max_f} frames "
                    f"(minimum {MIN_VIDEO_FRAMES} required)"
                )
        models = [m for m in models if m in video_capable]
        if not models:
            logging.warning("No models left after video capability filter")
            return

    for model_key in models:
        suffix = getattr(args, "output_suffix", "")
        output_path = os.path.join(output_dir, f"{model_key}{suffix}_inference.json")
        run_inference_for_model(
            model_key=model_key,
            samples=samples,
            output_path=output_path,
            max_new_tokens=args.max_new_tokens,
            rate_limit_delay=args.rate_limit_delay,
            resume=not args.no_resume,
            save_every=args.save_every,
            no_system_prompt=args.no_system_prompt,
            minimal=args.minimal,
        )


def main():
    parser = argparse.ArgumentParser(description="Run VLM inference for evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini"],
        help=f"Models to evaluate. Available: {list_models()}",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["brain", "knee"],
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["image", "video", "all"],
        default="all",
        help="Evaluation mode: 'image' (image-level QA+grounding), 'video' (video-level QA+grounding), 'all'",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max new tokens for generation (uses model config if not set)",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=1.0,
        help="Seconds between API calls",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from existing results"
    )
    parser.add_argument(
        "--save-every", type=int, default=5, help="Save checkpoint every N samples"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Override output directory"
    )
    parser.add_argument(
        "--split",
        choices=["val", "train"],
        default="val",
        help="Data split to evaluate on (default: val)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Filter to specific tasks (e.g. --tasks localization captioning)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit to first N cases (volume_ids) per dataset",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=None,
        help="Limit to N questions per task per case",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Skip the system prompt (send empty string to runner)",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Use minimal prompts (no ICL, no JSON schema, no task instructions)",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix to append to output filename (e.g. '_nosys' -> model_nosys_inference.json)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="Start sample index for sharded inference (inclusive)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="End sample index for sharded inference (exclusive)",
    )

    args = parser.parse_args()
    ensure_dirs()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Dispatch by eval mode
    if args.eval_mode == "all":
        for mode in ["image", "video"]:
            _run_single_mode(args, mode)
    else:
        _run_single_mode(args, args.eval_mode)


if __name__ == "__main__":
    main()
