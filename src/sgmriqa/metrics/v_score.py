"""V-Score: Visual grounding metric using bounding box IoU with Hungarian matching.

Supports two modes:
- Image-level: matches predicted bboxes to GT bboxes on a single image.
- Volume-level (frame-aware): matches predicted bboxes to GT bboxes only within
  the same frame. Bboxes on different frames cannot match, even if they overlap
  spatially. This prevents false positives when a finding on frame 5 has the same
  (x,y,w,h) as a different finding on frame 10.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from sgmriqa.metrics.utils import (
    compute_iou,
    extract_answer_text,
    gt_bboxes_from_sample,
    gt_frames_from_sample,
    parse_bboxes,
    parse_bboxes_with_frames,
)


def compute_v_score(
    prediction: str,
    gt_bboxes_raw: List[Dict],
    iou_threshold: float = 0.5,
    pred_bboxes_parsed: List[Dict] = None,
    image_width: int = None,
    image_height: int = None,
) -> Dict:
    """Compute V-Score by matching predicted bboxes to GT bboxes.

    Uses Hungarian algorithm to find optimal assignment, then computes mIoU.
    For volume-level samples (GT bboxes with 'frame' field), matching is
    frame-aware: bboxes on different frames get IoU=0.

    Args:
        prediction: Model output text containing <bbx> tags.
        gt_bboxes_raw: Ground truth bboxes as list of dicts with x, y, width, height,
            and optionally 'frame' for volume-level evaluation.
        iou_threshold: IoU threshold for counting a match as correct.
        pred_bboxes_parsed: Pre-parsed bboxes from structured JSON output.
            Each dict has a 'bbox' key with [x, y, w, h]. If provided, skips
            regex parsing of prediction text.
        image_width: Image width in pixels for denormalizing coords (used by text parser).
        image_height: Image height in pixels for denormalizing coords.

    Returns:
        Dict with 'v_score' (mIoU over matched GT boxes) and 'details'.
    """
    # Strip think tags before parsing bboxes
    prediction = extract_answer_text(prediction)

    gt_bboxes = gt_bboxes_from_sample(gt_bboxes_raw)
    gt_frames = gt_frames_from_sample(gt_bboxes_raw)

    # Check if this is volume-level (GT has frame info)
    is_volume = any(f is not None for f in gt_frames)

    # Parse predicted bboxes
    if pred_bboxes_parsed is not None:
        pred_bboxes = [
            [b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]]
            for b in pred_bboxes_parsed
            if isinstance(b.get("bbox"), (list, tuple)) and len(b["bbox"]) >= 4
        ]
        pred_frames: List[Optional[int]] = [
            b.get("frame") for b in pred_bboxes_parsed
            if isinstance(b.get("bbox"), (list, tuple)) and len(b["bbox"]) >= 4
        ]
    elif is_volume:
        # Use frame-aware parser for volume-level
        parsed = parse_bboxes_with_frames(
            prediction, image_width=image_width, image_height=image_height
        )
        pred_frames = [f for f, _ in parsed]
        pred_bboxes = [bb for _, bb in parsed]
    else:
        pred_bboxes = parse_bboxes(
            prediction, image_width=image_width, image_height=image_height
        )
        pred_frames = [None] * len(pred_bboxes)

    # No GT bboxes -> skip (not a grounding question)
    if not gt_bboxes:
        return {
            "v_score": None,
            "details": {"reason": "no_gt_bboxes", "pred_count": len(pred_bboxes)},
        }

    # No predicted bboxes -> score 0
    if not pred_bboxes:
        return {
            "v_score": 0.0,
            "details": {
                "reason": "no_pred_bboxes",
                "gt_count": len(gt_bboxes),
                "pred_count": 0,
            },
        }

    n_gt = len(gt_bboxes)
    n_pred = len(pred_bboxes)

    # Build IoU cost matrix
    iou_matrix = np.zeros((n_gt, n_pred))
    for i, gt_box in enumerate(gt_bboxes):
        for j, pred_box in enumerate(pred_bboxes):
            # For volume-level: only allow matching within same frame
            if is_volume and gt_frames[i] is not None and pred_frames[j] is not None:
                if gt_frames[i] != pred_frames[j]:
                    iou_matrix[i, j] = 0.0
                    continue
            iou_matrix[i, j] = compute_iou(gt_box, pred_box)

    # Hungarian matching (minimize negative IoU)
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    # Compute matched IoUs
    matched_ious = []
    matches = []
    for gi, pi in zip(gt_indices, pred_indices):
        iou_val = float(iou_matrix[gi, pi])
        matched_ious.append(iou_val)
        match_info = {
            "gt_idx": int(gi),
            "pred_idx": int(pi),
            "iou": iou_val,
            "gt_bbox": gt_bboxes[gi],
            "pred_bbox": pred_bboxes[pi],
        }
        if is_volume:
            match_info["gt_frame"] = gt_frames[gi]
            match_info["pred_frame"] = pred_frames[pi]
        matches.append(match_info)

    # For unmatched GT boxes, IoU = 0
    unmatched_gt = set(range(n_gt)) - set(gt_indices)
    for gi in unmatched_gt:
        matched_ious.append(0.0)

    # mIoU over all GT boxes
    miou = float(np.mean(matched_ious)) if matched_ious else 0.0

    # Count how many matches exceed threshold
    hits = sum(1 for iou_val in matched_ious if iou_val >= iou_threshold)

    # Multi-threshold Precision / Recall / F1 (mAP-style)
    # Since models don't output confidence scores, we compute detection metrics
    # at multiple IoU thresholds using the Hungarian-matched pairs.
    thresholds = [0.1, 0.25, 0.5]
    detection_metrics = {}
    for t in thresholds:
        tp = sum(1 for gi, pi in zip(gt_indices, pred_indices) if iou_matrix[gi, pi] >= t)
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        detection_metrics[f"P@{t}"] = precision
        detection_metrics[f"R@{t}"] = recall
        detection_metrics[f"F1@{t}"] = f1

    # mAP: mean F1 across thresholds (summary statistic)
    map_score = float(np.mean([detection_metrics[f"F1@{t}"] for t in thresholds]))

    return {
        "v_score": miou,
        "map": map_score,
        "details": {
            "miou": miou,
            "hits": hits,
            "gt_count": n_gt,
            "pred_count": n_pred,
            "matches": matches,
            "iou_threshold": iou_threshold,
            "frame_aware": is_volume,
            "detection_metrics": detection_metrics,
        },
    }
