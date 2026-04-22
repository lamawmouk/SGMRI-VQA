"""Utility functions for metrics: bbox parsing, IoU computation, choice parsing.

Supports multiple bounding box output formats from different VLMs:
- <bbx>[x, y, w, h]</bbx>  (GPT-4o, Gemini, default) — pixel coordinates
- <|box_start|>(x1,y1),(x2,y2)<|box_end|>  (Qwen2-VL) — 0-1000 normalized, needs denorm
- [x, y, w, h]  (generic bracket format) — pixel coordinates
- {"x", "y", "width", "height"}  (JSON objects) — pixel coordinates
- Gemini structured JSON: {"bboxes": [{"min_x", "min_y", "max_x", "max_y"}]} — normalized [0,1]
"""

import json
import re
from typing import Dict, List, Optional, Tuple


def parse_bboxes(text: str, image_width: int = None, image_height: int = None) -> List[List[float]]:
    """Extract bounding boxes from text, supporting multiple VLM output formats.

    Thin wrapper around parse_bboxes_with_frames that strips frame info.

    Returns:
        List of [x, y, w, h] bounding boxes in pixel coordinates.
    """
    return [bbox for _, bbox in parse_bboxes_with_frames(text, image_width, image_height)]


def parse_bboxes_with_frames(
    text: str, image_width: int = None, image_height: int = None,
) -> List[Tuple[Optional[int], List[float]]]:
    """Extract bounding boxes WITH frame numbers from model output.

    Tries each format in order of specificity. Returns list of
    (frame_number, [x, y, w, h]) tuples. frame_number is None if no
    frame reference is found near the bbox.

    Supported formats:
        - "Frame 7: <bbx>[x, y, w, h]</bbx>"
        - "Frame 7, <|box_start|>(x1,y1),(x2,y2)<|box_end|>"
        - "Frame 3: bbox(x1,y1,x2,y2)"
        - "Frame 3: {"bbox_2d": [x1,y1,x2,y2]}"
        - "Frame 3: {"x":N, "y":N, "width":N, "height":N}"
        - "Frame 3: [x, y, w, h]"
    """
    results = []

    # 0. Gemini structured JSON: {"bboxes": [{"min_x", "min_y", "max_x", "max_y", ...}]}
    #    Coords are normalized [0,1]; denormalize to pixel coords when image dims provided.
    try:
        parsed_json = json.loads(text)
        if isinstance(parsed_json, dict) and "bboxes" in parsed_json:
            for b in parsed_json["bboxes"]:
                if isinstance(b, dict) and all(k in b for k in ("min_x", "min_y", "max_x", "max_y")):
                    min_x = float(b["min_x"])
                    min_y = float(b["min_y"])
                    max_x = float(b["max_x"])
                    max_y = float(b["max_y"])
                    # Denormalize from [0,1] to pixel coords
                    if image_width and image_height and max(min_x, min_y, max_x, max_y) <= 1.0:
                        min_x *= image_width
                        min_y *= image_height
                        max_x *= image_width
                        max_y *= image_height
                    w = max_x - min_x
                    h = max_y - min_y
                    if w > 0 and h > 0:
                        frame = int(b["frame"]) if "frame" in b else None
                        results.append((frame, [min_x, min_y, w, h]))
            if results:
                return results
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 1. <bbx>[x, y, w, h]</bbx> format (GPT-4o, Gemini, default)
    bbx_pattern = r"(?:[Ff]rame\s+(\d+)\s*[:\-,]?\s*)?<bb[ox]+>\s*\[([^\]]+)\]\s*<[/\\]bb[ox]+>"
    last_frame = None
    for m in re.finditer(bbx_pattern, text):
        if m.group(1):
            last_frame = int(m.group(1))
        frame = int(m.group(1)) if m.group(1) else last_frame
        coords = _parse_coords(m.group(2))
        if coords and len(coords) == 4:
            results.append((frame, coords))

    if results:
        return results

    # 2. Qwen2-VL format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    #    Always 0-1000 normalized; denormalize when image dimensions provided.
    qwen_pattern = (
        r"(?:[Ff]rame\s+(\d+)\s*[:\-,]?\s*)?"
        r"<\|box_start\|>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*<\|box_end\|>"
    )
    last_frame = None
    for m in re.finditer(qwen_pattern, text):
        if m.group(1):
            last_frame = int(m.group(1))
        frame = int(m.group(1)) if m.group(1) else last_frame
        try:
            x1, y1, x2, y2 = float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            if image_width and image_height:
                x1 = x1 / 1000 * image_width
                y1 = y1 / 1000 * image_height
                x2 = x2 / 1000 * image_width
                y2 = y2 / 1000 * image_height
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                results.append((frame, [x1, y1, w, h]))
        except (ValueError, IndexError):
            continue

    if results:
        return results

    # 3. bbox(x1,y1,x2,y2) text format (Qwen free-text output)
    bbox_func_pattern = (
        r"(?:[Ff]rame\s+(\d+)\s*[:\-,]?\s*)?"
        r"bbox\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
    )
    last_frame = None
    for m in re.finditer(bbox_func_pattern, text, re.IGNORECASE):
        if m.group(1):
            last_frame = int(m.group(1))
        frame = int(m.group(1)) if m.group(1) else last_frame
        try:
            x1, y1, x2, y2 = float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                results.append((frame, [x1, y1, w, h]))
        except (ValueError, IndexError):
            continue

    if results:
        return results

    # 4. (x1, y1, x2, y2) or (x1,y1),(x2,y2) format
    xyxy_pattern = r"(?:[Ff]rame\s+(\d+)\s*[:\-,]?\s*)?\((\d+)\s*,\s*(\d+)\)\s*,?\s*\((\d+)\s*,\s*(\d+)\)"
    last_frame = None
    for m in re.finditer(xyxy_pattern, text):
        if m.group(1):
            last_frame = int(m.group(1))
        frame = int(m.group(1)) if m.group(1) else last_frame
        try:
            x1, y1, x2, y2 = float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                results.append((frame, [x1, y1, w, h]))
        except (ValueError, IndexError):
            continue

    if results:
        return results

    # 5. Qwen bbox_2d format: {"bbox_2d": [x1, y1, x2, y2]}
    #    Coords are 0-1000 normalized; denormalize when image dimensions provided.
    bbox_2d_frame_pattern = r'(?:[Ff]rame\s+(\d+)\s*[:\-,]?\s*)?\{[^{}]*"bbox_2d"[^{}]*\}'
    last_frame = None
    for m in re.finditer(bbox_2d_frame_pattern, text):
        if m.group(1):
            last_frame = int(m.group(1))
        frame = int(m.group(1)) if m.group(1) else last_frame
        try:
            json_str = re.search(r'\{[^{}]*\}', m.group(0)).group(0)
            obj = json.loads(json_str)
            coords = obj["bbox_2d"]
            if len(coords) == 4:
                x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                if image_width and image_height and max(x1, y1, x2, y2) > max(image_width, image_height):
                    x1 = x1 / 1000 * image_width
                    y1 = y1 / 1000 * image_height
                    x2 = x2 / 1000 * image_width
                    y2 = y2 / 1000 * image_height
                w, h = x2 - x1, y2 - y1
                if w > 0 and h > 0:
                    results.append((frame, [x1, y1, w, h]))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue

    if results:
        return results

    # 6. JSON object format: {"x": N, "y": N, "width": N, "height": N}
    json_obj_pattern = r'\{[^{}]*\}'
    for obj_str in re.findall(json_obj_pattern, text):
        try:
            obj = json.loads(obj_str)
            if all(k in obj for k in ("x", "y", "width", "height")):
                x, y, w, h = float(obj["x"]), float(obj["y"]), float(obj["width"]), float(obj["height"])
                frame = int(obj["frame"]) if "frame" in obj else None
                if w > 0 and h > 0:
                    results.append((frame, [x, y, w, h]))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue

    if results:
        return results

    # 7. Generic bracket format: [x1, y1, x2, y2] or [x, y, w, h]
    #    Detect xyxy (x2 > x1 and y2 > y1) and convert to xywh.
    #    Denormalize from 0-1000 if coords exceed image dims and dims are provided.
    frame_bracket_pattern = r"(?:[Ff]rame\s+(\d+)\s*[:\-,]?\s*)?\[(\d+[\d.,\s]*\d*)\]"
    last_frame = None
    for m in re.finditer(frame_bracket_pattern, text):
        if m.group(1):
            last_frame = int(m.group(1))
        frame = int(m.group(1)) if m.group(1) else last_frame
        coords = _parse_coords(m.group(2))
        if coords and len(coords) == 4:
            if all(0 <= c <= 10000 for c in coords) and coords[2] > 0 and coords[3] > 0:
                x1, y1, v3, v4 = coords
                # Detect xyxy: if v3 > x1 and v4 > y1, likely (x1,y1,x2,y2)
                if v3 > x1 and v4 > y1:
                    coords = [x1, y1, v3 - x1, v4 - y1]
                # Denormalize from 0-1000 if coords exceed image dimensions
                if image_width and image_height and max(coords) > max(image_width, image_height):
                    coords = [
                        coords[0] / 1000 * image_width,
                        coords[1] / 1000 * image_height,
                        coords[2] / 1000 * image_width,
                        coords[3] / 1000 * image_height,
                    ]
                results.append((frame, coords))

    return results


def _parse_coords(coord_str: str) -> Optional[List[float]]:
    """Parse a comma-separated coordinate string into floats."""
    try:
        return [float(x.strip()) for x in coord_str.split(",")]
    except ValueError:
        return None


def bbox_xywh_to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    """Convert [x, y, w, h] to (x1, y1, x2, y2)."""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two boxes in [x, y, w, h] format."""
    ax1, ay1, ax2, ay2 = bbox_xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def parse_choice_letters(text: str) -> set:
    """Extract choice letters like (A), (B), (C), (D) from text.

    Filters out negated choices: "It is not (A)" patterns are excluded
    so only affirmed choices are returned.
    """
    # Remove negated choices: "It is not (X) ..." sentences
    cleaned = re.sub(
        r"[Ii]t\s+is\s+not\s+\([A-Da-d]\)[^.]*\.?", "", text
    )

    matches = re.findall(r"\(([A-Da-d])\)", cleaned)
    if matches:
        return {m.upper() for m in matches}

    # Fallback: standalone letters at word boundaries
    matches = re.findall(r"\b([A-Da-d])\b", cleaned)
    return {m.upper() for m in matches}


def parse_yes_no(text: str) -> Optional[bool]:
    """Parse a yes/no answer from text. Returns True/False/None."""
    text_lower = text.strip().lower()
    first_word = text_lower.split()[0] if text_lower.split() else ""
    first_word = first_word.strip(".,!;:")

    if first_word in ("yes", "y"):
        return True
    if first_word in ("no", "n"):
        return False

    # "my answer is: Yes/No" pattern (common in fine-tuned models)
    m = re.search(r"(?:my answer is|the answer is)[:\s]*\**\s*(yes|no)\b", text_lower)
    if m:
        return m.group(1) == "yes"

    has_yes = "yes" in text_lower
    has_no = "no" in text_lower

    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False

    return None


def parse_structured_output(text: str):
    """Try to parse structured JSON from model output.

    Returns dict with 'reasoning', 'answer', 'bboxes' keys, or None.
    Not used in minimal prompt mode (which is what all paper results use).
    """
    return None


def extract_answer_text(text: str) -> str:
    """Extract the final answer, removing <think>...</think> blocks."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def gt_bboxes_from_sample(bboxes: List[Dict]) -> List[List[float]]:
    """Convert ground truth bboxes from dict format to [x, y, w, h] lists."""
    result = []
    for bb in bboxes:
        try:
            result.append([
                float(bb["x"]),
                float(bb["y"]),
                float(bb["width"]),
                float(bb["height"]),
            ])
        except (KeyError, ValueError):
            continue
    return result


def gt_frames_from_sample(bboxes: List[Dict]) -> List[Optional[int]]:
    """Extract frame numbers from ground truth bboxes (for volume-level matching)."""
    result = []
    for bb in bboxes:
        frame = bb.get("frame") or bb.get("slice_num")
        result.append(int(frame) if frame is not None else None)
    return result
