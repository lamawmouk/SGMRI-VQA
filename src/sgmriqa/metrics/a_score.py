"""A-Score: Answer accuracy metric.

- closed_ended: exact match yes/no -> 0 or 1
- single_choice: exact match option letter -> 0 or 1
- multiple_choice: F1 on selected option set -> 0-1
- open_ended / chain_of_thought: keyword recall + semantic similarity
"""

from typing import Dict, List, Optional

from sgmriqa.metrics.utils import extract_answer_text, parse_choice_letters, parse_yes_no


def compute_a_score(
    prediction: str,
    gt_answer: str,
    qa_type: str,
    gt_labels: Optional[List[str]] = None,
    semantic_model=None,
) -> Dict:
    """Compute A-Score for a single prediction.

    Args:
        prediction: Model output text.
        gt_answer: Ground truth answer.
        qa_type: Question type (closed_ended, single_choice, etc.).
        gt_labels: Ground truth labels/keywords for open-ended recall.
        semantic_model: Loaded sentence-transformers model (optional).

    Returns:
        Dict with 'a_score' (float 0-1) and 'details'.
    """
    if qa_type == "closed_ended":
        return _score_closed_ended(prediction, gt_answer)
    elif qa_type == "single_choice":
        return _score_single_choice(prediction, gt_answer)
    elif qa_type in ("multiple_choice", "multi_choice"):
        return _score_multiple_choice(prediction, gt_answer)
    elif qa_type in ("open_ended", "chain_of_thought"):
        return _score_open_ended(prediction, gt_answer, gt_labels, semantic_model)
    else:
        return {"a_score": 0.0, "details": {"error": f"Unknown qa_type: {qa_type}"}}


def _score_closed_ended(prediction: str, gt_answer: str) -> Dict:
    """Exact match on yes/no, with numeric fallback for counting questions."""
    pred_yn = parse_yes_no(prediction)
    gt_yn = parse_yes_no(gt_answer)

    # Yes/No match
    if pred_yn is not None and gt_yn is not None:
        match = pred_yn == gt_yn
        return {
            "a_score": 1.0 if match else 0.0,
            "details": {"pred_parsed": pred_yn, "gt_parsed": gt_yn, "method": "closed_ended"},
        }

    # Fallback: numeric match (for counting questions where GT is a number)
    gt_num = _extract_number(gt_answer)
    if gt_num is not None:
        pred_num = _extract_number(prediction)
        match = pred_num is not None and pred_num == gt_num
        return {
            "a_score": 1.0 if match else 0.0,
            "details": {
                "pred_parsed": pred_num,
                "gt_parsed": gt_num,
                "method": "closed_ended_numeric",
            },
        }

    return {
        "a_score": 0.0,
        "details": {"pred_parsed": pred_yn, "gt_parsed": gt_yn, "method": "closed_ended"},
    }


def _extract_number(text: str) -> Optional[int]:
    """Extract a standalone integer from text.

    Tries the full text as a number first, then looks for a number after
    'the final answer is:' pattern, then falls back to the last number found.
    """
    import re

    text = text.strip()

    # Direct number
    try:
        return int(text)
    except ValueError:
        pass

    # After "the final answer is:" pattern
    m = re.search(r"[Tt]herefore,?\s+the\s+final\s+answer\s+is:\s*\*{0,2}(\d+)\*{0,2}", text)
    if m:
        return int(m.group(1))

    # Last number in text
    numbers = re.findall(r"\b(\d+)\b", text)
    if numbers:
        return int(numbers[-1])

    # "no abnormal findings" / "no findings" / "none" implies 0
    if re.search(r"\bno\b.*\b(?:abnormal|finding|abnormalit)", text, re.IGNORECASE) or \
       re.search(r"\bnone\b", text, re.IGNORECASE) or \
       re.search(r"\bzero\b", text, re.IGNORECASE):
        return 0

    return None


def _score_single_choice(prediction: str, gt_answer: str) -> Dict:
    """Exact match on selected letter."""
    pred_letters = parse_choice_letters(prediction)
    gt_letters = parse_choice_letters(gt_answer)

    # For single choice, take the first letter from each
    pred_letter = sorted(pred_letters)[0] if pred_letters else None
    gt_letter = sorted(gt_letters)[0] if gt_letters else None

    match = pred_letter is not None and pred_letter == gt_letter
    return {
        "a_score": 1.0 if match else 0.0,
        "details": {
            "pred_letter": pred_letter,
            "gt_letter": gt_letter,
            "method": "single_choice",
        },
    }


def _score_multiple_choice(prediction: str, gt_answer: str) -> Dict:
    """F1 on the set of selected options."""
    pred_set = parse_choice_letters(prediction)
    gt_set = parse_choice_letters(gt_answer)

    if not gt_set:
        return {
            "a_score": 0.0,
            "details": {"error": "No GT choices parsed", "method": "multiple_choice"},
        }

    if not pred_set:
        return {
            "a_score": 0.0,
            "details": {
                "pred_set": list(pred_set),
                "gt_set": list(gt_set),
                "precision": 0.0,
                "recall": 0.0,
                "method": "multiple_choice",
            },
        }

    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "a_score": f1,
        "details": {
            "pred_set": sorted(pred_set),
            "gt_set": sorted(gt_set),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "method": "multiple_choice",
        },
    }


def _score_open_ended(
    prediction: str,
    gt_answer: str,
    gt_labels: Optional[List[str]] = None,
    semantic_model=None,
) -> Dict:
    """Score open-ended answers using keyword recall + semantic similarity."""
    # Extract answer text (remove <think> blocks if present)
    pred_text = extract_answer_text(prediction).lower()
    gt_text = extract_answer_text(gt_answer).lower()

    details = {"method": "open_ended"}
    scores = []

    # 1. Keyword recall from GT labels
    if gt_labels:
        matched = sum(1 for label in gt_labels if label.lower() in pred_text)
        keyword_recall = matched / len(gt_labels) if gt_labels else 0.0
        details["keyword_recall"] = keyword_recall
        details["matched_keywords"] = matched
        details["total_keywords"] = len(gt_labels)
        scores.append(keyword_recall)

    # 2. Semantic similarity using sentence-transformers
    if semantic_model is not None:
        try:
            embeddings = semantic_model.encode([pred_text, gt_text])
            from numpy import dot
            from numpy.linalg import norm

            sim = float(dot(embeddings[0], embeddings[1]) / (
                norm(embeddings[0]) * norm(embeddings[1]) + 1e-8
            ))
            sim = max(0.0, sim)  # Clamp to non-negative
            details["semantic_similarity"] = sim
            scores.append(sim)
        except Exception as e:
            details["semantic_error"] = str(e)

    if scores:
        a_score = sum(scores) / len(scores)
    else:
        # Fallback: simple word overlap
        pred_words = set(pred_text.split())
        gt_words = set(gt_text.split())
        if gt_words:
            overlap = len(pred_words & gt_words) / len(gt_words)
        else:
            overlap = 0.0
        a_score = overlap
        details["word_overlap"] = overlap

    details["a_score"] = a_score
    return {"a_score": a_score, "details": details}
