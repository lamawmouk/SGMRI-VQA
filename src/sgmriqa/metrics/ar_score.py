"""AR-Score: Answer+Reasoning quality metric.

Combines:
- GPTScore: GPT-4o judges candidate vs reference (accuracy/relevance/helpfulness)
- NLG metrics: BERTScore F1, BLEU (smoothed), ROUGE-L
"""

import os
from typing import Dict, Optional

from dotenv import load_dotenv


def compute_ar_score(
    prediction: str,
    gt_answer: str,
    gt_reasoning: str,
    gpt_judge_model: str = "gpt-4o",
    use_bertscore: bool = True,
    use_bleu: bool = True,
    use_rouge: bool = True,
) -> Dict:
    """Compute AR-Score combining GPTScore and NLG metrics.

    Args:
        prediction: Model output text.
        gt_answer: Ground truth answer.
        gt_reasoning: Ground truth reasoning.
        gpt_judge_model: Model to use for GPT judging.
        use_bertscore: Whether to compute BERTScore.
        use_bleu: Whether to compute BLEU.
        use_rouge: Whether to compute ROUGE-L.

    Returns:
        Dict with 'ar_score' (float 0-1) and 'details'.
    """
    reference = gt_answer.strip()
    candidate = prediction.strip()

    if not candidate:
        return {
            "ar_score": 0.0,
            "details": {"error": "empty_prediction"},
        }

    scores = {}
    details = {}

    # 1. GPTScore
    gpt_result = _compute_gpt_score(candidate, reference, gpt_judge_model)
    if gpt_result is not None:
        scores["gpt_score"] = gpt_result["normalized_score"]
        details["gpt_score"] = gpt_result

    # 2. NLG metrics
    if use_bertscore:
        bs = _compute_bertscore(candidate, reference)
        if bs is not None:
            scores["bertscore_f1"] = bs
            details["bertscore_f1"] = bs

    if use_bleu:
        bl = _compute_bleu(candidate, reference)
        if bl is not None:
            scores["bleu"] = bl
            details["bleu"] = bl

    if use_rouge:
        rl = _compute_rouge_l(candidate, reference)
        if rl is not None:
            scores["rouge_l"] = rl
            details["rouge_l"] = rl

    # Aggregate: weighted average (GPT judge + NLG metrics per paper)
    if scores:
        if "gpt_score" in scores:
            weights = {"gpt_score": 0.4}
            nlg_keys = [k for k in scores if k != "gpt_score"]
            nlg_weight = 0.6 / len(nlg_keys) if nlg_keys else 0.0
            for k in nlg_keys:
                weights[k] = nlg_weight
        else:
            weights = {k: 1.0 / len(scores) for k in scores}

        ar_score = sum(scores[k] * weights[k] for k in scores)
    else:
        ar_score = 0.0

    return {"ar_score": ar_score, "details": details}


def _compute_gpt_score(
    candidate: str,
    reference: str,
    model: str = "gpt-4o",
) -> Optional[Dict]:
    """Use GPT to judge candidate vs reference on accuracy/relevance/helpfulness."""
    try:
        import openai

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        client = openai.OpenAI(api_key=api_key)

        judge_prompt = f"""You are evaluating a medical imaging AI's response against a reference answer.

Reference answer:
{reference}

Candidate answer:
{candidate}

Rate the candidate on these criteria (1-10 each):
1. Accuracy: How factually correct is the candidate compared to the reference?
2. Relevance: How relevant is the candidate's content to the question?
3. Helpfulness: How complete and clinically useful is the candidate's response?

Respond with ONLY three numbers separated by commas, e.g.: 7,8,6"""

        # Reasoning models (gpt-5, o1, o3, etc.) use max_completion_tokens, no temperature
        is_reasoning = any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": judge_prompt}],
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = 1000
        else:
            kwargs["temperature"] = 0.0
            kwargs["max_tokens"] = 20

        response = client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content.strip()
        parts = [float(x.strip()) for x in text.split(",")]
        if len(parts) == 3:
            raw_score = sum(parts) / 3.0 / 10.0  # Normalize to 0-1

            # Self-score normalization: score the reference against itself
            ref_kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": judge_prompt.replace(
                            f"Candidate answer:\n{candidate}",
                            f"Candidate answer:\n{reference}",
                        ),
                    }
                ],
            }
            if is_reasoning:
                ref_kwargs["max_completion_tokens"] = 1000
            else:
                ref_kwargs["temperature"] = 0.0
                ref_kwargs["max_tokens"] = 20

            ref_response = client.chat.completions.create(**ref_kwargs)
            ref_text = ref_response.choices[0].message.content.strip()
            ref_parts = [float(x.strip()) for x in ref_text.split(",")]
            if len(ref_parts) == 3:
                ref_score = sum(ref_parts) / 3.0 / 10.0
                normalized = raw_score / ref_score if ref_score > 0 else raw_score
                normalized = min(normalized, 1.0)
            else:
                normalized = raw_score

            return {
                "raw_score": raw_score,
                "ref_self_score": ref_score if len(ref_parts) == 3 else None,
                "normalized_score": normalized,
                "accuracy": parts[0],
                "relevance": parts[1],
                "helpfulness": parts[2],
            }
    except Exception as e:
        return {"error": str(e), "normalized_score": 0.0}

    return None


def _compute_bertscore(candidate: str, reference: str) -> Optional[float]:
    """Compute BERTScore F1."""
    try:
        from bert_score import score as bert_score

        P, R, F1 = bert_score(
            [candidate], [reference], lang="en", rescale_with_baseline=True
        )
        return float(F1[0])
    except ImportError:
        return None
    except Exception:
        return None


def _compute_bleu(candidate: str, reference: str) -> Optional[float]:
    """Compute smoothed BLEU score."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        smoothie = SmoothingFunction().method1
        return float(sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie))
    except ImportError:
        return None
    except Exception:
        return None


def _compute_rouge_l(candidate: str, reference: str) -> Optional[float]:
    """Compute ROUGE-L F1 score."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        result = scorer.score(reference, candidate)
        return float(result["rougeL"].fmeasure)
    except ImportError:
        return None
    except Exception:
        return None
