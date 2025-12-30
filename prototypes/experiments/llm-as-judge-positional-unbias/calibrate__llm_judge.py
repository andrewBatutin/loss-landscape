"""
LLM-as-Judge Position Bias Calibration Experiment

Tests the claim from "LLMs are Bayesian in Expectation":
- Permutation averaging (k=20) should reduce position bias
- Expected: ~4x variance reduction

Usage:
    python eval_harness.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
    python eval_harness.py --model mlx-community/Llama-3.2-3B-Instruct-4bit
"""

import argparse
import json
import random
import time
from dataclasses import dataclass
from typing import Literal

from eval.test_pairs import get_pairs, get_balanced_pairs


@dataclass
class JudgeResult:
    pair_id: str
    ground_truth: str  # "a" or "b"
    pred_single_ab: str  # prediction with A first
    pred_single_ba: str  # prediction with B first (flipped back)
    pred_calibrated: float  # 0.0 to 1.0 confidence for A
    is_consistent: bool  # single predictions agree?


def load_model(model_name: str):
    """Load MLX model and tokenizer."""
    try:
        from mlx_lm import load
        print(f"Loading {model_name}...")
        model, tokenizer = load(model_name)
        return model, tokenizer
    except ImportError:
        print("MLX not available. Install with: pip install mlx-lm")
        return None, None


def judge_single(model, tokenizer, resp_a: str, resp_b: str, generate_fn) -> str:
    """Single forward pass, return 'A' or 'B'."""
    prompt = f"""Which response is better? Answer with just the letter A or B.

Question context doesn't matter - just compare these two responses:

Response A: {resp_a}

Response B: {resp_b}

Better response (A or B):"""

    output = generate_fn(model, tokenizer, prompt=prompt, max_tokens=5)
    output = output.strip().upper()

    # Parse response
    if "A" in output and "B" not in output:
        return "A"
    elif "B" in output and "A" not in output:
        return "B"
    elif output.startswith("A"):
        return "A"
    elif output.startswith("B"):
        return "B"
    else:
        # Fallback: random (marks as unparseable)
        return random.choice(["A", "B"])


def judge_calibrated(model, tokenizer, resp_a: str, resp_b: str, generate_fn, k: int = 20) -> float:
    """
    Shuffle k times, return confidence score for A.

    Key insight: run both orderings and average to cancel position bias.
    """
    votes_for_a = 0

    for i in range(k):
        if i % 2 == 0:
            # Original order: A first
            choice = judge_single(model, tokenizer, resp_a, resp_b, generate_fn)
            votes_for_a += (choice == "A")
        else:
            # Swapped order: B first
            choice = judge_single(model, tokenizer, resp_b, resp_a, generate_fn)
            # If model says "A", it means it prefers B (since B is in position A)
            votes_for_a += (choice == "B")

    return votes_for_a / k


def evaluate_pair(model, tokenizer, pair: dict, generate_fn, k: int = 20) -> JudgeResult:
    """Evaluate a single pair with all methods."""
    resp_a = pair["response_a"]
    resp_b = pair["response_b"]
    ground_truth = pair["better"]

    # Single pass: A first
    pred_ab = judge_single(model, tokenizer, resp_a, resp_b, generate_fn)
    pred_ab_normalized = "a" if pred_ab == "A" else "b"

    # Single pass: B first (swapped)
    pred_ba_raw = judge_single(model, tokenizer, resp_b, resp_a, generate_fn)
    # Flip back: if model said "A" when B was first, it prefers B
    pred_ba_normalized = "b" if pred_ba_raw == "A" else "a"

    # Calibrated: k shuffles
    confidence_a = judge_calibrated(model, tokenizer, resp_a, resp_b, generate_fn, k)

    # Consistency check
    is_consistent = (pred_ab_normalized == pred_ba_normalized)

    return JudgeResult(
        pair_id=pair["id"],
        ground_truth=ground_truth,
        pred_single_ab=pred_ab_normalized,
        pred_single_ba=pred_ba_normalized,
        pred_calibrated=confidence_a,
        is_consistent=is_consistent,
    )


def compute_metrics(results: list[JudgeResult]) -> dict:
    """Compute all metrics from results."""
    n = len(results)

    # Accuracy metrics
    correct_ab = sum(1 for r in results if r.pred_single_ab == r.ground_truth)
    correct_ba = sum(1 for r in results if r.pred_single_ba == r.ground_truth)
    correct_calibrated = sum(1 for r in results if
                             (r.pred_calibrated > 0.5 and r.ground_truth == "a") or
                             (r.pred_calibrated < 0.5 and r.ground_truth == "b") or
                             (r.pred_calibrated == 0.5)  # tie goes to coin flip, count as 0.5
                             )

    # For ties, count as half correct
    ties = sum(1 for r in results if r.pred_calibrated == 0.5)
    correct_calibrated = correct_calibrated - ties + ties * 0.5

    # Position bias metrics
    consistent = sum(1 for r in results if r.is_consistent)

    # Position preference (does model prefer first position?)
    prefers_first_ab = sum(1 for r in results if r.pred_single_ab == "a")
    prefers_first_ba = sum(1 for r in results if r.pred_single_ba == "b")  # B was first

    # Confidence distribution
    confidences = [r.pred_calibrated for r in results]
    high_confidence = sum(1 for c in confidences if c > 0.7 or c < 0.3)

    return {
        "n": n,
        "accuracy_single_ab": correct_ab / n,
        "accuracy_single_ba": correct_ba / n,
        "accuracy_calibrated": correct_calibrated / n,
        "accuracy_improvement": (correct_calibrated / n) - ((correct_ab + correct_ba) / (2 * n)),
        "position_consistency": consistent / n,
        "position_bias_ab": prefers_first_ab / n,  # how often picks A when A is first
        "position_bias_ba": prefers_first_ba / n,  # how often picks B when B is first
        "first_position_preference": (prefers_first_ab + prefers_first_ba) / (2 * n),
        "high_confidence_rate": high_confidence / n,
        "mean_confidence": sum(confidences) / n,
    }


def print_results(metrics: dict, results: list[JudgeResult]):
    """Pretty print results."""
    print("\n" + "=" * 60)
    print("LLM-as-Judge Calibration Results")
    print("=" * 60)

    print(f"\n📊 ACCURACY")
    print(f"  Single pass (A first): {metrics['accuracy_single_ab']:.1%}")
    print(f"  Single pass (B first): {metrics['accuracy_single_ba']:.1%}")
    print(f"  Calibrated (k=20):     {metrics['accuracy_calibrated']:.1%}")
    print(f"  Improvement:           {metrics['accuracy_improvement']:+.1%}")

    print(f"\n📍 POSITION BIAS")
    print(f"  Consistency (same answer both orders): {metrics['position_consistency']:.1%}")
    print(f"  First-position preference:             {metrics['first_position_preference']:.1%}")
    print(f"    - Picks A when A is first:           {metrics['position_bias_ab']:.1%}")
    print(f"    - Picks B when B is first:           {metrics['position_bias_ba']:.1%}")

    print(f"\n🎯 CALIBRATION")
    print(f"  High confidence (>70% or <30%): {metrics['high_confidence_rate']:.1%}")
    print(f"  Mean confidence for A:          {metrics['mean_confidence']:.2f}")

    # Show worst position bias cases
    inconsistent = [r for r in results if not r.is_consistent]
    if inconsistent:
        print(f"\n⚠️  INCONSISTENT CASES ({len(inconsistent)} pairs flipped based on order):")
        for r in inconsistent[:5]:
            print(f"    {r.pair_id}: AB→{r.pred_single_ab}, BA→{r.pred_single_ba}, truth={r.ground_truth}")


def run_mock_evaluation():
    """Run with mock model for testing the harness."""
    print("Running mock evaluation (no model)...")

    pairs = get_balanced_pairs(n=10)

    # Mock results with simulated position bias
    results = []
    for pair in pairs:
        # Simulate: 70% first-position bias
        first_bias = random.random() < 0.7

        pred_ab = "a" if first_bias else ("a" if random.random() < 0.6 else "b")
        pred_ba = "b" if first_bias else ("b" if random.random() < 0.6 else "a")

        # Calibrated: closer to ground truth
        gt = pair["better"]
        confidence = 0.65 if gt == "a" else 0.35
        confidence += random.uniform(-0.15, 0.15)
        confidence = max(0, min(1, confidence))

        results.append(JudgeResult(
            pair_id=pair["id"],
            ground_truth=gt,
            pred_single_ab=pred_ab,
            pred_single_ba=pred_ba,
            pred_calibrated=confidence,
            is_consistent=(pred_ab == pred_ba),
        ))

    metrics = compute_metrics(results)
    print_results(metrics, results)
    return metrics, results


def logged_generate(model, tokenizer, prompt, **kwargs):
    from mlx_lm import generate

    print(f"\n{'=' * 40}")
    print(f"INPUT:\n{prompt[:200]}...")

    output = generate(model, tokenizer, prompt=prompt, **kwargs)

    print(f"OUTPUT: {output}")
    print(f"{'=' * 40}\n")

    return output


def run_evaluation(model_name: str, n_pairs: int = 30, k: int = 20):
    """Run full evaluation with real model."""

    model, tokenizer = load_model(model_name)
    if model is None:
        return run_mock_evaluation()

    pairs = get_balanced_pairs(n=n_pairs)
    print(f"Evaluating {len(pairs)} pairs with k={k} calibration rounds...")

    results = []
    start_time = time.time()

    for i, pair in enumerate(pairs):
        result = evaluate_pair(model, tokenizer, pair, logged_generate, k=k)
        results.append(result)

        # Progress
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(pairs) - i - 1)
            print(f"  [{i + 1}/{len(pairs)}] ETA: {eta:.0f}s")

    metrics = compute_metrics(results)
    print_results(metrics, results)

    # Save results
    output = {
        "model": model_name,
        "n_pairs": n_pairs,
        "k": k,
        "metrics": metrics,
        "results": [
            {
                "pair_id": r.pair_id,
                "ground_truth": r.ground_truth,
                "pred_single_ab": r.pred_single_ab,
                "pred_single_ba": r.pred_single_ba,
                "pred_calibrated": r.pred_calibrated,
                "is_consistent": r.is_consistent,
            }
            for r in results
        ]
    }

    output_file = f"results_{model_name.split('/')[-1]}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Calibration Experiment")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-8bit",
                        help="MLX model to evaluate")
    parser.add_argument("--n-pairs", type=int, default=30,
                        help="Number of test pairs to evaluate")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of calibration rounds")
    parser.add_argument("--mock", action="store_true",
                        help="Run mock evaluation without model")

    args = parser.parse_args()

    if args.mock:
        run_mock_evaluation()
    else:
        run_evaluation(args.model, args.n_pairs, args.k)


if __name__ == "__main__":
    main()
