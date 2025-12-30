"""
LLM-as-Judge Position Bias Calibration Experiment

Tests the claim from "LLMs are Bayesian in Expectation":
- Permutation averaging (k=20) should reduce position bias
- Expected: ~4x variance reduction

Usage:
    python calibrate_llm_judge.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
    python calibrate_llm_judge.py --model mlx-community/Llama-3.2-3B-Instruct-4bit
"""

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from typing import Callable

from eval.test_pairs import get_pairs, get_balanced_pairs


@dataclass
class JudgeResult:
    pair_id: str
    category: str
    ground_truth: str  # "a" or "b"
    pred_single_ab: str  # prediction with A first
    pred_single_ba: str  # prediction with B first (flipped back)
    pred_calibrated: float  # 0.0 to 1.0 confidence for A
    is_consistent: bool  # single predictions agree?

    @property
    def correct_ab(self) -> bool:
        return self.pred_single_ab == self.ground_truth

    @property
    def correct_ba(self) -> bool:
        return self.pred_single_ba == self.ground_truth

    @property
    def correct_calibrated(self) -> bool:
        if self.pred_calibrated == 0.5:
            return random.random() < 0.5  # tie = coin flip
        return (self.pred_calibrated > 0.5) == (self.ground_truth == "a")


@dataclass
class EvalMetrics:
    """Structured metrics with clear semantics."""
    n: int

    # Accuracy
    acc_ab: float  # A shown first
    acc_ba: float  # B shown first
    acc_calibrated: float  # k=20 averaging

    # Position bias
    consistency: float  # same answer both orders
    first_pos_pref: float  # prefers first position (>0.5 = primacy, <0.5 = recency)

    # Calibration quality
    high_confidence_rate: float  # >70% or <30%
    mean_confidence_a: float

    # Per-category accuracy (calibrated)
    by_category: dict = field(default_factory=dict)


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
        return random.choice(["A", "B"])


def judge_calibrated(model, tokenizer, resp_a: str, resp_b: str, generate_fn, k: int = 20) -> float:
    """Shuffle k times, return confidence score for A."""
    votes_for_a = 0

    for i in range(k):
        if i % 2 == 0:
            choice = judge_single(model, tokenizer, resp_a, resp_b, generate_fn)
            votes_for_a += (choice == "A")
        else:
            choice = judge_single(model, tokenizer, resp_b, resp_a, generate_fn)
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
    pred_ba_normalized = "b" if pred_ba_raw == "A" else "a"

    # Calibrated: k shuffles
    confidence_a = judge_calibrated(model, tokenizer, resp_a, resp_b, generate_fn, k)

    return JudgeResult(
        pair_id=pair["id"],
        category=pair["category"],
        ground_truth=ground_truth,
        pred_single_ab=pred_ab_normalized,
        pred_single_ba=pred_ba_normalized,
        pred_calibrated=confidence_a,
        is_consistent=(pred_ab_normalized == pred_ba_normalized),
    )


def compute_metrics(results: list[JudgeResult]) -> EvalMetrics:
    """Compute all metrics from results."""
    n = len(results)

    # Accuracy
    acc_ab = sum(r.correct_ab for r in results) / n
    acc_ba = sum(r.correct_ba for r in results) / n
    acc_calibrated = sum(r.correct_calibrated for r in results) / n

    # Position bias
    consistency = sum(r.is_consistent for r in results) / n
    picks_first_ab = sum(1 for r in results if r.pred_single_ab == "a") / n
    picks_first_ba = sum(1 for r in results if r.pred_single_ba == "b") / n
    first_pos_pref = (picks_first_ab + picks_first_ba) / 2

    # Calibration quality
    confidences = [r.pred_calibrated for r in results]
    high_conf = sum(1 for c in confidences if c > 0.7 or c < 0.3) / n
    mean_conf = sum(confidences) / n

    # Per-category
    categories = set(r.category for r in results)
    by_category = {}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        by_category[cat] = {
            "n": len(cat_results),
            "acc_ab": sum(r.correct_ab for r in cat_results) / len(cat_results),
            "acc_ba": sum(r.correct_ba for r in cat_results) / len(cat_results),
            "acc_calibrated": sum(r.correct_calibrated for r in cat_results) / len(cat_results),
            "consistency": sum(r.is_consistent for r in cat_results) / len(cat_results),
        }

    return EvalMetrics(
        n=n,
        acc_ab=acc_ab,
        acc_ba=acc_ba,
        acc_calibrated=acc_calibrated,
        consistency=consistency,
        first_pos_pref=first_pos_pref,
        high_confidence_rate=high_conf,
        mean_confidence_a=mean_conf,
        by_category=by_category,
    )


def print_report(model_name: str, metrics: EvalMetrics, results: list[JudgeResult], k: int):
    """Print structured evaluation report."""

    # Header
    print("\n")
    print("╔" + "═" * 70 + "╗")
    print("║" + " LLM-AS-JUDGE POSITION BIAS CALIBRATION REPORT ".center(70) + "║")
    print("╚" + "═" * 70 + "╝")

    # Model info
    print(f"\n📋 EXPERIMENT SETUP")
    print(f"   Model:            {model_name}")
    print(f"   Test pairs:       {metrics.n}")
    print(f"   Calibration k:    {k}")

    # TL;DR
    print(f"\n{'─' * 72}")
    print(f"📌 TL;DR")
    print(f"{'─' * 72}")

    bias_type = "PRIMACY (prefers first)" if metrics.first_pos_pref > 0.5 else "RECENCY (prefers second)"
    bias_strength = abs(metrics.first_pos_pref - 0.5) * 2  # 0-1 scale

    if bias_strength > 0.6:
        bias_level = "SEVERE"
    elif bias_strength > 0.3:
        bias_level = "MODERATE"
    else:
        bias_level = "WEAK"

    improvement = metrics.acc_calibrated - (metrics.acc_ab + metrics.acc_ba) / 2
    best_single = max(metrics.acc_ab, metrics.acc_ba)
    worst_single = min(metrics.acc_ab, metrics.acc_ba)

    print(f"   Position bias:    {bias_level} {bias_type} ({metrics.first_pos_pref:.0%})")
    print(f"   Best ordering:    {best_single:.1%} accuracy")
    print(f"   Worst ordering:   {worst_single:.1%} accuracy")
    print(f"   Calibrated:       {metrics.acc_calibrated:.1%} accuracy")
    print(f"   Verdict:          ", end="")

    if metrics.acc_calibrated >= best_single - 0.02:
        print("✅ Calibration matches or beats best ordering")
    elif metrics.acc_calibrated >= (metrics.acc_ab + metrics.acc_ba) / 2:
        print("⚠️  Calibration helps but doesn't beat lucky ordering")
    else:
        print("❌ Calibration doesn't help (model too weak?)")

    # Accuracy breakdown
    print(f"\n{'─' * 72}")
    print(f"📊 ACCURACY BREAKDOWN")
    print(f"{'─' * 72}")
    print(f"   {'Method':<25} {'Accuracy':>10} {'vs Calibrated':>15}")
    print(f"   {'─' * 25} {'─' * 10} {'─' * 15}")
    print(f"   {'Single pass (A first)':<25} {metrics.acc_ab:>10.1%} {metrics.acc_ab - metrics.acc_calibrated:>+15.1%}")
    print(f"   {'Single pass (B first)':<25} {metrics.acc_ba:>10.1%} {metrics.acc_ba - metrics.acc_calibrated:>+15.1%}")
    print(f"   {'Calibrated (k=' + str(k) + ')':<25} {metrics.acc_calibrated:>10.1%} {'baseline':>15}")
    print(f"   {'─' * 25} {'─' * 10} {'─' * 15}")
    print(f"   {'Oracle (best ordering)':<25} {best_single:>10.1%}")
    print(f"   {'Random baseline':<25} {'50.0%':>10}")

    # Position bias analysis
    print(f"\n{'─' * 72}")
    print(f"📍 POSITION BIAS ANALYSIS")
    print(f"{'─' * 72}")

    picks_a_when_first = sum(1 for r in results if r.pred_single_ab == "a") / metrics.n
    picks_b_when_first = sum(1 for r in results if r.pred_single_ba == "b") / metrics.n

    print(f"   Consistency (same answer both orders):  {metrics.consistency:.1%}")
    print(f"   Flipped answers:                        {1 - metrics.consistency:.1%} ({int((1-metrics.consistency) * metrics.n)} pairs)")
    print()
    print(f"   Position preference breakdown:")
    print(f"     • Picks A when A is first:            {picks_a_when_first:.1%}")
    print(f"     • Picks B when B is first:            {picks_b_when_first:.1%}")
    print(f"     • Overall first-position preference:  {metrics.first_pos_pref:.1%}")
    print()

    # Visual bias meter
    bias_pct = int(metrics.first_pos_pref * 20)
    meter = "RECENCY │" + "█" * (10 - bias_pct) + "░" * bias_pct + "│ PRIMACY"
    print(f"   {meter}")
    print(f"           {'↑':^21}")
    print(f"           {f'{metrics.first_pos_pref:.0%}':^21}")

    # Per-category breakdown
    print(f"\n{'─' * 72}")
    print(f"📂 PER-CATEGORY RESULTS")
    print(f"{'─' * 72}")
    print(f"   {'Category':<15} {'N':>4} {'Acc(AB)':>9} {'Acc(BA)':>9} {'Acc(Cal)':>9} {'Consist':>9}")
    print(f"   {'─' * 15} {'─' * 4} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 9}")

    for cat, stats in sorted(metrics.by_category.items()):
        print(f"   {cat:<15} {stats['n']:>4} {stats['acc_ab']:>9.1%} {stats['acc_ba']:>9.1%} {stats['acc_calibrated']:>9.1%} {stats['consistency']:>9.1%}")

    # Calibration quality
    print(f"\n{'─' * 72}")
    print(f"🎯 CALIBRATION QUALITY")
    print(f"{'─' * 72}")
    print(f"   High confidence (>70% or <30%):  {metrics.high_confidence_rate:.1%}")
    print(f"   Mean confidence for A:           {metrics.mean_confidence_a:.2f}")

    # Confidence distribution
    conf_buckets = {"<30%": 0, "30-50%": 0, "50%": 0, "50-70%": 0, ">70%": 0}
    for r in results:
        c = r.pred_calibrated
        if c < 0.3:
            conf_buckets["<30%"] += 1
        elif c < 0.5:
            conf_buckets["30-50%"] += 1
        elif c == 0.5:
            conf_buckets["50%"] += 1
        elif c < 0.7:
            conf_buckets["50-70%"] += 1
        else:
            conf_buckets[">70%"] += 1

    print(f"\n   Confidence distribution:")
    for bucket, count in conf_buckets.items():
        bar = "█" * (count * 2)
        print(f"     {bucket:>6}: {bar:<20} ({count})")

    # Inconsistent cases
    inconsistent = [r for r in results if not r.is_consistent]
    if inconsistent:
        print(f"\n{'─' * 72}")
        print(f"⚠️  INCONSISTENT CASES (flipped based on position)")
        print(f"{'─' * 72}")
        print(f"   {'ID':<12} {'Category':<12} {'AB→':>4} {'BA→':>4} {'Truth':>6} {'Calibrated':>10}")
        print(f"   {'─' * 12} {'─' * 12} {'─' * 4} {'─' * 4} {'─' * 6} {'─' * 10}")
        for r in inconsistent[:10]:
            cal_pred = "A" if r.pred_calibrated > 0.5 else ("B" if r.pred_calibrated < 0.5 else "TIE")
            print(f"   {r.pair_id:<12} {r.category:<12} {r.pred_single_ab:>4} {r.pred_single_ba:>4} {r.ground_truth:>6} {r.pred_calibrated:>10.0%}")
        if len(inconsistent) > 10:
            print(f"   ... and {len(inconsistent) - 10} more")

    # Recommendations
    print(f"\n{'─' * 72}")
    print(f"💡 RECOMMENDATIONS")
    print(f"{'─' * 72}")

    if metrics.consistency < 0.6:
        print(f"   ⚠️  High inconsistency ({1-metrics.consistency:.0%} flip rate)")
        print(f"      → Calibration is essential for this model")
        print(f"      → Consider using k={k*2} for critical decisions")

    if metrics.acc_calibrated < 0.6:
        print(f"   ⚠️  Low calibrated accuracy ({metrics.acc_calibrated:.0%})")
        print(f"      → Model may be too weak for judge task")
        print(f"      → Consider larger model")

    if metrics.acc_calibrated >= best_single - 0.02:
        print(f"   ✅ Calibration effective")
        print(f"      → Safe to deploy with k={k} averaging")
        print(f"      → Cost: {k}x inference vs single pass")

    print(f"\n{'═' * 72}\n")


def make_generate_logger(log_file: str = None, verbose: bool = False) -> Callable:
    """Create a generate function with optional logging."""
    from mlx_lm import generate

    def logged_generate(model, tokenizer, prompt, **kwargs):
        output = generate(model, tokenizer, prompt=prompt, **kwargs)

        if verbose:
            print(f"\n{'─' * 40}")
            print(f"INPUT:\n{prompt[:300]}...")
            print(f"OUTPUT: {output}")
            print(f"{'─' * 40}\n")

        if log_file:
            import json
            from datetime import datetime
            entry = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "output": output,
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

        return output

    return logged_generate


def run_evaluation(model_name: str, n_pairs: int = 30, k: int = 20, verbose: bool = False):
    """Run full evaluation with real model."""

    model, tokenizer = load_model(model_name)
    if model is None:
        print("Failed to load model")
        return None, None

    pairs = get_balanced_pairs(n=n_pairs)
    print(f"Evaluating {len(pairs)} pairs with k={k} calibration rounds...")

    generate_fn = make_generate_logger(verbose=verbose)

    results = []
    start_time = time.time()

    for i, pair in enumerate(pairs):
        result = evaluate_pair(model, tokenizer, pair, generate_fn, k=k)
        results.append(result)

        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (i + 1) * (len(pairs) - i - 1)
            print(f"  [{i + 1}/{len(pairs)}] ETA: {eta:.0f}s")

    metrics = compute_metrics(results)
    print_report(model_name, metrics, results, k)

    # Save results
    output = {
        "model": model_name,
        "n_pairs": n_pairs,
        "k": k,
        "metrics": {
            "accuracy_ab": metrics.acc_ab,
            "accuracy_ba": metrics.acc_ba,
            "accuracy_calibrated": metrics.acc_calibrated,
            "consistency": metrics.consistency,
            "first_position_preference": metrics.first_pos_pref,
            "high_confidence_rate": metrics.high_confidence_rate,
            "mean_confidence_a": metrics.mean_confidence_a,
            "by_category": metrics.by_category,
        },
        "results": [
            {
                "pair_id": r.pair_id,
                "category": r.category,
                "ground_truth": r.ground_truth,
                "pred_single_ab": r.pred_single_ab,
                "pred_single_ba": r.pred_single_ba,
                "pred_calibrated": r.pred_calibrated,
                "is_consistent": r.is_consistent,
            }
            for r in results
        ]
    }

    output_file = f"prototypes/experiments/llm-as-judge-positional-unbias/results/results_{model_name.split('/')[-1]}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_file}")

    return metrics, results


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Calibration Experiment")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-Instruct-4bit",
                        help="MLX model to evaluate")
    parser.add_argument("--n-pairs", type=int, default=30,
                        help="Number of test pairs to evaluate")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of calibration rounds")
    parser.add_argument("--verbose", action="store_true",
                        help="Print model inputs/outputs")

    args = parser.parse_args()
    run_evaluation(args.model, args.n_pairs, args.k, args.verbose)


if __name__ == "__main__":
    main()
