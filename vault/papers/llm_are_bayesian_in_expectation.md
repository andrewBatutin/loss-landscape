# LLMs are Bayesian in Expectation, Not in Realization

## Source

https://arxiv.org/abs/2507.11768

## One-liner

Optimal chain-of-thought length scales as √n, not linearly — most CoT burns 5-10x too many tokens.

## Load-bearing insight

Positional encoding degradation penalizes long chains. There's a sweet spot where more thinking starts hurting. The formula:
```
k* = c × √n × log(1/ε)
```

## What most miss

The CoT formula is **not empirically validated** — paper defers this to follow-up work. The theoretical derivation is solid (grounded in positional encoding analysis), but constants are model/task-specific and need estimation.

## Concepts

[[concepts/chain-of-thought]] · [[concepts/positional-encoding]] · [[concepts/kolmogorov-complexity]]

## Project

[[projects/cot-length-validation]] - test √n scaling on GSM8K

## Essence
```python
# Optimal CoT length (theoretical)
def optimal_k(n, epsilon=0.1, c=1.0):
    """
    n: context length
    epsilon: error tolerance (0.1 = 90% of max performance)
    c: model-specific constant (estimate empirically)
    """
    return c * (n ** 0.5) * log2(1 / epsilon)

# Example: n=1000, ε=0.1
# k* ≈ 40 tokens (vs typical 200-500)

# The key insight: there's a MAXIMUM useful CoT length
# Beyond this, positional degradation hurts more than reasoning helps
```

## Code
[[concepts/optimal-cot-length]]
[cot_optimal.py](../prototypes/experiments/cot-length/cot_optimal.py)