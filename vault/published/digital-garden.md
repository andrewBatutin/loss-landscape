# Digital Garden

## Meta

- Date: 2025-12-30
- Platform: LinkedIn, Substack
- Project: [[projects/project-name]]
- Demo: [link](https://github.com/andrewBatutin/loss-landscape)

## Content

I read 50+ papers last year. Retained maybe 5 insights.

Problem: papers are write-once. Understanding requires iteration.

Solution: a research garden that forces extraction.

Just shipped mine: https://github.com/andrewBatutin/loss-landscape

Structure:
- concepts/ → one insight per file
- papers/ → what most people miss  
- projects/ → hypothesis → code → finding
- experiments/ → runnable reproductions

Rule: nothing goes in without answering "what's the load-bearing intuition?"

---

**First project: LLM-as-Judge position bias**

Why can't we just use LLM-as-judge directly?

Transformers have positional encodings. Order matters. "A vs B" ≠ "B vs A" even with identical content.

A Bayesian judge would satisfy: P(A better | A first, B second) = P(A better | B first, A second)

Transformers violate this. They're trained on ordered sequences, so position leaks into judgment.

**The fix: permutation averaging**

Run both orderings, average the votes:
```
Score = 0.5 × judge(A,B) + 0.5 × judge(B,A)
```

Position bias cancels. Content signal remains.

Paper "LLMs are Bayesian in Expectation" proves this formally: transformers minimize expected complexity over orderings, achieving Bayesian-optimal compression *on average* even though individual predictions are order-dependent.

**What I found testing this:**

| Model | Consistency | After calibration |
|-------|-------------|-------------------|
| Llama-3B | 50% | Still 50% (no signal to preserve) |
| GPT-4o-mini | 90% | 97% accuracy |

The insight: calibration doesn't create signal, it reveals whether signal exists. Small models have nothing to average — they just learn "pick first."

Frontier models have real judgment. Calibration protects the 10% of edge cases where position would have flipped the answer.

**Cost-benefit:**
- 20x compute for +10% accuracy on edge cases
- Worth it for high-stakes evals, overkill for bulk filtering

Code included. Test your models in 5 minutes.

## Performance

- Impressions: —
- Engagement: —
- Notable comments: —
