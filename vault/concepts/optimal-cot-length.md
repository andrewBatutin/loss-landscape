# CoT Optimal Length 

## Intuition

CoT has optimal context length and it scales `sqrt(n)` where n - context length
This is dictated by positional encoding degrading performance on long chains.

- Short CoT - not enough reasoning -> bad answer
- Long CoT - positional noise drowns out signal -> bad answer
- Sweet spot = `sqrt(n)` scaling

## Essence

```
k* = c × √n × log(1/ε)
```
- k* - optimal number of CoT tokens
- n - context length 
- eps - error tolerance (how much performance are you willing to lose)
- c - model specific constant (needs estimation)

Averaging predictions transformer becomes True Bayesian - without computational overhead of calculating Bayesian inference.
Transformer implicitly operates as Bayesian inference formin during test time compute priors and updated it as part of ICL


## Findings

You can calibrate your predictor by shuffling the input n times and running through transformer to get position independent prediction
That can be used to harden LLM-as-Judge or other LLM based classifiers. 

## Connections

- [[concepts/positional-encoding]] - positional encoding
- [[concepts/bayesian-martingale-property]] -  bayesian martingale property
- [[concepts/prediction-is-compression]] - If you can compress you can predict
- [[concepts/in-context-learning]] - ICL
- [[concepts/llm-as-judge]] - LLM-as-Judge

## Open Questions

### What is optimal `c` for specific model?

```
c = sqrt(α / (H_CoT × (B_0 - B_opt)))

where:
  α      = benefit curvature (~5 bits typical)
  H_CoT  = entropy of reasoning tokens (~3 bits/token)
  B_0    = prediction loss with k=0 (no CoT)
  B_opt  = best achievable prediction loss
```
These are model and task specific. 

### Generalisation 

Need to test. 

## Summary 

| What we know | Confidence |
|--------------|------------|
| CoT has diminishing returns | ✅ High (obvious empirically) |
| There's an optimal k | ✅ High (too long = position noise) |
| Shape is √n | 🤷 Theoretical, not validated |
| Exact formula works | ❌ Not tested by authors |

**The prototype that matters:**

```python
# Actually measure this for one model on one task
# Report: does accuracy peak and decline?
# Report: does √n fit better than linear?
# Report: what's the estimated c?
```

## Papers

- [[papers/llm_are_bayesian_in_expectation.md]]
