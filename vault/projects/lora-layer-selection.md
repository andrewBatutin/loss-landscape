# LoRA Layer Selection

## Hypothesis

Not all layers need LoRA adaptation. Selective application can reduce parameters while maintaining accuracy.

## Status

✅ Complete - presented at Data Science Summit 2025

## Finding

71% parameter reduction by targeting only high-impact layers. Rare class accuracy maintained.

## Method

1. Train full LoRA baseline
2. Measure gradient norms per layer
3. Prune low-gradient layers
4. Validate on held-out rare classes

## Essence

```python
# The insight: attention layers in middle blocks dominate
important_layers = [
    f"model.layers.{i}.self_attn" 
    for i in range(8, 24)  # middle layers only
]
# Skip: embeddings, early layers, late layers, most FFN
```

## Links

- Paper: [[papers/lora-original]]
- Concepts: [[concepts/lora]] · [[concepts/layer-selection]]
- Code: [experiment](../prototypes/experiments/lora-layer-selection/)
- Demo: https://huggingface.co/spaces/AiCapitalist/lora-layer-viz
- GitHub: https://github.com/AiCapitalist/lora-layer-selection

## Post

[[published/lora-layer-selection]]
