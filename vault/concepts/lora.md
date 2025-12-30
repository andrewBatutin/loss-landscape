# LoRA

## Intuition

Full fine-tuning updates W by ΔW. But ΔW is low-rank in practice. So don't store full ΔW—decompose it into two small matrices.

## Essence

```python
# Standard: output = W @ x
# LoRA:     output = W @ x + (B @ A) @ x
#                    ↑ frozen   ↑ trainable, tiny

# W: (d, d) = millions of params
# A: (d, r) + B: (r, d) where r << d = thousands of params
```

## Connections

- [[concepts/quantization]] - QLoRA combines 4-bit base + LoRA
- [[concepts/layer-selection]] - not all layers need adaptation
- [[concepts/verifier-primacy]] - train verifier LoRAs, not generator

## Open Questions

- Why do attention layers need lower rank than FFN?
- Optimal rank scheduling during training?

## Papers

- [[papers/lora-original]]
- [[papers/qlora]]
