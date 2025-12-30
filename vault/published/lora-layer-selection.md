# LoRA Layer Selection

## Meta

- Date: 2025-01-15
- Platform: LinkedIn
- Project: [[projects/lora-layer-selection]]
- Demo: https://huggingface.co/spaces/AiCapitalist/lora-layer-viz

## Content

🔬 71% fewer LoRA parameters. Same accuracy.

Everyone slaps LoRA on all layers. We tested which actually matter.

The finding: middle attention layers (8-24) do the work. Early/late layers and most FFN? Skip them.

[Link to demo]

---

This started as a hunch during our Qwen fine-tuning. Gradient norms told the story—some layers barely moved.

The "apply everywhere" default is convenience, not optimization.

## Performance

- Impressions: —
- Engagement: —
- Comments: —
