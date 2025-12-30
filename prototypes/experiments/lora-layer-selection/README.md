---
title: LoRA Layer Selection
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# LoRA Layer Selection

**Hypothesis**: Not all layers need LoRA adaptation.

**Finding**: 71% parameter reduction by targeting middle attention layers (8-24). Rare class accuracy maintained.

[Project notes](https://github.com/AiCapitalist/digital-garden/blob/main/vault/projects/lora-layer-selection.md)
