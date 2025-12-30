# LoRA: Low-Rank Adaptation of Large Language Models

## Source

https://arxiv.org/abs/2106.09685

## One-liner

Freeze base weights, inject trainable low-rank matrices into attention layers.

## Load-bearing insight

Weight updates during fine-tuning have low intrinsic rank. You don't need full ΔW.

## What most miss

The rank r=8 works across most tasks. The paper tested up to r=64 with diminishing returns.

## Concepts

[[concepts/lora]] · [[concepts/fine-tuning]] · [[concepts/parameter-efficiency]]

## Project

[[projects/lora-layer-selection]] - which layers actually need adaptation?

## Essence

```python
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W.weight.requires_grad = False  # frozen
        
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        self.scale = alpha / rank
        
    def forward(self, x):
        return self.W(x) + self.B(self.A(x)) * self.scale
```

## Code

[lora_layer.py](../prototypes/experiments/lora-layer-selection/lora_layer.py)
