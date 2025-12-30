"""
LoRA Layer - The Essence

The insight: Weight updates during fine-tuning have low intrinsic rank.
Instead of updating full W, add a small trainable detour: B @ A
"""

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    output = W @ x + (B @ A) @ x
             ↑ frozen   ↑ trainable, tiny
    """
    
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        
        # Original weights - frozen
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W.weight.requires_grad = False
        
        # LoRA weights - trainable
        # A: (in_dim, rank), B: (rank, out_dim)
        # Total params: rank * (in_dim + out_dim) << in_dim * out_dim
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        
        # Scaling factor
        self.scale = alpha / rank
        
        # Init: A gaussian, B zero (start at original model)
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen path + trainable low-rank path
        return self.W(x) + self.B(self.A(x)) * self.scale


# That's it. The paper adds benchmarks and ablations.
# The insight is 20 lines.
