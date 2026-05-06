from dataclasses import dataclass

import torch


@dataclass
class AttentionOutput:
    output: torch.Tensor
    attention_weights: torch.Tensor
