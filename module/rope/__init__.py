from torch.nn import Module
import torch
import torch.nn as nn
import math
from einops import einsum, rearrange


class RoPE(Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None):
        super(RoPE, self).__init__()

    def forward(self, 
                x:torch.Tensor,
                token_positions: torch.Tensor) -> torch.Tensor:
        pass
