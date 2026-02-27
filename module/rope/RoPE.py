from torch.nn import Module
import torch
import torch.nn as nn
import math
from einops import einsum, rearrange


class RotaryPositionalEmbedding(Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device: torch.device | None = None):
        super(RotaryPositionalEmbedding, self).__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        token_positions = torch.arange(max_seq_len)
        ks = torch.arange(1, d_k//2+1)
        # tokens_pos_grad, ks_grad = torch.meshgrid(token_positions, ks, indexing='ij')
        # blocks = self.gen_block(tokens_pos_grad, ks_grad, theta, d_k)
        # blocks = torch.block_diag(blocks).to(device=device if device is not None else 'cpu')
        blocks = torch.stack([
            torch.block_diag(
                *[rearrange(x, '1 ... -> ...') for x in torch.chunk(self.gen_block(i, ks, theta, d_k), chunks=d_k//2, dim=0)]
            ) for i in range(max_seq_len)
        ], dim=0)
        self.register_buffer('rope_buffer', blocks, persistent=False)

    def gen_block(self, i:torch.Tensor, k:torch.Tensor, theta:float, d:int):
        '''
        i: token position,
        k: range from 1 to d_k//2
        '''
        angle = i / torch.pow(theta, (2*k-2)/d)
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        return torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin, cos], dim=-1),
        ], dim=-2)

    def forward(self, 
                x:torch.Tensor,
                token_positions: torch.Tensor) -> torch.Tensor:
        rope_selected = self.rope_buffer[token_positions, :, :]
        return einsum(
            rope_selected,
            x,
            '... seq_len d_k_1 d_k_2, ... seq_len d_k_2 -> ... seq_len d_k_1'
        )
