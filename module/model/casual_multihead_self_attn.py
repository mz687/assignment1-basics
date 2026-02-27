from torch.nn import Module
import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

from .linear import Linear
from module.rope import RotaryPositionalEmbedding
from module.functions import scaled_dot_product_attention

class CasualMultiheadSelfAttn(Module):
    def __init__(self, 
                 d_model:int, 
                 num_heads: int,
                 theta: float | None = None,
                 max_seq_len: int | None = None,
                 device: torch.device | None = None):
        '''
        d_model: the hidden dim of the model.
        num_heads: number of heads in multi-head self attn.
        theta, max_seq_len are for RoPE (optional).
        '''
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads

        self.q_proj = Linear(
            in_features=self.d_model,
            out_features=self.d_k*self.num_heads,
        )

        self.k_proj = Linear(
            in_features=self.d_model,
            out_features=self.d_k*self.num_heads,
        )

        self.v_proj = Linear(
            in_features=self.d_model,
            out_features=self.d_v*self.num_heads,
        )

        self.o_proj = Linear(
            in_features=self.d_v*self.num_heads,
            out_features=self.d_model
        )

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                theta = theta,
                d_k = self.d_k, 
                max_seq_len = max_seq_len,
                device=device
            )


    def forward(self, 
                x:torch.Tensor, 
                token_positions:torch.Tensor|None = None) -> torch.Tensor:
        assert x.shape[-1] == self.d_model

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        q_projed = self.q_proj(x)
        k_projed = self.k_proj(x)
        v_projed = self.v_proj(x)
        q_projed_chunks = torch.chunk(q_projed, chunks=self.num_heads, dim=-1)
        k_projed_chunks = torch.chunk(k_projed, chunks=self.num_heads, dim=-1)
        v_projed_chunks = torch.chunk(v_projed, chunks=self.num_heads, dim=-1)
        head_results = []
        for h in range(self.num_heads):
            if hasattr(self, 'rope'): # apply the same rope to every head's q and k (not v)
                token_positions = token_positions if token_positions is not None else torch.arange(seq_len)
                q_projed_chunk = self.rope(q_projed_chunks[h], token_positions)
                k_projed_chunk = self.rope(k_projed_chunks[h], token_positions)
            else:
                q_projed_chunk = q_projed_chunks[h]
                k_projed_chunk = k_projed_chunks[h]

            head_results.append(
                scaled_dot_product_attention(
                    q_projed_chunk,
                    k_projed_chunk,
                    v_projed_chunks[h],
                    mask
                )
            )
        concated = torch.concat(head_results, dim=-1)
        return self.o_proj(concated)