from torch.nn import Module
import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

from module.rope import RotaryPositionalEmbedding
from .casual_multihead_self_attn import CasualMultiheadSelfAttn
from .SwiGLU import SwiGLU
from module.RMSNorm import RMSNorm

class TransformerBlock(Module):
    '''
    This is the pre-norm transformer block.
    The RMSNorm is applied before MultiHeadAttention and FFN.
    Remember it has two residual connections.
    '''
    def __init__(self, 
                 d_model:int, 
                 num_heads:int, 
                 d_ff:int,
                 theta:float,
                 max_seq_len:int,
                 device:torch.device | None = None):
        '''
        d_model: dim of transformer block's input.
        num_heads: number of heads to use in multi-head attn.
        d_ff: dim of the FFN hidden layer
        '''
        super(TransformerBlock, self).__init__()

        self.swiglu = SwiGLU(
            d_model = d_model,
            d_ff = d_ff
        )
        self.mha = CasualMultiheadSelfAttn(
            d_model = d_model,
            num_heads = num_heads,
            theta = theta,
            max_seq_len = max_seq_len,
            device = device
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.norm1(x))
        x = x + self.swiglu(self.norm2(x))
        return x



