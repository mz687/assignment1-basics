from torch.nn import Module
import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

from .linear import Linear

class SwiGLU(Module):
    def __init__(self, 
                 d_model:int, 
                 d_ff:int, 
                 device:torch.device|None = None, 
                 dtype:torch.dtype|None = None):
        '''
        d_ff: canonical val: 8/3 * d_model
        SwiGLU(x) = SwiGLU(x, w1, w2, w3) = w2(SiLU(w1@x)*w3@x)
        SiLU(x) = x * sigmoid(x)
        GLU(x, w1, w2) = sigmoid(w1@x) * w2@x
        '''
        super(SwiGLU, self).__init__()
        self.linear1 = Linear(d_ff, d_model)
        self.linear2 = Linear(d_model, d_ff)
        self.linear3 = Linear(d_ff, d_model)

    def SiLU(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(torch.sigmoid(x), x, '..., ... -> ...')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear1.weights.data.shape == self.linear3.weights.data.shape
        return self.linear2(
            einsum(
                self.SiLU(self.linear1(x)),
                self.linear3(x),
                '..., ... -> ...'
            )
        )
        