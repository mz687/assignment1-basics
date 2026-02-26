from torch.nn import Module
import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

class RMSNorm(Module):
    def __init__(self, 
                 d_model: int,
                 eps:float = 1e-5,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        '''
        d_model: Hidden dim of the model,
        eps: hp, but usually fixed to 1e-5

        Each d_model dim has its own scaling factor gamma (a learnable param)
        Formula: RMSNorm(a_i) = a_i / RMS(a) * gamma_i,
        where a is the input tensor of shape (..., d_model), and RMS(a) = \sqrt( a^T \times a / d_model + eps)
        '''
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d_model = d_model

        self.weights = nn.Parameter(torch.empty(self.d_model)).to(dtype, device)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.ones_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) # avoid overflow in RMS's square

        rms = torch.sqrt(rearrange(
            einsum(x, x, '... d_model, ... d_model -> ...'), 
            '... -> ... 1'
        ) / self.d_model + self.eps)
        result = einsum(self.weights, x, '... d_model, ... d_model -> ... d_model') / rms

        return result.to(in_dtype)