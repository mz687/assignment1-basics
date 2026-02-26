from torch.nn import Module
import torch
import torch.nn as nn
import math

from einops import einsum

class Linear(Module):
    def __init__(self, 
                 in_features, 
                 out_features,
                 device:torch.device | None=None,
                 dtype:torch.dtype | None=None):
        '''
        in_features: number of input features.
        out_features: number of output features.
        Assume there is no bias.
        '''
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(
            torch.empty(out_features, in_features)
        ).to(dtype=dtype, device=device)

        self._init_parameters()
    
    def _init_parameters(self):
        '''
        Initialize the weight, so that it's normal(mean=0, var=2/(d_in+d_out)) 
        \in [-3*std, 3*std]
        '''
        var = 2/(self.in_features+self.out_features)
        std = math.sqrt(var)
        nn.init.trunc_normal_(
            self.weights,
            mean=0,
            std=std,
            a=-3*std,
            b=3*std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return  einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")