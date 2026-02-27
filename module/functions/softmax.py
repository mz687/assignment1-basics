from torch.nn import Module
import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    dim: the dim to apply softmax on.

    Performs softmax on x.
    Offset if the largest x per-row. 
    Subtract it for stability reason.
    '''
    offset = torch.max(x, dim=dim, keepdims=True).values
    x_offseted = x - offset
    exp_x = torch.exp(x_offseted)
    denom = exp_x.sum(dim=dim, keepdims=True)
    return exp_x / denom
