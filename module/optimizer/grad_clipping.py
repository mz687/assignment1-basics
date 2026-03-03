import torch
import math
from collections.abc import Iterable

def grad_clip(params: Iterable[torch.nn.Parameter],
        max_l2_norm: float,
        eps: float = 1e-6):
    '''
    Global-norm clipping: norm is computed based on all grad's norm 
    norm = \sqrt(\sum_i(||g_i||_2^2))
    '''
    norm = math.sqrt(
        sum([
            math.pow(torch.linalg.norm(p.grad), 2) 
            for p in params if p.grad is not None
        ])
    )
    if norm > max_l2_norm:
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(max_l2_norm / (norm + eps))