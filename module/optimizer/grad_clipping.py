import torch
import math

def grad_clip(params: list[torch.nn.Parameter],
        max_l2_norm: float,
        eps: float = 1e-6):
    for p in params:
        if p.grad is None:
            continue
        
        norm = torch.linalg.norm(p.grad)
        if norm > max_l2_norm:
            p.grad.mul_(max_l2_norm / (norm + eps))