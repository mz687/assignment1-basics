from torch.nn import Module
import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

from .softmax import softmax

def scaled_dot_product_attention(
    queries:torch.Tensor,
    keys:torch.Tensor,
    values:torch.Tensor,
    masks:torch.Tensor | None = None
) -> torch.Tensor:
    '''
    queries: (batch_size, .., seq_len, d_k)
    keys: (batch_size, .., seq_len, d_k)
    values: (batch_size, .., seq_len, d_v)
    masks: (seq_len, seq_len): If mask[i,j] is True, then if means the qk result should flow to v,
                                therefore, be careful with masked_fill_ which fills based on the True of mask.
    '''
    d_k = keys.shape[-1] 
    qk = einsum(
        queries,
        keys,
        'batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k'
    )
    qk.div_(math.sqrt(d_k)) # scaled qk^T
    if masks is not None:
        qk.masked_fill_(~masks, -torch.inf)

    return einsum(
        softmax(qk, dim=-1),
        values,
        'batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v'
    )

