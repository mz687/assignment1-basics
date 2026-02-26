from torch.nn import Module
import torch
import torch.nn as nn
import math
from einops import einsum

class Embedding(Module):
    def __init__(self, 
                 num_embeddings, 
                 embedding_dim, 
                 device:torch.device | None=None, 
                 dtype:torch.dtype | None=None):
        '''
        num_embeddings: int, size of vocab
        embedding_dim: d_model
        '''
        super(Embedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.weights = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim)
        ).to(dtype=dtype, device=device)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.trunc_normal_(
            self.weights,
            mean=0,
            std=1,
            a=-3,
            b=3
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        one_hot = nn.functional.one_hot(
            token_ids, num_classes=self.num_embeddings
        ).to(self.weights.dtype)
        return einsum(
            self.weights, 
            one_hot,
            'vocab_size d_model, ... vocab_size -> ... d_model'
        )
    
