from torch.nn import Module
import torch
import torch.nn as nn
import math
from einops import einsum, rearrange

from .transformer_block import TransformerBlock
from module.RMSNorm import RMSNorm
from .linear import Linear
from .embedding import Embedding
from module.functions import softmax

class TransformerLM(Module):
    '''
    This is the final model after putting all ingradients together.

    First is the embedding layer,
    then the transformer layers,
    then an RMSNorm,
    then a linear layer,
    finally a softmax
    '''

    def __init__(self, 
                 vocab_size:int, 
                 context_length:int, 
                 num_layers:int,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 theta: float,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        '''
        d_model: dim of transformer block's input.
        num_heads: number of heads to use in multi-head attn.
        d_ff: dim of the FFN hidden layer
        theta: for RoPE

        In addition to all the existing hyper-parameters, we have
        vocab_size: size of the vocabulary (for token embedding layer).
        context_length: max_seq_len (for positional embedding).
        num_layers: number of transformer blocks.
        '''
        super().__init__()

        self.num_layers = num_layers

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )

        self.transformer_blocks = [
            TransformerBlock(
                d_model = d_model,
                num_heads = num_heads,
                d_ff = d_ff,
                theta = theta,
                max_seq_len = context_length
            ) for _ in range(num_layers)
        ]

        self.output_norm = RMSNorm(
            d_model = d_model,
            device = device,
            dtype = dtype
        )

        self.output_embedding = Linear(
            in_features = d_model,
            out_features = vocab_size
        )

    def forward(self,
                in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)

        for i in range(self.num_layers):
            x = self.transformer_blocks[i](x)
        x = self.output_norm(x)
        x = self.output_embedding(x)
        return softmax(x, dim=-1)

