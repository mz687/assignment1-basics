import torch
from torch import Tensor
from jaxtyping import Float, Int

def cross_entropy_loss(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    batch_size, vocab_size = inputs.shape
    assert batch_size == targets.shape[0]
    # prenorm for stability
    maxes = torch.max(inputs, dim=1, keepdims=True)[0]
    inputs -= maxes
    targets_vals = torch.gather(inputs, dim=1, index=targets.unsqueeze(1))
    # compute the cross entropy loss
    cross_entropy_loss = torch.sum(torch.log(torch.sum(torch.exp(inputs), dim=1, keepdims=True)) - targets_vals, dim=0) / batch_size 
    return cross_entropy_loss