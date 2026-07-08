import torch
import numpy as np 
import os 
import typing

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    model_state_dict = model.state_dict()
    optm_state_dict = optimizer.state_dict()
    to_be_saved = {
        "model": model_state_dict,
        "optimizer": optm_state_dict,
        "iteration": iteration
    }
    torch.save(to_be_saved, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    saved_dict = torch.load(src)
    model_state_dict = saved_dict['model']
    optm_state_dict = saved_dict['optimizer']
    iteration = saved_dict['iteration']

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optm_state_dict)
    return iteration