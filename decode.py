import torch
import numpy as np
from module.functions import softmax
from module.model import TransformerLM
from module.data_loader import get_batch

@torch.no_grad()
def decode(
    model: torch.nn.Module,
    inputs: torch.Tensor, 
    max_number_of_generated_tokens: int,
    end_of_text_token_id: int,
    temperature: float, 
    top_p: int,
):
    '''
    support temperature and top-p sampling.
    user can define their own max number of generated tokens,
    but still ends if end_of_text_token_id is met.

    Currently does not support kv cache, so will be slow, as it repeats prefilling ops.
    '''

    assert temperature >= 0, print(f"temperature ({temperature}) cannot be negative!")
    finished = torch.zeros(inputs.shape[0], dtype=inputs.dtype, device=inputs.device)
    generations = inputs
    for _ in range(max_number_of_generated_tokens):
        inputs = generations[:, -context_length:]
        logits = model(inputs)
        generation = logits[:, -1, :]
        
        if temperature > 0:
            generation /= temperature
            probs = softmax(generation, dim=-1)
            if top_p < 1:
                sorted_probs, sorted_ids = torch.sort(
                    probs, dim=-1, descending=True
                )
                cumulated_probs = torch.cumsum(sorted_probs, dim=-1)
                removed = cumulated_probs - cumulated_probs >= top_p
                cumulated_probs.mask_fill(removed, 0)
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(sorted_probs, 1)
                next_token = torch.gather(sorted_ids, dim=-1, index=sampled)
            else:
                next_token = torch.multinomial(
                    probs, num_samples=1, replacement=False
                )
        else:
            next_token = torch.argmax(
                softmax(generation, dim=-1),
                dim=-1,
                keepdim=True
            )
        next_token = torch.where(
            finished[:, None],
            torch.full_like(next_token, end_of_text_token_id),
            next_token
        )
        generations = torch.concat(
            (generations, next_token), dim=1
        )
        finished |= next_token.squeeze(-1).eq(end_of_text_token_id)
        if finished.all():
            break
            
    return generations

if __name__ == '__main__':
    context_length = 256
    batch_size = 1
    vocab_size = 10_000
    num_layers = 4
    num_heads = 16
    d_model = 512
    d_ff = 1344
    rope_theta = 10_000
    dtype = torch.float32
    device = torch.device('cuda')

    model = TransformerLM(
        context_length=context_length,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        d_model=d_model,
        d_ff=d_ff,
        theta=rope_theta,
        dtype=dtype,
        device=device
    )
    model.eval()

    eval_data_np = np.load(
        '/pscratch/sd/m/mzheng/cs336_data/tinystories_valid.npy',
        mmap_mode='r'
    )

    inputs, _ = get_batch(
        dataset=eval_data_np,
        batch_size=batch_size,
        context_length=context_length,
        device='cuda'
    )

    generations = decode(
        inputs=inputs,
        model=model,
        max_number_of_generated_tokens=64,
        top_p=100,
        temperature=0,
        end_of_text_token=0
    )
