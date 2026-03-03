import math

def learning_rate_schedule(t: int,
                           max_learning_rate: float,
                           min_learning_rate: float,
                           warmup_iters: int,
                           cosine_cycle_iters: int):
    '''
    Implement the cosine lr scheduler used in Llama.
    '''
    if t < warmup_iters:
        lr_t = t / warmup_iters * max_learning_rate
    elif warmup_iters <= t <= cosine_cycle_iters:
        lr_t = min_learning_rate + 1/2 * (1 + math.cos((t - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        lr_t = min_learning_rate
    return lr_t