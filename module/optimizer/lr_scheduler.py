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

class LRScheduler:
    '''
    Only support cosine lr decay for this assignment.
    '''

    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        max_learning_rate: float,
        min_learning_rate: float,
        cosine_cycle_iters: int,
        warmup_iters: int,
        t: int = 1
    ):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.cosine_cycle_iters = cosine_cycle_iters
        self.warmup_iters = warmup_iters

        # support load checkpoint to reset the iter in scheduler
        self.t = t

        self.optimizer = optimizer
    
    def get_last_lr(self) -> list[float]:
        lrs = []
        for group in self.optimizer.param_groups:
            lrs.append(group['lr'])
        return lrs
    
    def step(self):
        lr = learning_rate_schedule(
            t=self.t,
            max_learning_rate=self.max_learning_rate,
            min_learning_rate=self.min_learning_rate,
            warmup_iters=self.warmup_iters,
            cosine_cycle_iters=self.cosine_cycle_iters
        )

        for group in self.optimizer.param_groups:
            group['lr'] = lr
        
        self.t += 1