import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr: float,
                 betas: tuple[float],
                 eps: float,
                 weight_decay: float):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                step = state.get('step', 0)
                exp_avg = state.get('exp_avg', torch.zeros_like(p))
                exp_avg_sq = state.get('exp_avg_sq', torch.zeros_like(p))

                # update exp_avg and exp_avg_sq
                step += 1
                exp_avg.mul_(beta1)
                exp_avg.add_(p.grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2)
                exp_avg_sq.addcmul_(p.grad, p.grad, value=1-beta2)
                lr_t = lr * math.sqrt(1-math.pow(beta2, step)) / (1-math.pow(beta1, step))
                denom = torch.sqrt(exp_avg_sq) + eps
                update = lr_t * exp_avg / denom + lr * weight_decay * p 

                state['step'] = step
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq

                p.data.sub_(update)
