from dataclasses import dataclass

@dataclass(kw_only=True)
class OptimizerConfigs:
    lr: float | None = None
    adam_beta1: float | None = None
    adam_beta2: float | None = None
    adam_eps: float | None = None
    weight_decay: float | None = None
    clip_grad: float | None = None

    min_lr: float | None = None
    warmup_iters: int | None = None

