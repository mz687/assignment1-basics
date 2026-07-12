from dataclasses import dataclass 

@dataclass(kw_only=True)
class ModelConfigs:
    vocab_size: int | None = None
    context_length: int | None = None
    num_layers: int | None = None
    d_model: int | None = None
    num_heads: int | None = None
    d_ff: int | None = None
    theta: float | None = None
    

