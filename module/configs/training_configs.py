from dataclasses import dataclass

@dataclass(kw_only=True)
class TrainingConfigs:
    batch_size: int | None = None
    train_iters: int | None = None
    save: str | None = None
    load: str | None = None
    save_interval: int | None = None
    data_path: str | None = None

