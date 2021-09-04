from dataclasses import dataclass, field
from typing import List


@dataclass
class Args:
    trial_name: str
    main_funcname: str
    debug: bool
    device: str
    model_name: str
    lr: float
    step_scheduler: bool
    epoch_scheduler: bool
    seed: int
    start_epoch: int
    epochs: int
    batch_size: int
    max_length: int = 512
    num_workers: int
    save_path: str
    kfold_nsplit: int
    dir_path: str
    loss: str
    fname_train: str
    fname_test: str
    fname_sub: str
    thr: float = field(default=None)
    id_to_0: List[int] = field(default=None)
    id_to_1: List[int] = field(default=None)

