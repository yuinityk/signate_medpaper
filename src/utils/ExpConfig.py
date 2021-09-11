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
    max_length: int = 512

@dataclass
class SubmitArgs:
    trial_name: str
    main_funcname: str
    debug: bool
    device: str
    model_name: str
    seed: int
    batch_size: int
    num_workers: int
    save_path: str
    kfold_nsplit: int
    dir_path: str
    ensemble_type: str
    fname_train: str
    fname_test: str
    fname_sub: str
    thr: float = field(default=None)
    id_to_0: List[int] = field(default=None)
    id_to_1: List[int] = field(default=None)
    max_length: int = 512

