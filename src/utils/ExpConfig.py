from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class Args:
    trial_name: str
    main_funcname: str
    debug: bool
    device: str
    model_name: str
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
    lr: Any = field(default=2e-5)
    dropout: Any = field(default=None)
    thr: float = field(default=None)
    id_to_0: List[int] = field(default=None)
    id_to_1: List[int] = field(default=None)
    max_length: int = 512
    base_model_name: str = field(default=None)
    margin: float = field(default=None)
    path_idx_1_1: str = field(default=None)
    path_idx_1_0: str = field(default=None)
    path_idx_0_0: str = field(default=None)

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
    fname_train: str
    fname_test: str
    fname_sub: str
    dropout: Any = field(default=None)
    thr: float = field(default=None)
    id_to_0: List[int] = field(default=None)
    id_to_1: List[int] = field(default=None)
    max_length: int = 512

