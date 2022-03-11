from typing import Iterator
import torch

class TensorCollector:

    def dump_to_disk(self, value: object) -> int:
        pass
    
    def load_from_disk(self, fidx: int):
        pass

    def clear_disk():
        pass

    def __init__(
        self, device: str, working_directory: str, shuffle: True) -> None:
        self.container   = []
        self.working_dir = working_directory
        self.device      = str(device).lower()
        self.shuffle     = shuffle

    def __iter__(self) -> Iterator:
        if self.device == 'disk': raise PermissionError(
            'Can not iterate a tensor collector which device = disk.')
        return self.container.__iter__()

    @ torch.no_grad()
    def push(self, value: torch.Tensor):
        if self.device == 'keep':
            self.container.append(value)
        if self.device == 'cpu':
            self.container.append(value.detach().cpu())
        if self.device == 'disk':
            self.container.append(self.dump_to_disk(value))
    
    def pop(self, idx = None) -> torch.Tensor:
        if self.shuffle:
            pass