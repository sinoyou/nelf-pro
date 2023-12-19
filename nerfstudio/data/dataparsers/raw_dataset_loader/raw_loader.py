from abc import abstractmethod
import os
import numpy as np
from pathlib import Path, PurePath
from PIL import Image

class RawLoader:
    def __init__(self, data_dir: Path, downscale_factor: int = None, partition_index: tuple = None, **kwargs):
        self.data_dir = data_dir
        self.downscale_factor = downscale_factor
        self.partition_index = partition_index
        self.other_args = kwargs
    
    def get_loaded_data(self, **args) -> dict:
        raise NotImplementedError
    
    def get_downscale_factor(self) -> int:
        return self.downscale_factor
    
    @abstractmethod
    def get_train_val_indices(self, eval_interval):
        raise NotImplementedError