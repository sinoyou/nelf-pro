import os
from nerfstudio.configs.base_config import Config

def check_load_step(config: Config):
    """pre check whether load_dir exists and return load_step is specified. """
    load_dir = config.trainer.load_dir
    if load_dir is None:
        load_dir_try =  config.get_checkpoint_dir()
        if load_dir_try.exists():
            load_dir = load_dir_try
    if load_dir is not None:
        load_step = config.trainer.load_step
        if load_step is None:
            # NOTE: this is specific to the checkpoint name format
            load_step = sorted(int(x.replace('-', '.').split('.')[-2]) for x in os.listdir(load_dir))[-1]
        return load_step
    else:
        return None