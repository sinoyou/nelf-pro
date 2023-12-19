from nerfstudio.configs.base_config import PrintableConfig

def merge_customized_config(default_config: PrintableConfig, customized_config: dict) -> PrintableConfig:
    for k, v in customized_config.items():
        if isinstance(v, dict) and isinstance(default_config.__dict__[k], PrintableConfig):
            merge_customized_config(default_config.__dict__[k], v)
        else:
            assert k in default_config.__dict__, f"Key {k} is not in default config."
            default_config.__dict__[k] = v
    return default_config