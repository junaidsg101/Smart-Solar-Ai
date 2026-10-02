import os
import toml


def load_config(config_path: str = "config.toml") -> dict:
    if os.path.exists(config_path):
        return toml.load(config_path)
    return {}


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

