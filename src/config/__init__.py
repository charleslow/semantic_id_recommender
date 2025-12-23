from omegaconf import OmegaConf
from pathlib import Path


def load_config(config_path: str | Path | None = None) -> OmegaConf:
    """Load configuration from yaml file."""
    if config_path is None:
        config_path = Path(__file__).parent / "default.yaml"
    return OmegaConf.load(config_path)


__all__ = ["load_config"]
