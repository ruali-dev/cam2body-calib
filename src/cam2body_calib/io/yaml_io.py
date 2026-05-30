"""YAML I/O utilities."""

from pathlib import Path

import yaml


def read_yaml(path: str | Path) -> dict:
    """Read and parse a YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed dict from YAML content.

    Raises:
        FileNotFoundError: If file does not exist.
        yaml.YAMLError: If YAML is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
