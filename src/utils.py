"""Shared utility functions for the GalaxyPPG HRV pipeline.

Provides configuration loading for the pipeline entry points.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> dict:
    """Load pipeline configuration from a YAML file.

    Args:
        config_path: Path to the ``config.yaml`` file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
