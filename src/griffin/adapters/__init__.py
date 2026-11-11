"""Public adapters for the single typed-core architecture."""

from .cli import main
from .dispatch import run_task_config, run_task_path

__all__ = ["main", "run_task_config", "run_task_path"]
