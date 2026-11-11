"""Typed-core CLI entrypoint.

Public mode:
    python -m griffin --config FILE
"""

from __future__ import annotations

import argparse
import sys

from griffin.infra.runtime_logging import configure_cli_logging

from .dispatch import run_task_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="griffin")
    parser.add_argument("--config", required=True, help="path to a YAML/JSON task config")
    args = parser.parse_args(argv)
    runtime_logger = configure_cli_logging()

    try:
        run_task_path(args.config, runtime_logger=runtime_logger)
        return 0
    except Exception as exc:
        runtime_logger.error("[任务] 执行失败：%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
