from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    *,
    name: str,
    log_file: Path,
    level: int = logging.INFO,
    to_console: bool = False,
) -> logging.Logger:
    """
    Create a logger writing to `log_file`.
    Safe to call multiple times (no duplicate handlers).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(formatter)
    fh.setLevel(level)
    logger.addHandler(fh)

    if to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)

    logger.info("Logger initialized")
    return logger


def get_run_logger(
    *,
    output_dir: str,
    run_log_name: Optional[str],
    to_console: bool,
) -> logging.Logger:
    """
    One log file for the entire run.
    """
    name = run_log_name or "drsci_run"
    log_path = Path(output_dir) / "logs" / f"{name}.log"
    return setup_logger(
        name=f"drsci.run.{name}",
        log_file=log_path,
        to_console=to_console,
    )


def get_instance_logger(
    *,
    instance_name: str,
    output_dir: str,
    to_console: bool,
) -> logging.Logger:
    """
    One log file per instance.
    """
    base = Path(instance_name).stem
    log_path = Path(output_dir) / "logs" / f"{base}.log"
    return setup_logger(
        name=f"drsci.instance.{base}",
        log_file=log_path,
        to_console=to_console,
    )
