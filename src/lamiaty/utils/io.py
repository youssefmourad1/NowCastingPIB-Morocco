"""
I/O helpers — Parquet, Excel, and JSON utilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    path = Path(path)
    df = pd.read_parquet(path, engine="pyarrow")
    logger.debug("Read %s (%d rows × %d cols)", path, len(df), len(df.columns))
    return df


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame to Parquet, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    logger.info("Wrote %s (%d rows × %d cols)", path, len(df), len(df.columns))


def read_json(path: str | Path) -> dict:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: str | Path, indent: int = 2) -> None:
    """Write a dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.debug("Wrote JSON to %s", path)


def read_excel(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read an Excel file."""
    return pd.read_excel(path, engine="openpyxl", **kwargs)
