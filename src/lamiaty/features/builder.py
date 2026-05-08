"""
Feature matrix builder — assembles the final DFM-ready feature set.

Reads data_sources.yaml metadata (via Settings) and enforces the
include_in_model flag, ensuring excluded series (e.g., Investissement_Etat
pending TGR confirmation) never reach the estimation step.
"""

from __future__ import annotations

import logging

import pandas as pd

from lamiaty.config.settings import Settings

logger = logging.getLogger(__name__)


def build_feature_matrix(panel: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Filter and order the model panel to the DFM feature set.

    Drops any columns flagged include_in_model: false in data_sources.yaml.
    Separates the target variable (va_construction) to facilitate model setup.

    Args:
        panel: Full mixed-frequency panel from run_pipeline().
        settings: Settings object with data_sources metadata.

    Returns:
        DataFrame containing only DFM-eligible series (including the target).
        Same DatetimeIndex as input.
    """
    ds = settings.data_sources
    excluded = [
        key for key, cfg in ds.items()
        if not cfg.get("include_in_model", True)
    ]

    included_cols = [c for c in panel.columns if c not in excluded]
    excluded_present = [c for c in excluded if c in panel.columns]

    if excluded_present:
        logger.info(
            "Excluding from feature matrix (include_in_model=false): %s",
            excluded_present,
        )

    feature_matrix = panel[included_cols].copy()
    logger.info(
        "Feature matrix: %d rows × %d columns (excluded %d)",
        len(feature_matrix),
        len(feature_matrix.columns),
        len(excluded_present),
    )
    return feature_matrix


def get_target_series(panel: pd.DataFrame, target_col: str = "va_construction") -> pd.Series:
    """Extract the target variable (VA CONSTRUCTION) from the panel.

    Args:
        panel: Mixed-frequency panel.
        target_col: Column name of the target series.

    Returns:
        Series with quarterly values at quarter-end months and NaN elsewhere.

    Raises:
        KeyError: If target_col is not found in the panel.
    """
    if target_col not in panel.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in panel. "
            f"Available columns: {panel.columns.tolist()}"
        )
    return panel[target_col].copy()
