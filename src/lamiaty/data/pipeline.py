"""
Data pipeline orchestrator — runs the full sequence from raw files to model_panel.parquet.

Pipeline stages:
  1. Ingestion    — load raw files via loader.py
  2. Correction   — apply data fixes via corrections.py → save data/interim/
  3. Transform    — yoy log-diff + standardize; quarterly series → quarter-end
  4. Alignment    — assemble mixed-frequency panel → save data/processed/model_panel.parquet

Usage:
    from lamiaty.config import load_settings
    from lamiaty.data.pipeline import run_pipeline

    settings = load_settings("configs")
    panel = run_pipeline(settings)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from lamiaty.config.settings import Settings
from lamiaty.data.alignment import build_mixed_frequency_panel, build_monthly_index
from lamiaty.data.corrections import apply_all_corrections
from lamiaty.data.loader import load_all_raw
from lamiaty.utils.logging import log_stage
from lamiaty.data.transforms import (
    assign_quarterly_to_month_end,
    standardize,
    yoy_log_diff,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_pipeline(settings: Settings) -> pd.DataFrame:
    """Run the full data pipeline end-to-end.

    Args:
        settings: Loaded Settings object (from load_settings()).

    Returns:
        Mixed-frequency panel DataFrame saved to data/processed/model_panel.parquet.
        Shape: (n_months, n_series) where n_months covers pipeline.panel_start → panel_end.
    """
    logger.info("=== BTP Nowcasting Data Pipeline — Start ===")

    with log_stage("Stage 1: Ingestion", "lamiaty.data.pipeline"):
        raw = run_ingestion_stage(settings)
    with log_stage("Stage 2: Corrections", "lamiaty.data.pipeline"):
        interim = run_correction_stage(raw, settings)
    with log_stage("Stage 3: Transforms", "lamiaty.data.pipeline"):
        transformed = run_transform_stage(interim, settings)
    with log_stage("Stage 4: Alignment → model_panel.parquet", "lamiaty.data.pipeline"):
        panel = run_alignment_stage(transformed, settings)

    logger.info(
        "=== Pipeline Complete — shape=%s, NaN per col: %s ===",
        panel.shape,
        panel.isnull().sum().to_dict(),
    )
    return panel


# ---------------------------------------------------------------------------
# Stage 1 — Ingestion
# ---------------------------------------------------------------------------


def run_ingestion_stage(settings: Settings) -> dict[str, pd.DataFrame]:
    """Load all raw data files.

    Returns:
        Dict with keys: 'base_btp', 'moroccan_shares', 'masi_json', 'extract3'
    """
    logger.info("--- Stage 1: Ingestion ---")
    raw = load_all_raw(settings.paths)
    logger.info("Loaded raw sources: %s", list(raw.keys()))
    return raw


# ---------------------------------------------------------------------------
# Stage 2 — Corrections
# ---------------------------------------------------------------------------


def run_correction_stage(
    raw: dict[str, pd.DataFrame],
    settings: Settings,
) -> dict[str, pd.DataFrame]:
    """Apply data corrections to the base BTP dataset.

    Saves corrected base_btp to data/interim/base_btp_corrected.parquet if
    pipeline.save_interim is True.

    Returns:
        Same dict as input, with 'base_btp' replaced by the corrected DataFrame.
    """
    logger.info("--- Stage 2: Corrections ---")
    interim = dict(raw)

    if "base_btp" in interim:
        corrected = apply_all_corrections(interim["base_btp"], settings.corrections)
        interim["base_btp"] = corrected

        if settings.pipeline.save_interim:
            out_path = settings.paths.interim_dir / "base_btp_corrected.parquet"
            _save_parquet(corrected, out_path)

    return interim


# ---------------------------------------------------------------------------
# Stage 3 — Transforms
# ---------------------------------------------------------------------------


def run_transform_stage(
    interim: dict[str, pd.DataFrame],
    settings: Settings,
) -> dict[str, pd.Series]:
    """Apply yoy log-differencing, standardization, and quarterly assignment.

    Reads data_sources.yaml metadata (via settings.data_sources) to determine
    which transform to apply to each series. Series with include_in_model: false
    are still transformed but excluded from the returned dict.

    Returns:
        Dict of column_name → transformed, standardized Series.
    """
    logger.info("--- Stage 3: Transforms ---")
    result: dict[str, pd.Series] = {}

    if "base_btp" not in interim:
        logger.warning("No base_btp found in interim data — skipping transforms")
        return result

    df = interim["base_btp"]
    ds = settings.data_sources  # dict keyed by series name from data_sources.yaml

    # Column name mapping: internal name → config key
    col_to_config = {
        "consommation_ciment": "consommation_ciment",
        "credits_equipement": "credits_equipement",
        "credits_immobilier": "credits_immobilier",
        "va_construction": "va_construction",
        "ipai": "ipai",
        "lafarge_index": "lafarge_index",
        "investissement_etat": "investissement_etat",
        "creation_emploi": "creation_emploi",
    }

    for col in df.columns:
        config_key = col_to_config.get(col)
        if config_key is None:
            logger.debug("No config entry for column '%s' — skipping", col)
            continue

        series_cfg = ds.get(config_key, {})
        include = series_cfg.get("include_in_model", True)
        transform = series_cfg.get("transform", "yoy_log_diff")
        frequency = series_cfg.get("frequency", "monthly")

        # Skip transforms entirely for excluded series — avoids log(negative)
        # warnings from series like investissement_etat that have negative values
        if not include:
            logger.info(
                "Skipping '%s' from model panel (include_in_model=false). "
                "Reason: %s",
                col,
                series_cfg.get("exclusion_reason", "not specified"),
            )
            continue

        series = df[col].copy()

        if frequency == "quarterly":
            # Assign to last month of quarter; NaN elsewhere (§3.3 / §4.4)
            series = assign_quarterly_to_month_end(series)
            logger.debug("Quarterly assignment applied to '%s'", col)
        else:
            # Monthly series: apply yoy log-diff
            if transform == "yoy_log_diff":
                series = yoy_log_diff(series)
                series = standardize(series)
                logger.debug("yoy_log_diff + standardize applied to '%s'", col)
            elif transform == "none":
                logger.debug("No transform for '%s'", col)

        result[col] = series

    logger.info("Transformed %d series for model panel", len(result))
    return result


# ---------------------------------------------------------------------------
# Stage 4 — Panel alignment
# ---------------------------------------------------------------------------


def run_alignment_stage(
    transformed: dict[str, pd.Series],
    settings: Settings,
) -> pd.DataFrame:
    """Assemble and save the mixed-frequency panel.

    Saves to data/processed/model_panel.parquet.

    Returns:
        Final DataFrame with monthly DatetimeIndex and one column per series.
    """
    logger.info("--- Stage 4: Panel Alignment ---")

    target_index = build_monthly_index(
        start=settings.pipeline.panel_start,
        end=settings.pipeline.panel_end,
    )

    panel = build_mixed_frequency_panel(transformed, target_index)

    out_path = settings.paths.processed_dir / "model_panel.parquet"
    _save_parquet(panel, out_path)

    return panel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    logger.info("Saved %s (%d rows × %d cols)", path, len(df), len(df.columns))
