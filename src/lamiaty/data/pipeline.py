"""
Data pipeline orchestrator — runs the full sequence from raw files to model_panel.parquet.

Pipeline stages:
  1. Ingestion    — load raw files via loader.py
  2. Correction   — apply data fixes via corrections.py → save data/interim/
  3. Transform    — stationarity-oriented transforms + standardize
  4. Alignment    — assemble mixed-frequency panel → save data/processed/model_panel.parquet

This version is improved for BTP nowcasting:
  - supports multiple transforms: yoy_log_diff, log_diff, diff, pct_change, none
  - handles non-positive values safely for log transforms
  - standardizes only after transformation
  - keeps quarterly assignment logic
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from lamiaty.config.settings import Settings
from lamiaty.data.alignment import build_mixed_frequency_panel, build_monthly_index
from lamiaty.data.corrections import apply_all_corrections
from lamiaty.data.loader import load_all_raw
from lamiaty.data.transforms import (
    assign_quarterly_to_month_end,
    standardize,
    yoy_log_diff,
)
from lamiaty.utils.logging import log_stage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_pipeline(settings: Settings) -> pd.DataFrame:
    """Run the full data pipeline end-to-end."""
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
    """Load all raw data files."""
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
    """
    Apply data corrections to the base BTP dataset.

    Expected examples:
      - investissement_etat cumulative-to-monthly correction
      - cement unit-break fix
      - Lafarge string cleanup
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
    """
    Apply stationarity-oriented transformations and standardization.

    Reads settings.data_sources to decide:
      - include_in_model
      - transform
      - frequency

    Supported transforms:
      - yoy_log_diff
      - log_diff
      - diff
      - pct_change
      - none
    """
    logger.info("--- Stage 3: Transforms ---")
    result: dict[str, pd.Series] = {}

    if "base_btp" not in interim:
        logger.warning("No base_btp found in interim data — skipping transforms")
        return result

    df = interim["base_btp"].copy()
    ds = settings.data_sources

    # Map DataFrame columns to config keys
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
        do_standardize = series_cfg.get("standardize", True)

        if not include:
            logger.info(
                "Skipping '%s' from model panel (include_in_model=false). Reason: %s",
                col,
                series_cfg.get("exclusion_reason", "not specified"),
            )
            continue

        series = pd.to_numeric(df[col], errors="coerce").copy()
        series.name = col

        # Step A — quarterly placement first
        if frequency == "quarterly":
            series = assign_quarterly_to_month_end(series)
            logger.debug("Quarterly assignment applied to '%s'", col)

        # Step B — transform
        series = apply_transform(series, transform=transform)

        # Step C — standardize
        if do_standardize:
            series = standardize(series)

        result[col] = series
        
        logger.info(
            "Transformed '%s' | frequency=%s | transform=%s | standardize=%s | non-NaN=%d",
            col,
            frequency,
            transform,
            do_standardize,
            series.dropna().shape[0],
        )

    logger.info("Transformed %d series for model panel", len(result))
    return result


def apply_transform(series: pd.Series, transform: str) -> pd.Series:
    """
    Apply the requested transformation to a series.
    Improved to handle quarterly series in monthly panels (ignores NaNs for diff/pct_change).
    """
    s = pd.to_numeric(series, errors="coerce").copy()
    s.name = series.name

    if transform == "none":
        return s

    # Pour les séries avec beaucoup de NaN (trimestrielles), on doit calculer 
    # la variation sur les valeurs non-nulles puis re-indexer.
    s_clean = s.dropna()
    
    if transform == "diff":
        out = s_clean.diff()
    elif transform == "pct_change":
        out = s_clean.pct_change()
    elif transform == "log":
        return safe_log(s)
    elif transform == "log_diff":
        out = safe_log(s_clean).diff()
    elif transform == "log_diff_4":
        out = safe_log(s_clean).diff(4)
    elif transform == "yoy_log_diff":
        return yoy_log_diff(s)
    else:
        raise ValueError(
            f"Unknown transform '{transform}' for series '{series.name}'. "
            "Allowed: none, diff, pct_change, log, log_diff, log_diff_4, yoy_log_diff"
        )

    return out.reindex(series.index)
def safe_log(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").copy()

    invalid_mask = s <= 0
    if invalid_mask.any():
        logger.warning(
            "Series '%s' contains %d non-positive values; replaced with NaN before log transform",
            series.name,
            int(invalid_mask.sum()),
        )
        s[invalid_mask] = np.nan

    out = np.log(s)
    out.name = series.name
    return out


# ---------------------------------------------------------------------------
# Stage 4 — Panel alignment
# ---------------------------------------------------------------------------

def run_alignment_stage(
    transformed: dict[str, pd.Series],
    settings: Settings,
) -> pd.DataFrame:
    """
    Assemble and save the mixed-frequency panel.
    """
    logger.info("--- Stage 4: Panel Alignment ---")

    target_index = build_monthly_index(
        start=settings.pipeline.panel_start,
        end=settings.pipeline.panel_end,
    )

    panel = build_mixed_frequency_panel(transformed, target_index)

    # Cible ML : croissance annuelle log de la VA construction.
    # On garde va_construction en niveau pour le DFM.
    if "va_construction" in panel.columns:
        va_level = pd.to_numeric(panel["va_construction"], errors="coerce")
        va_level = va_level.where(va_level > 0)
        # Calculer le YoY sur les observations non-nulles uniquement (4 quarters = 12 mois)
        va_log = np.log(va_level.dropna())
        panel["va_construction_yoy"] = va_log.diff(4).reindex(panel.index)
    else:
        logger.warning("va_construction absente du panel : va_construction_yoy non créée")

    out_path = settings.paths.processed_dir / "model_panel.parquet"
    _save_parquet(panel, out_path)

    return panel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_parquet(df: pd.DataFrame | pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df.to_parquet(path, engine="pyarrow")
    logger.info("Saved %s (%d rows × %d cols)", path, len(df), len(df.columns))