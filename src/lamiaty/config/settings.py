"""
Configuration loading — single entry point for all YAML configs.

Every other module receives a Settings object via dependency injection.
No other module reads YAML files or hardcodes paths/parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-settings dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PathSettings:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    vintages_dir: Path
    external_dir: Path
    # Source file names (relative to raw_dir)
    base_btp: str = "base_BTP_modifiee.xlsx"
    moroccan_shares_csv: str = "Moroccan All Shares Historical Data.csv"
    masi_json_files: list[str] = field(default_factory=lambda: ["extract.json", "extract2.json"])
    extract3_csv: str = "extract3.csv"

    @property
    def base_btp_path(self) -> Path:
        return self.raw_dir / self.base_btp

    @property
    def moroccan_shares_path(self) -> Path:
        return self.raw_dir / self.moroccan_shares_csv

    @property
    def masi_json_paths(self) -> list[Path]:
        return [self.raw_dir / f for f in self.masi_json_files]

    @property
    def extract3_path(self) -> Path:
        return self.raw_dir / self.extract3_csv


@dataclass
class CementBreakSettings:
    break_date: str
    correction_factor: float
    direction: str
    status: str
    confirmed_by: Any  # None or str


@dataclass
class InvestissementSettings:
    treatment: str  # "monthly_diff" | "keep_as_is"
    confirmed_by: Any


@dataclass
class LafargeFixSettings:
    strip_commas: bool
    cast_to_float: bool


@dataclass
class CorrectionSettings:
    cement_break: CementBreakSettings
    investissement_etat: InvestissementSettings
    lafarge_string_fix: LafargeFixSettings


@dataclass
class MixedFrequencySettings:
    quarterly_series: list[str]
    aggregation: str


@dataclass
class SampleSettings:
    in_sample_start: str
    in_sample_end: str
    full_sample_start: str
    full_sample_end: str


@dataclass
class BacktestSettings:
    origin_start: str
    origin_end: str
    update_days: list[int]
    rolling_window_years: int


@dataclass
class ModelSettings:
    n_factors: int
    n_lags: int
    max_iterations: int
    tolerance: float
    mixed_frequency: MixedFrequencySettings
    sample: SampleSettings
    backtest: BacktestSettings


@dataclass
class PipelineSettings:
    run_corrections: bool
    run_transforms: bool
    save_interim: bool
    output_format: str
    panel_start: str
    panel_end: str


@dataclass
class Settings:
    paths: PathSettings
    corrections: CorrectionSettings
    model: ModelSettings
    pipeline: PipelineSettings
    data_sources: dict[str, Any]
    publication_calendar: dict[str, Any]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_settings(config_dir: str | Path = "configs", project_root: str | Path | None = None) -> Settings:
    """Load and validate all YAML configuration files.

    Args:
        config_dir: Path to the configs/ directory (relative to project_root or absolute).
        project_root: Project root directory. Defaults to the current working directory.

    Returns:
        A fully populated Settings object.
    """
    if project_root is None:
        project_root = Path.cwd()
    project_root = Path(project_root)
    config_dir = Path(config_dir)
    if not config_dir.is_absolute():
        config_dir = project_root / config_dir

    pipeline_cfg = _load_yaml(config_dir / "pipeline.yaml")
    corrections_cfg = _load_yaml(config_dir / "corrections.yaml")
    model_cfg = _load_yaml(config_dir / "model.yaml")
    data_sources_cfg = _load_yaml(config_dir / "data_sources.yaml")
    pub_cal_cfg = _load_yaml(config_dir / "publication_calendar.yaml")

    # Resolve paths relative to project root
    raw_dir = project_root / pipeline_cfg["paths"]["raw_dir"]
    interim_dir = project_root / pipeline_cfg["paths"]["interim_dir"]
    processed_dir = project_root / pipeline_cfg["paths"]["processed_dir"]
    vintages_dir = project_root / pipeline_cfg["paths"]["vintages_dir"]
    external_dir = project_root / pipeline_cfg["paths"]["external_dir"]

    srcs = pipeline_cfg.get("sources", {})
    paths = PathSettings(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        vintages_dir=vintages_dir,
        external_dir=external_dir,
        base_btp=srcs.get("base_btp", "base_BTP_modifiee.xlsx"),
        moroccan_shares_csv=srcs.get("moroccan_shares_csv", "Moroccan All Shares Historical Data.csv"),
        masi_json_files=srcs.get("masi_json_files", ["extract.json", "extract2.json"]),
        extract3_csv=srcs.get("extract3_csv", "extract3.csv"),
    )

    cb = corrections_cfg["cement_break"]
    corrections = CorrectionSettings(
        cement_break=CementBreakSettings(
            break_date=cb["break_date"],
            correction_factor=float(cb["correction_factor"]),
            direction=cb["direction"],
            status=cb.get("status", "PLACEHOLDER"),
            confirmed_by=cb.get("confirmed_by"),
        ),
        investissement_etat=InvestissementSettings(
            treatment=corrections_cfg["investissement_etat"]["treatment"],
            confirmed_by=corrections_cfg["investissement_etat"].get("confirmed_by"),
        ),
        lafarge_string_fix=LafargeFixSettings(
            strip_commas=corrections_cfg["lafarge_string_fix"]["strip_commas"],
            cast_to_float=corrections_cfg["lafarge_string_fix"]["cast_to_float"],
        ),
    )

    dfm_cfg = model_cfg["dfm"]
    mf_cfg = dfm_cfg["mixed_frequency"]
    sample_cfg = model_cfg["sample"]
    bt_cfg = model_cfg["backtest"]
    model = ModelSettings(
        n_factors=dfm_cfg["n_factors"],
        n_lags=dfm_cfg["n_lags"],
        max_iterations=dfm_cfg["max_iterations"],
        tolerance=float(dfm_cfg["tolerance"]),
        mixed_frequency=MixedFrequencySettings(
            quarterly_series=mf_cfg["quarterly_series"],
            aggregation=mf_cfg["aggregation"],
        ),
        sample=SampleSettings(
            in_sample_start=sample_cfg["in_sample_start"],
            in_sample_end=sample_cfg["in_sample_end"],
            full_sample_start=sample_cfg["full_sample_start"],
            full_sample_end=sample_cfg["full_sample_end"],
        ),
        backtest=BacktestSettings(
            origin_start=bt_cfg["origin_start"],
            origin_end=bt_cfg["origin_end"],
            update_days=bt_cfg["update_days"],
            rolling_window_years=bt_cfg["rolling_window_years"],
        ),
    )

    pl_cfg = pipeline_cfg["pipeline"]
    pipeline = PipelineSettings(
        run_corrections=pl_cfg["run_corrections"],
        run_transforms=pl_cfg["run_transforms"],
        save_interim=pl_cfg["save_interim"],
        output_format=pl_cfg["output_format"],
        panel_start=pl_cfg.get("panel_start", "2007-01"),
        panel_end=pl_cfg.get("panel_end", "2026-02"),
    )

    return Settings(
        paths=paths,
        corrections=corrections,
        model=model,
        pipeline=pipeline,
        data_sources=data_sources_cfg.get("series", {}),
        publication_calendar=pub_cal_cfg.get("series", {}),
    )


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
