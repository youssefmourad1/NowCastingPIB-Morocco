"""
Pseudo-vintage builder — Phase 3 stub.

Will reconstruct the dataset as it would have been available at any historical
date, using the publication calendar from configs/publication_calendar.yaml.

Protocol (§5.3.1 of Implementation Plan):
  For each forecast origin (7th and 21st of each month, Jan 2015 – Dec 2024):
    1. Apply publication lags to determine which observations were available
    2. Reconstruct the information set Ω_v as of that date
    3. Pass to DFM for estimation + nowcast generation
    4. Store nowcast + 90% CI + news decomposition

Key constraint: ~40 quarterly evaluation points for VA CONSTRUCTION
(vs. ~60 in the Kenya study), requiring careful sub-period analysis.
"""


def build_vintage(panel, forecast_origin, publication_calendar):
    """Reconstruct the dataset as-of forecast_origin.

    Phase 3 implementation.

    Args:
        panel: Full model panel (all data).
        forecast_origin: pd.Timestamp — the date at which we simulate knowledge.
        publication_calendar: Dict of series → publication lag settings.

    Returns:
        Subset panel with future observations set to NaN.
    """
    raise NotImplementedError("Vintage builder implemented in Phase 3.")
