"""
Dynamic Factor Model — Phase 1 stub with complete API definition.

This module defines the DynamicFactorModel class interface following the
specification in §4 of the Morocco BTP Nowcasting Implementation Plan.
The estimation code is implemented in Phase 2.

Model equations (Danov et al. 2026, Equations 1–6):
  Observation:     y_t = Λ F_t + ε_t
  Factor dynamics: F_t = Γ F_{t-1} + u_t,     u_t  ~ i.i.d. N(0, Ξ)
  Idiosyncratic:   ε_t = Θ ε_{t-1} + η_t,     η_t  ~ i.i.d. N(0, Σ)

Mixed-frequency aggregation constraint (§4.4):
  VA^q_t = VA^m_t + VA^m_{t-1} + VA^m_{t-2}
         = λ_VA (f_t + f_{t-1} + f_{t-2}) + ε_t + ε_{t-1} + ε_{t-2}

Estimation algorithm:
  EM-Kalman filter/smoother following Banbura & Modugno (2014).
  Handles arbitrary patterns of missing data natively — no pre-imputation needed.
  Key advantage: quarterly series (VA CONSTRUCTION, IPAI, employment) are
  treated as partially observed without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from lamiaty.config.settings import ModelSettings


@dataclass
class DFMResults:
    """Container for DFM estimation results (populated in Phase 2).

    Attributes:
        loadings: Factor loading matrix Λ (n_series × n_factors).
        factor_dynamics: Transition matrix Γ (n_factors × n_factors).
        idiosyncratic_ar: Diagonal AR(1) coefficients Θ for each series.
        factors_smoothed: Smoothed factor estimates F_{t|T} (n_months × n_factors).
        variance_shares: Fraction of each series' variance explained by common factors.
        log_likelihood: Final log-likelihood at EM convergence.
        n_iterations: Number of EM iterations until convergence.
    """

    loadings: Any = None
    factor_dynamics: Any = None
    idiosyncratic_ar: Any = None
    factors_smoothed: Any = None
    variance_shares: dict[str, float] = field(default_factory=dict)
    log_likelihood: float = float("nan")
    n_iterations: int = 0


class DynamicFactorModel:
    """Nowcasting DFM for Morocco's Construction Sector Value Added.

    Implements the Dynamic Factor Model of Danov, Giannone, Kabundi, Okou &
    Spilimbergo (2026) adapted to the Moroccan BTP sector.

    Phase 1: Class interface defined; estimation stubs raise NotImplementedError.
    Phase 2: EM-Kalman estimation implemented in model/kalman.py.
    Phase 3: News decomposition implemented in model/news.py.

    Args:
        n_factors: Number of latent common factors k. Expected: 2–3 for BTP Morocco.
            Factor 1 — BTP cycle (ciment, crédits immo, LafargeHolcim, emploi)
            Factor 2 — Public investment (crédits équipement, marchés publics)
            Factor 3 — Seasonal/Ramadan (optional)
        n_lags: Number of lags in the factor VAR dynamics (default: 1).
        settings: ModelSettings loaded from configs/model.yaml.
    """

    def __init__(
        self,
        n_factors: int = 2,
        n_lags: int = 1,
        settings: ModelSettings | None = None,
    ) -> None:
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.settings = settings
        self.results_: DFMResults | None = None
        self._is_fitted = False

    def fit(self, panel: pd.DataFrame) -> "DynamicFactorModel":
        """Estimate DFM parameters via EM-Kalman algorithm.

        Phase 2 implementation. The panel must contain:
          - Monthly series (yoy log-differenced, standardized) as regular columns.
          - Quarterly series (VA CONSTRUCTION, IPAI, employment) with NaN in
            non-quarter-end months (output of transforms.assign_quarterly_to_month_end).

        The EM algorithm from Banbura & Modugno (2014) handles the missing
        data pattern natively — no pre-interpolation of quarterly series.

        Args:
            panel: Mixed-frequency panel from run_pipeline(). Shape: (T, N).

        Returns:
            Self (fitted model).

        Raises:
            NotImplementedError: Phase 2 not yet implemented.
        """
        raise NotImplementedError(
            "DFM estimation is implemented in Phase 2. "
            "See configs/model.yaml for the target specification."
        )

    def nowcast(self, panel: pd.DataFrame) -> pd.Series:
        """Generate nowcast of VA CONSTRUCTION for the current quarter.

        Applies the trained Kalman filter to the panel (which may include
        recently published indicators with NaN for not-yet-released series)
        and returns the expected value of VA^q_t given available information.

        Args:
            panel: Mixed-frequency panel as-of the nowcast date. May contain
                   NaN for indicators not yet published.

        Returns:
            Series with nowcast values indexed by quarter-end dates.
            Includes 90% confidence intervals as metadata.

        Raises:
            NotImplementedError: Phase 2 not yet implemented.
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before nowcast().")
        raise NotImplementedError("Nowcasting implemented in Phase 2.")

    def news_decomposition(
        self,
        panel_prev: pd.DataFrame,
        panel_new: pd.DataFrame,
    ) -> pd.DataFrame:
        """Decompose the revision in the nowcast into per-variable news contributions.

        Implements Equation (11) of Danov et al. (2026):
          E[VA^q_t | Ω_{v+1}] - E[VA^q_t | Ω_v]  =  Σ_j δ_{v+1,j} × (y_{t_j} - E[y_{t_j} | Ω_v])
                  Revision                                    Weight          News (surprise)

        Args:
            panel_prev: Information set at previous update Ω_v.
            panel_new: Information set at new update Ω_{v+1}.

        Returns:
            DataFrame with columns ['series', 'news', 'weight', 'contribution'],
            sorted by absolute contribution descending.

        Raises:
            NotImplementedError: Phase 3 not yet implemented.
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before news_decomposition().")
        raise NotImplementedError("News decomposition implemented in Phase 3.")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"DynamicFactorModel(n_factors={self.n_factors}, n_lags={self.n_lags}, status={status})"
