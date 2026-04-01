"""
Dynamic Factor Model — Phase 2 implementation.

Backend: statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ
  Implements Banbura & Modugno (2014) EM-Kalman with native mixed-frequency
  support (no pre-interpolation of quarterly series required).

Model equations (Danov et al. 2026, Equations 1–6):
  Observation:     y_t = Λ F_t + ε_t
  Factor dynamics: F_t = Γ F_{t-1} + u_t,     u_t  ~ i.i.d. N(0, Ξ)
  Idiosyncratic:   ε_t = Θ ε_{t-1} + η_t,     η_t  ~ i.i.d. N(0, Σ)

Mixed-frequency aggregation constraint (§4.4):
  VA^q_t = VA^m_t + VA^m_{t-1} + VA^m_{t-2}
         = λ_VA (f_t + f_{t-1} + f_{t-2}) + ε_t + ε_{t-1} + ε_{t-2}
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lamiaty.config.settings import ModelSettings

logger = logging.getLogger(__name__)


@dataclass
class DFMResults:
    """Container for DFM estimation results.

    Attributes:
        loadings: Factor loading matrix Λ (n_series × n_factors).
        factor_dynamics: Transition matrix Γ (n_factors × n_factors).
        idiosyncratic_ar: AR(1) coefficients for idiosyncratic components (dict).
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

    Wraps statsmodels DynamicFactorMQ (Banbura & Modugno 2014) to estimate
    a k-factor model on a mixed-frequency panel.  Quarterly series (VA
    CONSTRUCTION, IPAI, creation_emploi) enter with NaN in non-quarter-end
    months; the EM algorithm handles the missing-data pattern natively.

    Args:
        n_factors: Number of latent common factors k (default 2).
            Factor 1 — BTP cycle (ciment, crédits immo, LafargeHolcim, emploi)
            Factor 2 — Public investment (crédits équipement)
        n_lags: Lag order of the factor VAR dynamics (default 1).
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
        self._sm_result = None
        self._monthly_cols: list[str] = []
        self._quarterly_cols: list[str] = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fit(self, panel: pd.DataFrame) -> "DynamicFactorModel":
        """Estimate DFM parameters via EM-Kalman algorithm.

        The panel must contain mixed-frequency series:
          - Monthly series (yoy log-differenced, standardized): regular cols.
          - Quarterly series (VA CONSTRUCTION, IPAI, employment): NaN in
            non-quarter-end months (output of assign_quarterly_to_month_end).

        Estimation window is restricted to settings.sample.in_sample_start –
        in_sample_end (default 2010-01 – 2019-12) to use a pre-COVID baseline.

        Args:
            panel: Mixed-frequency panel from run_pipeline(). Shape: (T, N).

        Returns:
            Self (fitted model, supports chaining).
        """
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

        s = self.settings

        # ── 1. Split monthly vs quarterly columns ──────────────────────────
        quarterly_cols = (
            [c for c in (s.mixed_frequency.quarterly_series or []) if c in panel.columns]
            if s else []
        )
        monthly_cols = [c for c in panel.columns if c not in quarterly_cols]

        self._monthly_cols = monthly_cols
        self._quarterly_cols = quarterly_cols

        endog_m = panel[monthly_cols]

        # ── 2. Quarterly data: extract observed rows, shift to quarter-end ─
        endog_q = self._prepare_quarterly(panel, quarterly_cols)

        # ── 3. Restrict to in-sample window ────────────────────────────────
        if s:
            start, end = s.sample.in_sample_start, s.sample.in_sample_end
            endog_m_is = endog_m.loc[start:end]
            endog_q_is = endog_q.loc[start:end] if len(endog_q) > 0 else endog_q
        else:
            endog_m_is, endog_q_is = endog_m, endog_q

        maxiter  = s.max_iterations if s else 500
        tol      = s.tolerance if s else 1e-6

        logger.info(
            "Fitting DFM: k=%d, p=%d, monthly=%s (%d rows), quarterly=%s (%d rows)",
            self.n_factors, self.n_lags,
            monthly_cols, len(endog_m_is),
            quarterly_cols, len(endog_q_is),
        )

        # ── 4. Build statsmodels model ─────────────────────────────────────
        sm_kwargs: dict[str, Any] = dict(
            endog=endog_m_is,
            factors=self.n_factors,
            factor_orders=self.n_lags,
            # Quarterly series are in original units (MDH) while monthly series
            # are yoy log-diff + standardized.  Let statsmodels standardize
            # all series internally to prevent scale mismatch in the EM.
            standardize=True,
        )
        if len(endog_q_is) > 0:
            sm_kwargs["endog_quarterly"] = endog_q_is

        sm_model = DynamicFactorMQ(**sm_kwargs)

        # ── 5. Fit via EM — capture convergence warnings ───────────────────
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            self._sm_result = sm_model.fit(
                method="em",
                maxiter=maxiter,
                tolerance=tol,
                disp=False,
            )

        for w in caught_warnings:
            logger.warning("statsmodels EM: %s", str(w.message))

        self._is_fitted = True
        self.results_ = self._extract_results(self._sm_result)

        logger.info(
            "DFM fitted: llf=%.4f, iterations=%d, variance_shares=%s",
            self.results_.log_likelihood,
            self.results_.n_iterations,
            {k: f"{v:.2%}" for k, v in self.results_.variance_shares.items()},
        )
        return self

    def nowcast(self, panel: pd.DataFrame) -> pd.Series:
        """Generate nowcast of VA CONSTRUCTION for all available quarters.

        Applies the trained Kalman filter to the panel (which may contain NaN
        for not-yet-released series) and returns the smoothed prediction for
        VA CONSTRUCTION at each quarter-end date.

        Args:
            panel: Mixed-frequency panel as-of the nowcast date.  NaN allowed
                   for indicators not yet published.

        Returns:
            Series of nowcast values indexed by quarter-end dates (QE freq).
            Attributes 'ci_lower' and 'ci_upper' contain 90% CI bounds.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before nowcast().")

        endog_m  = panel[self._monthly_cols] if self._monthly_cols else panel
        endog_q  = self._prepare_quarterly(panel, self._quarterly_cols)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            apply_kwargs: dict[str, Any] = {"endog": endog_m}
            if len(endog_q) > 0:
                apply_kwargs["endog_quarterly"] = endog_q
            applied = self._sm_result.apply(**apply_kwargs)

        for w in caught_warnings:
            logger.warning("statsmodels apply: %s", str(w.message))

        pred     = applied.get_prediction()
        pred_m   = pred.predicted_mean
        ci       = pred.conf_int(alpha=0.10)

        va_col = "va_construction"
        if va_col not in pred_m.columns:
            raise ValueError(
                f"'{va_col}' not in predicted columns: {pred_m.columns.tolist()}"
            )

        # Keep only quarter-end months (3, 6, 9, 12) for the target
        all_va = pred_m[va_col]
        q_mask = all_va.index.month.isin([3, 6, 9, 12])
        va_pred  = all_va[q_mask].dropna()
        ci_lower = ci[f"lower {va_col}"].reindex(va_pred.index)
        ci_upper = ci[f"upper {va_col}"].reindex(va_pred.index)

        result = va_pred.rename("nowcast")
        result.attrs["ci_lower"] = ci_lower
        result.attrs["ci_upper"] = ci_upper

        logger.info(
            "Nowcast: %d quarter-end predictions, latest=%s → %.2f",
            len(result),
            result.index[-1].strftime("%Y-Q%q") if len(result) else "—",
            result.iloc[-1] if len(result) else float("nan"),
        )
        return result

    def news_decomposition(
        self,
        panel_prev: pd.DataFrame,
        panel_new: pd.DataFrame,
        impact_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Decompose the nowcast revision into per-variable news contributions.

        Implements Equation (11) of Danov et al. (2026):
          Revision = Σ_j  weight_j × news_j
        where news_j = y_{j,t} − E[y_{j,t} | Ω_prev]  (surprise)
              weight_j = ∂E[VA^q | Ω_new] / ∂y_{j,t}  (Kalman gain)

        Args:
            panel_prev: Information set at previous update Ω_prev.
            panel_new: Information set at new update Ω_new.
            impact_date: Quarter-end date for which to compute news.
                         Defaults to the latest available quarter-end.

        Returns:
            DataFrame with columns ['series', 'date', 'news', 'weight', 'contribution'],
            sorted by |contribution| descending, including a 'TOTAL' row.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before news_decomposition().")

        from lamiaty.model.news import compute_news_decomposition
        return compute_news_decomposition(
            panel_prev, panel_new, self, impact_date=impact_date
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _prepare_quarterly(
        panel: pd.DataFrame, quarterly_cols: list[str]
    ) -> pd.DataFrame:
        """Extract quarterly obs and shift index to quarter-end for statsmodels.

        statsmodels DynamicFactorMQ requires the quarterly DataFrame to have a
        DatetimeIndex with quarterly frequency (freqstr starting with 'Q').
        We shift the month-start index (e.g., 2015-03-01) to quarter-end
        (2015-03-31) and set freq='QE'.
        """
        if not quarterly_cols:
            return pd.DataFrame()
        q_raw = panel[quarterly_cols].dropna(how="all")
        if len(q_raw) == 0:
            return q_raw
        q = q_raw.copy()
        # Shift month-start → quarter-end and set quarterly frequency
        q.index = pd.DatetimeIndex(
            q.index + pd.offsets.MonthEnd(0), freq="QE"
        )
        return q

    def _extract_results(self, sm_result) -> DFMResults:
        """Populate DFMResults from a fitted statsmodels DynamicFactorMQ result."""
        params = sm_result.params

        # ── Loadings Λ (N × k) ─────────────────────────────────────────────
        all_series = self._monthly_cols + self._quarterly_cols
        loading_raw = {
            k: v for k, v in params.items() if k.startswith("loading.")
        }
        # params named: 'loading.{factor_idx}->{series_name}'
        loadings_dict: dict[str, list[float]] = {s: [] for s in all_series}
        for k, v in loading_raw.items():
            _, rhs = k.split(".", 1)           # '0->ciment'
            _, series = rhs.split("->", 1)
            if series in loadings_dict:
                loadings_dict[series].append(float(v))

        loadings_df = pd.DataFrame(
            {s: vals for s, vals in loadings_dict.items() if len(vals) == self.n_factors},
            index=[f"f{i + 1}" for i in range(self.n_factors)],
        ).T  # (n_series, n_factors)

        # ── Factor dynamics Γ (k × k) ───────────────────────────────────────
        gamma_raw = {k: v for k, v in params.items() if k.startswith("L1.")}
        gamma = np.zeros((self.n_factors, self.n_factors))
        for k, v in gamma_raw.items():
            body = k[3:]  # strip 'L1.'
            if "->" not in body:
                continue
            from_s, to_s = body.split("->", 1)
            try:
                i, j = int(from_s), int(to_s)
                if i < self.n_factors and j < self.n_factors:
                    gamma[j, i] = float(v)
            except ValueError:
                pass

        # ── Smoothed factors F_{t|T} ────────────────────────────────────────
        factors_smoothed = sm_result.factors.smoothed.copy()
        factors_smoothed.columns = [f"f{i + 1}" for i in range(self.n_factors)]

        # ── Variance shares ─────────────────────────────────────────────────
        sigma2_raw = {k: v for k, v in params.items() if k.startswith("sigma2.")}
        factor_var = factors_smoothed.var().values  # empirical (k,)
        variance_shares: dict[str, float] = {}
        for s in all_series:
            if s not in loadings_df.index:
                continue
            lam      = loadings_df.loc[s].values          # (k,)
            explained = float(lam @ np.diag(factor_var) @ lam)
            idio_var  = float(sigma2_raw.get(f"sigma2.{s}", 0.0))
            total     = explained + idio_var
            variance_shares[s] = round(explained / total, 4) if total > 0 else 0.0

        # ── Idiosyncratic AR coefficients ───────────────────────────────────
        ar_params = {k: float(v) for k, v in params.items() if "L1.eps_" in k}

        return DFMResults(
            loadings=loadings_df,
            factor_dynamics=gamma,
            idiosyncratic_ar=ar_params,
            factors_smoothed=factors_smoothed,
            variance_shares=variance_shares,
            log_likelihood=float(sm_result.llf),
            n_iterations=int(sm_result.mle_retvals.get("iter", 0)),
        )

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"DynamicFactorModel(n_factors={self.n_factors}, "
            f"n_lags={self.n_lags}, status={status})"
        )
