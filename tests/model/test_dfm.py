"""Tests for DynamicFactorModel (Phase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lamiaty.model.dfm import DynamicFactorModel, DFMResults


class TestDFMFit:

    def test_fit_returns_self(self, fitted_dfm):
        assert isinstance(fitted_dfm, DynamicFactorModel)

    def test_is_fitted_flag(self, fitted_dfm):
        assert fitted_dfm._is_fitted is True

    def test_results_populated(self, fitted_dfm):
        assert fitted_dfm.results_ is not None
        assert isinstance(fitted_dfm.results_, DFMResults)

    def test_loadings_shape(self, fitted_dfm):
        """Loadings matrix: (n_series, n_factors)."""
        loadings = fitted_dfm.results_.loadings
        assert loadings is not None
        # 2 monthly + 1 quarterly = 3 series, 1 factor
        assert loadings.shape == (3, 1), f"Expected (3,1), got {loadings.shape}"

    def test_factors_smoothed_shape(self, synthetic_panel, fitted_dfm):
        """Smoothed factors: (T_insample, n_factors)."""
        fs = fitted_dfm.results_.factors_smoothed
        assert fs is not None
        # 60 months in-sample, 1 factor
        assert fs.shape[1] == 1
        assert fs.shape[0] > 0

    def test_loglikelihood_finite(self, fitted_dfm):
        assert not np.isnan(fitted_dfm.results_.log_likelihood)
        assert np.isfinite(fitted_dfm.results_.log_likelihood)

    def test_variance_shares_bounded(self, fitted_dfm):
        """Variance shares must be in [0, 1] for each series."""
        shares = fitted_dfm.results_.variance_shares
        assert len(shares) > 0
        for series, share in shares.items():
            assert 0.0 <= share <= 1.0, f"{series}: share={share} out of [0,1]"

    def test_monthly_quarterly_split(self, fitted_dfm):
        assert "monthly_a" in fitted_dfm._monthly_cols
        assert "monthly_b" in fitted_dfm._monthly_cols
        assert "va_construction" in fitted_dfm._quarterly_cols

    def test_iterations_positive(self, fitted_dfm):
        assert fitted_dfm.results_.n_iterations >= 0

    def test_unfitted_raises_on_nowcast(self):
        """Calling nowcast() before fit() raises RuntimeError."""
        dfm = DynamicFactorModel(n_factors=1)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            dfm.nowcast(pd.DataFrame())


class TestDFMNowcast:

    def test_nowcast_returns_series(self, synthetic_panel, fitted_dfm):
        nc = fitted_dfm.nowcast(synthetic_panel)
        assert isinstance(nc, pd.Series)
        assert nc.name == "nowcast"

    def test_nowcast_non_empty(self, synthetic_panel, fitted_dfm):
        nc = fitted_dfm.nowcast(synthetic_panel)
        assert len(nc) > 0

    def test_nowcast_has_ci_attrs(self, synthetic_panel, fitted_dfm):
        nc = fitted_dfm.nowcast(synthetic_panel)
        assert "ci_lower" in nc.attrs
        assert "ci_upper" in nc.attrs

    def test_nowcast_ci_ordering(self, synthetic_panel, fitted_dfm):
        """CI lower bound ≤ point estimate ≤ CI upper bound."""
        nc = fitted_dfm.nowcast(synthetic_panel)
        lo = nc.attrs["ci_lower"]
        hi = nc.attrs["ci_upper"]
        common = nc.index.intersection(lo.index).intersection(hi.index)
        assert (lo.loc[common] <= nc.loc[common] + 1e-8).all()
        assert (nc.loc[common] <= hi.loc[common] + 1e-8).all()

    def test_nowcast_at_quarter_end_months(self, synthetic_panel, fitted_dfm):
        """va_construction predictions only at quarter-end months."""
        nc = fitted_dfm.nowcast(synthetic_panel)
        # The index may be month-start (MS) corresponding to Mar/Jun/Sep/Dec
        # or quarter-end; all must correspond to quarter-end months
        if hasattr(nc.index, "month"):
            assert nc.index.month.isin([3, 6, 9, 12]).all(), (
                f"Unexpected months in nowcast index: {nc.index.month.unique()}"
            )
