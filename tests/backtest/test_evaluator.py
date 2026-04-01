"""Tests for forecast evaluation metrics (Phase 3)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lamiaty.backtest.evaluator import (
    compute_rmsfe,
    compute_mafe,
    compute_theil_u,
    diebold_mariano,
    benchmark_random_walk,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_nowcast():
    idx = pd.date_range("2018-03-31", periods=8, freq="QE")
    realized = pd.Series([10.0, 11.0, 12.0, 11.5, 10.5, 9.5, 10.0, 11.0], index=idx)
    return realized, realized.copy()  # (realized, nowcast)


@pytest.fixture
def imperfect_nowcast():
    idx = pd.date_range("2018-03-31", periods=8, freq="QE")
    realized = pd.Series([10.0, 11.0, 12.0, 11.5, 10.5, 9.5, 10.0, 11.0], index=idx)
    errors   = pd.Series([1.0, -1.0, 2.0, 0.0, -1.0, 1.0, -2.0, 0.0], index=idx)
    nowcast  = realized + errors
    return realized, nowcast


# ── compute_rmsfe ─────────────────────────────────────────────────────────────

class TestRMSFE:

    def test_perfect_forecast_is_zero(self, perfect_nowcast):
        realized, nowcast = perfect_nowcast
        assert compute_rmsfe(nowcast, realized) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self, imperfect_nowcast):
        """errors = [1, -1, 2, 0, -1, 1, -2, 0] → RMSFE = sqrt(12/8) = sqrt(1.5)."""
        realized, nowcast = imperfect_nowcast
        expected = math.sqrt(sum(e**2 for e in [1, -1, 2, 0, -1, 1, -2, 0]) / 8)
        assert compute_rmsfe(nowcast, realized) == pytest.approx(expected, rel=1e-6)

    def test_empty_returns_nan(self):
        assert math.isnan(compute_rmsfe(pd.Series(dtype=float), pd.Series(dtype=float)))

    def test_all_nan_returns_nan(self):
        idx = pd.date_range("2020-03-31", periods=4, freq="QE")
        assert math.isnan(
            compute_rmsfe(
                pd.Series([float("nan")] * 4, index=idx),
                pd.Series([1.0, 2.0, 3.0, 4.0], index=idx),
            )
        )


# ── compute_mafe ──────────────────────────────────────────────────────────────

class TestMAFE:

    def test_perfect_forecast_is_zero(self, perfect_nowcast):
        realized, nowcast = perfect_nowcast
        assert compute_mafe(nowcast, realized) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self, imperfect_nowcast):
        """errors = [1, -1, 2, 0, -1, 1, -2, 0] → MAFE = 8/8 = 1.0."""
        realized, nowcast = imperfect_nowcast
        assert compute_mafe(nowcast, realized) == pytest.approx(1.0, rel=1e-6)


# ── compute_theil_u ───────────────────────────────────────────────────────────

class TestTheilU:

    def test_equal_to_benchmark_gives_one(self, imperfect_nowcast):
        """When model = benchmark, Theil U = 1.0."""
        realized, nowcast = imperfect_nowcast
        assert compute_theil_u(nowcast, realized, nowcast) == pytest.approx(1.0, rel=1e-6)

    def test_perfect_model_gives_zero(self, perfect_nowcast, imperfect_nowcast):
        """Perfect model vs imperfect benchmark → Theil U = 0."""
        realized, _ = imperfect_nowcast
        _, perfect  = perfect_nowcast
        assert compute_theil_u(perfect, realized, realized + 1) == pytest.approx(0.0, abs=1e-8)

    def test_zero_benchmark_rmsfe_returns_nan(self, perfect_nowcast):
        realized, nowcast = perfect_nowcast
        result = compute_theil_u(nowcast, realized, realized)  # perfect benchmark
        assert math.isnan(result)


# ── diebold_mariano ───────────────────────────────────────────────────────────

class TestDieboldMariano:

    def test_output_keys(self, imperfect_nowcast):
        realized, nowcast = imperfect_nowcast
        errors_model = nowcast - realized
        errors_bench = realized.shift(4).fillna(method="ffill") - realized
        result = diebold_mariano(errors_model.dropna(), errors_bench.dropna())
        assert {"statistic", "pvalue", "n_obs"} <= result.keys()

    def test_sign_symmetric(self, imperfect_nowcast):
        """Swapping model and benchmark flips the sign of the DM statistic."""
        realized, nowcast = imperfect_nowcast
        e1 = nowcast - realized
        e2 = (realized + 2) - realized  # constant +2 error benchmark

        dm_12 = diebold_mariano(e1, e2)
        dm_21 = diebold_mariano(e2, e1)

        if not (math.isnan(dm_12["statistic"]) or math.isnan(dm_21["statistic"])):
            assert dm_12["statistic"] == pytest.approx(-dm_21["statistic"], rel=1e-6)

    def test_pvalue_in_range(self, imperfect_nowcast):
        realized, nowcast = imperfect_nowcast
        e1 = nowcast - realized
        e2 = (realized + 1.5) - realized
        result = diebold_mariano(e1, e2)
        if not math.isnan(result["pvalue"]):
            assert 0.0 <= result["pvalue"] <= 1.0


# ── benchmark_random_walk ─────────────────────────────────────────────────────

class TestBenchmarkRandomWalk:

    def test_shift_4_quarters(self):
        idx = pd.date_range("2018-03-31", periods=8, freq="QE")
        va  = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], index=idx)
        rw  = benchmark_random_walk(va)
        # First 4 should be NaN, rest should be original values shifted by 4
        assert rw.iloc[:4].isna().all()
        assert rw.iloc[4] == pytest.approx(va.iloc[0])
        assert rw.iloc[5] == pytest.approx(va.iloc[1])
