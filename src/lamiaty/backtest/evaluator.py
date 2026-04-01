"""
Forecast evaluation metrics — Phase 3 stub.

Will compute RMSFE, MAFE, bias, Theil U, and Diebold-Mariano statistics
for comparing the DFM nowcast against benchmarks.

Benchmarks (§5.3.2):
  - Random walk: VA_{t} = VA_{t-4} (naïve y-o-y stable)
  - AR(1) on VA BTP
  - Bridge equation: OLS of VA_BTP on ciment + LafargeHolcim
  - HCP / Bank Al-Maghrib published forecasts (if available)
"""
import numpy as np


def compute_rmsfe(nowcasts, realized):
    """Root Mean Squared Forecast Error. Phase 3 implementation."""
    raise NotImplementedError("Evaluator implemented in Phase 3.")


def compute_mafe(nowcasts, realized):
    """Mean Absolute Forecast Error. Phase 3 implementation."""
    raise NotImplementedError("Evaluator implemented in Phase 3.")


def compute_theil_u(nowcasts, realized, naive_forecasts):
    """Theil U statistic vs. naïve benchmark. Phase 3 implementation."""
    raise NotImplementedError("Evaluator implemented in Phase 3.")


def diebold_mariano(errors_model, errors_benchmark, h=1):
    """Diebold-Mariano test for equal predictive accuracy. Phase 3 implementation."""
    raise NotImplementedError("Evaluator implemented in Phase 3.")
