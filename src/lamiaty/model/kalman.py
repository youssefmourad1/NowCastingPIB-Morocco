"""
State-space reference and companion form helpers.

The actual Kalman filter and EM algorithm are implemented by
statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ
(Banbura & Modugno 2014).  This module documents the state-space
matrices that DynamicFactorMQ builds internally and provides
build_companion_form() for inspection / educational purposes.

State-space form used by DynamicFactorMQ:
  Transition:    α_t  = T α_{t-1}  + η_t,    Cov(η_t) = Q
  Observation:   y_t  = Z α_t      + ε_t,    Cov(ε_t) = H

For a DFM with k factors, p VAR lags, N series:
  State vector:  α_t = (F_t', F_{t-1}', …, F_{t-p+1}', ε_t')'
    where F_t ∈ R^k, ε_t ∈ R^N  (idiosyncratic component)
  T  = companion matrix (kp + N) × (kp + N)
  Z  = [Λ | 0_{N×k(p-1)} | I_N]  observation matrix
  Q  = block-diag(Ξ at top-left, Σ at bottom-right)
  H  = 0  (idiosyncratic variance absorbed into state)

References:
  Banbura, M. & Modugno, M. (2014). Maximum Likelihood Estimation of
  Factor Models on Datasets with Arbitrary Pattern of Missing Data.
  Journal of Applied Econometrics, 29(1), 133–160.

  Bok, B. et al. (2018). Macroeconomic Nowcasting and Forecasting
  with Big Data. Annual Review of Economics, 10, 615–643.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_companion_form(
    loadings: pd.DataFrame,
    gamma: np.ndarray,
    sigma_diag: np.ndarray,
    xi: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the companion-form state-space matrices for a fitted DFM.

    Useful for verifying the model or computing impulse responses manually.

    Args:
        loadings:   Factor loading matrix Λ, shape (N, k).
        gamma:      Factor transition matrix Γ, shape (k, k).
        sigma_diag: Diagonal idiosyncratic variances σ²_i, shape (N,).
        xi:         Factor innovation covariance Ξ, shape (k, k).
                    If None, assumes identity.

    Returns:
        Dict with keys:
          'T'  — state transition matrix (k + N, k + N)
          'Z'  — observation matrix (N, k + N)
          'Q'  — state innovation covariance (k + N, k + N)
          'H'  — observation noise covariance (N, N) — all zeros
    """
    N = loadings.shape[0]
    k = loadings.shape[1]
    d = k + N  # state dimension (1 lag for simplicity)

    # Transition matrix
    T = np.zeros((d, d))
    T[:k, :k] = gamma                   # factor VAR block
    T[k:, k:] = np.zeros((N, N))        # idiosyncratic AR (0 for white noise)

    # Observation matrix
    Z = np.zeros((N, d))
    Z[:, :k] = loadings.values          # Λ
    Z[:, k:] = np.eye(N)                # idiosyncratic pass-through

    # State innovation covariance
    Q = np.zeros((d, d))
    Q[:k, :k] = xi if xi is not None else np.eye(k)
    Q[k:, k:] = np.diag(sigma_diag)    # diagonal Σ

    # Observation noise (zero — idiosyncratic in state)
    H = np.zeros((N, N))

    return {"T": T, "Z": Z, "Q": Q, "H": H}


class KalmanFilter:
    """Reference class documenting the Kalman filter algorithm.

    The production implementation uses statsmodels' optimised C-level
    Kalman filter internally.  This class exists for documentation
    and educational inspection only.

    Algorithm (Kalman 1960, Durbin & Koopman 2012):
      Predict:
        α_{t|t-1} = T α_{t-1|t-1}
        P_{t|t-1} = T P_{t-1|t-1} T' + Q

      Update (observed subset O_t ⊆ {1,…,N}):
        ν_t        = y_{O_t,t} − Z_{O_t} α_{t|t-1}
        F_t        = Z_{O_t} P_{t|t-1} Z_{O_t}' + H_{O_t,O_t}
        K_t        = P_{t|t-1} Z_{O_t}' F_t^{−1}
        α_{t|t}    = α_{t|t-1} + K_t ν_t
        P_{t|t}    = (I − K_t Z_{O_t}) P_{t|t-1}

      Smoother (Rauch-Tung-Striebel):
        L_t        = T P_{t|t} T' + Q  (used for backward pass)
        α_{t|T}    = α_{t|t} + P_{t|t} T' P_{t+1|t}^{−1} (α_{t+1|T} − T α_{t|t})
    """

    def filter(self, *args, **kwargs):
        raise NotImplementedError(
            "Use statsmodels DynamicFactorMQ directly for production filtering."
        )

    def smooth(self, *args, **kwargs):
        raise NotImplementedError(
            "Use statsmodels DynamicFactorMQ directly for production smoothing."
        )


class EMAlgorithm:
    """Reference class documenting the EM algorithm for DFM estimation.

    The production implementation uses statsmodels' EM routine.

    Algorithm (Banbura & Modugno 2014):
      Initialisation:
        Λ⁰ ← PCA loadings on complete-case panel
        F⁰ ← PC scores (extrapolated to full panel)
        Γ⁰ ← 0.9 · I_k   (near unit-root initialisation)
        Σ⁰ ← diag(var(y_i − Λ⁰ F⁰))

      E-step (given Λ, Γ, Σ, Ξ):
        Run Kalman filter and smoother → E[F_t|Y], Cov[F_t|Y], E[F_t F_{t-1}'|Y]

      M-step (update parameters):
        Λ_new ← closed-form OLS on smoothed factors (handles NaN per series)
        Σ_new ← residual variance (diagonal)
        Γ_new ← closed-form VAR regression on smoothed factors
        Ξ_new ← residual VAR covariance

      Convergence:
        |log L_{n+1} − log L_n| / |log L_n| < tolerance
    """

    def run(self, *args, **kwargs):
        raise NotImplementedError(
            "Use statsmodels DynamicFactorMQ directly for production EM estimation."
        )
