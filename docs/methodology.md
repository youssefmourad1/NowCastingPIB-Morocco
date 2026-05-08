# Methodology

## Model: Dynamic Factor Model (DFM)

### Reference
Danov, N., Giannone, D., Kabundi, A., Okou, C., Spilimbergo, A. (2026).
*Nowcasting GDP Growth for Kenya.* IMF Working Paper WP/26/32.

### Model equations

**Observation equation:**
```
y_t = Λ F_t + ε_t
```

**Factor dynamics:**
```
F_t = Γ F_{t-1} + u_t,   u_t ~ i.i.d. N(0, Ξ)
```

**Idiosyncratic component:**
```
ε_t = Θ ε_{t-1} + η_t,   η_t ~ i.i.d. N(0, Σ)
```

Where:
- `y_t` is the (M×1) vector of standardised indicators at time t
- `Λ` is the (M×k) factor loading matrix
- `F_t` is the (k×1) vector of latent common factors
- `ε_t` is the idiosyncratic component, AR(1) with diagonal Θ and Σ

### Mixed-frequency aggregation constraint

VA CONSTRUCTION is published quarterly but enters the monthly panel.
The temporal aggregation constraint (Equation 9 of Danov et al.):

```
VA^q_t = VA^m_t + VA^m_{t-1} + VA^m_{t-2}
       = λ_VA (f_t + f_{t-1} + f_{t-2}) + ε_t + ε_{t-1} + ε_{t-2}
```

The quarterly observation appears only in the last month of the quarter (March, June,
September, December) in the panel. The Kalman smoother infers the monthly path.

### Estimation

EM-Kalman algorithm of Banbura & Modugno (2014). Handles arbitrary missing data
patterns natively — quarterly series and publication lags require no pre-imputation.

### News decomposition

Revision between two information sets Ω_v and Ω_{v+1}:

```
E[VA^q_t | Ω_{v+1}] - E[VA^q_t | Ω_v]  =  Σ_j δ_{v+1,j} × (y_{t_j} - E[y_{t_j} | Ω_v])
```

Identifies which newly published indicator drove the latest nowcast revision.

## References

- Banbura, M. & Modugno, M. (2014). *Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data.* Journal of Applied Econometrics, 29(1), 133–160.
- Bok et al. (2018). *Macroeconomic Nowcasting and Forecasting with Big Data.* Annual Review of Economics, 10, 615–643.
- Stock, J.H. & Watson, M.W. (2025). *Recovering from COVID.* Brookings Papers on Economic Activity.
- Alessi, L., Barigozzi, M., Capasso, M. (2010). *Improved Penalization for Determining the Number of Factors.* Statistics & Probability Letters, 80(23–24), 1806–1813.
