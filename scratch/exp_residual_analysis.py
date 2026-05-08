import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def run_residual_analysis():
    print("=== Residual Analysis: What is the AR(1) missing? ===")
    
    panel_path = _project_root / "data/processed/model_panel.parquet"
    df = pd.read_parquet(panel_path)
    target_col = "va_construction_yoy"
    
    # 1. Fit AR(1) baseline
    df['L1_target'] = df[target_col].shift(3)
    data = df.dropna(subset=[target_col, 'L1_target']).copy()
    
    X_ar = sm.add_constant(data[['L1_target']])
    y = data[target_col]
    ar_model = sm.OLS(y, X_ar).fit()
    
    # 2. Get Residuals (The "Innovation")
    data['residual'] = ar_model.resid
    
    # 3. Correlate Features with Residuals
    exog_cols = ['consommation_ciment', 'credits_equipement', 'credits_immobilier', 'lafarge_index']
    corrs = []
    
    for col in exog_cols:
        for lag in range(7):
            c = data['residual'].corr(data[col].shift(lag))
            corrs.append({"Feature": col, "Lag": lag, "Corr_with_Residual": c})
            
    corr_df = pd.DataFrame(corrs).sort_values("Corr_with_Residual", key=abs, ascending=False)
    print("\n--- Correlation of Indicators with AR(1) Residuals ---")
    print(corr_df.head(10))
    
    # 4. Try to improve the model with the top residual-correlated feature
    top_feat = corr_df.iloc[0]['Feature']
    top_lag = int(corr_df.iloc[0]['Lag'])
    
    data['top_indicator'] = data[top_feat].shift(top_lag)
    X_plus = sm.add_constant(data[['L1_target', 'top_indicator']].dropna())
    y_plus = data.loc[X_plus.index, target_col]
    
    plus_model = sm.OLS(y_plus, X_plus).fit()
    print(f"\n--- Improved Model (AR1 + {top_feat} Lag {top_lag}) ---")
    print(f"R2: {plus_model.rsquared:.4f}")
    print(f"Adj R2: {plus_model.rsquared_adj:.4f}")

if __name__ == "__main__":
    run_residual_analysis()
