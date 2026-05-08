import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

# 1. Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def run_experiment():
    print("=== Experiment: U-MIDAS (Lags) ===")
    
    # 2. Load data
    panel_path = _project_root / "data/processed/model_panel.parquet"
    if not panel_path.exists():
        print(f"Error: Panel not found at {panel_path}")
        return
        
    df = pd.read_parquet(panel_path)
    
    # Target: va_construction_yoy
    target_col = "va_construction_yoy"
    feature_cols = ['consommation_ciment', 'credits_equipement', 'credits_immobilier', 'lafarge_index']
    
    # 3. Create Lags (U-MIDAS approach)
    # For each indicator, we take lags 0, 1, 2, 3
    lagged_features = []
    for col in feature_cols:
        for lag in range(4):
            lag_name = f"{col}_lag{lag}"
            df[lag_name] = df[col].shift(lag)
            lagged_features.append(lag_name)
    
    # 4. Filter to Quarter Ends
    # The DFM panel has target at 3,6,9,12
    data = df.dropna(subset=[target_col]).copy()
    
    if len(data) < 10:
        print(f"Error: Not enough observations ({len(data)}).")
        return
        
    X = data[lagged_features].ffill().fillna(0)
    y = data[target_col]
    X = sm.add_constant(X)
    
    print(f"Dataset: {len(data)} observations, {len(lagged_features)} features.")
    
    # 5. OLS
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    # 6. Evaluation
    y_pred = model.predict(X)
    print(f"\nIn-sample R2: {r2_score(y, y_pred):.4f}")
    print(f"In-sample RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

if __name__ == "__main__":
    run_experiment()
