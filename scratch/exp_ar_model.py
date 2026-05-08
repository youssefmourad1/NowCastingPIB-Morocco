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
    print("=== Experiment: AR(1) and AR(2) Baseline ===")
    
    # 2. Load data
    panel_path = _project_root / "data/processed/model_panel.parquet"
    df = pd.read_parquet(panel_path)
    
    # Target
    target_col = "va_construction_yoy"
    y_full = df[target_col].dropna()
    
    # 3. Create Lags of target
    data = pd.DataFrame(y_full)
    data['L1'] = data[target_col].shift(1)
    data['L2'] = data[target_col].shift(2)
    data = data.dropna()
    
    print(f"Dataset: {len(data)} quarters.")
    
    # 4. Fit AR(1)
    print("\n--- AR(1) Results ---")
    model1 = sm.OLS(data[target_col], sm.add_constant(data[['L1']])).fit()
    print(model1.summary())
    
    # 5. Fit AR(2)
    print("\n--- AR(2) Results ---")
    model2 = sm.OLS(data[target_col], sm.add_constant(data[['L1', 'L2']])).fit()
    print(model2.summary())

if __name__ == "__main__":
    run_experiment()
