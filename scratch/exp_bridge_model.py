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
    print("=== Experiment: Bridge Model (Quarterly Aggregation) ===")
    
    # 2. Load data
    panel_path = _project_root / "data/processed/model_panel.parquet"
    if not panel_path.exists():
        print(f"Error: Panel not found at {panel_path}")
        return
        
    df = pd.read_parquet(panel_path)
    
    # Target: va_construction_yoy
    target_col = "va_construction_yoy"
    feature_cols = ['consommation_ciment', 'credits_equipement', 'credits_immobilier', 'lafarge_index']
    
    # 3. Aggregate Monthly to Quarterly
    # We want to take the mean of the 3 months in each quarter
    q_features = df[feature_cols].resample('QE').mean()
    
    # Target is already at quarter ends
    q_target = df[target_col].dropna()
    q_target.index = q_target.index.to_period('Q').to_timestamp(how='end').normalize()
    q_features.index = q_features.index.to_period('Q').to_timestamp(how='end').normalize()
    
    # 4. Align
    print(f"q_target head:\n{q_target.head()}")
    print(f"q_features head:\n{q_features.head()}")
    data = pd.concat([q_target, q_features], axis=1).dropna()
    
    if len(data) < 10:
        print(f"Error: Not enough observations ({len(data)}).")
        return
        
    X = data[feature_cols]
    y = data[target_col]
    X = sm.add_constant(X)
    
    print(f"Dataset: {len(data)} quarters.")
    
    # 5. OLS
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    # 6. Evaluation (In-sample for now, as data is very small)
    y_pred = model.predict(X)
    print(f"\nIn-sample R2: {r2_score(y, y_pred):.4f}")
    print(f"In-sample RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

if __name__ == "__main__":
    run_experiment()
