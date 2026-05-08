import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def run_combined_best():
    print("=== Final Attempt: Combining Best Indicators & Lags ===")
    
    panel_path = _project_root / "data/processed/model_panel.parquet"
    df = pd.read_parquet(panel_path)
    target_col = "va_construction_yoy"
    
    # 1. Engineer the "Golden Features" found in exploration
    df['L1_target'] = df[target_col].shift(3)
    df['Ciment_L4'] = df['consommation_ciment'].shift(4)
    df['Credits_Immo_L5'] = df['credits_immobilier'].shift(5)
    df['Lafarge_L2'] = df['lafarge_index'].shift(2)
    
    features = ['L1_target', 'Ciment_L4', 'Credits_Immo_L5', 'Lafarge_L2']
    data = df.dropna(subset=[target_col] + features).copy()
    
    X = sm.add_constant(data[features])
    y = data[target_col]
    
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    print(f"\nFINAL R2: {model.rsquared:.4f}")
    print(f"FINAL Adjusted R2: {model.rsquared_adj:.4f}")

if __name__ == "__main__":
    run_combined_best()
