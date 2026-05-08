import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def run_time_interaction():
    print("=== Experiment: Time-Varying Correlation (Post-2018 Interaction) ===")
    
    panel_path = _project_root / "data/processed/model_panel.parquet"
    df = pd.read_parquet(panel_path)
    target_col = "va_construction_yoy"
    
    # Base
    df['L1_target'] = df[target_col].shift(3)
    df['is_covid'] = ((df.index >= '2020-03-01') & (df.index <= '2020-09-30')).astype(int)
    
    # Time Interaction
    df['is_post_2018'] = (df.index.year >= 2018).astype(int)
    df['ciment_recent'] = df['consommation_ciment'] * df['is_post_2018']
    
    data = df.dropna(subset=[target_col, 'L1_target']).copy()
    
    X = sm.add_constant(data[['L1_target', 'is_covid', 'consommation_ciment', 'ciment_recent']].ffill().fillna(0))
    y = data[target_col]
    
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    print(f"\nTIME-INTERACTION R2: {model.rsquared:.4f}")

if __name__ == "__main__":
    run_time_interaction()
