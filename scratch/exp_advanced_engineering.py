import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

# Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def run_advanced_engineering():
    print("=== Experiment: Advanced Engineering (Interactions, PCA, Acceleration) ===")
    
    panel_path = _project_root / "data/processed/model_panel.parquet"
    df = pd.read_parquet(panel_path)
    target_col = "va_construction_yoy"
    
    # 1. Base Cleaning & Persistence
    df['L1_target'] = df[target_col].shift(3)
    
    # 2. Advanced Features
    # A. Acceleration (Momentum)
    df['ciment_accel'] = df['consommation_ciment'].diff()
    
    # B. Interactions (The "Fuel" Effect)
    df['ciment_x_credits'] = df['consommation_ciment'] * df['credits_immobilier']
    
    # C. PCA (The "Construction Pulse")
    exog_cols = ['consommation_ciment', 'credits_immobilier', 'lafarge_index']
    pca_data = df[exog_cols].ffill().fillna(0)
    pca = PCA(n_components=1)
    df['construction_pulse'] = pca.fit_transform(pca_data)
    
    # D. Dummies
    df['is_covid'] = ((df.index >= '2020-03-01') & (df.index <= '2020-09-30')).astype(int)
    
    # 3. Define Candidate Models
    features_sets = {
        "Base (AR1 + Covid)": ['L1_target', 'is_covid'],
        "Interactions": ['L1_target', 'is_covid', 'ciment_x_credits'],
        "Acceleration": ['L1_target', 'is_covid', 'ciment_accel'],
        "PCA Pulse": ['L1_target', 'is_covid', 'construction_pulse'],
        "Combined Super-Model": ['L1_target', 'is_covid', 'ciment_x_credits', 'ciment_accel', 'construction_pulse']
    }
    
    data = df.dropna(subset=[target_col, 'L1_target']).copy()
    
    for name, f_list in features_sets.items():
        X = sm.add_constant(data[f_list].ffill().fillna(0))
        y = data[target_col]
        model = sm.OLS(y, X).fit()
        print(f"\n--- Model: {name} ---")
        print(f"R2: {model.rsquared:.4f} | Adj R2: {model.rsquared_adj:.4f}")

if __name__ == "__main__":
    run_advanced_engineering()
