import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

# Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def run_nonlinear_oos():
    print("=== Non-Linear Models: Out-of-Sample Performance ===")
    
    panel_path = _project_root / "data/processed/model_panel.parquet"
    df = pd.read_parquet(panel_path)
    target_col = "va_construction_yoy"
    
    df['L1_target'] = df[target_col].shift(3)
    df['Ciment_L1'] = df['consommation_ciment'].shift(1)
    df['Credits_L1'] = df['credits_immobilier'].shift(1)
    
    data = df.dropna(subset=[target_col, 'L1_target']).copy()
    features = ['L1_target', 'Ciment_L1', 'Credits_L1']
    X = data[features].ffill().fillna(0)
    y = data[target_col]
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=42)
    }
    
    for name, model in models.items():
        oos_preds = []
        true_vals = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            oos_preds.extend(model.predict(X_test))
            true_vals.extend(y_test)
            
        oos_r2 = r2_score(true_vals, oos_preds)
        print(f"{name} Out-of-sample R2: {oos_r2:.4f}")

if __name__ == "__main__":
    run_nonlinear_oos()
