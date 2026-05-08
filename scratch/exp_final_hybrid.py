import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

# 1. Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def run_experiment():
    print("=== Experiment: Elastic Net on U-MIDAS Features ===")
    
    # 2. Load data
    panel_path = _project_root / "data/processed/model_panel.parquet"
    df = pd.read_parquet(panel_path)
    
    # Target
    target_col = "va_construction_yoy"
    feature_cols = ['consommation_ciment', 'credits_equipement', 'credits_immobilier', 'lafarge_index']
    
    # 3. Create Lags (Lags 0 to 5)
    lagged_features = []
    for col in feature_cols:
        for lag in range(6): # Test deeper lags
            lag_name = f"{col}_lag{lag}"
            df[lag_name] = df[col].shift(lag)
            lagged_features.append(lag_name)
    
    # 4. Filter to Quarter Ends
    data = df.dropna(subset=[target_col]).copy()
    X = data[lagged_features].ffill().fillna(0)
    y = data[target_col]
    
    print(f"Dataset: {len(X)} observations, {len(lagged_features)} features.")
    
    # 5. TimeSeries CV
    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        results.append({
            "R2": r2_score(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        })
        
    res_df = pd.DataFrame(results)
    print("\n--- Out-of-Sample Metrics (TSC-CV) ---")
    print(res_df.to_string())
    print(f"\nAverage OOS R2: {res_df['R2'].mean():.4f}")
    
    # 6. Final Fit and Feature Selection
    model_final = ElasticNetCV(cv=5, random_state=42).fit(X, y)
    coeffs = pd.Series(model_final.coef_, index=lagged_features)
    selected = coeffs[coeffs != 0].sort_values(ascending=False)
    
    print("\n--- Selected Features (Non-Zero) ---")
    if selected.empty:
        print("None selected (Model predicted the mean).")
    else:
        print(selected)

if __name__ == "__main__":
    run_experiment()
