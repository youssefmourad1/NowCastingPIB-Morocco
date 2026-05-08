import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# 1. Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def run_experiment():
    print("=== Experiment: Elastic Net for GDP Nowcasting ===")
    
    # 2. Load data
    panel_path = _project_root / "data/processed/model_panel.parquet"
    if not panel_path.exists():
        print(f"Error: Panel not found at {panel_path}")
        return
        
    df = pd.read_parquet(panel_path)
    
    # Target: va_construction_yoy (Growth rate)
    # Features: Everything else
    target_col = "va_construction_yoy"
    feature_cols = [c for c in df.columns if c not in [target_col, "va_construction"]]
    
    # 3. Preprocessing for ML
    # Drop rows where target is NaN (usually quarterly observations)
    data = df.dropna(subset=[target_col]).copy()
    
    if len(data) < 10:
        print(f"Error: Not enough observations ({len(data)}) for training.")
        return
        
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    
    # Handle NaNs in features (if any)
    X = X.ffill().fillna(0)
    
    print(f"Dataset: {len(X)} observations, {len(feature_cols)} features.")
    print(f"Features: {feature_cols}")
    
    # 4. Cross-Validation Setup
    tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 5))
    
    results = []
    
    # 5. Training and Evaluation
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Fit ElasticNet with built-in Cross-Validation for l1_ratio and alpha
        model = ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
            cv=5,
            max_iter=10000,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        results.append({
            "R2": r2,
            "RMSE": rmse,
            "MAPE": mape,
            "Alpha": model.alpha_,
            "L1_ratio": model.l1_ratio_
        })

    # 6. Final Results
    res_df = pd.DataFrame(results)
    print("\n--- Cross-Validation Metrics ---")
    print(res_df.to_string())
    print("\nAverage Metrics:")
    print(res_df.mean())
    
    # 7. Final Fit on all data to see coefficients
    model_final = ElasticNetCV(cv=5, random_state=42).fit(X, y)
    coeffs = pd.Series(model_final.coef_, index=feature_cols).sort_values(ascending=False)
    
    print("\n--- Final Model Coefficients ---")
    print(coeffs)
    print(f"Final Alpha: {model_final.alpha_:.6f}")
    print(f"Final L1 Ratio: {model_final.l1_ratio_:.2f}")

if __name__ == "__main__":
    run_experiment()
