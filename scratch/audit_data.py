import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def audit_data_quality():
    print("=== Data Quality Audit: Cleaning Strategy Analysis ===")
    
    # 1. Load panel
    panel_path = _project_root / "data/processed/model_panel.parquet"
    df = pd.read_parquet(panel_path)
    
    results = []
    
    # 2. Audit each column
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 10: continue
        
        # A. Stationarity (ADF Test)
        adf_res = adfuller(series)
        is_stationary = adf_res[1] < 0.05
        
        # B. Outliers (Z-score > 3)
        z_scores = (series - series.mean()) / series.std()
        outliers_count = (z_scores.abs() > 3).sum()
        extreme_dates = series.index[z_scores.abs() > 3].tolist()
        
        # C. Missingness
        null_pct = df[col].isna().mean() * 100
        
        results.append({
            "Variable": col,
            "Stationary": is_stationary,
            "ADF_p": f"{adf_res[1]:.4f}",
            "Outliers": outliers_count,
            "Null_%": f"{null_pct:.1f}%"
        })
        
        if outliers_count > 0:
            print(f"\nOutliers in {col}: {len(extreme_dates)} found at {extreme_dates}")

    audit_df = pd.DataFrame(results)
    print("\n--- Summary Audit Table ---")
    print(audit_df.to_string())
    
    # 3. Correlation Stability Check (Pre vs Post 2018)
    print("\n--- Correlation Stability (va_construction vs Others) ---")
    target = "va_construction_yoy"
    for col in ['consommation_ciment', 'credits_immobilier']:
        if col in df.columns and target in df.columns:
            full_corr = df[col].corr(df[target])
            pre_corr  = df[df.index < '2018-01-01'][col].corr(df[target])
            post_corr = df[df.index >= '2018-01-01'][col].corr(df[target])
            print(f"{col}: Full={full_corr:.2f}, Pre-2018={pre_corr:.2f}, Post-2018={post_corr:.2f}")

if __name__ == "__main__":
    audit_data_quality()
