import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

# Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Apply the cleaning strategy identified in the audit."""
    df_clean = df.copy()
    
    # 1. Winsorization (Cap outliers at 3 sigma)
    exog_cols = ['consommation_ciment', 'credits_equipement', 'credits_immobilier', 'lafarge_index']
    for col in exog_cols:
        if col in df_clean.columns:
            m, s = df_clean[col].mean(), df_clean[col].std()
            df_clean[col] = df_clean[col].clip(lower=m - 3*s, upper=m + 3*s)
            
    # 2. Temporal Smoothing (3-month rolling mean to denoise monthly data)
    for col in exog_cols:
        df_clean[f"{col}_smooth"] = df_clean[col].rolling(window=3, min_periods=1).mean()
        
    # 3. Dummy Variables
    # A. December Dummy (for year-end credit artifacts)
    df_clean['is_december'] = (df_clean.index.month == 12).astype(int)
    
    # B. COVID Dummy (2020 Q2 and Q3)
    df_clean['is_covid'] = ((df_clean.index >= '2020-03-01') & (df_clean.index <= '2020-09-30')).astype(int)
    
    # 4. Stationarity Adjustments (Difference of Growth Rates if needed)
    # Experimenting with the 'Acceleration' of cement
    df_clean['ciment_accel'] = df_clean['consommation_ciment'].diff()
    
    # 5. Mandatory AR(1) Term
    df_clean['L1_target'] = df_clean[target_col].shift(3)
    
    return df_clean

def run_cleaned_experiment():
    print("=== Experiment: Cleaned Data & Advanced Features ===")
    
    panel_path = _project_root / "data/processed/model_panel.parquet"
    df_raw = pd.read_parquet(panel_path)
    target_col = "va_construction_yoy"
    
    # 1. Apply Cleaning
    df = clean_data(df_raw, target_col)
    
    # 2. Select the "Optimized Feature Set"
    features = [
        'L1_target',           # Persistence
        'consommation_ciment_smooth', # Denoised leading indicator
        'credits_immobilier_smooth',  # Financing signal
        'is_december',         # Accounting adjustment
        'is_covid'             # Structural shock adjustment
    ]
    
    data = df.dropna(subset=[target_col, 'L1_target']).copy()
    X = data[features].ffill().fillna(0)
    y = data[target_col]
    
    # 3. In-Sample Evaluation
    X_ols = sm.add_constant(X)
    model = sm.OLS(y, X_ols).fit()
    print(model.summary())
    
    print(f"\nCLEANED In-sample R2: {model.rsquared:.4f}")
    print(f"CLEANED Adj R2: {model.rsquared_adj:.4f}")
    
    # 4. Out-of-Sample Validation (The true test)
    tscv = TimeSeriesSplit(n_splits=5)
    oos_preds, true_vals = [], []
    
    for train_index, test_index in tscv.split(X):
        X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
        y_tr, y_te = y.iloc[train_index], y.iloc[test_index]
        
        m_te = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
        oos_preds.extend(m_te.predict(sm.add_constant(X_te)))
        true_vals.extend(y_te)
        
    print(f"CLEANED Out-of-sample R2: {r2_score(true_vals, oos_preds):.4f}")

if __name__ == "__main__":
    run_cleaned_experiment()
