import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def run_deep_exploration():
    print("=== Deep Exploration: Structural Breaks & Non-Linearity ===")
    
    # 1. Load data
    panel_path = _project_root / "data/processed/model_panel.parquet"
    df = pd.read_parquet(panel_path)
    
    target_col = "va_construction_yoy"
    
    # Features
    df['L1_target'] = df[target_col].shift(3)
    df['Ciment_L1'] = df['consommation_ciment'].shift(1)
    df['Credits_L1'] = df['credits_immobilier'].shift(1)
    
    data = df.dropna(subset=[target_col, 'L1_target']).copy()
    
    # Define COVID period
    covid_mask = (data.index >= '2020-01-01') & (data.index <= '2021-12-31')
    
    # 2. Split samples
    pre_covid = data[data.index < '2020-01-01']
    full_sample = data.copy()
    no_covid = data[~covid_mask]
    
    features = ['L1_target', 'Ciment_L1', 'Credits_L1']
    
    def evaluate(sample, name):
        X = sm.add_constant(sample[features])
        y = sample[target_col]
        model = sm.OLS(y, X).fit()
        print(f"\n--- {name} (N={len(sample)}) ---")
        print(f"R2: {model.rsquared:.4f}")
        print(f"Adj R2: {model.rsquared_adj:.4f}")
        return model

    # 3. Baseline Runs
    evaluate(pre_covid, "Pre-COVID (2010-2019)")
    evaluate(full_sample, "Full Sample (with COVID)")
    evaluate(no_covid, "No COVID (2010-2024 excluding 2020-21)")
    
    # 4. Dummy Variable for COVID
    data['is_covid'] = covid_mask.astype(int)
    X_dummy = sm.add_constant(data[features + ['is_covid']])
    y_dummy = data[target_col]
    model_dummy = sm.OLS(y_dummy, X_dummy).fit()
    print("\n--- Full Sample with COVID Dummy ---")
    print(f"R2: {model_dummy.rsquared:.4f}")
    
    # 5. Non-Linear (Gradient Boosting) on Full Sample
    print("\n--- Non-Linear (Gradient Boosting) ---")
    gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    gbm.fit(data[features], data[target_col])
    print(f"GBM In-sample R2: {r2_score(data[target_col], gbm.predict(data[features])):.4f}")

if __name__ == "__main__":
    run_deep_exploration()
