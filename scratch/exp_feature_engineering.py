import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import r2_score

# Setup paths
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

def run_feature_engineering_exploration():
    print("=== Advanced Feature Engineering Exploration for R2 Optimization ===")
    
    # 1. Load data
    panel_path = _project_root / "data/processed/model_panel.parquet"
    df_raw = pd.read_parquet(panel_path)
    
    target_col = "va_construction_yoy"
    base_features = ['consommation_ciment', 'credits_equipement', 'credits_immobilier', 'lafarge_index']
    
    # 2. Base Dataset (Target Lags are mandatory based on previous results)
    df = df_raw.copy()
    # IMPORTANT: Target is quarterly (NaN in non-quarter months). 
    # Must shift by 3 months to get the previous quarter.
    df['L1_target'] = df[target_col].shift(3) 
    df['L2_target'] = df[target_col].shift(6)
    
    # 3. Feature Generation
    new_features = []
    
    # A. Lags (1 to 6 months)
    for col in base_features:
        for lag in [1, 2, 3, 6]:
            name = f"{col}_lag{lag}"
            df[name] = df[col].shift(lag)
            new_features.append(name)
            
    # B. Moving Averages (3-month and 6-month)
    for col in base_features:
        for window in [3, 6]:
            name = f"{col}_ma{window}"
            df[name] = df[col].rolling(window=window).mean()
            new_features.append(name)
            
    # C. Momentum (Change in indicators)
    for col in base_features:
        name = f"{col}_momentum"
        df[name] = df[col].diff()
        new_features.append(name)

    # D. Interactions (Between main drivers: Cement and Credits)
    df['cement_x_credits_immo'] = df['consommation_ciment'] * df['credits_immobilier']
    new_features.append('cement_x_credits_immo')
    
    # 4. Filter and Clean
    all_features = ['L1_target', 'L2_target'] + new_features
    data = df.dropna(subset=[target_col]).copy()
    
    # Drop rows with too many NaNs in generated features
    data = data.dropna(subset=['L2_target'])
    
    X = data[all_features].ffill().fillna(0)
    y = data[target_col]
    
    print(f"Total features generated: {len(all_features)}")
    print(f"Total observations: {len(data)}")
    
    # 5. Greedy Selection / Regularization
    # Since we have many features and few observations, we MUST use Lasso for selection
    model = LassoCV(cv=5, max_iter=10000, random_state=42).fit(X, y)
    
    coeffs = pd.Series(model.coef_, index=all_features)
    selected = coeffs[coeffs.abs() > 1e-5].sort_values(ascending=False)
    
    print("\n--- Top Selected Features (Lasso) ---")
    print(selected)
    
    # 6. Evaluate the best model (In-sample for ceiling analysis)
    y_pred = model.predict(X)
    r2_final = r2_score(y, y_pred)
    print(f"\nFinal In-sample R2: {r2_final:.4f}")
    
    # 7. Check if we can reach 0.7 with OLS on selected features
    if not selected.empty:
        best_features = selected.index.tolist()
        ols_model = sm.OLS(y, sm.add_constant(X[best_features])).fit()
        print("\n--- OLS on Selected Features ---")
        print(f"Adjusted R2: {ols_model.rsquared_adj:.4f}")
        print(f"R2: {ols_model.rsquared:.4f}")
        
    # 8. Test Polynomials on the 3 strongest features
    if len(selected) >= 3:
        top_3 = selected.index[:3].tolist()
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X[top_3])
        poly_names = poly.get_feature_names_out(top_3)
        
        poly_model = sm.OLS(y, sm.add_constant(X_poly)).fit()
        print("\n--- Polynomial (Deg 2) on Top 3 Features ---")
        print(f"Polynomial R2: {poly_model.rsquared:.4f}")
        print(f"Polynomial Adjusted R2: {poly_model.rsquared_adj:.4f}")

if __name__ == "__main__":
    run_feature_engineering_exploration()
