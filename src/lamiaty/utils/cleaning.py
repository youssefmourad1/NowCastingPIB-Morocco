"""
Cleaning and Feature Engineering utilities for the Nowcasting pipeline.
Includes outlier handling, smoothing, and structural dummies.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_nowcasting_panel(
    df: pd.DataFrame, 
    target_col: str = "va_construction_yoy",
    winsorize_sigma: float = 3.0,
    smooth_window: int = 3
) -> pd.DataFrame:
    """
    Apply a comprehensive cleaning and engineering pipeline to the model panel.
    
    Args:
        df: Raw model panel (monthly-aligned).
        target_col: The primary prediction target.
        winsorize_sigma: Threshold for outlier clipping.
        smooth_window: Window for rolling mean smoothing.
        
    Returns:
        DataFrame with cleaned and engineered features.
    """
    df_clean = df.copy()
    
    # 1. Outlier Handling (Winsorization)
    # We apply this to exogenous indicators to prevent extreme shocks from distorting models.
    exog_cols = [c for c in df.columns if c not in [target_col, "va_construction"]]
    
    for col in exog_cols:
        if col in df_clean.columns:
            series = df_clean[col].dropna()
            if len(series) > 0:
                m, s = series.mean(), series.std()
                lower, upper = m - winsorize_sigma * s, m + winsorize_sigma * s
                df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
                
    # 2. Temporal Smoothing (Denoising)
    # Construction signals are often noisy month-to-month. Rolling means help extract the trend.
    smoothed_cols = []
    for col in exog_cols:
        smooth_name = f"{col}_smooth"
        df_clean[smooth_name] = df_clean[col].rolling(window=smooth_window, min_periods=1).mean()
        smoothed_cols.append(smooth_name)
        
    # 3. Structural Dummies
    # A. COVID Dummy (2020 Q2-Q3)
    df_clean['is_covid'] = ((df_clean.index >= '2020-03-01') & (df_clean.index <= '2020-09-30')).astype(float)
    
    # B. December Dummy (Year-end credit cleanup artifact)
    df_clean['is_december'] = (df_clean.index.month == 12).astype(float)
    
    # 4. Auto-Regressive Term (Persistence)
    # The previous quarter's GDP is the strongest predictor for construction.
    df_clean['L1_target'] = df_clean[target_col].shift(3)
    
    logger.info(
        "Cleaned panel: applied winsorization (%.1f sigma), smoothing (%d mon), "
        "and added structural dummies.", winsorize_sigma, smooth_window
    )
    
    return df_clean

def get_golden_features(df: pd.DataFrame) -> list[str]:
    """Return the list of recommended features after cleaning."""
    base_smoothed = [c for c in df.columns if c.endswith("_smooth")]
    dummies = ['is_covid', 'is_december', 'L1_target']
    return base_smoothed + dummies
