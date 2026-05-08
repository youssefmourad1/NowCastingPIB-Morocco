
import pandas as pd
import numpy as np
from lamiaty.features.stationarity import TRANSFORM_RULES

# Mocking the panel load or finding where it's loaded
# I'll try to find a data file or just list some data
import os

data_dir = "/Users/Apple/Desktop/projects/NowCastingPIB-Morocco/NowCastingPIB-Morocco/data/processed"
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    print(f"Files in processed: {files}")
    for f in files:
        if f.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(data_dir, f))
            print(f"\nFile: {f}")
            print(df.info())
            print(df.head())
            print("\nValue counts for va_construction (non-null):")
            if 'va_construction' in df.columns:
                print(df['va_construction'].dropna().shape)
            break
