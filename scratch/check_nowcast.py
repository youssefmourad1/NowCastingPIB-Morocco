import sys
from pathlib import Path

# Fix path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

import pandas as pd
import numpy as np
from lamiaty.config import load_settings
from lamiaty.data.pipeline import run_pipeline
from lamiaty.model.dfm import DynamicFactorModel

# Load settings and run pipeline
settings = load_settings(project_root=_project_root)
panel = run_pipeline(settings)

# Fit DFM
dfm = DynamicFactorModel(
    n_factors=settings.model.n_factors,
    n_lags=settings.model.n_lags,
    settings=settings.model
)
dfm.fit(panel)

# Run historical nowcast
nc = dfm.historical_nowcast(panel)

print("\n--- Nowcast results ---")
print(f"Number of nowcasts: {len(nc)}")
if len(nc) > 0:
    print("Last 5 nowcasts:")
    print(nc.tail(5))
    print("\nCI lower tail:")
    print(nc.attrs['ci_lower'].tail(5))
    print("\nCI upper tail:")
    print(nc.attrs['ci_upper'].tail(5))
else:
    print("ERROR: No nowcasts generated!")

