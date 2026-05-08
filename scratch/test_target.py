from lamiaty.config import load_settings
from lamiaty.data.pipeline import run_pipeline

settings = load_settings()
panel = run_pipeline(settings)
print("panel freq:", panel.index.inferred_freq)
y = panel['va_construction_yoy'].dropna()
print("y_raw count:", len(y))
print("y_raw head:\n", y.head())
