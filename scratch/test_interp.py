from lamiaty.config import load_settings
from lamiaty.data.pipeline import run_pipeline

settings = load_settings()
panel = run_pipeline(settings)
target = "va_construction_yoy"

y = panel[target].resample("ME").last().interpolate(method="linear").dropna()
print("y count:", len(y))
print(y.head(10))
