from lamiaty.config import load_settings
from lamiaty.data.pipeline import run_pipeline
import pandas as pd

settings = load_settings()
panel = run_pipeline(settings)
target = "va_construction_yoy"

excluded = {target, "va_construction", "ipai", "creation_emploi", "investissement_etat"}
feature_cols = [c for c in panel.columns if c not in excluded]

resample_rule = "ME"
q = panel[feature_cols].resample(resample_rule).mean()

for col in feature_cols:
    q[f"{col}_lag1"] = q[col].shift(1)
    q[f"{col}_lag2"] = q[col].shift(2)

X_cols = feature_cols + [f"{c}_lag1" for c in feature_cols] + [f"{c}_lag2" for c in feature_cols]

y = panel[target].resample(resample_rule).last().interpolate(method="linear").dropna()

model_df = pd.concat([q[X_cols], y], axis=1).dropna()
print("After dropna:", len(model_df))
