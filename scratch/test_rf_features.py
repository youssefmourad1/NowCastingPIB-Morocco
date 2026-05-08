import sys
import warnings
import pandas as pd
from lamiaty.config import load_settings
from lamiaty.data.pipeline import run_pipeline

settings = load_settings()
panel = run_pipeline(settings)
from lamiaty.app.ml_models_page import _prepare_ml_data, train_random_forest

data = _prepare_ml_data(hash(str(panel.shape)), panel, freq="M")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
feature_names = data["feature_names"]

rf_res = train_random_forest(X_train, y_train, X_test, y_test)

print("len feature_names:", len(feature_names))
print("len importances:", len(rf_res.get("feature_importances", [])))

if "error" in rf_res:
    print("Error:", rf_res["error"])
