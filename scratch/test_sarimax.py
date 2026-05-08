from lamiaty.config import load_settings
from lamiaty.data.pipeline import run_pipeline
from lamiaty.app.ml_models_page import _prepare_ml_data
import warnings
warnings.filterwarnings('ignore')

settings = load_settings()
panel = run_pipeline(settings)
data = _prepare_ml_data(hash(str(panel.shape)), panel, freq="Q")

X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(y_train, exog=X_train, order=(1,0,1), enforce_stationarity=False)
res = model.fit(disp=False)
p_tr = res.predict(start=0, end=len(y_train)-1, exog=X_train)
p_te = res.forecast(steps=len(y_test), exog=X_test)
print(len(p_tr), len(p_te))
print("SARIMAX SUCCESS!")
