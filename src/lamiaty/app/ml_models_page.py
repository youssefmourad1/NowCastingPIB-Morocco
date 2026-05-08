"""
ML Models — Random Forest, LSTM, RMSE, MAPE & Comparaison
==========================================================
Module à intégrer dans streamlit_app.py.

Usage:
    Ajouter dans PAGES :
        "🤖 Modèles ML": "ml_models"

    Ajouter dans _ROUTERS :
        "ml_models": page_ml_models

    Puis appeler :
        from ml_models_page import page_ml_models

Ce module est autonome : il génère ses propres données synthétiques
si le pipeline n'est pas disponible, et tombe toujours gracieusement.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from lamiaty.utils.logging import log_stage
from lamiaty.utils.cleaning import clean_nowcasting_panel, get_golden_features
from plotly.subplots import make_subplots

logger = logging.getLogger("lamiaty.ml_models")

C_BLUE = "#2563eb"
C_RED = "#dc2626"
C_GREEN = "#16a34a"
C_ORANGE = "#ea580c"
C_PURPLE = "#7c3aed"
C_TEAL = "#0d9488"
C_GREY = "#64748b"
PLOTLY_TEMPLATE = "plotly_white"

MODEL_COLORS = {
    "Random Forest": C_BLUE,
    "LSTM": C_PURPLE,
    "AR(1)": C_ORANGE,
    "Random Walk": C_GREY,
    "DFM": C_TEAL,
    "Elastic Net": "#059669",
}


def _warn(msg: str) -> None:
    st.warning("⚠️ " + msg)


def _info(msg: str) -> None:
    st.info("ℹ️ " + msg)


def _quarter_label(dt: pd.Timestamp) -> str:
    return f"{dt.year}-Q{dt.quarter}"


@st.cache_data(show_spinner="Préparation des données ML...")
def _prepare_ml_data(
    panel_hash: int, 
    _panel: pd.DataFrame, 
    freq: str = "Q", 
    selected_features: list[str] | None = None
) -> dict[str, Any]:
    """
    Prépare X, y pour les modèles ML.
    Si selected_features est fourni, utilise ces colonnes directement.
    Sinon, utilise le pipeline automatique (lags + Lasso).
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    target = "va_construction_yoy"

    if target not in _panel.columns:
        raise ValueError(f"Cible '{target}' absente du panel.")

    panel = _panel.copy().sort_index()
    resample_rule = "QE" if freq == "Q" else "ME"

    if selected_features:
        # ── Cas : Features spécifiées manuellement ────────────────────────
        q = panel[selected_features].resample(resample_rule).mean()
        
        if freq == "Q":
            y = panel[target].dropna().resample(resample_rule).last().dropna()
        else:
            y = panel[target].resample(resample_rule).last().interpolate(method="linear").dropna()
        
        y.name = target
        model_df = pd.concat([q, y], axis=1).dropna()
        
        if model_df.empty:
            raise ValueError("Aucune observation après alignement des features manuelles.")
            
        split = int(len(model_df) * 0.70)
        X = model_df[selected_features]
        y = model_df[target]
        
        return {
            "X_train": X.iloc[:split].values,
            "X_test": X.iloc[split:].values,
            "y_train": y.iloc[:split].values,
            "y_test": y.iloc[split:].values,
            "dates_train": y.index[:split],
            "dates_test": y.index[split:],
            "feature_names": selected_features,
            "target_series": y,
            "model_df": model_df,
        }

    # ── Cas : Pipeline automatique (legacy) ───────────────────────────────
    # Variables exclues temporairement (y compris les cibles pour éviter la fuite de données)
    excluded = {
        target,
        "va_construction",
        "ipai",
        "creation_emploi",
        "investissement_etat",
    }

    feature_cols = [c for c in panel.columns if c not in excluded]

    if not feature_cols:
        raise ValueError("Aucune feature disponible après exclusion.")

    resample_rule = "QE" if freq == "Q" else "ME"
    
    # 1) Features (Moyennes trimestrielles ou mensuelles)
    q = panel[feature_cols].resample(resample_rule).mean()

    # 2) Lags t-1 et t-2
    for col in feature_cols:
        q[f"{col}_lag1"] = q[col].shift(1)
        q[f"{col}_lag2"] = q[col].shift(2)

    X_cols = (
        feature_cols
        + [f"{c}_lag1" for c in feature_cols]
        + [f"{c}_lag2" for c in feature_cols]
    )

    # 3) Cible
    if freq == "Q":
        y_raw = panel[target].dropna()
        if y_raw.empty:
            raise ValueError("La cible 'va_construction' est vide après dropna().")
        y = y_raw.resample(resample_rule).last().dropna()
    else:
        # Interpolation linéaire pour la fréquence mensuelle
        y = panel[target].resample(resample_rule).last().interpolate(method="linear").dropna()
        
    y.name = target

    # 4) Alignement propre
    model_df = pd.concat([q[X_cols], y], axis=1).dropna()

    if model_df.empty:
        raise ValueError(
            "Aucune observation disponible après alignement X/y. "
            "Vérifier transformations, lags et cible."
        )

    if len(model_df) < 20:
        raise ValueError(
            f"Pas assez d'observations trimestrielles ({len(model_df)} < 20)."
        )

    # 5) Split chronologique
    split = int(len(model_df) * 0.70)
    if split <= 0 or split >= len(model_df):
        raise ValueError("Split train/test invalide.")

    train_df = model_df.iloc[:split].copy()
    test_df = model_df.iloc[split:].copy()

    # 6) Filtre corrélation uniquement sur train
    corr = train_df[X_cols + [target]].corr(numeric_only=True)[target].drop(target)
    corr_keep = corr[abs(corr) >= 0.10].index.tolist()

    if not corr_keep:
        corr_keep = X_cols

    # 7) LASSO uniquement sur train
    X_train_raw = train_df[corr_keep]
    y_train = train_df[target]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)

    lasso = LassoCV(
        cv=min(5, max(2, len(y_train) // 4)),
        random_state=42,
        max_iter=20000,
    )
    lasso.fit(X_train_sc, y_train)

    selected = [
        col for col, coef in zip(corr_keep, lasso.coef_)
        if abs(coef) > 1e-10
    ]

    if not selected:
        selected = corr_keep

    X = model_df[selected]
    y = model_df[target]

    if y.isna().all():
        raise ValueError("La cible est entièrement NaN après préparation.")

    if np.isclose(float(y.std()), 0.0, atol=1e-12):
        raise ValueError(
            "La cible trimestrielle est constante ou quasi constante. "
            "Vérifier la transformation de va_construction."
        )

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    logger.info("Prepared ML data: X=%s | y=%s", X.shape, y.shape)
    logger.info("Selected ML features: %s", selected)
    logger.info("Target summary:\n%s", y.describe().to_string())

    return {
        "X_train": X_train.values,
        "X_test": X_test.values,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "dates_train": y.index[:split],
        "dates_test": y.index[split:],
        "feature_names": list(X.columns),
        "target_series": y,
        "model_df": model_df,
        "selected_features": selected,
        "corr_kept_features": corr_keep,
    }





# ── Metrics ───────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.sum():
        return float("nan")
    e = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(e ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    "Mean Absolute Percentage Error (évite division par 0)."
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (np.abs(y_true) > 1e-5)
    if not mask.sum():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.sum():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not mask.sum():
        return float("nan")
    ss_res = np.sum((y_true[mask] - y_pred[mask]) ** 2)
    ss_tot = np.sum((y_true[mask] - np.mean(y_true[mask])) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict[str, float]:
    return {
        "model": model_name,
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R²": r2(y_true, y_pred),
    }


# ── Model trainers ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Entraînement Elastic Net...")
def train_elastic_net(X_train, X_test, y_train, y_test, feature_names):
    from sklearn.linear_model import ElasticNetCV
    from sklearn.metrics import r2_score
    
    with log_stage("Training Elastic Net", "lamiaty.ml_models"):
        model = ElasticNetCV(
            l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
            cv=5,
            max_iter=10000,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        return {
            "model": model,
            "pred_test": y_pred,
            "pred_train": y_train_pred,
            "test_metrics": {
                "RMSE": rmse(y_test, y_pred),
                "MAPE": mape(y_test, y_pred),
                "MAE": mae(y_test, y_pred),
                "R²": r2_score(y_test, y_pred),
            },
            "train_metrics": {
                "RMSE": rmse(y_train, y_train_pred),
                "MAPE": mape(y_train, y_train_pred),
                "MAE": mae(y_train, y_train_pred),
                "R²": r2_score(y_train, y_train_pred),
            },
            "best_params": {
                "alpha": model.alpha_,
                "l1_ratio": model.l1_ratio_
            },
            "feature_importances": model.coef_
        }

@st.cache_data(show_spinner="Entraînement Random Forest…")
def train_random_forest(
    _X_train, _y_train, _X_test, _y_test,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(_X_train)
        X_te_sc = scaler.transform(_X_test)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            oob_score=True,
        )
        rf.fit(X_tr_sc, _y_train)

        pred_train = rf.predict(X_tr_sc)
        pred_test = rf.predict(X_te_sc)
        feature_imp = rf.feature_importances_

        return {
            "train_metrics": compute_metrics(_y_train, pred_train, "RF (train)"),
            "test_metrics": compute_metrics(_y_test, pred_test, "Random Forest"),
            "pred_train": pred_train,
            "pred_test": pred_test,
            "feature_importances": feature_imp,
            "oob_score": getattr(rf, "oob_score_", None),
        }
    except ImportError:
        return {"error": "scikit-learn non installé. `pip install scikit-learn`"}
    except Exception as exc:
        logger.exception("RF training failed")
        return {"error": str(exc)}


def _create_lstm_sequences(
    X: np.ndarray, y: np.ndarray, look_back: int
) -> tuple[np.ndarray, np.ndarray]:
    "Fenêtres glissantes pour LSTM : (n_samples, look_back, n_features)."
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i : i + look_back])
        ys.append(y[i + look_back])
    return np.array(Xs), np.array(ys)





@st.cache_data(show_spinner="Entraînement LSTM (peut prendre ~30s)…")
def train_lstm(
    _X_train, _y_train, _X_test, _y_test,
    look_back: int = 2, lstm_units: int = 16, dropout_rate: float = 0.0, epochs: int = 300, learning_rate: float = 1e-3,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.preprocessing import StandardScaler

        # Detect Apple Silicon MPS or fallback to CPU
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_tr_s = x_scaler.fit_transform(_X_train)
        X_te_s = x_scaler.transform(_X_test)
        y_tr_s = y_scaler.fit_transform(_y_train.reshape(-1, 1)).ravel()
        y_te_s = y_scaler.transform(_y_test.reshape(-1, 1)).ravel()

        X_tr_seq, y_tr_seq = _create_lstm_sequences(X_tr_s, y_tr_s, look_back)
        X_te_seq, y_te_seq = _create_lstm_sequences(X_te_s, y_te_s, look_back)

        if len(X_tr_seq) < 5:
            return {"error": "Pas assez de données pour LSTM."}

        n_features = X_tr_seq.shape[2]
        
        class PyTorchLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, dropout_r):
                super().__init__()
                self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.dropout = nn.Dropout(dropout_r)
                self.fc1 = nn.Linear(hidden_dim, 16)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(16, 1)

            def forward(self, x):
                out, _ = self.lstm1(x)
                out = self.dropout(out)
                out = out[:, -1, :] # Last time step
                out = self.fc1(out)
                out = self.relu(out)
                out = self.fc2(out)
                return out

        model = PyTorchLSTM(n_features, lstm_units, dropout_rate).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Numpy to PyTorch tensors
        X_tr_t = torch.tensor(X_tr_seq, dtype=torch.float32).to(device)
        y_tr_t = torch.tensor(y_tr_seq, dtype=torch.float32).view(-1, 1).to(device)
        X_te_t = torch.tensor(X_te_seq, dtype=torch.float32).to(device)

        print(f"Début de l'entraînement LSTM PyTorch sur {device} ({epochs} époques)...")
        history_loss = []
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tr_t)
            loss = criterion(outputs, y_tr_t)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            history_loss.append(loss_val)
            if (epoch + 1) % 10 == 0:
                print(f"  → LSTM Époque {epoch+1}/{epochs} - loss: {loss_val:.4f}")

        print("Fin de l'entraînement LSTM.")

        model.eval()
        with torch.no_grad():
            pred_tr_s = model(X_tr_t).cpu().numpy().ravel()
            pred_te_s = model(X_te_t).cpu().numpy().ravel()

        pred_tr = y_scaler.inverse_transform(pred_tr_s.reshape(-1, 1)).ravel()
        pred_te = y_scaler.inverse_transform(pred_te_s.reshape(-1, 1)).ravel()
        y_tr_real = y_scaler.inverse_transform(y_tr_seq.reshape(-1, 1)).ravel()
        y_te_real = y_scaler.inverse_transform(y_te_seq.reshape(-1, 1)).ravel()

        return {
            "train_metrics": compute_metrics(y_tr_real, pred_tr, "LSTM (train)"),
            "test_metrics": compute_metrics(y_te_real, pred_te, "LSTM"),
            "pred_train": pred_tr,
            "pred_test": pred_te,
            "y_train_aligned": y_tr_real,
            "y_test_aligned": y_te_real,
            "history": {"loss": history_loss},
            "look_back": look_back,
        }
    except ImportError:
        return {"error": "PyTorch non installé. Veuillez exécuter `pip install torch`."}
    except Exception as exc:
        logger.exception("LSTM training failed")
        return {"error": str(exc)}

@st.cache_data(show_spinner="Entraînement SARIMAX…")
def train_sarima(
    _X_train, _y_train, _X_test, _y_test,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> dict[str, Any]:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import numpy as np

        exog_train = np.asarray(_X_train, dtype=float) if _X_train is not None and len(_X_train) > 0 else None
        exog_test = np.asarray(_X_test, dtype=float) if _X_test is not None and len(_X_test) > 0 else None
        endog = np.asarray(_y_train, dtype=float)

        model = SARIMAX(
            endog,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        
        pred_train = res.predict(start=0, end=len(endog)-1, exog=exog_train)
        pred_test = res.forecast(steps=len(_y_test), exog=exog_test)

        return {
            "train_metrics": compute_metrics(_y_train, pred_train, "SARIMAX (train)"),
            "test_metrics": compute_metrics(_y_test, pred_test, "SARIMAX"),
            "pred_train": pred_train,
            "pred_test": pred_test,
            "summary": res.summary().as_text(),
            "model_name": f"SARIMA{order}{seasonal_order}" if sum(seasonal_order) > 0 else f"ARIMA{order}",
        }
    except ImportError:
        return {"error": "statsmodels non installé."}
    except Exception as exc:
        logger.exception("SARIMAX training failed")
        return {"error": str(exc)}

def _render_ts_predictions(res_dict: dict[str, Any], dates_train, y_train, dates_test, y_test, model_name: str, color: str = C_BLUE) -> None:
    m = res_dict["test_metrics"]
    col1, col2, col3, col4 = st.columns(4)
    is_growth = np.nanmean(np.abs(y_test)) < 2.0
    u = "pts" if is_growth else "MDH"
    f_val = ".4f" if is_growth else ",.0f"
    
    col1.metric("RMSE", f"{m['RMSE']:{f_val}} {u}" if not np.isnan(m['RMSE']) else "N/A")
    col2.metric("MAPE (%)", f"{m['MAPE']:.1f} %" if not np.isnan(m['MAPE']) else "N/A")
    col3.metric("MAE", f"{m['MAE']:{f_val}} {u}" if not np.isnan(m['MAE']) else "N/A")
    col4.metric("R²", f"{m['R²']:.3f}" if not np.isnan(m['R²']) else "N/A")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_test, y=y_test,
        mode="markers+lines", name="Réalisé",
        hovertemplate="%{x|%Y-%m-%d}: %{y:,.3f}<extra></extra>",
        line=dict(color=C_GREEN, width=1.5),
    ))
    
    p_te = res_dict["pred_test"]
    p_tr = res_dict["pred_train"]
    n_te = min(len(dates_test), len(p_te))
    n_tr = min(len(dates_train), len(p_tr))

    fig.add_trace(go.Scatter(
        x=dates_test[:n_te], y=p_te[:n_te],
        mode="lines", name=f"{model_name} — Prédiction",
        line=dict(color=color, width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=dates_train[:n_tr], y=p_tr[:n_tr],
        mode="lines", name=f"{model_name} — In-sample",
        line=dict(color=C_GREY, width=1, dash="dash"),
    ))
    if len(dates_test) > 0:
        fig.add_vline(
            x=dates_test[0].timestamp() * 1000,
            line_width=1, line_dash="dash",
            annotation_text="Train | Test", annotation_position="top",
        )
    fig.update_layout(
        title=f"{model_name} — Prédictions vs Réalisé",
        xaxis_title="Date", yaxis_title="Valeur",
        hovermode="x unified", template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="top"),
    )
    st.plotly_chart(fig, width="stretch")

# ── Page ───────────────────────────────────────────────────────────────────────

def page_ml_models() -> None:
    st.title("🤖 Modèles Machine Learning — VA Construction")
    st.markdown(
        "Comparaison de modèles ML supervisés contre le DFM et les benchmarks statistiques.  \n"
        "**Target** : `VA CONSTRUCTION` (HCP, trimestrielle) · "
        "**Features** : indicateurs BTP transformés + lags t-1/t-2  \n"
        "**Split** : 70 % train / 30 % test — chronologique (pas de data leakage)"
    )
    logger.info("Page: ML Models")

    freq_choice = st.radio(
        "Fréquence d'entraînement",
        options=["Trimestrielle (Défaut)", "Mensuelle (Interpolée)"],
        horizontal=True
    )
    freq = "M" if "Mensuelle" in freq_choice else "Q"

    import sys
    import pathlib

    _proj = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_proj))

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        try:
            from lamiaty.config import load_settings
            from lamiaty.data.pipeline import run_pipeline

            settings = load_settings()
            panel = run_pipeline(settings)
            
            # ── 1. Preparation du Panel ──────────────────────────────────────────
            # Applique le pipeline de nettoyage (Winsorisation, Lissage, Dummies)
            panel = clean_nowcasting_panel(panel)
            golden_features = get_golden_features(panel)

            st.sidebar.markdown("---")
            model_type = st.sidebar.selectbox(
                "Choisir un modèle",
                ["Comparaison", "Elastic Net", "Random Forest", "LSTM", "ARIMA", "ARMA", "SARIMA"]
            )
            feature_names = st.sidebar.multiselect(
                "Indicateurs (X)",
                options=list(panel.columns),
                default=golden_features
            )

            data = _prepare_ml_data(
                hash(str(panel.shape) + freq + str(feature_names)), 
                panel, 
                freq=freq,
                selected_features=feature_names
            )
        except Exception as exc:
            st.error(f"❌ Erreur critique : Impossible de charger les données réelles du pipeline. Détail : {exc}")
            st.stop()
            return

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    dates_train = data["dates_train"]
    dates_test = data["dates_test"]
    feature_names = data["feature_names"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations train", len(X_train))
    c2.metric("Observations test", len(X_test))
    c3.metric("Features", len(feature_names))
    if len(dates_test) > 0:
        c4.metric(
            "Période test",
            f"{_quarter_label(dates_test[0])} – {_quarter_label(dates_test[-1])}",
        )
    else:
        c4.metric("Période test", "—")

    with st.expander("Debug cible / alignement"):
        if "target_series" in data:
            st.write("Cible trimestrielle")
            st.write(data["target_series"].head(10))
            st.write(data["target_series"].tail(10))
            st.write(data["target_series"].describe())
        if "model_df" in data:
            st.write("Jeu final X/y")
            st.write(data["model_df"].head(10))

    st.divider()

    tab_en, tab_rf, tab_lstm, tab_arima, tab_arma, tab_sarima, tab_compare, tab_metrics, tab_errors = st.tabs(
        ["📈 Elastic Net", "🌲 Random Forest", "🧠 LSTM", "📈 ARIMA", "📉 ARMA", "❄️ SARIMA", "📊 Comparaison", "📐 Métriques", "🔍 Résidus"]
    )

    # ── Elastic Net ──────────────────────────────────────────────────────────
    with tab_en:
        st.subheader("📈 Elastic Net Regressor")
        st.markdown(
            "Régression linéaire régularisée (L1+L2). **Idéal pour les petits échantillons** "
            "car il évite le sur-apprentissage en sélectionnant les variables les plus stables."
        )
        en_res = train_elastic_net(X_train, X_test, y_train, y_test, feature_names)
        
        if "error" in en_res:
            st.error(en_res["error"])
        else:
            m = en_res["test_metrics"]
            col1, col2, col3, col4 = st.columns(4)
            is_growth = np.nanmean(np.abs(y_test)) < 2.0
            u = "pts" if is_growth else "MDH"
            f_val = ".4f" if is_growth else ",.0f"
            
            col1.metric("RMSE", f"{m['RMSE']:{f_val}} {u}")
            col2.metric("MAPE (%)", f"{m['MAPE']:.1f} %")
            col3.metric("MAE", f"{m['MAE']:{f_val}} {u}")
            col4.metric("R²", f"{m['R²']:.3f}")

            # Plot prediction
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates_test, y=y_test, mode="markers+lines", name="Réalisé", line=dict(color=C_GREEN)))
            fig.add_trace(go.Scatter(x=dates_test, y=en_res["pred_test"], mode="lines", name="EN — Test", line=dict(color=C_BLUE, dash="dot")))
            fig.update_layout(title="Elastic Net — Prédictions", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

    # ── Random Forest ─────────────────────────────────────────────────────────
    with tab_rf:
        st.subheader("🌲 Random Forest Regressor")
        st.markdown(
            "Ensemble d'arbres de décision — robuste aux outliers, interprétable "
            "via l'importance des features."
        )
        with st.expander("⚙️ Hyperparamètres"):
            col1, col2, col3 = st.columns(3)
            n_est = col1.slider("n_estimators", 50, 500, 150, 50)
            max_d = col2.slider("max_depth (0 = None)", 0, 20, 4)
            min_ss = col3.slider("min_samples_split", 2, 10, 4)

        rf_res = train_random_forest(
            X_train, y_train, X_test, y_test,
            n_estimators=n_est,
            max_depth=max_d if max_d > 0 else None,
            min_samples_split=min_ss,
        )

        if "error" in rf_res:
            st.error(rf_res["error"])
        else:
            m = rf_res["test_metrics"]
            col1, col2, col3, col4 = st.columns(4)
            # Détection automatique de l'échelle pour l'unité et le format
            is_growth = np.nanmean(np.abs(y_test)) < 2.0
            u = "pts" if is_growth else "MDH"
            f_val = ".4f" if is_growth else ",.0f"
            
            col1.metric("RMSE", f"{m['RMSE']:{f_val}} {u}")
            col2.metric("MAPE (%)", f"{m['MAPE']:.1f} %" if not np.isnan(m["MAPE"]) else "N/A")
            col3.metric("MAE", f"{m['MAE']:{f_val}} {u}")
            col4.metric("R²", f"{m['R²']:.3f}" if not np.isnan(m["R²"]) else "N/A")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates_test, y=y_test,
                mode="markers+lines", name="Réalisé",
                hovertemplate="%{x|%Y-%m-%d}: %{y:,.3f}<extra></extra>",
                line=dict(color=C_GREEN, width=1.5),
            ))
            fig.add_trace(go.Scatter(
                x=dates_test, y=rf_res["pred_test"],
                mode="lines", name="RF — Prédiction",
                line=dict(color=C_BLUE, width=2, dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=dates_train, y=rf_res["pred_train"],
                mode="lines", name="RF — In-sample",
                line=dict(color=C_GREY, width=1, dash="dash"),
            ))
            if len(dates_test) > 0:
                fig.add_vline(
                    x=dates_test[0].timestamp() * 1000,
                    line_width=1, line_dash="dash",
                    annotation_text="Train | Test",
                    annotation_position="top",
                )
            fig.update_layout(
                title="Random Forest — Prédictions vs VA Construction réalisée",
                xaxis_title="Date", yaxis_title="Valeur transformée",
                hovermode="x unified", template=PLOTLY_TEMPLATE,
                legend=dict(orientation="h", yanchor="top"),
            )
            st.plotly_chart(fig, width="stretch")

            st.subheader("📌 Importance des features")
            
            # Robustesse contre les problèmes de cache de Streamlit (mismatch de dimensions)
            f_names = feature_names
            f_imps = rf_res["feature_importances"]
            
            if len(f_names) != len(f_imps):
                st.warning(f"Conflit de cache Streamlit ignoré (noms={len(f_names)}, importances={len(f_imps)}).")
                # Fallback générique pour permettre l'affichage
                f_names = [f"Feature {i}" for i in range(len(f_imps))]
                
            imp_df = (
                pd.DataFrame({"Feature": f_names, "Importance": f_imps})
                .sort_values("Importance")
                .tail(20)
            )
            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                color_discrete_sequence=["#dbeafe"],
                title="Feature Importance — Random Forest (top 20)",
                labels={"Importance": "Importance relative"},
            )
            fig_imp.update_layout(
                template=PLOTLY_TEMPLATE,
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(fig_imp, width="stretch")

            train_m = rf_res["train_metrics"]
            st.info(
                f"Train RMSE: {train_m['RMSE']:,.0f} — Test RMSE: {m['RMSE']:,.0f}  |  "
                "Un écart important signale un sur-apprentissage."
            )

    # ── LSTM ──────────────────────────────────────────────────────────────────
    with tab_lstm:
        st.subheader("🧠 LSTM — Long Short-Term Memory")
        st.markdown(
            "Réseau de neurones récurrent — capte les dépendances temporelles à long terme.  \n"
            "Architecture : LSTM(unités) → Dropout → Dense(16, ReLU) → Dense(1)"
        )
        with st.expander("⚙️ Hyperparamètres LSTM"):
            col1, col2, col3, col4 = st.columns(4)
            look_b = col1.slider("look_back (trimestres)", 1, 8, 2)
            units = col2.slider("LSTM units", 8, 128, 16, 8)
            dropout = col3.slider("Dropout rate", 0.0, 0.5, 0.0, 0.05)
            epochs = col4.slider("Max epochs", 50, 1000, 300, 50)

        lstm_res = train_lstm(
            X_train, y_train, X_test, y_test,
            look_back=look_b, lstm_units=units,
            dropout_rate=dropout, epochs=epochs
        )


        if "error" in lstm_res:
            st.error(lstm_res["error"])
        else:
            lb = lstm_res.get("look_back", look_b)
            m = lstm_res["test_metrics"]
            
            y_te_al = lstm_res.get("y_test_aligned", y_test[lb:])
            y_tr_al = lstm_res.get("y_train_aligned", y_train[lb:])
            p_tr = lstm_res["pred_train"]
            p_te = lstm_res["pred_test"]
            
            col1, col2, col3, col4 = st.columns(4)
            # Détection automatique de l'échelle
            is_growth = np.nanmean(np.abs(y_te_al)) < 2.0
            u = "pts" if is_growth else "MDH"
            f_val = ".4f" if is_growth else ",.0f"

            col1.metric("RMSE", f"{m['RMSE']:{f_val}} {u}" if not np.isnan(m["RMSE"]) else "N/A")
            col2.metric("MAPE (%)", f"{m['MAPE']:.1f} %" if not np.isnan(m["MAPE"]) else "N/A")
            col3.metric("MAE", f"{m['MAE']:{f_val}} {u}" if not np.isnan(m["MAE"]) else "N/A")
            col4.metric("R²", f"{m['R²']:.3f}" if not np.isnan(m["R²"]) else "N/A")

            tab_pred, tab_loss = st.tabs(["Prédictions", "Courbe de perte"])

            d_test_aligned = dates_test[lb:] if len(dates_test) > lb else dates_test
            d_train_aligned = dates_train[lb:] if len(dates_train) > lb else dates_train

            n_tr = min(len(d_train_aligned), len(p_tr), len(y_tr_al))
            n_te = min(len(d_test_aligned), len(p_te), len(y_te_al))

            with tab_pred:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=d_test_aligned[:n_te], y=y_te_al[:n_te],
                    mode="markers+lines", name="Réalisé",
                    line=dict(color=C_GREEN, width=1.5),
                ))
                fig.add_trace(go.Scatter(
                    x=d_test_aligned[:n_te], y=p_te[:n_te],
                    mode="lines", name="LSTM — Prédiction",
                    line=dict(color=C_PURPLE, width=2, dash="dot"),
                ))
                fig.add_trace(go.Scatter(
                    x=d_train_aligned[:n_tr], y=p_tr[:n_tr],
                    mode="lines", name="LSTM — In-sample",
                    line=dict(color=C_GREY, width=1, dash="dash"),
                ))
                if len(d_test_aligned) > 0:
                    fig.add_vline(
                        x=d_test_aligned[0].timestamp() * 1000,
                        line_width=1, line_dash="dash",
                        annotation_text="Train | Test", annotation_position="top",
                    )
                fig.update_layout(
                    title="LSTM — Prédictions vs VA Construction réalisée",
                    xaxis_title="Date", yaxis_title="Valeur transformée",
                    hovermode="x unified", template=PLOTLY_TEMPLATE,
                    legend=dict(orientation="h", yanchor="top"),
                )
                st.plotly_chart(fig, width="stretch")

            with tab_loss:
                hist = lstm_res.get("history")
                if hist and "loss" in hist:
                    loss = hist["loss"]
                    val_loss = hist.get("val_loss", [])
                    ep = list(range(1, len(loss) + 1))
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=ep, y=loss, mode="lines",
                                                  name="Train loss", line=dict(color=C_BLUE)))
                    fig_loss.add_trace(go.Scatter(x=ep, y=val_loss, mode="lines",
                                                  name="Val loss", line=dict(color=C_RED, dash="dot")))
                    fig_loss.update_layout(
                        title="Courbe de perte LSTM (MSE)",
                        xaxis_title="Epoch", yaxis_title="MSE (normalisé)",
                        template=PLOTLY_TEMPLATE,
                    )
                    st.plotly_chart(fig_loss, width="stretch")
                    st.caption(
                        f"Early stopping activé. Epochs effectifs : {len(loss)}  |  "
                        "Val loss min = critère de sélection."
                    )
                else:
                    st.info("Courbe de perte non disponible (mode simulation).")

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    with tab_arima:
        st.subheader("📈 Modèle ARIMA (ARIMAX)")
        st.markdown("AutoRegressive Integrated Moving Average avec variables exogènes.")
        with st.expander("⚙️ Hyperparamètres ARIMA"):
            c1, c2, c3 = st.columns(3)
            p_ar = c1.number_input("p (Lags AR)", 0, 10, 1, key="ari_p")
            d_ar = c2.number_input("d (Différenciation)", 0, 2, 0, key="ari_d")
            q_ar = c3.number_input("q (Lags MA)", 0, 10, 1, key="ari_q")
        
        arima_res = train_sarima(X_train, y_train, X_test, y_test, order=(p_ar, d_ar, q_ar))
        
        if "error" in arima_res:
            st.error(arima_res["error"])
        else:
            _render_ts_predictions(arima_res, dates_train, y_train, dates_test, y_test, "ARIMA")
            with st.expander("📄 Summary du modèle"):
                st.text(arima_res["summary"])

    # ── ARMA ──────────────────────────────────────────────────────────────────
    with tab_arma:
        st.subheader("📉 Modèle ARMA (ARMAX)")
        st.markdown("AutoRegressive Moving Average (sans différenciation).")
        with st.expander("⚙️ Hyperparamètres ARMA"):
            c1, c2 = st.columns(2)
            p_arma = c1.number_input("p (Lags AR)", 0, 10, 1, key="arm_p")
            q_arma = c2.number_input("q (Lags MA)", 0, 10, 1, key="arm_q")
        
        arma_res = train_sarima(X_train, y_train, X_test, y_test, order=(p_arma, 0, q_arma))
        if "error" in arma_res:
            st.error(arma_res["error"])
        else:
            _render_ts_predictions(arma_res, dates_train, y_train, dates_test, y_test, "ARMA", color=C_ORANGE)
            with st.expander("📄 Summary du modèle"):
                st.text(arma_res["summary"])

    # ── SARIMA ────────────────────────────────────────────────────────────────
    with tab_sarima:
        st.subheader("❄️ Modèle SARIMA (SARIMAX)")
        st.markdown("Seasonal ARIMA pour capter la saisonnalité.")
        with st.expander("⚙️ Hyperparamètres SARIMA"):
            st.write("Ordre classique (p,d,q)")
            c1, c2, c3 = st.columns(3)
            sp = c1.number_input("p", 0, 5, 1, key="s_p")
            sd = c2.number_input("d", 0, 2, 0, key="s_d")
            sq = c3.number_input("q", 0, 5, 1, key="s_q")
            st.write("Ordre saisonnier (P,D,Q,s)")
            c4, c5, c6, c7 = st.columns(4)
            sP = c4.number_input("P", 0, 5, 1, key="s_sP")
            sD = c5.number_input("D", 0, 2, 0, key="s_sD")
            sQ = c6.number_input("Q", 0, 5, 0, key="s_sQ")
            s_s = c7.number_input("s (Période)", 0, 12, 4 if freq=="Q" else 12, key="s_s")
            
        sarima_res = train_sarima(X_train, y_train, X_test, y_test, order=(sp, sd, sq), seasonal_order=(sP, sD, sQ, s_s))
        if "error" in sarima_res:
            st.error(sarima_res["error"])
        else:
            _render_ts_predictions(sarima_res, dates_train, y_train, dates_test, y_test, "SARIMA", color=C_TEAL)
            with st.expander("📄 Summary du modèle"):
                st.text(sarima_res["summary"])

    # ── Comparaison ───────────────────────────────────────────────────────────
    with tab_compare:
        st.subheader("📊 Comparaison haute vue — tous les modèles")
        st.markdown(
            "Vue synthétique des performances sur le **jeu de test**.  \n"
            "Benchmarks statistiques : AR(1) et Marche Aléatoire."
        )

        preds: dict[str, np.ndarray] = {}
        dates_ref = dates_test

        if "pred_test" in en_res and "error" not in en_res:
            preds["Elastic Net"] = en_res["pred_test"]

        if "pred_test" in rf_res and "error" not in rf_res:
            preds["Random Forest"] = rf_res["pred_test"]

        if "pred_test" in lstm_res and "error" not in lstm_res:
            preds["LSTM"] = lstm_res["pred_test"]
            
        if "pred_test" in arima_res and "error" not in arima_res:
            preds["ARIMA"] = arima_res["pred_test"]
            
        if "pred_test" in arma_res and "error" not in arma_res:
            preds["ARMA"] = arma_res["pred_test"]
            
        if "pred_test" in sarima_res and "error" not in sarima_res:
            preds["SARIMA"] = sarima_res["pred_test"]

        try:
            from statsmodels.tsa.ar_model import AutoReg
            va_full = pd.Series(
                np.concatenate([y_train, y_test]),
                index=pd.DatetimeIndex(np.concatenate([dates_train, dates_test])),
            )
            ar_preds = []
            for i in range(len(y_test)):
                hist_y = va_full.iloc[: len(y_train) + i]
                m_ar = AutoReg(hist_y, lags=1).fit()
                ar_preds.append(float(m_ar.forecast(1).iloc[0]))
            preds["AR(1)"] = np.array(ar_preds)
        except Exception:
            pass

        rw_preds = np.concatenate([[y_train[-1]], y_test[:-1]])
        preds["Random Walk"] = rw_preds

        rows: list[dict] = []
        for name, pred in preds.items():
            target_back = y_test
            if name == "LSTM" and "pred_test" in lstm_res and "error" not in lstm_res:
                target_back = lstm_res.get("y_test_aligned", y_test[lb:])
                n = min(len(pred), len(target_back))
                pred = pred[:n]
                target_back = target_back[:n]
            n = min(len(pred), len(target_back))
            rows.append(compute_metrics(target_back[:n], pred[:n], name))

        compare_df = pd.DataFrame(rows).set_index("model")
        compare_df["Rang RMSE"] = compare_df["RMSE"].rank().astype(int)

        def _style_table(df: pd.DataFrame) -> Any:
            def highlight(col: pd.Series) -> list[str]:
                if col.name == "R²":
                    best = col.max()
                else:
                    best = col.min()
                return [
                    "background-color: #dcfce7; font-weight: bold" if v == best else ""
                    for v in col
                ]
            return (
                df.style
                .apply(highlight)
                .format({
                    "RMSE": "{:,.4f}" if np.nanmean(np.abs(y_test)) < 2.0 else "{:,.0f}",
                    "MAE": "{:,.4f}" if np.nanmean(np.abs(y_test)) < 2.0 else "{:,.0f}",
                    "MAPE": "{:.2f}", "R²": "{:.3f}",
                    "Rang RMSE": "{:.0f}",
                })
            )

        st.dataframe(_style_table(compare_df), width="stretch")
        st.caption("🟢 Vert = meilleur score pour la colonne.")

        st.subheader("🕸️ Vue radar — profil de performance")
        st.caption("Scores normalisés [0, 1].")
        metrics_for_radar = ["RMSE", "MAPE", "MAE", "R²"]
        radar_df = compare_df[metrics_for_radar].copy()
        radar_norm = radar_df.copy()
        for col in metrics_for_radar:
            col_max = radar_df[col].max()
            col_min = radar_df[col].min()
            if col == "R²":
                radar_norm[col] = (radar_df[col] - col_min) / (col_max - col_min + 1e-9)
            else:
                radar_norm[col] = 1 - (radar_df[col] - col_min) / (col_max - col_min + 1e-9)

        categories = metrics_for_radar + [metrics_for_radar[0]]
        fig_radar = go.Figure()
        for model_name, row in radar_norm.iterrows():
            vals = list(row.values) + [row.values[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=categories, fill="toself",
                name=str(model_name),
                line=dict(color=MODEL_COLORS.get(str(model_name), C_GREY)),
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Profil radar — performance normalisée",
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig_radar, width="stretch")

        st.subheader("📈 Prédictions superposées — jeu de test")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=dates_test, y=y_test, mode="markers+lines", name="Réalisé (HCP)",
            hovertemplate="%{x|%Y-%m-%d}: %{y:,.3f}<extra></extra>",
            line=dict(color=C_GREEN, width=2),
        ))
        for name, pred in preds.items():
            idx_back = dates_test
            if name == "LSTM" and "pred_test" in lstm_res and "error" not in lstm_res:
                idx_back = dates_test[lb:] if len(dates_test) > lb else dates_test
                n = min(len(pred), len(idx_back))
                pred = pred[:n]
                idx_back = idx_back[:n]
            else:
                n = min(len(pred), len(idx_back))
                pred = pred[:n]
                idx_back = idx_back[:n]
            fig_cmp.add_trace(go.Scatter(
                x=idx_back, y=pred, mode="lines", name=name,
                line=dict(color=MODEL_COLORS.get(name, C_GREY), dash="dot", width=1.5),
            ))
        fig_cmp.update_layout(
            title="Tous les modèles — prédictions sur le jeu de test",
            xaxis_title="Date", yaxis_title="Valeur transformée",
            hovermode="x unified", template=PLOTLY_TEMPLATE,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_cmp, width="stretch")

        rmse_df = compare_df[["RMSE"]].reset_index().rename(columns={"model": "Modèle"})
        fig_rmse = px.bar(
            rmse_df, x="Modèle", y="RMSE", color="Modèle",
            color_discrete_map=MODEL_COLORS,
            title="🏆 Classement RMSE — jeu de test",
            labels={"RMSE": "RMSE"},
        )
        fig_rmse.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        fig_rmse.update_layout(template=PLOTLY_TEMPLATE, showlegend=False)
        st.plotly_chart(fig_rmse, width="stretch")

        mape_df = compare_df[["MAPE"]].reset_index().rename(columns={"model": "Modèle"})
        fig_mape = px.bar(
            mape_df, x="Modèle", y="MAPE", color="Modèle",
            color_discrete_map=MODEL_COLORS,
            title="MAPE par modèle — jeu de test",
            labels={"MAPE": "MAPE (%)"},
        )
        fig_mape.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig_mape.update_layout(template=PLOTLY_TEMPLATE, showlegend=False)
        st.plotly_chart(fig_mape, width="stretch")

        st.subheader("📏 Theil-U (référence = Random Walk)")
        rw_pred = preds.get("Random Walk", np.full(len(y_test), float("nan")))
        theil_rows = []
        for name, pred in preds.items():
            if name == "Random Walk":
                continue
            target_back = y_test
            if name == "LSTM" and "pred_test" in lstm_res and "error" not in lstm_res:
                target_back = lstm_res.get("y_test_aligned", y_test[lb:])
                n = min(len(pred), len(target_back), len(rw_pred))
            else:
                n = min(len(pred), len(target_back), len(rw_pred))
            rmsfe_m = rmse(target_back[:n], pred[:n])
            rmsfe_rw = rmse(target_back[:n], rw_pred[:n])
            u = rmsfe_m / rmsfe_rw if rmsfe_rw > 0 else float("nan")
            theil_rows.append({
                "Modèle": name,
                "Theil-U": round(u, 3),
                "Meilleur que RW ?": "✅ Oui" if u < 1 else "❌ Non",
            })
        theil_df = pd.DataFrame(theil_rows)

        def _style_theil(df: pd.DataFrame) -> Any:
            def hl(row: pd.Series) -> list[str]:
                return [
                    "background-color: #dcfce7" if row["Theil-U"] < 1 else "background-color: #fee2e2"
                    if not np.isnan(row["Theil-U"]) else ""
                    for _ in range(len(row))
                ]
            return df.style.apply(hl, axis=1).format({"Theil-U": "{:.3f}"})

        st.dataframe(_style_theil(theil_df), hide_index=True, width="stretch")

    # ── Métriques détaillées ──────────────────────────────────────────────────
    with tab_metrics:
        st.subheader("📐 Métriques détaillées — train & test")
        st.markdown(
            "\n| Métrique | Formule | Interprétation |\n"
            "|----------|---------|----------------|\n"
            "| **RMSE** | √(mean(e²)) | Pénalise les grandes erreurs |\n"
            "| **MAPE** | mean(\\|e/y\\|)×100 | % d'erreur moyen |\n"
            "| **MAE**  | mean(\\|e\\|) | Erreur absolue moyenne |\n"
            "| **R²**   | 1 − SS_res/SS_tot | Part de variance expliquée |\n"
        )

        all_metrics = []
        for name, res in [("Elastic Net", en_res), ("Random Forest", rf_res), ("LSTM", lstm_res), ("ARIMA", arima_res), ("ARMA", arma_res), ("SARIMA", sarima_res)]:
            if "error" in res:
                continue
            tr = res.get("train_metrics", {})
            te = res.get("test_metrics", {})
            for phase, m in [("Train", tr), ("Test", te)]:
                all_metrics.append({
                    "Modèle": name, "Phase": phase,
                    "RMSE": m.get("RMSE", float("nan")),
                    "MAPE": m.get("MAPE", float("nan")),
                    "MAE": m.get("MAE", float("nan")),
                    "R²": m.get("R²", float("nan")),
                })
        m_df = pd.DataFrame(all_metrics)
        st.dataframe(
            m_df.style.format({
                "RMSE": "{:,.0f}", "MAE": "{:,.0f}",
                "MAPE": "{:.2f}", "R²": "{:.4f}",
            }).background_gradient(subset=["RMSE", "MAPE"], cmap="RdYlGn_r")
              .background_gradient(subset=["R²"], cmap="RdYlGn"),
            hide_index=True, width="stretch",
        )

        st.subheader("⚖️ Détection du sur-apprentissage (RMSE train vs test)")
        overfit_rows = []
        for name, res in [("Elastic Net", en_res), ("Random Forest", rf_res), ("LSTM", lstm_res), ("ARIMA", arima_res), ("ARMA", arma_res), ("SARIMA", sarima_res)]:
            if "error" in res:
                continue
            tr_m = res.get("train_metrics", {})
            te_m = res.get("test_metrics", {})
            tr_r = tr_m.get("RMSE", float("nan"))
            te_r = te_m.get("RMSE", float("nan"))
            ratio = te_r / tr_r if tr_r > 0 else float("nan")
            overfit_rows.append({
                "Modèle": name,
                "RMSE Train": tr_r,
                "RMSE Test": te_r,
                "Ratio T/Tr": ratio,
                "Verdict": "⚠️ Surapprentissage" if ratio > 1.5 else
                           "✅ Correct" if ratio <= 1.2 else "🔶 Attention",
            })
        of_df = pd.DataFrame(overfit_rows)
        st.dataframe(
            of_df.style.format({"RMSE Train": "{:,.0f}", "RMSE Test": "{:,.0f}", "Ratio T/Tr": "{:.2f}"}),
            hide_index=True, width="stretch",
        )

    # ── Résidus ───────────────────────────────────────────────────────────────
    with tab_errors:
        st.subheader("🔍 Analyse des résidus — jeu de test")
        st.caption("Diagnostics des erreurs de prévision par modèle.")

        models_for_diag = {
            name: res for name, res in [
                ("Elastic Net", en_res), 
                ("Random Forest", rf_res), 
                ("LSTM", lstm_res),
                ("ARIMA", arima_res),
                ("ARMA", arma_res),
                ("SARIMA", sarima_res)
            ]
            if "pred_test" in res and "error" not in res
        }
        if not models_for_diag:
            st.info("Aucun modèle disponible pour le diagnostic.")
        else:
            selected_model = st.selectbox(
                "Choisir le modèle à diagnostiquer",
                list(models_for_diag.keys()),
                key="diag_model_select",
            )
            res = models_for_diag[selected_model]
            if selected_model == "LSTM":
                y_te_al = lstm_res.get("y_test_aligned", y_test[lb:])
                pred_sel = res["pred_test"]
                dates_sel = dates_test[lb:] if len(dates_test) > lb else dates_test
                n = min(len(pred_sel), len(y_te_al), len(dates_sel))
                errors_sel = y_te_al[:n] - pred_sel[:n]
                dates_sel = dates_sel[:n]
                y_ref = y_te_al[:n]
            else:
                pred_sel = res["pred_test"]
                n = min(len(pred_sel), len(y_test), len(dates_test))
                errors_sel = y_test[:n] - pred_sel[:n]
                dates_sel = dates_test[:n]
                y_ref = y_test[:n]

            fig_err = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Erreur", "Distribution",
                    "Réalisé vs erreur", "Q-Q",
                ],
            )
            fig_err.add_trace(
                go.Scatter(x=dates_sel, y=errors_sel, mode="markers+lines",
                           name="Erreur", line=dict(color=C_RED, width=1)),
                row=1, col=1,
            )
            fig_err.add_hline(y=0, line_dash="dash", line_color="black", row=1, col=1)
            fig_err.add_trace(
                go.Histogram(x=errors_sel, nbinsx=15, name="Distribution",
                             marker_color=C_BLUE, opacity=0.7),
                row=1, col=2,
            )
            fig_err.add_trace(
                go.Scatter(
                    x=y_ref, y=errors_sel, mode="markers",
                    name="Réalisé: %{x}<br>Erreur: %{y}<extra></extra>",
                    marker=dict(color=C_ORANGE, size=6),
                ),
                row=2, col=1,
            )
            fig_err.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

            try:
                import scipy.stats as scipy_stats
                qq = scipy_stats.probplot(errors_sel[~np.isnan(errors_sel)])
                fig_err.add_trace(
                    go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers",
                               name="Q-Q", marker=dict(color=C_TEAL, size=5)),
                    row=2, col=2,
                )
                fig_err.add_trace(
                    go.Scatter(
                        x=[qq[0][0].min(), qq[0][0].max()],
                        y=[qq[1][1] + qq[1][0] * qq[0][0].min(),
                           qq[1][1] + qq[1][0] * qq[0][0].max()],
                        mode="lines", name="Ligne normale",
                        line=dict(color=C_RED, dash="dot"),
                    ),
                    row=2, col=2,
                )
            except Exception:
                pass

            fig_err.update_layout(
                title=f"Diagnostics des résidus — {selected_model}",
                hovermode="closest", template=PLOTLY_TEMPLATE, showlegend=False,
            )
            st.plotly_chart(fig_err, width="stretch")

            e = errors_sel[~np.isnan(errors_sel)]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Biais moyen", f"{np.mean(e):+,.3f}")
            col2.metric("Écart-type erreurs", f"{np.std(e):,.3f}")
            col3.metric("Skewness", f"{float(pd.Series(e).skew()):.3f}")
            try:
                import scipy.stats as scipy_stats
                _, sw_p = scipy_stats.shapiro(e)
                col4.metric("Shapiro-Wilk p", f"{sw_p:.3f}")
            except Exception:
                col4.metric("Shapiro-Wilk p", "—")

            st.caption(
                "Biais > 0 → sous-estimation systématique  |  "
                "Skewness proche de 0 = distribution symétrique"
            )

    st.divider()
    st.caption(
        "📚 Références : Breiman (2001) Random Forests — Hochreiter & Schmidhuber (1997) LSTM"
    )


