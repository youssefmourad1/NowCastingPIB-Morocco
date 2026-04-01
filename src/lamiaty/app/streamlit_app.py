"""
Morocco BTP Nowcasting — Streamlit Dashboard
=============================================
Entry point: `streamlit run src/lamiaty/app/streamlit_app.py`

All charts are interactive (Plotly). Structured log stream shown in sidebar.

Pages:
  1. Accueil          — project overview and data summary
  2. Audit des données — §2 descriptive statistics and correlations
  3. Rupture ciment   — §3.1 cement unit break analysis
  4. Investissement   — §3.2 Investissement_Etat diagnosis
  5. Validation pipeline — model_panel.parquet + missing heatmap
  6. Stationnarité    — ADF/KPSS battery on the transformed panel
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Ensure package is importable regardless of working directory
_project_root = Path(__file__).resolve().parents[3]
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Logging — must be configured before any lamiaty imports ──────────────────
from lamiaty.utils.logging import setup_logging, get_streamlit_buffer

_LOG_DIR = _project_root / "logs"
setup_logging(
    level=20,  # INFO
    log_file=_LOG_DIR / "app.log",
    json_file=_LOG_DIR / "app.jsonl",
    enable_streamlit_buffer=True,
)

import logging
_app_log = logging.getLogger("lamiaty.app")
_app_log.info("Streamlit app started")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTP Nowcasting Maroc",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#2563eb"
C_RED    = "#dc2626"
C_GREEN  = "#16a34a"
C_ORANGE = "#ea580c"
C_GREY   = "#64748b"
PLOTLY_TEMPLATE = "plotly_white"

# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement de la configuration...")
def get_settings():
    from lamiaty.config import load_settings
    _app_log.info("Loading settings from configs/")
    return load_settings(config_dir=_project_root / "configs", project_root=_project_root)


@st.cache_data(show_spinner="Lecture de la base BTP brute...")
def get_raw_df():
    from lamiaty.data.loader import load_base_btp
    settings = get_settings()
    _app_log.info("Loading raw BTP DataFrame")
    return load_base_btp(settings.paths.base_btp_path)


@st.cache_data(show_spinner="Application des corrections...")
def get_corrected_df():
    from lamiaty.data.corrections import apply_all_corrections
    settings = get_settings()
    raw = get_raw_df()
    _app_log.info("Applying data corrections")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return apply_all_corrections(raw, settings.corrections)


@st.cache_data(show_spinner="Exécution du pipeline complet...")
def get_panel():
    from lamiaty.data.pipeline import run_pipeline
    settings = get_settings()
    _app_log.info("Running full pipeline")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return run_pipeline(settings)


# ── Sidebar ───────────────────────────────────────────────────────────────────

PAGES = {
    "🏠 Accueil": "home",
    "📊 Audit des données": "audit",
    "🔴 Rupture ciment (§3.1)": "cement",
    "🟠 Investissement État (§3.2)": "investissement",
    "✅ Validation pipeline": "pipeline",
    "📈 Stationnarité": "stationarity",
    "🎯 Estimation DFM": "dfm",
    "🔮 Nowcast": "nowcast",
    "📉 Backtesting": "backtest",
    "📰 News decomposition": "news",
}

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/2c/Flag_of_Morocco.svg", width=55)
    st.title("BTP Nowcasting")
    st.caption("VA Construction — Maroc | Phase 1")
    st.divider()
    page_label = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    page = PAGES[page_label]

    # ── Log viewer ────────────────────────────────────────────────────────
    st.divider()
    with st.expander("🪵 Pipeline logs", expanded=False):
        buf = get_streamlit_buffer()
        log_level = st.selectbox(
            "Niveau minimum",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1,
            key="log_level_select",
        )
        level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        lines = buf.get_lines(min_level=level_map[log_level])
        if lines:
            for line in lines[-30:]:
                st.markdown(line)
        else:
            st.caption("Aucun log pour ce niveau.")
        if st.button("🗑️ Vider les logs"):
            buf.clear()

    st.divider()
    st.caption(
        "Méthode : Dynamic Factor Model  \n"
        "Réf : Danov et al. (2026) IMF WP/26/32  \n"
        "Phase 1 ✅ · Phase 2 ✅ · Phase 3 ✅"
    )


def _warn(msg):
    st.warning(f"⚠️ {msg}", icon="⚠️")


# ── Page: Accueil ─────────────────────────────────────────────────────────────

def page_home():
    st.title("🏗️ Nowcasting de la VA BTP au Maroc")
    st.markdown(
        """
        **Objectif :** Estimer en temps réel la Valeur Ajoutée du secteur Construction
        à l'aide d'un **Dynamic Factor Model (DFM)** — Danov et al. (2026), IMF WP/26/32.

        **Variable cible :** `VA CONSTRUCTION` (HCP, Comptes Nationaux Trimestriels)
        """
    )

    settings = get_settings()
    raw = get_raw_df()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(raw):,}")
    c2.metric("Variables", len(raw.columns))
    c3.metric("Période", f"{raw.index[0].strftime('%b %Y')} – {raw.index[-1].strftime('%b %Y')}")
    c4.metric("Variable cible", "VA CONSTRUCTION")

    # Phase roadmap
    st.divider()
    st.subheader("Feuille de route")
    phases = pd.DataFrame({
        "Phase": ["1 — Infrastructure données", "2 — Estimation DFM",
                  "3 — Backtesting pseudo-temps réel",
                  "4 — Données alternatives", "5 — Production"],
        "Statut": ["✅ Complète", "⏳ Planifiée", "⏳ Planifiée", "⏳ Planifiée", "⏳ Planifiée"],
        "Description": [
            "Corrections, pipeline, panel mixte fréquence",
            "EM-Kalman, facteurs latents k=2",
            "Pseudo-vintages, news decomposition, RMSFE",
            "Paiements numériques, données satellite",
            "Pipeline automatisé, bulletin bimensuel",
        ],
    })
    st.dataframe(phases, hide_index=True, use_container_width=True)

    # ── Interactive overview chart of VA CONSTRUCTION ─────────────────────
    st.divider()
    st.subheader("VA CONSTRUCTION — série brute")
    va = raw["va_construction"].dropna()
    fig = px.line(
        x=va.index, y=va.values,
        labels={"x": "Date", "y": "MDH (prix courants)"},
        title="Valeur Ajoutée Construction (HCP) — données brutes trimestrielles",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(line_color=C_BLUE, line_width=2)
    fig.add_vrect(x0="2020-01-01", x1="2021-01-01",
                  fillcolor="red", opacity=0.08, line_width=0,
                  annotation_text="COVID", annotation_position="top left")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Status of corrections
    st.divider()
    st.subheader("Statut des corrections")
    cb = settings.corrections.cement_break
    inv = settings.corrections.investissement_etat
    corrections_df = pd.DataFrame({
        "Correction": ["Rupture ciment (×759)", "Investissement État (monthly_diff)", "LafargeHolcim (string → float)"],
        "Statut": [cb.status, "APPLIED", "APPLIED"],
        "Confirmé par": [
            cb.confirmed_by or "❌ Placeholder — APC requis",
            inv.confirmed_by or "❌ Non confirmé — TGR/MEF requis",
            "✅ Automatique",
        ],
    })
    st.dataframe(corrections_df, hide_index=True, use_container_width=True)
    _warn("Le facteur ciment (×759) est un PLACEHOLDER. Confirmer avec l'APC avant production.")


# ── Page: Audit ───────────────────────────────────────────────────────────────

def page_audit():
    st.title("📊 Audit de la base de données (§2)")
    _app_log.info("Page: Audit des données")
    raw = get_raw_df()

    tab_stats, tab_missing, tab_corr, tab_series = st.tabs(
        ["Statistiques", "Valeurs manquantes", "Corrélations", "Séries brutes"]
    )

    with tab_stats:
        st.subheader("Statistiques descriptives")
        st.dataframe(raw.describe().T.round(2), use_container_width=True)

    with tab_missing:
        st.subheader("Valeurs manquantes par série")
        miss = raw.isnull().sum().reset_index()
        miss.columns = ["Série", "n_missing"]
        miss["pct (%)"] = (miss["n_missing"] / len(raw) * 100).round(1)
        fig = px.bar(
            miss, x="n_missing", y="Série", orientation="h",
            text="n_missing",
            color="n_missing",
            color_continuous_scale=["#dbeafe", C_BLUE],
            title="Valeurs manquantes par série",
            labels={"n_missing": "Nombre de NaN", "Série": ""},
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(miss, hide_index=True, use_container_width=True)

    with tab_corr:
        st.subheader("Corrélations avec VA CONSTRUCTION (y-o-y)")
        from lamiaty.data.transforms import assign_quarterly_to_month_end, yoy_log_diff

        df_c = get_corrected_df().copy()
        monthly = ["consommation_ciment","credits_equipement","credits_immobilier","lafarge_index"]
        quarterly = ["va_construction","ipai","creation_emploi"]
        for col in monthly:
            if col in df_c.columns:
                df_c[col] = yoy_log_diff(df_c[col])
        for col in quarterly:
            if col in df_c.columns:
                df_c[col] = assign_quarterly_to_month_end(df_c[col])

        target = df_c["va_construction"].dropna()
        corrs = {}
        for col in [c for c in df_c.columns if c != "va_construction"]:
            aligned = pd.concat([target, df_c[col]], axis=1).dropna()
            if len(aligned) >= 10:
                corrs[col] = round(aligned.iloc[:,0].corr(aligned.iloc[:,1]), 3)

        corr_df = pd.DataFrame({"Série": list(corrs.keys()), "Corrélation": list(corrs.values())})
        corr_df = corr_df.sort_values("Corrélation")
        fig = px.bar(
            corr_df, x="Corrélation", y="Série", orientation="h",
            color="Corrélation",
            color_continuous_scale=[C_RED, "#f1f5f9", C_GREEN],
            range_color=[-0.5, 0.5],
            text="Corrélation",
            title="Corrélations Pearson avec VA Construction (y-o-y)",
            template=PLOTLY_TEMPLATE,
        )
        fig.add_vline(x=0, line_width=1, line_color="black")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Corrélations globalement faibles → signal davantage dynamique (lags) "
                   "ou variables clés manquantes (permis, mises en chantier — §6 du plan).")

    with tab_series:
        st.subheader("Séries brutes — visualisation interactive")
        numeric = raw.select_dtypes("number")
        all_opts = numeric.columns.tolist()
        preferred = ["va_construction", "consommation_ciment", "credits_immobilier"]
        default_sel = [c for c in preferred if c in all_opts] or all_opts[:2]
        selected = st.multiselect(
            "Choisir les séries à afficher",
            options=all_opts,
            default=default_sel,
        )
        if selected:
            fig = make_subplots(
                rows=len(selected), cols=1,
                shared_xaxes=True,
                subplot_titles=selected,
                vertical_spacing=0.06,
            )
            colours = px.colors.qualitative.Plotly
            for i, col in enumerate(selected):
                s = numeric[col].dropna()
                fig.add_trace(
                    go.Scatter(x=s.index, y=s.values, name=col,
                               line=dict(color=colours[i % len(colours)], width=1.5),
                               hovertemplate="%{x|%b %Y}: %{y:,.1f}<extra></extra>"),
                    row=i+1, col=1,
                )
            fig.update_layout(
                height=250 * len(selected),
                showlegend=False,
                hovermode="x unified",
                template=PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Page: Cement ──────────────────────────────────────────────────────────────

def page_cement():
    st.title("🔴 Rupture d'unité — Consommation ciment (§3.1)")
    _app_log.info("Page: Rupture ciment")
    settings = get_settings()
    raw = get_raw_df()
    cb = settings.corrections.cement_break

    _warn(f"Facteur de correction = **{cb.correction_factor:.0f}** — PLACEHOLDER. Confirmer avec l'APC.")

    from lamiaty.data.corrections import fix_cement_unit_break

    cement_raw = raw["consommation_ciment"]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cement_corr = fix_cement_unit_break(
            cement_raw, cb.break_date, cb.correction_factor, cb.confirmed_by
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Date de rupture", cb.break_date)
    c2.metric("Facteur (placeholder)", f"×{cb.correction_factor:.0f}")
    c3.metric("Statut", cb.status)

    # Side-by-side interactive comparison
    tab_raw, tab_corr, tab_zoom = st.tabs(["Série brute", "Série corrigée", "Zoom autour de la rupture"])

    break_ts     = pd.Timestamp(cb.break_date)          # Timestamp for arithmetic
    break_ts_str = break_ts.strftime("%Y-%m-%d")         # string for Plotly

    with tab_raw:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cement_raw.index, y=cement_raw.values,
            mode="lines", name="Brut", line=dict(color=C_RED, width=1.5),
            hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>",
        ))
        fig.add_shape(type="line", x0=break_ts_str, x1=break_ts_str, y0=0, y1=1, yref="paper",
                      line=dict(dash="dash", color=C_GREY))
        fig.add_annotation(x=break_ts_str, y=1, yref="paper", text=f"Rupture {cb.break_date}",
                           showarrow=False, xanchor="left", font=dict(color=C_GREY))
        fig.update_layout(title="Consommation ciment — BRUT (rupture ~×759 visible)",
                          xaxis_title="Date", yaxis_title="Unité brute",
                          template=PLOTLY_TEMPLATE, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab_corr:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cement_corr.index, y=cement_corr.values,
            mode="lines", name="Corrigé", line=dict(color=C_BLUE, width=1.5),
            hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>",
        ))
        fig.add_shape(type="line", x0=break_ts_str, x1=break_ts_str, y0=0, y1=1, yref="paper",
                      line=dict(dash="dash", color=C_GREY))
        fig.add_annotation(x=break_ts_str, y=1, yref="paper",
                           text=f"Correction ×{cb.correction_factor:.0f}",
                           showarrow=False, xanchor="left", font=dict(color=C_GREY))
        fig.update_layout(title=f"Consommation ciment — CORRIGÉE (×{cb.correction_factor:.0f} avant {cb.break_date})",
                          xaxis_title="Date", yaxis_title="Tonnes (unité homogène)",
                          template=PLOTLY_TEMPLATE, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab_zoom:
        window = cement_raw.loc[
            (cement_raw.index >= break_ts - pd.DateOffset(months=6)) &
            (cement_raw.index <= break_ts + pd.DateOffset(months=6))
        ]
        fig = go.Figure()
        pre  = window[window.index < break_ts]
        post = window[window.index >= break_ts]
        fig.add_trace(go.Bar(x=pre.index,  y=pre.values,  name="Avant rupture",
                             marker_color=C_ORANGE, hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Bar(x=post.index, y=post.values, name="Après rupture",
                             marker_color=C_RED,    hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>"))
        fig.add_shape(type="line", x0=break_ts_str, x1=break_ts_str, y0=0, y1=1, yref="paper",
                      line=dict(dash="dash", color="black"))
        fig.update_layout(title="Zoom ±6 mois autour de la rupture (valeurs brutes)",
                          barmode="overlay", template=PLOTLY_TEMPLATE, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        ratio = post.iloc[0] / pre.iloc[-1] if len(pre) > 0 and len(post) > 0 else None
        if ratio:
            st.metric("Ratio post/pré rupture", f"×{ratio:.0f}",
                      help="Doit être ≈ 759. Confirmer avec l'APC.")


# ── Page: Investissement ──────────────────────────────────────────────────────

def page_investissement():
    st.title("🟠 Diagnostic Investissement État (§3.2)")
    _app_log.info("Page: Investissement État")
    settings = get_settings()
    raw = get_raw_df()
    inv = settings.corrections.investissement_etat

    _warn("Investissement_Etat est **exclu du DFM** jusqu'à confirmation avec TGR/MEF.")

    from lamiaty.data.corrections import fix_investissement_etat

    s_raw = raw["investissement_etat"]
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        s_corr = fix_investissement_etat(s_raw, method=inv.treatment, confirmed_by=inv.confirmed_by)

    jan_mask = s_raw.index.month == 1

    tab_raw, tab_corr, tab_compare = st.tabs(["Brut", "Corrigé (monthly_diff)", "Janvier avant/après"])

    with tab_raw:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=s_raw.index, y=s_raw.values,
            mode="lines", name="Brut", line=dict(color=C_RED, width=1.5),
            hovertemplate="%{x|%b %Y}: %{y:,.0f} MDH<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=s_raw.index[jan_mask], y=s_raw.values[jan_mask],
            mode="markers", name="Janvier", marker=dict(color=C_RED, size=9, symbol="circle"),
            hovertemplate="Janvier %{x|%Y}: %{y:,.0f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="black")
        fig.update_layout(title="Investissement_Etat — BRUT (points = janvier, suspicion cumul YTD)",
                          xaxis_title="Date", yaxis_title="MDH",
                          template=PLOTLY_TEMPLATE, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab_corr:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=s_corr.index, y=s_corr.values,
            mode="lines", name="monthly_diff", line=dict(color=C_GREEN, width=1.5),
            hovertemplate="%{x|%b %Y}: %{y:,.0f} MDH<extra></extra>",
        ))
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="black")
        fig.update_layout(title="Investissement_Etat — CORRIGÉ (première différence = flux mensuel)",
                          xaxis_title="Date", yaxis_title="MDH (variation mensuelle)",
                          template=PLOTLY_TEMPLATE, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Action requise :** Confirmer la définition avec TGR/MEF puis mettre "
                "`confirmed_by: TGR` dans `configs/corrections.yaml`.")

    with tab_compare:
        jan_raw  = s_raw[jan_mask]
        jan_corr = s_corr[s_corr.index.month == 1].dropna()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=jan_raw.index.year,  y=jan_raw.values,
                             name="Brut", marker_color=C_RED, opacity=0.7))
        fig.add_trace(go.Bar(x=jan_corr.index.year, y=jan_corr.values,
                             name="Après monthly_diff", marker_color=C_GREEN, opacity=0.7))
        fig.update_layout(title="Valeurs de janvier — avant et après correction",
                          xaxis_title="Année", yaxis_title="MDH",
                          barmode="group", template=PLOTLY_TEMPLATE, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


# ── Page: Pipeline ────────────────────────────────────────────────────────────

def page_pipeline():
    st.title("✅ Validation du pipeline (model_panel.parquet)")
    _app_log.info("Page: Validation pipeline")

    with st.spinner("Exécution du pipeline..."):
        panel = get_panel()

    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes (mois)", len(panel))
    c2.metric("Colonnes (séries)", len(panel.columns))
    c3.metric("Période", f"{panel.index[0].strftime('%b %Y')} – {panel.index[-1].strftime('%b %Y')}")

    tab_preview, tab_heatmap, tab_series, tab_va = st.tabs(
        ["Aperçu", "Carte NaN", "Séries transformées", "VA CONSTRUCTION"]
    )

    with tab_preview:
        st.dataframe(panel.head(15).round(3), use_container_width=True)
        miss = panel.isnull().sum().reset_index()
        miss.columns = ["Série", "n_missing"]
        miss["pct (%)"] = (miss["n_missing"] / len(panel) * 100).round(1)
        st.subheader("Valeurs manquantes")
        st.dataframe(miss, hide_index=True, use_container_width=True)

    with tab_heatmap:
        st.caption("Bleu = observé · Blanc = NaN.  "
                   "Les séries trimestrielles (va_construction, ipai, creation_emploi) "
                   "affichent 2/3 de NaN — comportement attendu pour le filtre de Kalman.")
        # Interactive heatmap with Plotly
        z = (~panel.isnull()).astype(int).T.values
        fig = go.Figure(go.Heatmap(
            z=z,
            x=panel.index.strftime("%Y-%m"),
            y=panel.columns.tolist(),
            colorscale=[[0, "#f8fafc"], [1, C_BLUE]],
            showscale=False,
            hovertemplate="Date: %{x}<br>Série: %{y}<br>Observé: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title="Panel modèle — carte des données manquantes",
            xaxis_title="Date",
            height=350,
            template=PLOTLY_TEMPLATE,
        )
        # Thin out x-tick labels
        tick_step = max(1, len(panel) // 20)
        tickvals = panel.index[::tick_step].strftime("%Y-%m").tolist()
        fig.update_xaxes(tickvals=tickvals, tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with tab_series:
        st.subheader("Séries transformées (yoy log-diff, standardisées)")
        selected = st.multiselect(
            "Choisir les séries",
            options=[c for c in panel.columns if c != "va_construction"],
            default=["consommation_ciment", "credits_immobilier", "lafarge_index"],
            key="pipeline_series_select",
        )
        if selected:
            fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True,
                                subplot_titles=selected, vertical_spacing=0.07)
            colours = px.colors.qualitative.Plotly
            for i, col in enumerate(selected):
                s = panel[col].dropna()
                fig.add_trace(
                    go.Scatter(x=s.index, y=s.values, name=col,
                               line=dict(color=colours[i % len(colours)], width=1.4),
                               hovertemplate="%{x|%b %Y}: %{y:.3f}<extra></extra>"),
                    row=i+1, col=1,
                )
                fig.add_hline(y=0, row=i+1, col=1, line_width=0.8, line_dash="dot",
                              line_color=C_GREY)
            fig.update_layout(height=220*len(selected), showlegend=False,
                               hovermode="x unified", template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

    with tab_va:
        st.subheader("VA CONSTRUCTION — convention trimestrielle")
        st.caption("Valeur présente uniquement aux mois de fin de trimestre (Mar/Jun/Sep/Déc). "
                   "NaN dans les autres mois = comportement correct pour le DFM.")
        if "va_construction" in panel.columns:
            va = panel["va_construction"]
            va_obs = va.dropna()
            fig = go.Figure()
            # Add NaN positions as light background
            fig.add_trace(go.Scatter(
                x=va.index, y=[0]*len(va),
                mode="markers",
                marker=dict(color="#f1f5f9", size=4, symbol="line-ns-open"),
                name="NaN (non observé)", showlegend=True,
                hovertemplate="%{x|%b %Y}: NaN<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=va_obs.index, y=va_obs.values,
                mode="lines+markers",
                line=dict(color=C_BLUE, width=2),
                marker=dict(size=8, color=C_BLUE),
                name="VA CONSTRUCTION",
                hovertemplate="%{x|%b %Y}: %{y:,.0f} MDH<extra></extra>",
            ))
            fig.add_vrect(x0="2020-01-01", x1="2021-01-01",
                          fillcolor="red", opacity=0.07, line_width=0,
                          annotation_text="COVID-19")
            fig.update_layout(title="VA CONSTRUCTION dans le panel (assignée au dernier mois du trimestre)",
                               xaxis_title="Date", yaxis_title="MDH",
                               template=PLOTLY_TEMPLATE, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            sample = va.head(12)
            st.dataframe(
                sample.to_frame("va_construction").assign(
                    note=sample.index.map(
                        lambda d: "✅ Fin de trimestre" if d.month in [3,6,9,12]
                        else "⬜ NaN attendu"
                    )
                ), use_container_width=True
            )


# ── Page: Stationarity ────────────────────────────────────────────────────────

def page_stationarity():
    st.title("📈 Tests de stationnarité (ADF + KPSS)")
    _app_log.info("Page: Stationnarité")
    st.caption("Prérequis pour l'estimation DFM (Phase 2). "
               "Réalisé sur les séries yoy log-diff + standardisées.")

    with st.spinner("Calcul des tests de racine unitaire..."):
        panel = get_panel()
        from lamiaty.features.stationarity import run_stationarity_battery
        battery = run_stationarity_battery(panel.dropna(thresh=30, axis=1))


    c1, c2, c3 = st.columns(3)
    n_stat = (battery["verdict"] == "STATIONARY").sum()
    n_unit = (battery["verdict"] == "UNIT_ROOT").sum()
    n_amb  = len(battery) - n_stat - n_unit
    c1.metric("Stationnaires ✅", int(n_stat))
    c2.metric("Racine unitaire ❌", int(n_unit))
    c3.metric("Ambigu ⚠️", int(n_amb))

    # Interactive p-value chart
    tab_pval, tab_table = st.tabs(["P-valeurs", "Table complète"])

    with tab_pval:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=battery.index, x=battery["ADF_pvalue"],
            orientation="h", name="ADF p-valeur",
            marker_color=[C_GREEN if v < 0.05 else C_RED for v in battery["ADF_pvalue"]],
            hovertemplate="%{y}: ADF p=%{x:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            y=battery.index, x=battery["KPSS_pvalue"],
            orientation="h", name="KPSS p-valeur",
            marker_color=[C_GREEN if v > 0.05 else C_RED for v in battery["KPSS_pvalue"]],
            opacity=0.5,
            hovertemplate="%{y}: KPSS p=%{x:.4f}<extra></extra>",
        ))
        fig.add_vline(x=0.05, line_dash="dash", line_color="black",
                      annotation_text="α=0.05", annotation_position="top right")
        fig.update_layout(
            title="P-valeurs ADF et KPSS par série",
            xaxis_title="p-valeur",
            barmode="group",
            height=350,
            template=PLOTLY_TEMPLATE,
            hovermode="y unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("ADF : p < 0.05 → stationnaire (rejet H₀ racine unitaire).  "
                   "KPSS : p > 0.05 → stationnaire (non-rejet H₀ stationnarité).")

    with tab_table:
        def colour_row(row):
            if row["verdict"] == "STATIONARY":
                return [f"background-color: #dcfce7"] * len(row)
            elif row["verdict"] == "UNIT_ROOT":
                return [f"background-color: #fee2e2"] * len(row)
            return [f"background-color: #fef9c3"] * len(row)

        styled = battery.style.apply(colour_row, axis=1).format({
            "ADF_stat": "{:.3f}", "ADF_pvalue": "{:.4f}",
            "KPSS_stat": "{:.3f}", "KPSS_pvalue": "{:.4f}",
        })
        st.dataframe(styled, use_container_width=True)

    if n_unit > 0:
        non_stat = battery[battery["verdict"] == "UNIT_ROOT"].index.tolist()
        st.warning(f"Séries non-stationnaires : {non_stat}. "
                   "Vérifier la transformation avant estimation DFM (Phase 2).")


# ── Cached DFM loader ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Estimation DFM en cours (peut prendre ~30s)...")
def get_dfm():
    from lamiaty.model.dfm import DynamicFactorModel
    settings = get_settings()
    panel    = get_panel()
    _app_log.info("Fitting DFM: n_factors=%d", settings.model.n_factors)
    dfm = DynamicFactorModel(
        n_factors=settings.model.n_factors,
        n_lags=settings.model.n_lags,
        settings=settings.model,
    )
    dfm.fit(panel)
    _app_log.info("DFM fitted: llf=%.2f, iter=%d",
                  dfm.results_.log_likelihood, dfm.results_.n_iterations)
    return dfm


# ── Page: DFM Estimation ──────────────────────────────────────────────────────

def page_dfm():
    st.title("🎯 Estimation du Dynamic Factor Model (§4)")
    _app_log.info("Page: Estimation DFM")
    dfm = get_dfm()
    res = dfm.results_

    # Convergence summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Log-vraisemblance", f"{res.log_likelihood:.1f}")
    c2.metric("Itérations EM", res.n_iterations)
    c3.metric("Facteurs", dfm.n_factors)

    if res.n_iterations >= (dfm.settings.max_iterations if dfm.settings else 500):
        st.warning("⚠️ L'algorithme EM n'a pas convergé dans le nombre d'itérations maximal. "
                   "Les paramètres sont proches mais la tolérance n'est pas atteinte.")

    tab_factors, tab_loadings, tab_variance = st.tabs(
        ["📈 Facteurs communs", "🔢 Loadings (Λ)", "📊 Parts de variance"]
    )

    with tab_factors:
        st.subheader("Facteurs latents lissés F_{t|T}")
        factors_df = res.factors_smoothed
        fig = go.Figure()
        colours = [C_BLUE, C_ORANGE, C_GREEN]
        for i, col in enumerate(factors_df.columns):
            fig.add_trace(go.Scatter(
                x=factors_df.index.to_timestamp() if hasattr(factors_df.index, "to_timestamp") else factors_df.index,
                y=factors_df[col],
                name=col,
                line=dict(color=colours[i % len(colours)], width=2),
            ))
        fig.add_vrect(x0="2020-01-01", x1="2021-01-01",
                      fillcolor="red", opacity=0.07, line_width=0,
                      annotation_text="COVID", annotation_position="top left")
        fig.update_layout(
            title="Facteurs communs estimés (période d'estimation 2010-2019)",
            xaxis_title="Date", yaxis_title="Valeur normalisée",
            hovermode="x unified", template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Estimés par filtre de Kalman-EM sur la période in-sample 2010-01 à 2019-12 "
                   "(pré-COVID). Les valeurs hors échantillon sont des prédictions de filtre.")

    with tab_loadings:
        st.subheader("Matrice des loadings Λ — contribution de chaque série à chaque facteur")
        if res.loadings is not None:
            loadings = res.loadings
            fig = px.imshow(
                loadings,
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                aspect="auto",
                title="Loadings Λ (séries × facteurs) — rouge = négatif, bleu = positif",
                labels={"color": "Loading", "x": "Facteur", "y": "Série"},
                template=PLOTLY_TEMPLATE,
            )
            fig.update_traces(text=loadings.round(3).values, texttemplate="%{text}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(loadings.round(4), use_container_width=True)
        else:
            st.info("Loadings non disponibles.")

    with tab_variance:
        st.subheader("Part de variance expliquée par les facteurs communs")
        if res.variance_shares:
            vs = pd.Series(res.variance_shares).sort_values(ascending=True)
            fig = px.bar(
                x=vs.values * 100, y=vs.index,
                orientation="h",
                labels={"x": "Variance expliquée (%)", "y": "Série"},
                color=vs.values * 100,
                color_continuous_scale=["#dbeafe", C_BLUE],
                title="% de variance expliquée par les facteurs communs",
                template=PLOTLY_TEMPLATE,
            )
            fig.update_coloraxes(showscale=False)
            fig.add_vline(x=50, line_dash="dash", line_color=C_GREY,
                          annotation_text="50%", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)


# ── Page: Nowcast ─────────────────────────────────────────────────────────────

def page_nowcast():
    st.title("🔮 Nowcast — VA Construction")
    _app_log.info("Page: Nowcast")
    dfm   = get_dfm()
    panel = get_panel()

    nc = dfm.nowcast(panel)
    if len(nc) == 0:
        st.error("Aucune prédiction disponible.")
        return

    # Current nowcast
    last_q   = nc.index[-1]
    last_val = float(nc.iloc[-1])
    prev_val = float(nc.iloc[-2]) if len(nc) > 1 else None
    delta    = f"{(last_val - prev_val):+.1f} MDH" if prev_val else None

    st.metric(
        label=f"Nowcast VA Construction — {last_q}",
        value=f"{last_val:,.0f} MDH",
        delta=delta,
    )

    # Observed + Nowcast with CI
    raw       = get_raw_df()
    va_obs    = raw["va_construction"].dropna()

    ci_lower  = nc.attrs.get("ci_lower", pd.Series(dtype=float))
    ci_upper  = nc.attrs.get("ci_upper", pd.Series(dtype=float))

    # Convert period index to timestamp if needed
    def _to_ts(idx):
        if hasattr(idx, "to_timestamp"):
            return idx.to_timestamp()
        return idx

    nc_idx = _to_ts(nc.index)
    ci_lo_idx = _to_ts(ci_lower.index) if len(ci_lower) > 0 else nc_idx
    ci_hi_idx = _to_ts(ci_upper.index) if len(ci_upper) > 0 else nc_idx

    fig = go.Figure()

    # 90% CI band
    if len(ci_lower) > 0 and len(ci_upper) > 0:
        x_band = list(ci_lo_idx) + list(ci_hi_idx[::-1])
        y_band = list(ci_lower.values) + list(ci_upper.values[::-1])
        fig.add_trace(go.Scatter(
            x=x_band, y=y_band, fill="toself",
            fillcolor="rgba(37,99,235,0.12)", line=dict(width=0),
            name="IC 90%", showlegend=True,
        ))

    # Nowcast line
    fig.add_trace(go.Scatter(
        x=nc_idx, y=nc.values,
        mode="lines+markers", name="Nowcast DFM",
        line=dict(color=C_BLUE, width=2, dash="dot"),
        marker=dict(size=5),
    ))

    # Observed data (scatter)
    fig.add_trace(go.Scatter(
        x=va_obs.index, y=va_obs.values,
        mode="markers", name="Observé (HCP)",
        marker=dict(color=C_GREEN, size=6),
    ))

    fig.add_vrect(x0="2020-01-01", x1="2021-01-01",
                  fillcolor="red", opacity=0.07, line_width=0,
                  annotation_text="COVID", annotation_position="top left")
    fig.update_layout(
        title="Nowcast VA Construction vs Observé — DFM k=2",
        xaxis_title="Date", yaxis_title="MDH (prix courants)",
        hovermode="x unified", template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Nowcast table
    st.divider()
    st.subheader("Tableau du nowcast (derniers trimestres)")
    nc_df = pd.DataFrame({
        "Quarter":   [str(i) for i in nc_idx[-8:]],
        "Nowcast":   nc.iloc[-8:].round(0).values,
        "IC lower":  ci_lower.reindex(nc.index[-8:]).round(0).values if len(ci_lower) > 0 else ["—"] * 8,
        "IC upper":  ci_upper.reindex(nc.index[-8:]).round(0).values if len(ci_upper) > 0 else ["—"] * 8,
    })
    st.dataframe(nc_df, hide_index=True, use_container_width=True)

    st.info("ℹ️ Le nowcast est basé sur un DFM estimé sur la période 2010–2019 (pré-COVID). "
            "Phase 2 complète. Phase 3 (backtesting) disponible dans l'onglet suivant.")


# ── Page: Backtesting ─────────────────────────────────────────────────────────

def page_backtest():
    st.title("📉 Backtesting pseudo-temps réel (§5)")
    _app_log.info("Page: Backtesting")

    settings = get_settings()
    panel    = get_panel()

    from lamiaty.backtest.runner import load_backtest_results

    # Try to load cached results
    cached = load_backtest_results(settings.paths.vintages_dir)

    st.info(
        "Le backtesting itère sur ~240 origines de prévision (2015–2024, 7ème et 21ème "
        "de chaque mois). Chaque origine requiert un ré-estimation complète du DFM. "
        "Cliquer sur **Lancer le backtest** pour démarrer (peut prendre plusieurs minutes)."
    )

    col1, col2 = st.columns([1, 3])
    run_button = col1.button("▶ Lancer le backtest", type="primary")

    if run_button:
        with st.spinner("Backtest en cours…"):
            from lamiaty.backtest.runner import run_backtest
            cached = run_backtest(settings, panel, save_vintages=False)
            st.success(f"Backtest terminé : {len(cached)} origines évaluées.")

    if cached is None:
        st.warning("Aucun résultat de backtest disponible. Cliquez sur 'Lancer le backtest'.")
        return

    # Filter to rows with valid nowcast and realized
    results = cached.dropna(subset=["nowcast", "realized"])

    if len(results) == 0:
        st.warning("Pas assez de données réalisées pour évaluer les métriques.")
        return

    tab_perf, tab_chart, tab_dm = st.tabs(
        ["📊 Performance", "📈 Nowcast vs Réalisé", "🧪 Diebold-Mariano"]
    )

    with tab_perf:
        from lamiaty.backtest.evaluator import (
            compute_rmsfe, compute_mafe, compute_theil_u, diebold_mariano,
            benchmark_random_walk, benchmark_ar1,
        )
        nc   = results.set_index("target_quarter")["nowcast"]
        real = results.set_index("target_quarter")["realized"]

        rw   = benchmark_random_walk(real)
        ar1b = benchmark_ar1(real)

        perf_data = []
        for name, bench in [("DFM", nc), ("Random Walk", rw), ("AR(1)", ar1b)]:
            perf_data.append({
                "Modèle": name,
                "RMSFE":  f"{compute_rmsfe(bench, real):,.1f}" if name != "DFM" else f"{compute_rmsfe(nc, real):,.1f}",
                "MAFE":   f"{compute_mafe(bench, real):,.1f}",
                "Theil U": f"{compute_theil_u(nc if name == 'DFM' else bench, real, rw):.3f}",
            })
        st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

    with tab_chart:
        st.subheader("Nowcast vs Valeurs réalisées")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Nowcast vs Réalisé", "Erreur de prévision"],
                            vertical_spacing=0.08)

        fig.add_trace(go.Scatter(x=results["origin"], y=results["realized"],
                                 mode="markers+lines", name="Réalisé",
                                 line=dict(color=C_GREEN, width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=results["origin"], y=results["nowcast"],
                                 mode="markers", name="Nowcast DFM",
                                 marker=dict(color=C_BLUE, size=5)), row=1, col=1)
        fig.add_trace(go.Bar(x=results["origin"], y=results["error"],
                             name="Erreur", marker_color=C_ORANGE), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color=C_GREY, row=2, col=1)

        fig.update_layout(hovermode="x unified", template=PLOTLY_TEMPLATE,
                          height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab_dm:
        st.subheader("Test de Diebold-Mariano vs benchmarks")
        from lamiaty.backtest.evaluator import diebold_mariano, benchmark_random_walk, benchmark_ar1
        nc   = results.set_index("target_quarter")["nowcast"]
        real = results.set_index("target_quarter")["realized"]
        e_dfm = nc - real

        dm_rows = []
        for bench_name, bench in [("Random Walk", benchmark_random_walk(real)),
                                   ("AR(1)", benchmark_ar1(real))]:
            e_b = bench - real
            dm  = diebold_mariano(e_dfm.dropna(), e_b.dropna())
            sig = "✅ Significatif" if dm["pvalue"] < 0.05 else "—"
            dm_rows.append({
                "Benchmark": bench_name,
                "DM statistic": f"{dm['statistic']:.3f}" if not pd.isna(dm["statistic"]) else "N/A",
                "p-value": f"{dm['pvalue']:.4f}" if not pd.isna(dm["pvalue"]) else "N/A",
                "Significatif (5%)": sig,
                "n_obs": dm["n_obs"],
            })

        dm_df = pd.DataFrame(dm_rows)

        def _style_sig(row):
            return ["background-color: #dcfce7" if row["Significatif (5%)"] == "✅ Significatif"
                    else "" for _ in row]

        st.dataframe(dm_df.style.apply(_style_sig, axis=1),
                     hide_index=True, use_container_width=True)
        st.caption("H0 : précision prévisionnelle égale. DM < 0 → DFM meilleur que benchmark.")


# ── Page: News decomposition ──────────────────────────────────────────────────

def page_news():
    st.title("📰 Décomposition news — Attribution des révisions")
    _app_log.info("Page: News decomposition")

    dfm   = get_dfm()
    panel = get_panel()
    raw   = get_raw_df()

    st.markdown(
        "Décompose la **révision du nowcast** entre deux mises à jour en contributions "
        "par série — quelle publication a le plus impacté la dernière révision ?  \n"
        "Équation (11) de Danov et al. (2026) : **révision = Σ poids × news**"
    )

    # Let user pick the 'previous' information cutoff
    va_obs = raw["va_construction"].dropna()
    q_dates = sorted(va_obs.index.tolist(), reverse=True)

    if len(q_dates) < 2:
        st.warning("Pas assez de données pour la décomposition news.")
        return

    st.subheader("Simuler une révision entre deux mises à jour")
    c1, c2 = st.columns(2)
    prev_date = c1.selectbox(
        "Information set précédent (Ω_prev)",
        options=q_dates[1:],
        format_func=lambda d: d.strftime("%Y-%m"),
        index=0,
        key="news_prev",
    )
    new_date = c2.selectbox(
        "Information set nouveau (Ω_new)",
        options=[d for d in q_dates if d > prev_date],
        format_func=lambda d: d.strftime("%Y-%m"),
        index=0,
        key="news_new",
    )

    if st.button("Calculer la décomposition news", type="primary"):
        panel_prev = panel.loc[:prev_date]
        panel_new  = panel.loc[:new_date]

        with st.spinner("Calcul de la décomposition news…"):
            try:
                news_df = dfm.news_decomposition(panel_prev, panel_new)
                st.success(f"Révision totale : {news_df.loc[news_df['series'] == 'TOTAL', 'contribution'].iloc[0]:+.2f} MDH")
            except Exception as exc:
                st.error(f"Erreur : {exc}")
                _app_log.error("News decomposition failed: %s", exc, exc_info=True)
                return

        # Filter out TOTAL for chart
        detail = news_df[news_df["series"] != "TOTAL"].copy()
        total  = news_df[news_df["series"] == "TOTAL"]["contribution"].iloc[0]

        # Waterfall chart
        st.subheader("Graphique en cascade — contributions par publication")
        measures = ["relative"] * len(detail) + ["total"]
        x_labels = [f"{r['series']} ({pd.Timestamp(r['update_date']).strftime('%Y-%m') if pd.notna(r['update_date']) else ''})"
                    for _, r in detail.iterrows()] + ["TOTAL"]
        y_values = list(detail["contribution"].values) + [total]
        text_vals = [f"{v:+.2f}" for v in y_values]

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=measures,
            x=x_labels,
            y=y_values,
            text=text_vals,
            textposition="outside",
            connector={"line": {"color": C_GREY}},
            increasing={"marker": {"color": C_BLUE}},
            decreasing={"marker": {"color": C_RED}},
            totals={"marker": {"color": C_GREEN}},
        ))
        fig.update_layout(
            title=f"Décomposition news : {prev_date.strftime('%Y-Q%q')} → {new_date.strftime('%Y-Q%q')}",
            yaxis_title="Contribution (MDH)",
            template=PLOTLY_TEMPLATE,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Details table
        st.subheader("Détail des contributions")
        display_df = news_df.copy()
        display_df["update_date"] = display_df["update_date"].apply(
            lambda x: x.strftime("%Y-%m") if pd.notna(x) else "—"
        )
        st.dataframe(
            display_df.style.format({"news": "{:.4f}", "weight": "{:.4f}", "contribution": "{:.4f}"}, na_rep="—"),
            hide_index=True, use_container_width=True,
        )
    else:
        st.info("Sélectionner deux dates et cliquer sur **Calculer la décomposition news**.")


# ── Router ────────────────────────────────────────────────────────────────────

_ROUTERS = {
    "home":         page_home,
    "audit":        page_audit,
    "cement":       page_cement,
    "investissement": page_investissement,
    "pipeline":     page_pipeline,
    "stationarity": page_stationarity,
    "dfm":          page_dfm,
    "nowcast":      page_nowcast,
    "backtest":     page_backtest,
    "news":         page_news,
}

_ROUTERS[page]()
