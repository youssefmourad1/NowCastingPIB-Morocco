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
        "Phase 1 — Infrastructure données"
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
        selected = st.multiselect(
            "Choisir les séries à afficher",
            options=numeric.columns.tolist(),
            default=["va_construction", "consommation_ciment", "lafarge_index"],
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

    break_ts = pd.Timestamp(cb.break_date)

    with tab_raw:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cement_raw.index, y=cement_raw.values,
            mode="lines", name="Brut", line=dict(color=C_RED, width=1.5),
            hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>",
        ))
        fig.add_vline(x=break_ts, line_dash="dash", line_color=C_GREY,
                      annotation_text=f"Rupture {cb.break_date}", annotation_position="top right")
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
        fig.add_vline(x=break_ts, line_dash="dash", line_color=C_GREY,
                      annotation_text=f"Correction appliquée ×{cb.correction_factor:.0f}")
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
        fig.add_vline(x=break_ts, line_dash="dash", line_color="black")
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

    verdict_colour = {
        "STATIONARY": C_GREEN,
        "UNIT_ROOT": C_RED,
    }

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


# ── Router ────────────────────────────────────────────────────────────────────

_ROUTERS = {
    "home":         page_home,
    "audit":        page_audit,
    "cement":       page_cement,
    "investissement": page_investissement,
    "pipeline":     page_pipeline,
    "stationarity": page_stationarity,
}

_ROUTERS[page]()
