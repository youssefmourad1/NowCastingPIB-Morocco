"""
Data quality and pipeline validation charts.

Used in notebooks/00_data_audit.ipynb through 03_pipeline_validation.ipynb.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


def plot_series_with_break(
    series: pd.Series,
    break_date: str,
    title: str = "",
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Plot a series with a vertical line marking a structural break.

    Used to visualise the cement unit rupture (§3.1) before and after correction.

    Args:
        series: Series with DatetimeIndex.
        break_date: Break date string, e.g., "2022-04-01".
        title: Plot title.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series.index, series.values, lw=1.5, color="#2563eb", label=series.name or "series")
    ax.axvline(pd.Timestamp(break_date), color="#dc2626", lw=1.5, linestyle="--", label=f"Break: {break_date}")
    ax.set_title(title or f"Structural break at {break_date}", fontsize=12)
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_missing_heatmap(
    panel: pd.DataFrame,
    figsize: tuple[int, int] = (14, 6),
    title: str = "Missing data pattern (NaN = white)",
) -> plt.Figure:
    """Heatmap of NaN positions across the mixed-frequency panel.

    Immediately reveals the quarterly observation pattern for VA CONSTRUCTION,
    IPAI, and employment (two-thirds of rows are NaN for these series).

    Args:
        panel: Mixed-frequency panel from build_mixed_frequency_panel().
        figsize: Figure dimensions.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    mask = panel.isnull()
    # Plot 1 (observed) and NaN (missing) as a binary heatmap
    data_binary = (~mask).astype(int)
    sns.heatmap(
        data_binary.T,
        ax=ax,
        cmap=["#f1f5f9", "#1e40af"],
        cbar=False,
        linewidths=0,
        yticklabels=panel.columns.tolist(),
    )
    # Reduce x-tick density
    n_ticks = min(12, len(panel))
    step = max(1, len(panel) // n_ticks)
    tick_positions = list(range(0, len(panel), step))
    tick_labels = [panel.index[i].strftime("%Y-%m") for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_correlation_matrix(
    panel: pd.DataFrame,
    target_col: str = "va_construction",
    figsize: tuple[int, int] = (9, 5),
) -> plt.Figure:
    """Bar chart of Pearson correlations with the target variable (VA CONSTRUCTION).

    Replicates Table 2.4 of the Implementation Plan — correlations computed on
    year-over-year changes using the quarterly aggregation of monthly series.

    Args:
        panel: Model panel (already yoy-differenced and standardized).
        target_col: Column to correlate against.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    if target_col not in panel.columns:
        raise KeyError(f"Target column '{target_col}' not in panel.")

    target = panel[target_col].dropna()
    other_cols = [c for c in panel.columns if c != target_col]

    corrs = {}
    for col in other_cols:
        aligned = pd.concat([target, panel[col]], axis=1).dropna()
        if len(aligned) >= 10:
            corrs[col] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

    corr_series = pd.Series(corrs).sort_values()
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in corr_series.values]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(corr_series.index, corr_series.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Pearson correlation with VA Construction (y-o-y)")
    ax.set_title(f"Correlations with {target_col}", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    for bar, val in zip(bars, corr_series.values):
        ax.text(
            val + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def plot_series_panel(
    panel: pd.DataFrame,
    cols: list[str] | None = None,
    figsize: tuple[int, int] = (14, 10),
    title: str = "Model panel — all series",
) -> plt.Figure:
    """Plot all series in the panel on a grid of subplots.

    Args:
        panel: Mixed-frequency panel.
        cols: Subset of columns to plot. Defaults to all.
        figsize: Figure dimensions.
        title: Suptitle.

    Returns:
        Matplotlib Figure.
    """
    cols = cols or panel.columns.tolist()
    n = len(cols)
    ncols = 2
    nrows = (n + 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        s = panel[col].dropna()
        ax.plot(s.index, s.values, lw=1.2, color="#2563eb")
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig
