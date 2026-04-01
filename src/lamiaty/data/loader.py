"""
Raw data loaders — reads source files exactly as they are.

No cleaning, no type coercion beyond date parsing.
Every function returns a DataFrame with a DatetimeIndex (monthly period, first day).
Column names are normalised to the snake_case constants defined at the top of this module.

These functions replace the 4 ad-hoc scripts that previously existed in the project root:
  analyze_json.py          → absorbed into notebooks/00_data_audit.ipynb
  extract_masi.py          → load_masi_json() + transforms.resample_daily_to_monthly_first()
  extract_moroccan_shares.py → load_moroccan_shares_csv()
  upsample_masi.py         → load_extract3_csv() (file-reading only; repetition logic NOT migrated)

All Tennis AI path references have been eliminated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants — use these throughout the codebase to avoid typos
# ---------------------------------------------------------------------------

COL_DATE = "date"
COL_CEMENT = "consommation_ciment"
COL_CREDITS_EQUIP = "credits_equipement"
COL_CREDITS_IMMO = "credits_immobilier"
COL_VA_CONSTRUCTION = "va_construction"
COL_IPAI = "ipai"
COL_LAFARGE = "lafarge_index"
COL_INVESTISSEMENT = "investissement_etat"
COL_EMPLOI = "creation_emploi"

# Mapping from raw Excel column names → internal constants
_BTP_COLUMN_MAP = {
    "Date (MM/YYYY)": COL_DATE,
    "Consommation_ciment": COL_CEMENT,
    "credits_equipement": COL_CREDITS_EQUIP,
    "credits_immobilier": COL_CREDITS_IMMO,
    "VA CONSTRUCTION": COL_VA_CONSTRUCTION,
    "L'IPAI": COL_IPAI,
    "Indice_societes_construction_LAFARGEHOLCIM": COL_LAFARGE,
    "Investissement_Etat": COL_INVESTISSEMENT,
    "Creation nette d emploi": COL_EMPLOI,
}

ALL_BTP_COLUMNS = [
    COL_CEMENT,
    COL_CREDITS_EQUIP,
    COL_CREDITS_IMMO,
    COL_VA_CONSTRUCTION,
    COL_IPAI,
    COL_LAFARGE,
    COL_INVESTISSEMENT,
    COL_EMPLOI,
]


# ---------------------------------------------------------------------------
# Main BTP dataset
# ---------------------------------------------------------------------------


def load_base_btp(path: str | Path) -> pd.DataFrame:
    """Load base_BTP_modifiee.xlsx into a DataFrame with a monthly DatetimeIndex.

    The LafargeHolcim index column is intentionally left as dtype object (string
    with comma separators) — cleaning is handled downstream in corrections.py.

    Args:
        path: Absolute or relative path to the Excel file.

    Returns:
        DataFrame with DatetimeIndex (monthly, first day of month) and columns
        named according to the COL_* constants defined in this module.
    """
    path = Path(path)
    logger.info("Loading base BTP from %s", path)
    df = pd.read_excel(path, engine="openpyxl")

    # Rename columns
    df = df.rename(columns=_BTP_COLUMN_MAP)

    # Parse date column — format is "MM/YYYY"
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], format="%m/%Y")
    df = df.sort_values(COL_DATE).reset_index(drop=True)
    df = df.set_index(COL_DATE)
    df.index.name = "date"

    # Ensure numeric types for all columns except LafargeHolcim (intentionally left as-is)
    numeric_cols = [c for c in ALL_BTP_COLUMNS if c != COL_LAFARGE and c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Loaded %d rows, %d columns from base BTP", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# MASI JSON data (migrated from extract_masi.py)
# ---------------------------------------------------------------------------


def load_masi_json(path: str | Path) -> pd.DataFrame:
    """Load a MASI index JSON export into a DataFrame of daily records.

    The JSON structure has a top-level 'data' key containing a list of items,
    each with an 'attributes' dict holding 'field_seance_date' and 'field_index_value'.

    The monthly aggregation (first trading day of each month) is performed downstream
    in transforms.resample_daily_to_monthly_first().

    Args:
        path: Path to the JSON file.

    Returns:
        DataFrame with columns ['date', 'masi_index'] and a RangeIndex.
        Dates are parsed to datetime. Values are float.
    """
    path = Path(path)
    logger.info("Loading MASI JSON from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("data", [])
    records = []
    for item in items:
        attrs = item.get("attributes", {})
        date_str = attrs.get("field_seance_date")
        value = attrs.get("field_index_value")
        if date_str and value is not None:
            records.append({"date": date_str, "masi_index": value})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["masi_index"] = pd.to_numeric(df["masi_index"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Loaded %d daily MASI records from %s", len(df), path.name)
    return df


# ---------------------------------------------------------------------------
# Moroccan All Shares CSV (migrated from extract_moroccan_shares.py)
# ---------------------------------------------------------------------------


def load_moroccan_shares_csv(path: str | Path) -> pd.DataFrame:
    """Load the Moroccan All Shares Historical Data CSV.

    Handles the comma-as-thousands-separator format in the Price column
    (e.g., "16,655.58" → 16655.58). Date format is DD/MM/YYYY.

    The monthly aggregation is performed downstream in
    transforms.resample_daily_to_monthly_first().

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with columns ['date', 'price', 'open', 'high', 'low', 'vol', 'change_pct'].
        Dates are datetime. Price and numeric columns are float.
    """
    path = Path(path)
    logger.info("Loading Moroccan All Shares CSV from %s", path)

    df = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_").replace(".", "").replace("%", "pct") for c in df.columns]

    # Parse date (dayfirst format DD/MM/YYYY)
    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col is None:
        raise ValueError(f"No date column found in {path.name}. Columns: {df.columns.tolist()}")
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    # Strip commas from numeric columns stored as strings
    for col in df.columns:
        if col == "date":
            continue
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Loaded %d daily records from %s", len(df), path.name)
    return df


# ---------------------------------------------------------------------------
# Extract3 CSV — quarterly trimester data (migrated from upsample_masi.py)
# ---------------------------------------------------------------------------


def load_extract3_csv(path: str | Path) -> pd.DataFrame:
    """Load the extract3.csv date-list file.

    The actual file contains a single column 'Date (MM/YYYY)' with monthly
    date strings — it is a reference calendar of monthly periods.

    If the file has a second column it is retained as 'value'. If it follows
    the older "Year:Trimester" format (e.g., "2019:T1"), that is also handled.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with a 'date' column (DatetimeIndex) and optional 'value' column.
    """
    path = Path(path)
    logger.info("Loading extract3 CSV from %s", path)

    # Try common separators
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) >= 1:
                break
        except Exception:
            continue

    df.columns = [c.strip() for c in df.columns]
    first_col = df.columns[0]
    first_val = str(df[first_col].dropna().iloc[0]) if len(df) > 0 else ""

    if ":" in first_val:
        # Legacy "Year:Trimester" format
        split = df[first_col].str.split(":", expand=True)
        df["year"] = pd.to_numeric(split[0], errors="coerce")
        quarter_str = split[1].str.upper().str.replace("T", "", regex=False)
        df["quarter"] = pd.to_numeric(quarter_str, errors="coerce")
        # Map quarter → last month: Q1→3, Q2→6, Q3→9, Q4→12
        df["month"] = df["quarter"] * 3
        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str) + "-01",
            errors="coerce",
        )
        cols = ["date"]
        if len(df.columns) > 1 and df.columns[1] not in cols:
            df = df.rename(columns={df.columns[1]: "value"})
            cols.append("value")
        result = df[cols].dropna(subset=["date"])
    else:
        # MM/YYYY date-list format (actual file)
        df = df.rename(columns={first_col: "date"})
        df["date"] = pd.to_datetime(df["date"], format="%m/%Y", errors="coerce")
        result = df.dropna(subset=["date"]).reset_index(drop=True)

    result = result.set_index("date")
    result.index.name = "date"
    logger.info("Loaded %d records from %s", len(result), path.name)
    return result


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_all_raw(paths) -> dict[str, pd.DataFrame]:
    """Load the primary data source: base_BTP_modifiee.xlsx.

    All model variables (VA CONSTRUCTION, cement, credits, IPAI, LafargeHolcim,
    Investissement_Etat, employment) come from this single Excel file.

    Auxiliary files (MASI JSON, Moroccan shares CSV, extract3) are available
    via their dedicated loaders but are NOT used by the main DFM pipeline.

    Args:
        paths: PathSettings instance from Settings.

    Returns:
        Dict with key: 'base_btp' → corrected DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}
    result["base_btp"] = load_base_btp(paths.base_btp_path)
    logger.info("Loaded primary source: base_BTP_modifiee.xlsx (%d rows)", len(result["base_btp"]))
    return result
