"""
Tests for src/lamiaty/data/loader.py
"""

import pandas as pd
import pytest

from lamiaty.data.loader import (
    COL_CEMENT,
    COL_CREDITS_EQUIP,
    COL_CREDITS_IMMO,
    COL_EMPLOI,
    COL_INVESTISSEMENT,
    COL_IPAI,
    COL_LAFARGE,
    COL_VA_CONSTRUCTION,
    ALL_BTP_COLUMNS,
    load_base_btp,
)


@pytest.fixture
def loaded_df(mini_base_path):
    return load_base_btp(mini_base_path)


class TestLoadBaseBtp:
    def test_returns_dataframe(self, loaded_df):
        assert isinstance(loaded_df, pd.DataFrame)

    def test_datetime_index(self, loaded_df):
        assert isinstance(loaded_df.index, pd.DatetimeIndex)

    def test_monthly_frequency(self, loaded_df):
        # Consecutive rows should differ by one month
        diffs = loaded_df.index[1:] - loaded_df.index[:-1]
        assert all(d.days in range(28, 32) for d in diffs)

    def test_index_name(self, loaded_df):
        assert loaded_df.index.name == "date"

    def test_column_names_are_constants(self, loaded_df):
        expected = set(ALL_BTP_COLUMNS)
        actual = set(loaded_df.columns)
        assert expected == actual, f"Missing columns: {expected - actual}"

    def test_lafarge_raw_dtype_is_object(self, loaded_df):
        """Loader must NOT clean the LafargeHolcim column — stays as string."""
        assert loaded_df[COL_LAFARGE].dtype == object, (
            "LafargeHolcim column should remain as object (string) in raw loader output. "
            "Cleaning is handled by corrections.fix_lafarge_strings()."
        )

    def test_cement_is_numeric(self, loaded_df):
        assert pd.api.types.is_numeric_dtype(loaded_df[COL_CEMENT])

    def test_va_construction_is_numeric(self, loaded_df):
        assert pd.api.types.is_numeric_dtype(loaded_df[COL_VA_CONSTRUCTION])

    def test_30_rows(self, loaded_df):
        """Fixture has 30 rows (Jan 2020 – Jun 2022)."""
        assert len(loaded_df) == 30

    def test_sorted_ascending(self, loaded_df):
        assert loaded_df.index.is_monotonic_increasing
