"""
Tests for src/lamiaty/data/corrections.py
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from lamiaty.data.corrections import (
    fix_cement_unit_break,
    fix_investissement_etat,
    fix_lafarge_strings,
)


# ---------------------------------------------------------------------------
# fix_cement_unit_break
# ---------------------------------------------------------------------------


class TestFixCementUnitBreak:
    BREAK_DATE = "2022-04-01"
    FACTOR = 759.0

    def test_pre_break_values_scaled(self, cement_series_with_break):
        corrected = fix_cement_unit_break(
            cement_series_with_break, self.BREAK_DATE, self.FACTOR, confirmed_by="test"
        )
        pre_mask = corrected.index < pd.Timestamp(self.BREAK_DATE)
        expected = 1000.0 * self.FACTOR
        assert all(abs(v - expected) < 1.0 for v in corrected[pre_mask].values)

    def test_post_break_values_unchanged(self, cement_series_with_break):
        corrected = fix_cement_unit_break(
            cement_series_with_break, self.BREAK_DATE, self.FACTOR, confirmed_by="test"
        )
        post_mask = corrected.index >= pd.Timestamp(self.BREAK_DATE)
        # Post-break values in fixture are already 1000 * 759
        original_post = cement_series_with_break[post_mask]
        assert (corrected[post_mask] == original_post).all()

    def test_boundary_value_not_scaled(self, cement_series_with_break):
        """The break date itself is post-break — should NOT be scaled."""
        corrected = fix_cement_unit_break(
            cement_series_with_break, self.BREAK_DATE, self.FACTOR, confirmed_by="test"
        )
        assert corrected.loc[pd.Timestamp(self.BREAK_DATE)] == pytest.approx(
            cement_series_with_break.loc[pd.Timestamp(self.BREAK_DATE)]
        )

    def test_emits_warning_when_unconfirmed(self, cement_series_with_break):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fix_cement_unit_break(
                cement_series_with_break, self.BREAK_DATE, self.FACTOR, confirmed_by=None
            )
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "confirmed_by" in str(w[0].message).lower() or "not been confirmed" in str(w[0].message).lower()

    def test_no_warning_when_confirmed(self, cement_series_with_break):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fix_cement_unit_break(
                cement_series_with_break, self.BREAK_DATE, self.FACTOR, confirmed_by="APC"
            )
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0

    def test_raises_on_non_positive_factor(self, cement_series_with_break):
        with pytest.raises(TypeError):
            fix_cement_unit_break(cement_series_with_break, self.BREAK_DATE, -1.0, confirmed_by="test")

    def test_raises_on_non_numeric_factor(self, cement_series_with_break):
        with pytest.raises(TypeError):
            fix_cement_unit_break(cement_series_with_break, self.BREAK_DATE, "759", confirmed_by="test")

    def test_series_length_preserved(self, cement_series_with_break):
        corrected = fix_cement_unit_break(
            cement_series_with_break, self.BREAK_DATE, self.FACTOR, confirmed_by="test"
        )
        assert len(corrected) == len(cement_series_with_break)

    def test_index_preserved(self, cement_series_with_break):
        corrected = fix_cement_unit_break(
            cement_series_with_break, self.BREAK_DATE, self.FACTOR, confirmed_by="test"
        )
        assert (corrected.index == cement_series_with_break.index).all()


# ---------------------------------------------------------------------------
# fix_investissement_etat
# ---------------------------------------------------------------------------


class TestFixInvestissementEtat:
    def test_monthly_diff_eliminates_large_negative_january(
        self, investissement_series_with_negative_january
    ):
        """After monthly_diff, the large negative January value (-50000) should be
        replaced by a difference value (Jan - Dec_prev), which is no longer equal
        to the raw -50000 level."""
        corrected = fix_investissement_etat(
            investissement_series_with_negative_january,
            method="monthly_diff",
            confirmed_by="test",
        )
        raw = investissement_series_with_negative_january
        # Check that the January values in the corrected series differ from raw
        jan_corrected = corrected[corrected.index.month == 1].dropna()
        jan_raw = raw[raw.index.month == 1]
        # At least one January value must change (diff ≠ level)
        assert not (jan_corrected == jan_raw.reindex(jan_corrected.index)).all()

    def test_monthly_diff_first_value_is_nan(self, investissement_series_with_negative_january):
        corrected = fix_investissement_etat(
            investissement_series_with_negative_january,
            method="monthly_diff",
            confirmed_by="test",
        )
        assert pd.isna(corrected.iloc[0])

    def test_keep_as_is_preserves_values(self, investissement_series_with_negative_january):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            corrected = fix_investissement_etat(
                investissement_series_with_negative_january,
                method="keep_as_is",
                confirmed_by="test",
            )
        assert (corrected == investissement_series_with_negative_january).all()

    def test_emits_warning_when_unconfirmed(self, investissement_series_with_negative_january):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fix_investissement_etat(
                investissement_series_with_negative_january,
                method="monthly_diff",
                confirmed_by=None,
            )
            assert any(issubclass(x.category, UserWarning) for x in w)

    def test_raises_on_invalid_method(self, investissement_series_with_negative_january):
        with pytest.raises(ValueError, match="Unknown method"):
            fix_investissement_etat(
                investissement_series_with_negative_january,
                method="invalid_method",
                confirmed_by="test",
            )


# ---------------------------------------------------------------------------
# fix_lafarge_strings
# ---------------------------------------------------------------------------


class TestFixLafargeStrings:
    def test_strips_commas(self):
        s = pd.Series(["1,612.5", "2,445.0", "838.0"])
        result = fix_lafarge_strings(s)
        assert pd.api.types.is_float_dtype(result)
        assert result.iloc[0] == pytest.approx(1612.5)
        assert result.iloc[1] == pytest.approx(2445.0)
        assert result.iloc[2] == pytest.approx(838.0)

    def test_handles_no_comma(self):
        s = pd.Series(["1612", "2445"])
        result = fix_lafarge_strings(s)
        assert result.iloc[0] == pytest.approx(1612.0)

    def test_preserves_nan(self):
        s = pd.Series(["1,612", np.nan, "2,000"])
        result = fix_lafarge_strings(s)
        assert pd.isna(result.iloc[1])
        assert result.iloc[0] == pytest.approx(1612.0)

    def test_raises_on_unparseable_non_nan(self):
        s = pd.Series(["1,612", "abc_invalid", "2,000"])
        with pytest.raises(ValueError, match="could not parse"):
            fix_lafarge_strings(s)

    def test_already_float_passthrough(self):
        s = pd.Series([1612.0, 2445.0, 838.0])
        result = fix_lafarge_strings(s)
        assert result.dtype == float
        assert (result == s).all()

    def test_length_preserved(self):
        s = pd.Series(["1,612", "2,445", "838", "1,200"])
        result = fix_lafarge_strings(s)
        assert len(result) == len(s)
