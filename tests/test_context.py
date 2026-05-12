"""Tests for homogene.context"""

import pandas as pd
import pytest

from homogene.context import _build_context, _validate_columns


class TestValidateColumns:
    def test_passes(self):
        df = pd.DataFrame({"review": ["a"], "product": ["b"]})
        _validate_columns(["review"], df)

    def test_missing_raises(self):
        df = pd.DataFrame({"review": ["a"]})
        with pytest.raises(KeyError, match="missing"):
            _validate_columns(["missing"], df)

    def test_multiple_missing_raises(self):
        df = pd.DataFrame({"review": ["a"]})
        with pytest.raises(KeyError):
            _validate_columns(["author", "date"], df)

    def test_partial_missing_raises(self):
        df = pd.DataFrame({"review": ["a"]})
        with pytest.raises(KeyError):
            _validate_columns(["review", "missing"], df)

    def test_empty_columns_passes(self):
        df = pd.DataFrame({"review": ["a"]})
        _validate_columns([], df)

    def test_empty_df_no_columns_passes(self):
        _validate_columns([], pd.DataFrame())

    def test_empty_df_with_column_raises(self):
        with pytest.raises(KeyError):
            _validate_columns(["review"], pd.DataFrame())

    def test_multiple_valid_columns_passes(self):
        df = pd.DataFrame({"review": ["a"], "product": ["b"]})
        _validate_columns(["review", "product"], df)

    def test_duplicate_columns_passes(self):
        df = pd.DataFrame({"review": ["a"]})
        _validate_columns(["review", "review"], df)


class TestBuildContext:
    def test_single_column(self):
        row = pd.Series({"review": "Great!", "product": "Chair"})
        assert _build_context(["review"], row) == {"review": "Great!"}

    def test_multiple_columns(self):
        row = pd.Series({"review": "Great!", "product": "Chair"})
        assert _build_context(["review", "product"], row) == {"review": "Great!", "product": "Chair"}

    def test_empty_columns(self):
        row = pd.Series({"review": "Great!"})
        assert _build_context([], row) == {}

    def test_only_specified_columns_included(self):
        row = pd.Series({"review": "Great!", "product": "Chair", "price": "99"})
        result = _build_context(["review"], row)
        assert "product" not in result
        assert "price" not in result

    def test_duplicate_columns_deduplicated(self):
        row = pd.Series({"review": "Great!"})
        assert _build_context(["review", "review"], row) == {"review": "Great!"}


class TestBuildContextValueTypes:
    def test_string_value(self):
        row = pd.Series({"col": "hello"})
        assert _build_context(["col"], row) == {"col": "hello"}

    def test_integer_value(self):
        row = pd.Series({"col": 42})
        assert _build_context(["col"], row) == {"col": 42}

    def test_float_value(self):
        row = pd.Series({"col": 3.14})
        assert _build_context(["col"], row) == {"col": 3.14}

    def test_none_value(self):
        row = pd.Series({"col": None})
        result = _build_context(["col"], row)
        assert result["col"] is None or pd.isna(result["col"])

    def test_object_value(self):
        obj = {"nested": "dict"}
        row = pd.Series({"col": obj})
        assert _build_context(["col"], row) == {"col": obj}


