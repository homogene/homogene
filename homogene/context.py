"""Column validation and context building."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _validate_columns(columns: list[str], df: pd.DataFrame) -> None:
    """Raise `KeyError` if any column in `columns` is missing from `df`.

    Args:
        columns: Column names to check.
        df: `pd.DataFrame` whose columns are checked.

    Raises:
        KeyError: If any column in `columns` is not found in `df`.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )


def _build_context(columns: list[str], row: pd.Series) -> dict[str, Any]:
    """Build a context `dict` from specified columns and a single row.

    Args:
        columns: Column names to include.
        row: A single `pd.DataFrame` row as a `pd.Series`.

    Returns:
        `dict` mapping each column name to its value for this row.
    """
    return {col: row[col] for col in columns}


