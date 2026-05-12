"""Parallel execution engine for column processing processors."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm


def _run_pipeline(
    df: pd.DataFrame,
    func: Callable[[pd.Series], Any],
    output_col: str,
    error_col: str,
    max_workers: int | None = None,
    progress: bool = True,
    on_step: Callable[[pd.DataFrame], None] | None = None,
    step: int = 0,
    on_error: Literal["none", "raise"] = "none",
) -> pd.DataFrame:
    """Apply `func` to every row in `df` in parallel, returning a `pd.DataFrame` of results.

    Args:
        df: Source `pd.DataFrame`.
        func: Callable that receives a `pd.Series` (one row) and returns a value.
        output_col: Column name for successful results.
        error_col: Column name for error messages.
        max_workers: Maximum number of parallel threads. `None` uses `ThreadPoolExecutor`'s default.
        progress: Whether to show a progress bar.
        on_step: Callback called with partial results every `step` completions.
        step: Interval (in completed rows) at which `on_step` is called.
        on_error: `"none"` stores `None` for failed rows and continues;
            `"raise"` re-raises the first exception.

    Returns:
        A `pd.DataFrame` with `output_col` and `error_col` columns, aligned to `df`'s index.

    Raises:
        ValueError: If `on_error` is not `"none"` or `"raise"`.
    """
    if on_error not in ("none", "raise"):
        raise ValueError(f"'on_error' must be 'none' or 'raise', got '{on_error}'.")

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks before collecting any result
        futures = {executor.submit(func, row): idx for idx, row in df.iterrows()}

        # Collect results as futures complete
        with tqdm(total=len(futures), disable=not progress) as bar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = (future.result(), None)
                except Exception as e:
                    if on_error == "raise":
                        raise
                    results[idx] = (None, str(e))
                bar.update(1)
                if on_step is not None and step > 0 and len(results) % step == 0:
                    partial_gens = pd.Series(
                        {i: v[0] for i, v in results.items()}
                    ).reindex(df.index)
                    partial_errs = pd.Series(
                        {i: v[1] for i, v in results.items()}
                    ).reindex(df.index)
                    partial_df = pd.DataFrame({output_col: partial_gens, error_col: partial_errs})
                    on_step(partial_df)

    values = {idx: v[0] for idx, v in results.items()}
    errors = {idx: v[1] for idx, v in results.items()}

    return pd.DataFrame({
        # Use 'reindex' to restore original row order after non-deterministic thread completion
        output_col: pd.Series(values).reindex(df.index),
        error_col:  pd.Series(errors).reindex(df.index),
    })
