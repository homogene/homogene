"""Parallel execution engine for column processing processors."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm


def _run_pipeline(
    df: pd.DataFrame,
    func: Callable[[pd.Series], tuple[Any, float | None, int | None, int | None]],
    output_col: str,
    error_col: str,
    duration_col: str,
    cost_col: str,
    input_tokens_col: str,
    output_tokens_col: str,
    max_workers: int | None = None,
    progress: bool = True,
    on_step: Callable[[pd.DataFrame], None] | None = None,
    step: int = 0,
    on_error: Literal["none", "raise"] = "none",
) -> pd.DataFrame:
    """Apply `func` to every row in `df` in parallel, returning a `pd.DataFrame` of results.

    Args:
        df: Source `pd.DataFrame`.
        func: Callable that receives a `pd.Series` (one row) and returns
            `(value, cost, input_tokens, output_tokens)`.
        output_col: Column name for successful results.
        error_col: Column name for error messages.
        duration_col: Column name for per-row execution duration in seconds.
        cost_col: Column name for per-row cost in USD.
        input_tokens_col: Column name for per-row prompt token count.
        output_tokens_col: Column name for per-row completion token count.
        max_workers: Maximum number of parallel threads. `None` uses `ThreadPoolExecutor`'s default.
        progress: Whether to show a progress bar.
        on_step: Callback called with partial results every `step` completions.
        step: Interval (in completed rows) at which `on_step` is called.
        on_error: `"none"` stores `None` for failed rows and continues;
            `"raise"` re-raises the first exception.

    Returns:
        A `pd.DataFrame` with `output_col`, `error_col`, `duration_col`, `cost_col`,
        `input_tokens_col`, and `output_tokens_col` columns, aligned to `df`'s index.

    Raises:
        ValueError: If `on_error` is not `"none"` or `"raise"`.
    """
    if on_error not in ("none", "raise"):
        raise ValueError(f"'on_error' must be 'none' or 'raise', got '{on_error}'.")

    def timed_func(row: pd.Series) -> tuple[Any, float | None, int | None, int | None, float]:
        t0 = time.perf_counter()
        value, cost, input_tokens, output_tokens = func(row)
        return value, cost, input_tokens, output_tokens, time.perf_counter() - t0

    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks before collecting any result
        futures = {executor.submit(timed_func, row): idx for idx, row in df.iterrows()}

        # Collect results as futures complete
        with tqdm(total=len(futures), disable=not progress) as bar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    value, cost, input_tokens, output_tokens, duration = future.result()
                    results[idx] = (value, None, duration, cost, input_tokens, output_tokens)
                except Exception as e:
                    if on_error == "raise":
                        raise
                    results[idx] = (None, str(e), None, None, None, None)
                bar.update(1)
                if on_step is not None and step > 0 and len(results) % step == 0:
                    partial_df = pd.DataFrame({
                        output_col:         pd.Series({i: v[0] for i, v in results.items()}).reindex(df.index),
                        error_col:          pd.Series({i: v[1] for i, v in results.items()}).reindex(df.index),
                        duration_col:       pd.Series({i: v[2] for i, v in results.items()}).reindex(df.index),
                        cost_col:           pd.Series({i: v[3] for i, v in results.items()}).reindex(df.index),
                        input_tokens_col:   pd.Series({i: v[4] for i, v in results.items()}).reindex(df.index),
                        output_tokens_col:  pd.Series({i: v[5] for i, v in results.items()}).reindex(df.index),
                    })
                    on_step(partial_df)

    return pd.DataFrame({
        # Use 'reindex' to restore original row order after non-deterministic thread completion
        output_col:         pd.Series({i: v[0] for i, v in results.items()}).reindex(df.index),
        error_col:          pd.Series({i: v[1] for i, v in results.items()}).reindex(df.index),
        duration_col:       pd.Series({i: v[2] for i, v in results.items()}).reindex(df.index),
        cost_col:           pd.Series({i: v[3] for i, v in results.items()}).reindex(df.index),
        input_tokens_col:   pd.Series({i: v[4] for i, v in results.items()}).reindex(df.index),
        output_tokens_col:  pd.Series({i: v[5] for i, v in results.items()}).reindex(df.index),
    })
