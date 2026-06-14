"""Abstract base class for all homogene processors."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Literal

import pandas as pd

from homogene.estimator import StatsEstimator
from homogene.models.generative_model import GenerativeModel
from homogene.pipeline import _run_pipeline


class Processor(ABC):
    """Abstract base class for all homogene processors.

    Subclasses must implement `_run_row()` and define `OUTPUT_COL`.

    Args:
        df: Source `pd.DataFrame`.
        model: `GenerativeModel` instance.

    Attributes:
        INDEX_COL: Column name used to store the `pd.DataFrame` index when saving to CSV.
        ERROR_COL: Column name used to store error messages in the result `pd.DataFrame`.
        DURATION_COL: Column name used to store per-row execution duration in the result `pd.DataFrame`.
        COST_COL: Column name used to store per-row cost in USD in the result `pd.DataFrame`.
        OUTPUT_COL: Column name used to store the model output in the result `pd.DataFrame`.
        result: `pd.DataFrame` with output, error, and duration columns, set after `run()` is called. `None` before.
        stats: Aggregated run stats, set after `run()` is called. `None` before.
    """

    INDEX_COL = "index"
    ERROR_COL = "error"
    DURATION_COL = "duration"
    COST_COL = "cost"
    INPUT_TOKENS_COL = "input_tokens"
    OUTPUT_TOKENS_COL = "output_tokens"

    @property
    @abstractmethod
    def OUTPUT_COL(self) -> str:
        """Column name used to store the model output in the result `pd.DataFrame`."""
        ...

    def __init__(
        self,
        df: pd.DataFrame,
        model: GenerativeModel,
    ) -> None:
        self.df = df
        self.model = model
        self._result: pd.DataFrame | None = None
        self._total_duration: float | None = None

    @property
    def result(self) -> pd.DataFrame | None:
        """Result `pd.DataFrame` with output, error, and duration columns, set after `run()` is called. `None` before."""
        return self._result

    @property
    def stats(self) -> dict | None:
        """Aggregated run stats, set after `run()` is called. `None` before.

        Keys:
            total_succeeded: Number of rows that completed without error.
            total_failed: Number of rows that raised an error.
            total_duration: Wall-clock time of the full run, in seconds.
            total_cost: Total cost of the run in USD, or `None` if unavailable.
            total_input_tokens: Total prompt tokens, or `None` if unavailable.
            total_output_tokens: Total completion tokens, or `None` if unavailable.
        """
        if self._result is None:
            return None
        costs = self._result[self.COST_COL].dropna()
        input_tokens = self._result[self.INPUT_TOKENS_COL].dropna()
        output_tokens = self._result[self.OUTPUT_TOKENS_COL].dropna()
        return {
            "total_succeeded": int(self._result[self.ERROR_COL].isna().sum()),
            "total_failed": int(self._result[self.ERROR_COL].notna().sum()),
            "total_duration": self._total_duration,
            "total_cost": float(costs.sum()) if len(costs) > 0 else None,
            "total_input_tokens": int(input_tokens.sum()) if len(input_tokens) > 0 else None,
            "total_output_tokens": int(output_tokens.sum()) if len(output_tokens) > 0 else None,
        }

    def _get_prompt_length(self, row: pd.Series) -> int:
        """Return the prompt length in characters for a row.

        Override in subclasses for accurate estimation. Default sums the character
        lengths of all column values in the row.

        Args:
            row: A row from the source `pd.DataFrame`, provided as a `pd.Series`.

        Returns:
            Prompt length in characters.
        """
        return sum(len(str(v)) for v in row)

    def estimate_stats(self, max_workers: int = 1) -> dict:
        """Estimate stats for the full DataFrame based on the current sample run.

        Uses prompt length in characters as the scaling variable for cost and tokens
        (rule of 3), and mean per-row duration to estimate wall-clock time.

        Args:
            max_workers: Number of parallel workers to assume for duration estimation.

        Returns:
            Dict with the same keys as `stats`, estimated for the full `df`.

        Raises:
            RuntimeError: If called before `run()`.
            ValueError: If `max_workers` is less than 1.
        """
        if self._result is None:
            raise RuntimeError("Call 'run()' before 'estimate_stats()'.")
        if max_workers < 1:
            raise ValueError(f"'max_workers' must be a positive integer, got {max_workers}.")

        n_sample = len(self._result)
        sample_df = self.df.head(n_sample)
        sample_prompt_lengths = [self._get_prompt_length(row) for _, row in sample_df.iterrows()]
        full_prompt_lengths = [self._get_prompt_length(row) for _, row in self.df.iterrows()]
        sample_durations = self._result[self.DURATION_COL].dropna().tolist()

        return StatsEstimator(
            sample_stats=self.stats,
            sample_durations=sample_durations,
            sample_prompt_lengths=sample_prompt_lengths,
            full_prompt_lengths=full_prompt_lengths,
        ).estimate(max_workers=max_workers)

    @abstractmethod
    def _run_row(self, row: pd.Series) -> Any:
        """Process a single row and return the model output.

        Args:
            row: A row from the source `pd.DataFrame`, provided as a `pd.Series`.

        Returns:
            The model output for this row. Type depends on the subclass implementation.
        """
        ...

    def run(
        self,
        n: int | None = None,
        max_workers: int | None = None,
        show_progress: bool = True,
        on_error: Literal["none", "raise"] = "none",
        save_path: str | None = None,
        save_step: int | None = None,
    ) -> pd.Series:
        """Run on the full `pd.DataFrame`, or only the first `n` rows if specified.

        Args:
            n: Number of rows to process, starting from the first. Processes all rows if `None`.
            max_workers: Maximum number of parallel threads. `None` uses `ThreadPoolExecutor`'s default.
            show_progress: Whether to display a `tqdm` progress bar.
            on_error: `"none"` stores `None` for failed rows and continues;
                `"raise"` re-raises the first exception.
            save_path: File path for saving results as CSV. Saves incrementally if `save_step` is set,
                and always after run completes.
            save_step: Checkpoint interval in number of completed rows. Requires `save_path`.

        Returns:
            `pd.Series` of output values aligned to the input `pd.DataFrame`'s index.

        Raises:
            ValueError: If any of the following conditions are met:

                - `n` is negative.
                - `max_workers` is not `None` and is less than 1.
                - `on_error` is not a valid value.
                - `save_path` is set and does not end with `".csv"`.
                - `save_step` is set without `save_path`.
                - `save_step` is less than 1.
        """
        # Validate arguments
        if n is not None and n < 0:
            raise ValueError(f"'n' must be a non-negative integer, got {n}.")
        if max_workers is not None and max_workers < 1:
            raise ValueError(f"'max_workers' must be a positive integer, got {max_workers}.")
        if on_error not in ("none", "raise"):
            raise ValueError(f"'on_error' must be 'none' or 'raise', got '{on_error}'.")
        if save_path is not None and not save_path.endswith(".csv"):
            raise ValueError(f"Unsupported file format. Only '.csv' is supported, got '{save_path}'.")
        if save_step is not None and save_path is None:
            raise ValueError("'save_step' requires 'save_path' to be set.")
        if save_step is not None and save_step < 1:
            raise ValueError(f"'save_step' must be a positive integer, got {save_step}.")

        # Subset to the first n rows if specified
        df = self.df.head(n) if n is not None else self.df

        # Build a checkpoint callback to save progress incrementally
        on_step = None
        if save_path and save_step:
            def on_step(partial_df: pd.DataFrame) -> None:
                self._save(partial_df, save_path)

        # Process all rows in parallel
        t0 = time.perf_counter()
        self._result = _run_pipeline(
            df=df,
            func=self._run_row,
            output_col=self.OUTPUT_COL,
            error_col=self.ERROR_COL,
            duration_col=self.DURATION_COL,
            cost_col=self.COST_COL,
            input_tokens_col=self.INPUT_TOKENS_COL,
            output_tokens_col=self.OUTPUT_TOKENS_COL,
            max_workers=max_workers,
            progress=show_progress,
            on_step=on_step,
            step=save_step or 0,
            on_error=on_error,
        )

        self._total_duration = time.perf_counter() - t0

        # Save the final result to disk
        if save_path:
            self._save(self._result, save_path)

        # Print a run summary
        print(self._get_summary(self._result))

        return self._result[self.OUTPUT_COL]

    def _get_summary(self, result: pd.DataFrame) -> str:
        """Return a human-readable run summary string.

        Args:
            result: The result `pd.DataFrame` produced by `run()`.

        Returns:
            A run summary string.
        """
        total = len(result)
        succeeded = result[self.ERROR_COL].isna().sum()
        pct = f" ({succeeded/total:.0%})" if total > 0 else ""
        return f"{succeeded}/{total} rows succeeded{pct}"

    def _save(self, df: pd.DataFrame, path: str) -> None:
        """Save a `pd.DataFrame` to CSV with the index column named `INDEX_COL`.

        Args:
            df: `pd.DataFrame` to save.
            path: Destination file path. Must end with `".csv"`.

        Raises:
            ValueError: If `path` does not end with `".csv"`.
        """
        if not path.endswith(".csv"):
            raise ValueError(f"Unsupported file format. Only '.csv' is supported, got '{path}'.")
        df.rename_axis(self.INDEX_COL).to_csv(path, index=True)
