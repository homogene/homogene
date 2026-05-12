"""Abstract base class for all homogene processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import pandas as pd

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
        OUTPUT_COL: Column name used to store the model output in the result `pd.DataFrame`.
        result: `pd.DataFrame` with output and error columns, set after `run()` is called. `None` before.
    """

    INDEX_COL = "index"
    ERROR_COL = "error"

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

    @property
    def result(self) -> pd.DataFrame | None:
        """Result `pd.DataFrame` with output and error columns, set after `run()` is called. `None` before."""
        return self._result

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
        self._result = _run_pipeline(
            df=df,
            func=self._run_row,
            output_col=self.OUTPUT_COL,
            error_col=self.ERROR_COL,
            max_workers=max_workers,
            progress=show_progress,
            on_step=on_step,
            step=save_step or 0,
            on_error=on_error,
        )

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
