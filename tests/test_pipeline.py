"""Tests for homogene.pipeline"""

import time

import pandas as pd
import pytest

from homogene.pipeline import _run_pipeline


def make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "text": ["a", "b", "c", "d"],
        "group": ["x", "x", "y", "y"],
    })


def upper_fn(row: pd.Series) -> str:
    return row["text"].upper()


def failing_fn(row: pd.Series) -> str:
    raise ValueError(f"Failed on {row['text']}")


def partial_failing_fn(row: pd.Series) -> str:
    if row["text"] == "b":
        raise ValueError("bad row")
    return row["text"].upper()


OUTPUT_COL = "value"
ERROR_COL = "error"


def rr(df, func, **kwargs):
    """Shorthand that injects default column names for pipeline tests."""
    return _run_pipeline(df, func, output_col=OUTPUT_COL, error_col=ERROR_COL, **kwargs)


class TestRunRows:
    def test_returns_dataframe(self):
        result = rr(make_df(), upper_fn, progress=False)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [OUTPUT_COL, ERROR_COL]

    def test_correct_values(self):
        result = rr(make_df(), upper_fn, progress=False)
        assert list(result[OUTPUT_COL]) == ["A", "B", "C", "D"]

    def test_preserves_index(self):
        df = make_df()
        result = rr(df, upper_fn, progress=False)
        assert list(result.index) == list(df.index)

    def test_filtered_index_alignment(self):
        df = make_df().iloc[[0, 2]]  # index [0, 2] — row 1 dropped
        result = rr(df, upper_fn, progress=False)
        assert list(result.index) == [0, 2]
        assert result.loc[0, OUTPUT_COL] == "A"
        assert result.loc[2, OUTPUT_COL] == "C"

    def test_shuffled_index_alignment(self):
        df = make_df().iloc[[3, 1, 0, 2]]  # index [3, 1, 0, 2]
        result = rr(df, upper_fn, progress=False)
        assert list(result.index) == [3, 1, 0, 2]
        assert result.loc[0, OUTPUT_COL] == "A"
        assert result.loc[1, OUTPUT_COL] == "B"
        assert result.loc[3, OUTPUT_COL] == "D"

    def test_single_worker(self):
        result = rr(make_df(), upper_fn, max_workers=1, progress=False)
        assert list(result[OUTPUT_COL]) == ["A", "B", "C", "D"]

    def test_raises_on_failure(self):
        with pytest.raises(ValueError):
            rr(make_df(), failing_fn, on_error="raise", progress=False)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"text": []})
        result = rr(df, lambda row: row["text"], progress=False)
        assert len(result) == 0
        assert list(result.columns) == [OUTPUT_COL, ERROR_COL]

    def test_custom_column_names(self):
        result = _run_pipeline(make_df(), upper_fn, output_col="generation", error_col="error", progress=False)
        assert list(result.columns) == ["generation", "error"]


class TestRunRowsErrorHandling:
    def test_none_mode_stores_none_for_failed_rows(self):
        result = rr(make_df(), partial_failing_fn, on_error="none", progress=False)
        failing_idx = make_df()[make_df()["text"] == "b"].index[0]
        assert pd.isna(result.loc[failing_idx, OUTPUT_COL])

    def test_none_mode_stores_error_message(self):
        result = rr(make_df(), partial_failing_fn, on_error="none", progress=False)
        failing_idx = make_df()[make_df()["text"] == "b"].index[0]
        assert result.loc[failing_idx, ERROR_COL] == "bad row"

    def test_none_mode_successful_rows_unaffected(self):
        result = rr(make_df(), partial_failing_fn, on_error="none", progress=False)
        successful = result[result[ERROR_COL].isna()]
        assert list(successful[OUTPUT_COL]) == ["A", "C", "D"]
        assert successful[ERROR_COL].isna().all()

    def test_raise_mode_raises(self):
        with pytest.raises(ValueError, match="bad row"):
            rr(make_df(), partial_failing_fn, on_error="raise", progress=False)

    def test_successful_rows_have_none_error(self):
        result = rr(make_df(), upper_fn, progress=False)
        assert result[ERROR_COL].isna().all()

    def test_invalid_on_error_raises(self):
        with pytest.raises(ValueError, match="'on_error'"):
            rr(make_df(), upper_fn, on_error="invalid", progress=False)

    def test_none_mode_all_rows_fail(self):
        result = rr(make_df(), failing_fn, on_error="none", progress=False)
        assert result[OUTPUT_COL].isna().all()
        assert result[ERROR_COL].notna().all()


class TestRunRowsParallelism:
    def test_runs_faster_than_sequential(self):
        df = pd.DataFrame({"x": range(20)})

        def slow_fn(row):
            time.sleep(0.3)
            return row["x"]

        start = time.time()
        rr(df, slow_fn, max_workers=20, progress=False)
        elapsed = time.time() - start

        # 20 tasks × 0.3s sequential = 6.0s; parallel should finish in ~0.3s
        assert elapsed < 0.8, f"Expected parallel execution but took {elapsed:.2f}s"

    def test_single_worker_is_sequential(self):
        df = pd.DataFrame({"x": range(8)})

        def slow_fn(row):
            time.sleep(0.3)
            return row["x"]

        start = time.time()
        rr(df, slow_fn, max_workers=1, progress=False)
        elapsed = time.time() - start

        # 8 tasks × 0.3s = 2.4s minimum with 1 worker
        assert elapsed >= 2.0, f"Expected sequential execution but took {elapsed:.2f}s"

    def test_speedup_scales_with_workers(self):
        df = pd.DataFrame({"x": range(10)})

        def slow_fn(row):
            time.sleep(0.3)
            return row["x"]

        start = time.time()
        rr(df, slow_fn, max_workers=1, progress=False)
        sequential_time = time.time() - start

        start = time.time()
        rr(df, slow_fn, max_workers=10, progress=False)
        parallel_time = time.time() - start

        # parallel should be at least 5x faster than sequential
        assert parallel_time * 5 < sequential_time, (
            f"Expected 5x speedup: sequential={sequential_time:.2f}s, parallel={parallel_time:.2f}s"
        )


class TestRunRowsOnStep:
    def test_on_step_called_at_each_step(self):
        calls = []
        rr(make_df(), upper_fn, max_workers=1, progress=False,
           on_step=lambda partial_df: calls.append(partial_df[OUTPUT_COL].notna().sum()), step=1)
        assert calls == [1, 2, 3, 4]

    def test_on_step_called_at_step_interval(self):
        calls = []
        rr(make_df(), upper_fn, max_workers=1, progress=False,
           on_step=lambda partial_df: calls.append(partial_df[OUTPUT_COL].notna().sum()), step=2)
        assert calls == [2, 4]

    def test_on_step_receives_partial_dataframe(self):
        snapshots = []
        rr(make_df(), upper_fn, max_workers=1, progress=False,
           on_step=lambda partial_df: snapshots.append(partial_df.copy()), step=1)
        assert all(isinstance(s, pd.DataFrame) for s in snapshots)
        assert all(list(s.columns) == [OUTPUT_COL, ERROR_COL] for s in snapshots)

    def test_on_step_not_called_when_step_is_zero(self):
        calls = []
        rr(make_df(), upper_fn, max_workers=1, progress=False,
           on_step=lambda partial: calls.append(1), step=0)
        assert calls == []

    def test_on_step_none_does_not_raise(self):
        result = rr(make_df(), upper_fn, max_workers=1, progress=False, on_step=None, step=1)
        assert len(result) == 4

    def test_on_step_partial_df_index_matches_source(self):
        df = make_df()
        snapshots = []
        rr(df, upper_fn, max_workers=1, progress=False,
           on_step=lambda partial_df: snapshots.append(partial_df.index.tolist()), step=1)
        for snapshot in snapshots:
            assert all(idx in df.index for idx in snapshot)

    def test_on_step_not_called_when_step_exceeds_row_count(self):
        calls = []
        rr(make_df(), upper_fn, max_workers=1, progress=False,
           on_step=lambda partial_df: calls.append(1), step=10)
        assert calls == []
