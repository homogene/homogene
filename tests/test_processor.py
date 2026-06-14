"""Tests for homogene.processors.processor"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from homogene.processors.processor import Processor


def make_df() -> pd.DataFrame:
    return pd.DataFrame({"review": ["Great!", "Terrible.", "Okay."]})


class ConcreteProcessor(Processor):
    OUTPUT_COL = "value"

    def _run_row(self, row: pd.Series):
        return row.iloc[0], None, None, None


def make_processor(**kwargs) -> ConcreteProcessor:
    defaults = dict(df=make_df(), model=MagicMock())
    return ConcreteProcessor(**{**defaults, **kwargs})


class TestProcessorConstants:
    def test_error_col_value(self):
        assert Processor.ERROR_COL == "error"

    def test_index_col_value(self):
        assert Processor.INDEX_COL == "index"

    def test_output_col_value(self):
        assert ConcreteProcessor.OUTPUT_COL == "value"

    def test_cost_col_value(self):
        assert Processor.COST_COL == "cost"

    def test_input_tokens_col_value(self):
        assert Processor.INPUT_TOKENS_COL == "input_tokens"

    def test_output_tokens_col_value(self):
        assert Processor.OUTPUT_TOKENS_COL == "output_tokens"


class TestProcessorInterface:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            Processor(df=make_df(), model=MagicMock())

    def test_run_row_must_be_defined(self):
        class ProcessorWithoutRunRow(Processor):
            OUTPUT_COL = "value"

        with pytest.raises(TypeError):
            ProcessorWithoutRunRow(df=make_df(), model=MagicMock())

    def test_output_col_must_be_defined(self):
        class ProcessorWithoutOutputCol(Processor):
            def _run_row(self, row: pd.Series):
                return row.iloc[0]

        with pytest.raises(TypeError):
            ProcessorWithoutOutputCol(df=make_df(), model=MagicMock())

    def test_valid_subclass_can_be_instantiated(self):
        p = make_processor()
        assert isinstance(p, Processor)


class TestProcessorInit:
    def test_stores_df(self):
        df = make_df()
        p = ConcreteProcessor(df=df, model=MagicMock())
        assert p.df is df

    def test_stores_model(self):
        model = MagicMock()
        p = make_processor(model=model)
        assert p.model is model

    def test_result_is_none_before_run(self):
        p = make_processor()
        assert p.result is None

    def test_result_is_dataframe_after_run(self):
        p = make_processor()
        p.run()
        assert isinstance(p.result, pd.DataFrame)


class TestRunValidation:
    def test_negative_n_raises(self):
        p = make_processor()
        with pytest.raises(ValueError, match="'n'"):
            p.run(n=-1)

    def test_zero_max_workers_raises(self):
        p = make_processor()
        with pytest.raises(ValueError, match="'max_workers'"):
            p.run(max_workers=0)

    def test_negative_max_workers_raises(self):
        p = make_processor()
        with pytest.raises(ValueError, match="'max_workers'"):
            p.run(max_workers=-1)

    def test_save_step_without_save_path_raises(self):
        p = make_processor()
        with pytest.raises(ValueError, match="'save_step'"):
            p.run(save_step=1)

    def test_invalid_save_step_raises(self):
        p = make_processor()
        with pytest.raises(ValueError, match="'save_step'"):
            p.run(save_path="out.csv", save_step=0)

    def test_invalid_on_error_raises(self):
        p = make_processor()
        with pytest.raises(ValueError, match="'on_error'"):
            p.run(on_error="invalid")

    def test_non_csv_save_path_raises(self):
        p = make_processor()
        with pytest.raises(ValueError, match="'.csv'"):
            p.run(save_path="out.txt")

    def test_valid_on_error_none_accepted(self):
        p = make_processor()
        p.run(on_error="none")

    def test_valid_on_error_raise_accepted(self):
        p = make_processor()
        p.run(on_error="raise")


class TestRun:
    def test_returns_series(self):
        p = make_processor()
        assert isinstance(p.run(), pd.Series)

    def test_returns_output_col(self):
        p = make_processor()
        assert p.run().name == ConcreteProcessor.OUTPUT_COL

    def test_calls_run_row_for_each_row(self):
        calls = []
        p = make_processor()
        original = p._run_row
        p._run_row = lambda row: calls.append(1) or original(row)
        p.run()
        assert len(calls) == 3

    def test_default_n_processes_all_rows(self):
        df = pd.DataFrame({"review": [str(i) for i in range(10)]})
        p = make_processor(df=df)
        assert len(p.run()) == 10

    def test_n_limits_rows(self):
        p = make_processor()
        assert len(p.run(n=2)) == 2

    def test_n_zero_returns_empty(self):
        p = make_processor()
        assert len(p.run(n=0)) == 0

    def test_n_greater_than_df_length_processes_all(self):
        p = make_processor()
        assert len(p.run(n=100)) == 3

    def test_run_with_n_stores_result(self):
        p = make_processor()
        p.run(n=2)
        assert len(p.result) == 2

    def test_result_has_correct_columns(self):
        p = make_processor()
        p.run()
        assert list(p.result.columns) == [ConcreteProcessor.OUTPUT_COL, Processor.ERROR_COL, Processor.DURATION_COL, Processor.COST_COL, Processor.INPUT_TOKENS_COL, Processor.OUTPUT_TOKENS_COL]

    def test_result_index_matches_df(self):
        df = pd.DataFrame({"review": ["a", "b", "c"]}, index=[10, 20, 30])
        p = ConcreteProcessor(df=df, model=MagicMock())
        p.run()
        assert list(p.result.index) == [10, 20, 30]

    def test_original_df_not_modified(self):
        df = make_df()
        p = ConcreteProcessor(df=df, model=MagicMock())
        p.run()
        assert ConcreteProcessor.OUTPUT_COL not in df.columns

    def test_successful_rows_have_none_error(self):
        p = make_processor()
        p.run()
        assert p.result[Processor.ERROR_COL].isna().all()

    def test_error_column_populated_on_failure(self):
        p = make_processor()
        p._run_row = lambda _: (_ for _ in ()).throw(ValueError("fail"))
        p.run()
        assert p.result[Processor.ERROR_COL].notna().all()

    def test_on_error_raise_propagates(self):
        p = make_processor()
        p._run_row = lambda _: (_ for _ in ()).throw(ValueError("fail"))
        with pytest.raises(ValueError, match="fail"):
            p.run(on_error="raise")


class TestProcessorDuration:
    def test_duration_column_present(self):
        p = make_processor()
        p.run()
        assert Processor.DURATION_COL in p.result.columns

    def test_duration_values_are_floats(self):
        p = make_processor()
        p.run()
        assert p.result[Processor.DURATION_COL].dtype == float

    def test_duration_values_are_positive(self):
        p = make_processor()
        p.run()
        assert (p.result[Processor.DURATION_COL] >= 0).all()

    def test_duration_is_none_for_failed_rows(self):
        p = make_processor()
        p._run_row = lambda _: (_ for _ in ()).throw(ValueError("fail"))
        p.run()
        assert p.result[Processor.DURATION_COL].isna().all()


class TestProcessorStats:
    def test_stats_is_none_before_run(self):
        p = make_processor()
        assert p.stats is None

    def test_stats_is_dict_after_run(self):
        p = make_processor()
        p.run()
        assert isinstance(p.stats, dict)

    def test_stats_has_total_succeeded(self):
        p = make_processor()
        p.run()
        assert "total_succeeded" in p.stats

    def test_stats_has_total_failed(self):
        p = make_processor()
        p.run()
        assert "total_failed" in p.stats

    def test_total_succeeded_correct(self):
        p = make_processor()
        p.run()
        assert p.stats["total_succeeded"] == 3

    def test_total_failed_correct(self):
        p = make_processor()
        p.run()
        assert p.stats["total_failed"] == 0

    def test_total_failed_correct_on_error(self):
        p = make_processor()
        p._run_row = lambda _: (_ for _ in ()).throw(ValueError("fail"))
        p.run()
        assert p.stats["total_failed"] == 3
        assert p.stats["total_succeeded"] == 0

    def test_stats_has_total_duration(self):
        p = make_processor()
        p.run()
        assert "total_duration" in p.stats

    def test_total_duration_is_positive(self):
        p = make_processor()
        p.run()
        assert p.stats["total_duration"] >= 0

    def test_stats_has_total_cost(self):
        p = make_processor()
        p.run()
        assert "total_cost" in p.stats

    def test_total_cost_is_none_when_no_cost_returned(self):
        p = make_processor()
        p.run()
        assert p.stats["total_cost"] is None

    def test_stats_has_total_input_tokens(self):
        p = make_processor()
        p.run()
        assert "total_input_tokens" in p.stats

    def test_stats_has_total_output_tokens(self):
        p = make_processor()
        p.run()
        assert "total_output_tokens" in p.stats

    def test_total_input_tokens_is_none_when_not_returned(self):
        p = make_processor()
        p.run()
        assert p.stats["total_input_tokens"] is None

    def test_total_output_tokens_is_none_when_not_returned(self):
        p = make_processor()
        p.run()
        assert p.stats["total_output_tokens"] is None


class TestEstimateStats:
    def test_raises_before_run(self):
        p = make_processor()
        with pytest.raises(RuntimeError):
            p.estimate_stats()

    def test_returns_dict(self):
        p = make_processor()
        p.run(n=2)
        assert isinstance(p.estimate_stats(), dict)

    def test_has_all_keys(self):
        p = make_processor()
        p.run(n=2)
        result = p.estimate_stats()
        assert set(result.keys()) == {"total_succeeded", "total_failed", "total_duration", "total_cost", "total_input_tokens", "total_output_tokens"}

    def test_full_run_matches_stats(self):
        p = make_processor()
        p.run()
        result = p.estimate_stats()
        assert result["total_succeeded"] == p.stats["total_succeeded"]
        assert result["total_failed"] == p.stats["total_failed"]

    def test_invalid_max_workers_raises(self):
        p = make_processor()
        p.run(n=2)
        with pytest.raises(ValueError, match="'max_workers'"):
            p.estimate_stats(max_workers=0)


class TestGetSummary:
    def test_all_succeeded(self):
        p = make_processor()
        df = pd.DataFrame({"value": ["a", "b"], "error": [None, None]})
        assert p._get_summary(df) == "2/2 rows succeeded (100%)"

    def test_partial_success(self):
        p = make_processor()
        df = pd.DataFrame({"value": ["a", None], "error": [None, "err"]})
        assert p._get_summary(df) == "1/2 rows succeeded (50%)"

    def test_all_failed(self):
        p = make_processor()
        df = pd.DataFrame({"value": [None, None], "error": ["err", "err"]})
        assert p._get_summary(df) == "0/2 rows succeeded (0%)"

    def test_empty(self):
        p = make_processor()
        df = pd.DataFrame({"value": [], "error": []})
        assert p._get_summary(df) == "0/0 rows succeeded"


class TestRunSummary:
    def test_prints_summary_on_success(self, capsys):
        p = make_processor()
        p.run()
        assert "3/3 rows succeeded (100%)" in capsys.readouterr().out

    def test_prints_summary_with_errors(self, capsys):
        def fail_first(row):
            if row.name == 0:
                raise ValueError("boom")
            return row.iloc[0], None, None, None

        p = ConcreteProcessor(df=make_df(), model=MagicMock())
        p._run_row = fail_first
        p.run()
        assert "2/3 rows succeeded (67%)" in capsys.readouterr().out

    def test_prints_summary_with_n_zero(self, capsys):
        p = make_processor()
        p.run(n=0)
        assert "0/0 rows succeeded" in capsys.readouterr().out


class TestSave:
    def test_csv_saves_file(self, tmp_path):
        path = str(tmp_path / "out.csv")
        p = make_processor()
        p._save(pd.DataFrame({"value": ["a"]}), path)
        assert (tmp_path / "out.csv").exists()

    def test_csv_content_matches_input(self, tmp_path):
        path = str(tmp_path / "out.csv")
        p = make_processor()
        p._save(pd.DataFrame({"value": ["a", "b"]}), path)
        restored = pd.read_csv(path, index_col=Processor.INDEX_COL)
        assert list(restored["value"]) == ["a", "b"]

    def test_csv_index_preserved(self, tmp_path):
        path = str(tmp_path / "out.csv")
        p = make_processor()
        p._save(pd.DataFrame({"value": ["a", "b"]}, index=[3, 7]), path)
        restored = pd.read_csv(path, index_col=Processor.INDEX_COL)
        assert list(restored.index) == [3, 7]

    def test_non_csv_raises(self, tmp_path):
        path = str(tmp_path / "out.jsonl")
        p = make_processor()
        with pytest.raises(ValueError, match="'.csv'"):
            p._save(pd.DataFrame({"value": ["a"]}), path)

    def test_save_via_run(self, tmp_path):
        path = str(tmp_path / "results.csv")
        p = make_processor()
        p.run(save_path=path)
        assert (tmp_path / "results.csv").exists()

    def test_save_step_writes_incrementally(self, tmp_path):
        path = str(tmp_path / "results.csv")
        save_calls = []

        p = make_processor()
        original_save = p._save
        p._save = lambda df, p_: save_calls.append(len(df)) or original_save(df, p_)

        p.run(save_path=path, save_step=1, max_workers=1)

        # 3 rows with step=1 → 3 incremental saves + 1 final save
        assert len(save_calls) == 4
