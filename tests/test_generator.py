"""Tests for homogene.processors.generator"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel

from homogene.models.generative_model import GenerativeModel
from homogene.processors.generator import Generator
from homogene.processors.processor import Processor


def make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "review": ["Great!", "Terrible.", "Okay."],
        "product": ["Chair", "Table", "Lamp"],
    })


def make_model(response: str = "positive") -> MagicMock:
    m = MagicMock()
    m._generate.return_value = (response, None, None, None)
    return m


class TestGeneratorConstants:
    def test_output_col_value(self):
        assert Generator.OUTPUT_COL == "generation"


class TestGeneratorRepr:
    def test_repr_is_valid_json(self):
        import json
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model="gpt-5.5")
        parsed = json.loads(repr(generator))
        assert parsed["instruction"] == "Fix the review"
        assert parsed["columns"] == ["review"]
        assert parsed["df"] == "[3 rows x 2 columns]"
    def test_contains_instruction(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model="gpt-5.5")
        assert "Fix the review" in repr(generator)

    def test_contains_columns(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model="gpt-5.5")
        assert "review" in repr(generator)

    def test_contains_model_name(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model="gpt-5.5")
        assert "gpt-5.5" in repr(generator)

    def test_contains_df_shape(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model="gpt-5.5")
        assert "[3 rows x 2 columns]" in repr(generator)

    def test_schema_null_when_not_set(self):
        import json
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model="gpt-5.5")
        assert json.loads(repr(generator))["schema"] is None

    def test_schema_shown_when_set(self):
        from pydantic import BaseModel
        class Sentiment(BaseModel):
            label: str
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model="gpt-5.5", output_schema=Sentiment)
        assert "Sentiment" in repr(generator)

    def test_two_instances_differ(self):
        g1 = Generator(df=make_df(), columns=["review"], instruction="Fix typos", model="gpt-5.5")
        g2 = Generator(df=make_df(), columns=["review"], instruction="Classify sentiment", model="gpt-5.5")
        assert repr(g1) != repr(g2)

    def test_long_instruction_truncated(self):
        long = "A" * 100
        generator = Generator(df=make_df(), columns=["review"], instruction=long, model="gpt-5.5")
        assert "..." in repr(generator)


class TestGeneratorInit:
    def test_valid_init(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        assert generator is not None

    def test_invalid_column_raises(self):
        with pytest.raises(KeyError):
            Generator(df=make_df(), columns=["missing"], instruction="Fix the review", model=make_model())

    def test_string_column_normalized_to_list(self):
        generator = Generator(df=make_df(), columns="review", instruction="Fix the review", model=make_model())
        assert generator.columns == ["review"]

    def test_stores_columns(self):
        generator = Generator(df=make_df(), columns=["review", "product"], instruction="Fix the review", model=make_model())
        assert generator.columns == ["review", "product"]

    def test_stores_instruction(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        assert generator.instruction == "Fix the review"

    def test_string_model_accepted(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model="gpt-5.5")
        assert isinstance(generator.model, GenerativeModel)

    def test_string_model_stores_model_name(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model="gpt-5.5")
        assert generator.model.name == "gpt-5.5"

    def test_one_of_multiple_columns_invalid_raises(self):
        with pytest.raises(KeyError):
            Generator(df=make_df(), columns=["review", "missing"], instruction="Fix the review", model=make_model())


class TestGeneratorRun:
    def test_run_returns_series(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        result = generator.run()
        assert isinstance(result, pd.Series)
        assert result.name == Generator.OUTPUT_COL

    def test_run_correct_length(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        result = generator.run()
        assert len(result) == 3

    def test_run_stores_result(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        assert generator.result is None
        generator.run()
        assert isinstance(generator.result, pd.DataFrame)

    def test_run_with_n_returns_n_rows(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        result = generator.run(n=2)
        assert len(result) == 2

    def test_run_with_n_zero_returns_empty(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        result = generator.run(n=0)
        assert len(result) == 0
        assert result.name == Generator.OUTPUT_COL

    def test_run_with_n_calls_model_n_times(self):
        model = make_model()
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=model)
        generator.run(n=2)
        assert model._generate.call_count == 2

    def test_run_with_n_stores_result(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        generator.run(n=2)
        assert isinstance(generator.result, pd.DataFrame)

    def test_run_calls_model_for_each_row(self):
        model = make_model()
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=model)
        generator.run()
        assert model._generate.call_count == 3

    def test_run_passes_correct_instruction(self):
        captured = []
        model = MagicMock()
        model._generate.side_effect = lambda system_prompt, user_prompt: captured.append(system_prompt) or ("positive", None, None, None)

        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=model)
        generator.run()
        assert all(p == "Fix the review" for p in captured)

    def test_run_passes_correct_context(self):
        captured = []
        model = MagicMock()
        model._generate.side_effect = lambda system_prompt, user_prompt: captured.append(user_prompt) or ("positive", None, None, None)

        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=model)
        generator.run()
        assert '{"review": "Great!"}' in captured

    def test_original_df_not_modified(self):
        df = make_df()
        generator = Generator(df=df, columns=["review"], instruction="Fix the review", model=make_model())
        generator.run()
        assert Generator.OUTPUT_COL not in df.columns

    def test_result_has_correct_columns(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        generator.run()
        assert list(generator.result.columns) == [Generator.OUTPUT_COL, Processor.ERROR_COL, Processor.DURATION_COL, Processor.COST_COL, Processor.INPUT_TOKENS_COL, Processor.OUTPUT_TOKENS_COL]

    def test_result_index_matches_df_index(self):
        df = make_df()
        generator = Generator(df=df, columns=["review"], instruction="Fix the review", model=make_model())
        generator.run()
        assert list(generator.result.index) == list(df.index)

    def test_filtered_df_index_alignment(self):
        df = make_df().iloc[[0, 2]]  # index [0, 2] — row 1 dropped
        generator = Generator(df=df, columns=["review"], instruction="Fix the review", model=make_model("pos"))
        result = generator.run()
        assert list(result.index) == [0, 2]

    def test_shuffled_df_index_alignment(self):
        df = make_df().iloc[[2, 0, 1]]  # index [2, 0, 1]
        generator = Generator(df=df, columns=["review"], instruction="Fix the review", model=make_model("pos"))
        result = generator.run()
        assert list(result.index) == [2, 0, 1]


class TestGeneratorGetPromptLength:
    def test_returns_int(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        assert isinstance(generator._get_prompt_length(make_df().iloc[0]), int)

    def test_length_is_positive(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        assert generator._get_prompt_length(make_df().iloc[0]) > 0

    def test_longer_row_gives_longer_length(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        short = generator._get_prompt_length(pd.Series({"review": "ok", "product": "x"}))
        long = generator._get_prompt_length(pd.Series({"review": "A very long review " * 20, "product": "x"}))
        assert long > short

    def test_includes_instruction(self):
        g1 = Generator(df=make_df(), columns=["review"], instruction="Short", model=make_model())
        g2 = Generator(df=make_df(), columns=["review"], instruction="A much longer instruction " * 10, model=make_model())
        assert g2._get_prompt_length(make_df().iloc[0]) > g1._get_prompt_length(make_df().iloc[0])


class TestGeneratorGetPrompt:
    def test_default_row_is_first(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        assert "Great!" in generator.get_prompt().user

    def test_returns_prompt(self):
        from homogene.prompt import Prompt
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        assert isinstance(generator.get_prompt(0), Prompt)

    def test_system_message(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        assert generator.get_prompt(0).system == "Fix the review"

    def test_user_message(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        assert "Great!" in generator.get_prompt(0).user

    def test_correct_row(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        assert "Terrible." in generator.get_prompt(1).user

    def test_does_not_call_model(self):
        model = make_model()
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=model)
        generator.get_prompt(0)
        model._generate.assert_not_called()

    def test_multiple_columns_in_prompt(self):
        generator = Generator(df=make_df(), columns=["review", "product"], instruction="Fix the review", model=make_model())
        user = generator.get_prompt(0).user
        assert "Great!" in user
        assert "Chair" in user

    def test_out_of_bounds_raises(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        with pytest.raises(IndexError):
            generator.get_prompt(100)

    def test_empty_columns_produces_empty_context(self):
        generator = Generator(df=make_df(), columns=[], instruction="Fix the review", model=make_model())
        assert generator.get_prompt(0).user == "{}"



class TestGeneratorRunValidation:
    def test_invalid_max_workers_raises(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        with pytest.raises(ValueError, match="'max_workers'"):
            generator.run(max_workers=0)

    def test_negative_max_workers_raises(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        with pytest.raises(ValueError, match="'max_workers'"):
            generator.run(max_workers=-1)

    def test_negative_n_raises(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        with pytest.raises(ValueError, match="'n'"):
            generator.run(n=-1)

    def test_invalid_save_step_raises(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        with pytest.raises(ValueError, match="'save_step'"):
            generator.run(save_path="out.csv", save_step=0)

    def test_invalid_on_error_raises(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        with pytest.raises(ValueError, match="'on_error'"):
            generator.run(on_error="invalid")

    def test_non_csv_save_path_raises(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        with pytest.raises(ValueError, match="'.csv'"):
            generator.run(save_path="out.txt")

    def test_valid_on_error_none_accepted(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        generator.run(on_error="none")

    def test_valid_on_error_raise_accepted(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        generator.run(on_error="raise")



class TestGeneratorSave:
    def test_save_step_without_save_path_raises(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        with pytest.raises(ValueError):
            generator.run(save_step=2)

    def test_saves_file_on_run(self, tmp_path):
        path = tmp_path / "results.csv"
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        generator.run(save_path=str(path))
        assert path.exists()

    def test_saves_file_on_run_with_n(self, tmp_path):
        path = tmp_path / "results.csv"
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        generator.run(n=2, save_path=str(path))
        assert path.exists()

    def test_save_content_matches_result(self, tmp_path):
        path = tmp_path / "results.csv"
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model("pos"))
        result = generator.run(save_path=str(path))
        saved = pd.read_csv(path, index_col=Generator.INDEX_COL)
        assert list(saved[Generator.OUTPUT_COL]) == list(result)
        assert Processor.ERROR_COL in saved.columns

    def test_no_file_without_save_path(self, tmp_path):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        generator.run()

    def test_checkpoint_write_count(self, tmp_path):
        path = tmp_path / "results.csv"
        write_count = []
        with patch.object(pd.DataFrame, "to_csv", lambda *a, **kw: write_count.append(1)):
            generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
            generator.run(save_path=str(path), save_step=1)
        # 3 rows with step=1: 3 checkpoint saves + 1 final save
        assert len(write_count) == 4

    def test_checkpoint_respects_step_size(self, tmp_path):
        path = tmp_path / "results.csv"
        write_count = []
        with patch.object(pd.DataFrame, "to_csv", lambda *a, **kw: write_count.append(1)):
            generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
            generator.run(save_path=str(path), save_step=2)
        # 3 rows with step=2: checkpoint at row 2 only + 1 final save
        assert len(write_count) == 2


class Sentiment(BaseModel):
    label: str


def make_schema_model() -> GenerativeModel:
    model = GenerativeModel("gpt-5.5")
    model._generate_structured = MagicMock(return_value=(Sentiment(label="positive"), None, None, None))
    return model


class TestGeneratorOutputSchema:
    def test_output_schema_stored(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model(), output_schema=Sentiment)
        assert generator.output_schema is Sentiment

    def test_none_by_default(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        assert generator.output_schema is None

    def test_non_basemodel_raises(self):
        class NotAModel:
            pass
        with pytest.raises(TypeError, match="'output_schema'"):
            Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model(), output_schema=NotAModel)

    def test_non_class_raises(self):
        with pytest.raises((TypeError, Exception)):
            Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model(), output_schema="not_a_class")

    def test_generate_structured_called_per_row(self):
        model = make_schema_model()
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=model, output_schema=Sentiment)
        generator.run()
        assert model._generate_structured.call_count == 3

    def test_returns_typed_instances(self):
        model = make_schema_model()
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=model, output_schema=Sentiment)
        result = generator.run()
        assert all(isinstance(v, Sentiment) for v in result)

    def test_passes_schema_as_response_format(self):
        model = make_schema_model()
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=model, output_schema=Sentiment)
        generator.run()
        call_kwargs = model._generate_structured.call_args_list[0].kwargs
        assert call_kwargs["response_format"] is Sentiment

    def test_without_schema_calls_generate(self):
        model = make_model()
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=model)
        generator.run()
        assert model._generate.call_count == 3
        model._generate_structured.assert_not_called()

    def test_passes_correct_instruction(self):
        model = make_schema_model()
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=model, output_schema=Sentiment)
        generator.run()
        call_kwargs = model._generate_structured.call_args_list[0].kwargs
        assert call_kwargs["system_prompt"] == "Fix the review"

    def test_passes_correct_context(self):
        model = make_schema_model()
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=model, output_schema=Sentiment)
        generator.run()
        user_prompts = [c.kwargs["user_prompt"] for c in model._generate_structured.call_args_list]
        assert any("Great!" in ctx for ctx in user_prompts)


def make_failing_model() -> MagicMock:
    m = MagicMock()
    m._generate.side_effect = ValueError("API timeout")
    return m


class TestGeneratorOnError:
    def test_default_on_error_continues_on_failure(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_failing_model())
        result = generator.run()
        assert result.isna().all()

    def test_raise_on_error_propagates(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_failing_model())
        with pytest.raises(ValueError, match="API timeout"):
            generator.run(on_error="raise")

    def test_error_column_populated_on_failure(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_failing_model())
        generator.run()
        assert generator.result[Processor.ERROR_COL].notna().all()
        assert all("API timeout" in e for e in generator.result[Processor.ERROR_COL])

    def test_successful_rows_have_none_error(self):
        generator = Generator(df=make_df(), columns=["review"], instruction="Fix the review", model=make_model())
        generator.run()
        assert generator.result[Processor.ERROR_COL].isna().all()
