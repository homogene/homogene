"""Tests for homogene.models.generative_model"""

import json
from unittest.mock import MagicMock, patch

import litellm
import pytest
from pydantic import BaseModel

from homogene.models.generative_model import GenerativeModel


class TestLiteLLMFlags:
    def test_verbose_disabled(self):
        assert litellm.set_verbose is False

    def test_suppress_debug_info_enabled(self):
        assert litellm.suppress_debug_info is True


class Sentiment(BaseModel):
    label: str


def make_mock_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.choices[0].message.content = content
    return mock


class TestGenerativeModelInit:
    def test_stores_defaults(self):
        model = GenerativeModel("gpt-5.5")
        assert model.name == "gpt-5.5"
        assert model.temperature is None
        assert model.max_tokens is None
        assert model._api_key is None
        assert model.top_p is None
        assert model.seed is None

    def test_stores_custom_params(self):
        model = GenerativeModel("gpt-5.5", api_key="sk-123", temperature=0.5, max_tokens=500, top_p=0.9, seed=42)
        assert model.name == "gpt-5.5"
        assert model._api_key == "sk-123"
        assert model.temperature == 0.5
        assert model.max_tokens == 500
        assert model.top_p == 0.9
        assert model.seed == 42


class TestGenerativeModelRepr:
    def test_repr_is_valid_json(self):
        model = GenerativeModel("gpt-5.5")
        parsed = json.loads(repr(model))
        assert parsed["name"] == "gpt-5.5"

    def test_contains_name(self):
        model = GenerativeModel("gpt-5.5")
        assert "gpt-5.5" in repr(model)

    def test_contains_all_params(self):
        model = GenerativeModel("gpt-5.5", temperature=0.5, max_tokens=500, top_p=0.9, seed=42)
        r = repr(model)
        assert "0.5" in r
        assert "500" in r
        assert "0.9" in r
        assert "42" in r

    def test_default_params_shown(self):
        model = GenerativeModel("gpt-5.5")
        r = repr(model)
        assert "temperature" in r
        assert "max_tokens" in r
        assert "top_p" in r
        assert "seed" in r

    def test_default_values_are_null(self):
        model = GenerativeModel("gpt-5.5")
        parsed = json.loads(repr(model))
        assert parsed["temperature"] is None
        assert parsed["max_tokens"] is None
        assert parsed["top_p"] is None
        assert parsed["seed"] is None

    def test_api_key_not_in_repr(self):
        model = GenerativeModel("gpt-5.5", api_key="sk-secret")
        assert "sk-secret" not in repr(model)

    def test_two_instances_differ(self):
        m1 = GenerativeModel("gpt-5.5", temperature=0.5)
        m2 = GenerativeModel("gpt-5.5", temperature=1.0)
        assert repr(m1) != repr(m2)


class TestGenerativeModelGenerate:
    @patch("homogene.models.generative_model.litellm.completion")
    def test_returns_string(self, mock_completion):
        mock_completion.return_value = make_mock_response("positive")
        model = GenerativeModel("gpt-5.5")
        result = model.generate(system_prompt="Classify the review", user_prompt='{"review": "Great!"}')
        assert result == "positive"

    @patch("homogene.models.generative_model.litellm.completion")
    def test_system_message_is_system_prompt(self, mock_completion):
        mock_completion.return_value = make_mock_response("positive")
        model = GenerativeModel("gpt-5.5")
        model.generate(system_prompt="Classify the review", user_prompt='{"review": "Great!"}')
        messages = mock_completion.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Classify the review"

    @patch("homogene.models.generative_model.litellm.completion")
    def test_user_message_is_user_prompt(self, mock_completion):
        mock_completion.return_value = make_mock_response("positive")
        model = GenerativeModel("gpt-5.5")
        model.generate(system_prompt="Classify the review", user_prompt='{"review": "Great!"}')
        messages = mock_completion.call_args.kwargs["messages"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == '{"review": "Great!"}'

    def test_empty_system_prompt_raises(self):
        model = GenerativeModel("gpt-5.5")
        with pytest.raises(ValueError, match="'system_prompt'"):
            model.generate(system_prompt="", user_prompt="{}")

    def test_blank_system_prompt_raises(self):
        model = GenerativeModel("gpt-5.5")
        with pytest.raises(ValueError, match="'system_prompt'"):
            model.generate(system_prompt="   ", user_prompt="{}")

    @patch("homogene.models.generative_model.litellm.completion")
    def test_empty_user_prompt(self, mock_completion):
        mock_completion.return_value = make_mock_response("done")
        model = GenerativeModel("gpt-5.5")
        result = model.generate(system_prompt="Do something", user_prompt="")
        assert result == "done"

    @patch("homogene.models.generative_model.litellm.completion")
    def test_passes_correct_params(self, mock_completion):
        mock_completion.return_value = make_mock_response("ok")
        model = GenerativeModel("gpt-5.5", temperature=0.5, max_tokens=500, top_p=0.9, seed=42)
        model.generate(system_prompt="Classify", user_prompt='{"review": "Good"}')
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "gpt-5.5"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 500
        assert kwargs["top_p"] == 0.9
        assert kwargs["seed"] == 42

    @patch("homogene.models.generative_model.litellm.completion")
    def test_api_key_passed_to_litellm(self, mock_completion):
        mock_completion.return_value = make_mock_response("ok")
        model = GenerativeModel("gpt-5.5", api_key="sk-direct")
        model.generate(system_prompt="Classify", user_prompt="")
        assert mock_completion.call_args.kwargs["api_key"] == "sk-direct"

    @patch("homogene.models.generative_model.litellm.completion")
    def test_no_api_key_passes_none(self, mock_completion):
        mock_completion.return_value = make_mock_response("ok")
        model = GenerativeModel("gpt-5.5")
        model.generate(system_prompt="Classify", user_prompt="")
        assert mock_completion.call_args.kwargs["api_key"] is None

    @patch("homogene.models.generative_model.litellm.completion")
    def test_default_none_params_passed_to_litellm(self, mock_completion):
        mock_completion.return_value = make_mock_response("ok")
        model = GenerativeModel("gpt-5.5")
        model.generate(system_prompt="Classify", user_prompt="")
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["temperature"] is None
        assert kwargs["max_tokens"] is None
        assert kwargs["top_p"] is None
        assert kwargs["seed"] is None


class TestGenerativeModelGenerateStructured:
    @patch("homogene.models.generative_model._instructor_client.chat.completions.create_with_completion")
    def test_returns_pydantic_instance(self, mock_create):
        mock_create.return_value = (Sentiment(label="positive"), MagicMock())
        model = GenerativeModel("gpt-5.5")
        result = model.generate_structured(system_prompt="Classify", user_prompt="{}", response_format=Sentiment)
        assert isinstance(result, Sentiment)
        assert result.label == "positive"

    @patch("homogene.models.generative_model._instructor_client.chat.completions.create_with_completion")
    def test_response_model_passed_to_instructor(self, mock_create):
        mock_create.return_value = (Sentiment(label="positive"), MagicMock())
        model = GenerativeModel("gpt-5.5")
        model.generate_structured(system_prompt="Classify", user_prompt="{}", response_format=Sentiment)
        assert mock_create.call_args.kwargs["response_model"] is Sentiment

    @patch("homogene.models.generative_model._instructor_client.chat.completions.create_with_completion")
    def test_system_message_is_system_prompt(self, mock_create):
        mock_create.return_value = (Sentiment(label="positive"), MagicMock())
        model = GenerativeModel("gpt-5.5")
        model.generate_structured(system_prompt="Classify the review", user_prompt="{}", response_format=Sentiment)
        messages = mock_create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Classify the review"

    @patch("homogene.models.generative_model._instructor_client.chat.completions.create_with_completion")
    def test_user_message_is_user_prompt(self, mock_create):
        mock_create.return_value = (Sentiment(label="positive"), MagicMock())
        model = GenerativeModel("gpt-5.5")
        model.generate_structured(system_prompt="Classify", user_prompt='{"review": "Great!"}', response_format=Sentiment)
        messages = mock_create.call_args.kwargs["messages"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == '{"review": "Great!"}'

    def test_blank_system_prompt_raises(self):
        model = GenerativeModel("gpt-5.5")
        with pytest.raises(ValueError, match="'system_prompt'"):
            model.generate_structured(system_prompt="   ", user_prompt="{}", response_format=Sentiment)

    @patch("homogene.models.generative_model._instructor_client.chat.completions.create_with_completion")
    def test_passes_correct_params(self, mock_create):
        mock_create.return_value = (Sentiment(label="positive"), MagicMock())
        model = GenerativeModel("gpt-5.5", temperature=0.5, max_tokens=500, top_p=0.9, seed=42)
        model.generate_structured(system_prompt="Classify", user_prompt="{}", response_format=Sentiment)
        kwargs = mock_create.call_args.kwargs
        assert kwargs["model"] == "gpt-5.5"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 500
        assert kwargs["top_p"] == 0.9
        assert kwargs["seed"] == 42

    @patch("homogene.models.generative_model._instructor_client.chat.completions.create_with_completion")
    def test_api_key_passed(self, mock_create):
        mock_create.return_value = (Sentiment(label="positive"), MagicMock())
        model = GenerativeModel("gpt-5.5", api_key="sk-direct")
        model.generate_structured(system_prompt="Classify", user_prompt="{}", response_format=Sentiment)
        assert mock_create.call_args.kwargs["api_key"] == "sk-direct"

