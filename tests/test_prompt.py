"""Tests for homogene.prompt"""

import json

from homogene.prompt import Prompt


class TestPromptInit:
    def test_stores_system(self):
        p = Prompt("Classify.", {"review": "Great!"})
        assert p.system == "Classify."

    def test_user_is_json_string(self):
        p = Prompt("Classify.", {"review": "Great!"})
        assert json.loads(p.user) == {"review": "Great!"}

    def test_user_contains_all_columns(self):
        p = Prompt("Classify.", {"review": "Great!", "rating": 5})
        parsed = json.loads(p.user)
        assert parsed["review"] == "Great!"
        assert parsed["rating"] == 5

    def test_user_empty_context(self):
        p = Prompt("Classify.", {})
        assert p.user == "{}"

    def test_no_backslash_before_apostrophe(self):
        p = Prompt("Classify.", {"review": "it's great"})
        assert "\\'" not in p.user

    def test_non_serializable_value_falls_back_to_str(self):
        class Custom:
            def __str__(self): return "custom"
        p = Prompt("Classify.", {"col": Custom()})
        assert json.loads(p.user) == {"col": "custom"}

    def test_empty_instruction(self):
        p = Prompt("", {"review": "Great!"})
        assert p.system == ""


class TestPromptRepr:
    def test_repr_is_valid_json(self):
        p = Prompt("Classify.", {"review": "Great!"})
        assert isinstance(json.loads(repr(p)), list)

    def test_repr_has_two_messages(self):
        p = Prompt("Classify.", {"review": "Great!"})
        assert len(json.loads(repr(p))) == 2

    def test_repr_system_role_and_content(self):
        p = Prompt("Classify.", {"review": "Great!"})
        msg = json.loads(repr(p))[0]
        assert msg["role"] == "system"
        assert msg["content"] == "Classify."

    def test_repr_user_role_and_content(self):
        p = Prompt("Classify.", {"review": "Great!"})
        msg = json.loads(repr(p))[1]
        assert msg["role"] == "user"
        assert json.loads(msg["content"]) == {"review": "Great!"}

    def test_repr_with_empty_context(self):
        p = Prompt("Classify.", {})
        msg = json.loads(repr(p))[1]
        assert json.loads(msg["content"]) == {}
