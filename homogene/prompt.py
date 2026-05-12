"""Prompt composed of system and user messages sent to the LLM for a single row."""

from __future__ import annotations

import json
from typing import Any


class Prompt:
    """System and user messages sent to the LLM for a single row.

    Args:
        instruction: System prompt describing the task.
        context: `dict` mapping column names to their values for this row.

    Attributes:
        system: The instruction string, passed as the system message.
        user: The context serialized as a JSON string, passed as the user message.
    """

    def __init__(self, instruction: str, context: dict[str, Any]) -> None:
        self.system = instruction
        self.user = json.dumps(context, default=str)

    def __repr__(self) -> str:
        return json.dumps([
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ], indent=2, default=str)
