"""Generator that applies an instruction to each row of a `pd.DataFrame` independently, in parallel."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel

from homogene.context import _build_context, _validate_columns
from homogene.models.generative_model import GenerativeModel
from homogene.processors.processor import Processor
from homogene.prompt import Prompt


class Generator(Processor):
    """Applies an instruction to each row independently and returns a Series of results.

    Each row is processed in isolation — no context from other rows.
    Rows are processed in parallel using threads.

    Args:
        df: Source `pd.DataFrame`.
        columns: Column name(s) passed to the model as input context.
        instruction: Instruction string describing the task.
        model: `GenerativeModel` instance or model string (e.g., `"gpt-5.5"`).
        output_schema: Pydantic model class for structured output. When set, returns typed
            instances instead of strings. `None` returns plain strings.

    Attributes:
        result: `pd.DataFrame` with `"generation"` and `"error"` columns, set after `run()` is called. `None` before.

    Example:
        class Sentiment(BaseModel):
            label: str
            keywords: list[str]

        generator = Generator(
            df=df,
            columns=["review"],
            instruction="Classify the sentiment of the review",
            model="gpt-5.5",
            output_schema=Sentiment,
        )
        generator.run()
        # generator.result → pd.DataFrame with "generation" and "error" columns
        # generator.result[Generator.OUTPUT_COL] → pd.Series of Sentiment instances
    """

    OUTPUT_COL = "generation"

    def __init__(
        self,
        df: pd.DataFrame,
        columns: list[str] | str,
        instruction: str,
        model: GenerativeModel | str,
        output_schema: type[BaseModel] | None = None,
    ) -> None:
        # Allow passing a model string instead of a GenerativeModel instance
        model = GenerativeModel(model) if isinstance(model, str) else model

        # Normalize columns to a list
        columns = [columns] if isinstance(columns, str) else columns

        # Validate columns
        _validate_columns(columns, df)

        # Validate output_schema
        if output_schema is not None and not issubclass(output_schema, BaseModel):
            raise TypeError("'output_schema' must be a Pydantic BaseModel subclass.")

        # Set attributes
        super().__init__(df=df, model=model)
        self.columns = columns
        self.instruction = instruction
        self.output_schema = output_schema

    def __repr__(self) -> str:
        is_truncated = len(self.instruction) > 60
        instruction = self.instruction[:57] + "..." if is_truncated else self.instruction
        model = (
            json.loads(repr(self.model)) if isinstance(self.model, GenerativeModel) else self.model
        )
        return json.dumps({
            "instruction": instruction,
            "columns": self.columns,
            "model": model,
            "schema": self.output_schema.__name__ if self.output_schema else None,
            "df": f"[{len(self.df)} rows x {len(self.df.columns)} columns]",
        }, indent=2)

    def _run_row(self, row: pd.Series) -> Any:
        """Process a single row and return the model output.

        Args:
            row: A single `pd.DataFrame` row as a `pd.Series`.

        Returns:
            A string if `output_schema` is `None`, otherwise an instance of `output_schema`.
        """
        prompt = Prompt(self.instruction, _build_context(self.columns, row))

        if self.output_schema is not None:
            return self.model.generate_structured(
                system_prompt=prompt.system,
                user_prompt=prompt.user,
                response_format=self.output_schema,
            )
        return self.model.generate(
            system_prompt=prompt.system,
            user_prompt=prompt.user,
        )

    def get_prompt(self, i: int = 0) -> Prompt:
        """Return the `Prompt` for the row at position `i`.

        Args:
            i: Positional row number (0-based), not the DataFrame index label.

        Returns:
            A `Prompt` instance for the specified row.
        """
        return Prompt(self.instruction, _build_context(self.columns, self.df.iloc[i]))

