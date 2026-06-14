"""LiteLLM-backed generative model adapter."""

from __future__ import annotations

import json

import instructor
import litellm
from pydantic import BaseModel

litellm.set_verbose = False
litellm.suppress_debug_info = True

_instructor_client = instructor.from_litellm(litellm.completion)


class GenerativeModel:
    """A generative model adapter backed by LiteLLM.

    Supports any chat completion model LiteLLM supports (OpenAI, Anthropic, Google, Mistral, etc.).
    Standard API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
    `MISTRAL_API_KEY`, `TOGETHERAI_API_KEY`, etc.) are automatically picked up from the environment.

    Args:
        name: LiteLLM model string (e.g., `"gpt-5.5"`, `"claude-opus-4-7"`, `"gemini-3.1-pro"`,
            `"mistral-large-3"`, etc.). Must support chat completions (text generation). Embedding,
            image, and audio models are not supported.
        api_key: API key as a literal string. Omit to rely on environment variables.
        temperature: Sampling temperature. `None` uses LiteLLM's default.
        max_tokens: Maximum tokens in the response. `None` uses LiteLLM's default.
        top_p: Nucleus sampling probability mass. `None` uses LiteLLM's default.
        seed: Random seed for reproducibility. `None` uses LiteLLM's default.

    Example:
        >>> model = GenerativeModel("gpt-5.5")
        >>> model.generate(system_prompt="Fix typos", user_prompt='{"review": "Grate product"}')
        # "Great product"
    """

    def __init__(
        self,
        name: str,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.seed = seed

    def __repr__(self) -> str:
        return json.dumps({
            "name": self.name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
        }, indent=2)

    def _build_messages(self, system_prompt: str, user_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_params(self) -> dict:
        return {
            "model": self.name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "api_key": self._api_key,
        }

    def _extract_cost(self, response: object) -> float | None:
        try:
            return litellm.completion_cost(completion_response=response)
        except Exception:
            return None

    def _extract_tokens(self, response: object) -> tuple[int | None, int | None]:
        try:
            return response.usage.prompt_tokens, response.usage.completion_tokens
        except Exception:
            return None, None

    def _generate(self, system_prompt: str, user_prompt: str) -> tuple[str, float | None, int | None, int | None]:
        """Generate a plain-text response and return it with its cost and token counts.

        Args:
            system_prompt: System message describing the task.
            user_prompt: User message containing the input context.

        Returns:
            A tuple of `(content, cost, input_tokens, output_tokens)` where `cost` is in USD,
            and token counts are `None` if unavailable.

        Raises:
            ValueError: If `system_prompt` is blank.
            Exception: Any exception raised by LiteLLM is propagated as-is.
        """
        if not system_prompt.strip():
            raise ValueError("'system_prompt' must not be blank.")

        response = litellm.completion(
            messages=self._build_messages(system_prompt, user_prompt),
            **self._build_params(),
        )
        input_tokens, output_tokens = self._extract_tokens(response)
        return response.choices[0].message.content, self._extract_cost(response), input_tokens, output_tokens

    def _generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[BaseModel],
    ) -> tuple[BaseModel, float | None, int | None, int | None]:
        """Generate a structured response and return it with its cost and token counts.

        Args:
            system_prompt: System message describing the task.
            user_prompt: User message containing the input context.
            response_format: Pydantic model class the response is parsed and validated against.

        Returns:
            A tuple of `(instance, cost, input_tokens, output_tokens)` where `cost` is in USD,
            and token counts are `None` if unavailable.

        Raises:
            ValueError: If `system_prompt` is blank.
            Exception: Any exception raised by LiteLLM or Instructor is propagated as-is.
        """
        if not system_prompt.strip():
            raise ValueError("'system_prompt' must not be blank.")

        instance, completion = _instructor_client.chat.completions.create_with_completion(
            messages=self._build_messages(system_prompt, user_prompt),
            response_model=response_format,
            **self._build_params(),
        )
        input_tokens, output_tokens = self._extract_tokens(completion)
        return instance, self._extract_cost(completion), input_tokens, output_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a plain-text response for the given prompts.

        Args:
            system_prompt: System message describing the task.
            user_prompt: User message containing the input context.

        Returns:
            The model's response as a plain string.

        Raises:
            ValueError: If `system_prompt` is blank.
            Exception: Any exception raised by LiteLLM is propagated as-is.
        """
        content, _, _, _ = self._generate(system_prompt, user_prompt)
        return content

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[BaseModel],
    ) -> BaseModel:
        """Generate a structured response validated against a Pydantic model.

        Uses Instructor to reliably extract and validate typed structured output from the model.

        Args:
            system_prompt: System message describing the task.
            user_prompt: User message containing the input context.
            response_format: Pydantic model class the response is parsed and validated against.

        Returns:
            An instance of `response_format`.

        Raises:
            ValueError: If `system_prompt` is blank.
            Exception: Any exception raised by LiteLLM or Instructor is propagated as-is.
        """
        instance, _, _, _ = self._generate_structured(system_prompt, user_prompt, response_format)
        return instance
