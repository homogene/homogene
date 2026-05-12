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
    Standard API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) are automatically picked 
    up from the environment.

    Args:
        name: LiteLLM model string (e.g., `"gpt-5.5"`, `"claude-opus-4-7"`, etc.). Must support chat 
            completions (text generation).
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
        if not system_prompt.strip():
            raise ValueError("'system_prompt' must not be blank.")

        response = litellm.completion(
            model=self.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            seed=self.seed,
            api_key=self._api_key,
        )
        return response.choices[0].message.content

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
        if not system_prompt.strip():
            raise ValueError("'system_prompt' must not be blank.")

        return _instructor_client.chat.completions.create(
            model=self.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            seed=self.seed,
            api_key=self._api_key,
            response_model=response_format,
        )
