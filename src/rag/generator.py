"""LLM answer generation using an OpenAI-compatible API (e.g. Groq)."""

import logging
from typing import Optional

from openai import OpenAI

from src.config.prompts import RAG_PROMPT_TEMPLATE, RAG_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Generator:

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        base_url: str = "https://api.groq.com/openai/v1",
        temperature: float = 0.3,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt or RAG_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": RAG_PROMPT_TEMPLATE.format(
                    context=context, question=question
                ),
            },
        ]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )

        answer = response.choices[0].message.content or ""
        logger.debug(
            "Generated answer (model=%s, input_tokens=%d, output_tokens=%d)",
            self._model,
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )
        return answer
