from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str | None
    model: str


def get_llm_config() -> LLMConfig | None:
    if os.getenv("OPENAI_API_KEY"):
        return LLMConfig(
            provider="OpenAI-compatible",
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL") or None,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
    if os.getenv("GEMINI_API_KEY"):
        return LLMConfig(
            provider="Gemini",
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        )
    return None
