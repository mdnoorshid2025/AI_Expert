from langchain_ollama import ChatOllama

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL
)


def get_llm(
    temperature: float = 0.0
) -> ChatOllama:

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        validate_model_on_init=True
    )