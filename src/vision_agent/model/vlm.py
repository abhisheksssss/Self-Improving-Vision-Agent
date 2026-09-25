"""Groq provider via LangChain."""
from typing import Optional
from langchain_groq import ChatGroq


def create_groq(
    model_name: str = "qwen/qwen3.6-27b",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> ChatGroq:
    """Create a LangChain model for Groq API endpoint."""
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )
