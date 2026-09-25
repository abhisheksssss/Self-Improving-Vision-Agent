"""Gemini, OpenAI-compatible, and NVIDIA NIM providers via LangChain."""
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA


def create_gemini_model(
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> ChatGoogleGenerativeAI:
    """Create a LangChain Gemini model (works for both vision and reasoning)."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Please set it in your .env file.")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        # Disable automatic function calling to suppress AFC warnings
        # The agent manages its own JSON action parsing
        disable_streaming=False,
    )


def create_openai_compatible_model(
    model_name: str,
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Create a LangChain model for any OpenAI-compatible API endpoint (OpenRouter, Ollama)."""
    return ChatOpenAI(
        model=model_name,
        api_key=api_key or "no-key-required",
        base_url=base_url,
        temperature=temperature,
    )


def create_nvidia_nim(
    model_name: str = "nvidia/llama-3.1-nemotron-70b-instruct",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
):
    """Create a LangChain model for NVIDIA NIM endpoints."""
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is missing. Please set it in your .env file.")
    
    # Using ChatOpenAI targeting NVIDIA NIM endpoint provides standard OpenAI auth headers
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        temperature=temperature,
    )
