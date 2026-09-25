"""Model factory — returns LangChain BaseChatModel instances based on .env config."""
import logging
from langchain_core.language_models import BaseChatModel
from vision_agent.config import settings
from .llm import create_gemini_model, create_openai_compatible_model, create_nvidia_nim
from .vlm import create_groq

logger = logging.getLogger("sivac.models")


class ModelFactory:
    """Creates Vision and Reasoning LangChain models from .env configuration."""

    @staticmethod
    def get_vision_model() -> BaseChatModel:
        provider = settings.VISION_PROVIDER.lower()
        model_name = settings.VISION_MODEL
        logger.info(f"Initializing Vision Model: [{provider}] {model_name}")

        if provider == "gemini":
            return create_gemini_model(
                model_name=model_name,
                api_key=settings.GEMINI_API_KEY
            )
        elif provider == "groq":
            return create_groq(
                model_name=model_name,
                api_key=settings.GROQ_API_KEY
            )
        elif provider in ["nvidia", "nvidia_nim"]:
            return create_nvidia_nim(
                model_name=model_name,
                api_key=settings.NVIDIA_API_KEY
            )
        elif provider in ["openai_compatible", "openrouter", "ollama"]:
            return create_openai_compatible_model(
                model_name=model_name,
                api_key=settings.OPENROUTER_API_KEY or settings.OPENAI_COMPATIBLE_API_KEY,
                base_url=settings.OPENAI_COMPATIBLE_BASE_URL,
            )
        else:
            raise ValueError(f"Unsupported VISION_PROVIDER: '{provider}'")

    @staticmethod
    def get_reasoning_model() -> BaseChatModel:
        provider = settings.REASONING_PROVIDER.lower()
        model_name = settings.REASONING_MODEL
        logger.info(f"Initializing Reasoning Model: [{provider}] {model_name}")

        if provider == "gemini":
            return create_gemini_model(
                model_name=model_name,
                api_key=settings.GEMINI_API_KEY
            )
        elif provider == "groq":
            return create_groq(
                model_name=model_name,
                api_key=settings.GROQ_API_KEY
            )
        elif provider in ["nvidia", "nvidia_nim"]:
            return create_nvidia_nim(
                model_name=model_name,
                api_key=settings.NVIDIA_API_KEY
            )
        elif provider in ["openai_compatible", "openrouter", "ollama"]:
            return create_openai_compatible_model(
                model_name=model_name,
                api_key=settings.OPENROUTER_API_KEY or settings.OPENAI_COMPATIBLE_API_KEY,
                base_url=settings.OPENAI_COMPATIBLE_BASE_URL,
            )
        else:
            raise ValueError(f"Unsupported REASONING_PROVIDER: '{provider}'")
