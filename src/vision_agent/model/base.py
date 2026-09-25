"""LangChain-based model type aliases for SIVAC."""
from langchain_core.language_models import BaseChatModel

# Both VisionModel and ReasoningModel are just LangChain BaseChatModel instances.
# Gemini (ChatGoogleGenerativeAI) and OpenAI-compatible (ChatOpenAI) both inherit from it.
VisionModelType = BaseChatModel
ReasoningModelType = BaseChatModel
