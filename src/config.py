from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # model config
    model_api: Literal["anthropic", "openai", "openapi", "umans"] = "anthropic"
    """the model api to use. must be one of `anthropic`, `openai`, `openapi`, or `umans`"""
    model_name: str = "claude-sonnet-4-5-20250929"
    """the model to use with the given api"""
    model_api_key: str = ""
    """the model api key"""
    model_endpoint: str = ""
    """for openapi model apis, the endpoint to use"""

    # sub-agent model config (for summarize, research tools)
    sub_model_api: Literal["anthropic", "openai", "openapi", "umans"] = "anthropic"
    """the model api for sub-agent tasks"""
    sub_model_name: str = "claude-haiku-4-5-20251001"
    """the model for sub-agent tasks (default: Haiku for speed/cost)"""
    sub_model_api_key: str = ""
    """api key for sub-agent model (falls back to model_api_key if empty)"""
    sub_model_endpoint: str = ""
    """endpoint for sub-agent model (for openapi providers like Moonshot/Kimi)"""

    # local data
    data_dir: str = "data"
    """local directory for secrets, schedules, and other operator state"""

    # display
    context_limit: int = 200_000
    """approximate context window of the main model; used for the web UI context bar"""

    # discord bot chat
    discord_sessions_channel: str = "victrola-sessions"
    """name of the text channel the Discord bot watches for chat sessions"""

    discord_allowed_user_ids: str = ""
    """comma-separated Discord user IDs allowed to drive the agent via Discord.
    If empty, all users are allowed (with a startup warning)."""

    discord_chat_timeout_seconds: int = 300
    """timeout in seconds for agent.chat() calls from the Discord bot"""

    # exa web search
    exa_api_key: str = ""
    """api key for Exa web search"""

    # umans
    umans_endpoint: str = "https://api.code.umans.ai"
    """base url for the umans api"""

    # embeddings (Ollama local)
    embedding_endpoint: str = "http://localhost:11434"
    """Ollama API endpoint for embeddings"""
    embedding_model: str = "nomic-embed-text"
    """embedding model name (768 dims for nomic-embed-text)"""
    embedding_dimensions: int = 768
    """dimensionality of the embedding model"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


CONFIG = Config()
