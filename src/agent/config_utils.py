"""Shared configuration helpers for sub-agent key and endpoint resolution."""

import logging

from src.config import CONFIG

logger = logging.getLogger(__name__)


def resolve_sub_agent_key(
    model_api: str,
    model_api_key: str | None,
    sub_model_api: str,
    sub_model_api_key: str | None,
) -> str | None:
    """Resolve the sub-agent API key with provider-consistency validation.

    - If providers match and sub_model_api_key is empty, fall back to model_api_key.
    - If providers differ and sub_model_api_key is empty, log a warning and
      return None (graceful degradation — no sub-agent wired).
    - If sub_model_api_key is set, return it as-is.
    """
    if sub_model_api_key:
        return sub_model_api_key
    if sub_model_api == model_api:
        # Same provider — safe to reuse the main key
        return model_api_key
    # Different provider, no sub key — don't silently send the wrong key
    logger.warning(
        "Sub-agent provider (%s) differs from main provider (%s) but no "
        "sub_model_api_key is set. Sub-agent LLM will not be wired.",
        sub_model_api,
        model_api,
    )
    return None


def resolve_sub_agent_endpoint() -> str | None:
    """Resolve the sub-agent endpoint, defaulting to the Umans endpoint when needed.

    - If SUB_MODEL_ENDPOINT is set, use it.
    - If SUB_MODEL_API is "umans" and no explicit endpoint, default to CONFIG.umans_endpoint.
    - Otherwise, return None (SDK/provider defaults apply).
    """
    endpoint = CONFIG.sub_model_endpoint or None
    if endpoint is None and CONFIG.sub_model_api == "umans":
        endpoint = CONFIG.umans_endpoint or None
        if endpoint is None:
            raise ValueError(
                "umans_endpoint is required when sub_model_api is 'umans' "
                "and no explicit sub_model_endpoint is set "
                "(set UMANS_ENDPOINT or SUB_MODEL_ENDPOINT in .env)"
            )
    return endpoint
