"""Shared configuration helpers for sub-agent key resolution."""

import logging

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
