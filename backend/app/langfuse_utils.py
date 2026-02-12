from __future__ import annotations

import logging
import os
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


def build_langfuse_callback(settings: Settings) -> Any | None:
    """Return a Langfuse callback handler when enabled and configured.

    Supports both import paths used across Langfuse versions:
    - langfuse.callback.CallbackHandler
    - langfuse.langchain.CallbackHandler
    """
    if not settings.langfuse_enabled:
        return None

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "LANGFUSE_ENABLED is true but LANGFUSE_PUBLIC_KEY/SECRET_KEY is missing; tracing disabled"
        )
        return None

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    if settings.langfuse_host:
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)

    callback_cls: type[Any] | None = None
    try:
        from langfuse.callback import CallbackHandler as _CallbackHandler

        callback_cls = _CallbackHandler
    except Exception:
        try:
            from langfuse.langchain import CallbackHandler as _CallbackHandler

            callback_cls = _CallbackHandler
        except Exception as exc:
            logger.warning("Unable to import Langfuse callback handler: %s", exc)
            return None

    init_attempts: list[dict[str, Any]] = [
        {
            "public_key": settings.langfuse_public_key,
            "secret_key": settings.langfuse_secret_key,
            "host": settings.langfuse_host,
        },
        {},
    ]

    for kwargs in init_attempts:
        try:
            callback = callback_cls(**kwargs)
            logger.info("Langfuse callback initialized")
            return callback
        except TypeError:
            continue
        except Exception as exc:
            logger.warning("Langfuse callback init failed: %s", exc)
            return None

    logger.warning("Langfuse callback init failed due to constructor mismatch")
    return None
