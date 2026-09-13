"""Structured logging. Never use `print` (build plan §9)."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED: tuple[str, bool] | None = None


def configure(level: str = "INFO", json_output: bool = False) -> None:
    """Configure structlog. Idempotent for identical settings, else reconfigures.

    A plain "configure once" guard silently swallows later calls, which is how a
    CLI's --verbose and --json-logs flags end up doing nothing: importing any
    module that calls get_logger() locks in the defaults before the flags are
    parsed. Re-applying on a genuine change keeps the flags working.
    """
    global _CONFIGURED
    if (level.upper(), json_output) == _CONFIGURED:
        return

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        # False, deliberately. Caching binds a module-level logger to whatever
        # configuration existed at import time, so a later configure() from a
        # CLI flag would never reach it - which is the bug this guard exists to
        # avoid. The per-call resolution cost is irrelevant at this log volume.
        cache_logger_on_first_use=False,
    )
    _CONFIGURED = (level.upper(), json_output)


def get_logger(name: str) -> Any:
    """Return a bound structlog logger, applying defaults only if unconfigured.

    Crucially this does NOT re-apply the defaults when configuration already
    happened. Calling configure() unconditionally here would let any later
    import silently revert an explicit CLI choice back to console logging.
    """
    if _CONFIGURED is None:
        configure()
    return structlog.get_logger(name)
