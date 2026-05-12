"""Wrapper around ``clifpy.setup_logging`` that captures ``warnings.warn``
output into the same log files clifpy already maintains.

Drop-in replacement for ``from clifpy import setup_logging`` — same call
signature, same behavior on the clifpy side. Additions:

1. ``logging.captureWarnings(True)`` bridges ``warnings.warn`` → the
   ``py.warnings`` logger so ConvergenceWarning / RuntimeWarning /
   UserWarning lines stop bypassing the logging system.
2. Clifpy's existing handlers are mirrored onto ``py.warnings`` (a
   root-level logger, not under ``clifpy.*``, so it does NOT inherit
   clifpy's handler tree by default). The shared ``EmojiFormatter`` works
   on any LogRecord since it injects ``emoji`` / ``shortname`` at format
   time.
3. The benign polars ``"Sortedness of columns cannot be checked when
   'by' groups provided"`` UserWarning is filtered out — join_asof inputs
   at ``code/_sofa.py:798/947`` are pre-sorted at lines 793-794/942-943,
   so polars' inability to statically verify composite-key sortedness is
   uninformative noise.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

from clifpy.utils.logging_config import setup_logging as _clifpy_setup_logging


def setup_logging(
    output_directory: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    separate_error_log: bool = True,
) -> logging.Logger:
    """Run clifpy's setup_logging, then layer on warning capture."""
    clifpy_root = _clifpy_setup_logging(
        output_directory=output_directory,
        level=level,
        console_output=console_output,
        separate_error_log=separate_error_log,
    )

    logging.captureWarnings(True)

    # Re-assign on every call so we stay in sync with clifpy's idempotent
    # handler swap (clifpy nukes its handler list and rebuilds it).
    py_warn_logger = logging.getLogger('py.warnings')
    py_warn_logger.handlers = list(clifpy_root.handlers)
    py_warn_logger.setLevel(logging.WARNING)
    py_warn_logger.propagate = False

    warnings.filterwarnings(
        "ignore",
        message="Sortedness of columns cannot be checked when 'by' groups provided",
        category=UserWarning,
    )

    return clifpy_root
