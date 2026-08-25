"""
Structured logging configuration using loguru.
Logs to console (with colour) and to a rotating daily file under logs/.
"""

import sys
import io
from pathlib import Path
from loguru import logger


LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Force UTF-8 stdout on Windows (avoids CP1252 UnicodeEncodeError)
_stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def setup_logger(name: str = "finance_agent", level: str = "DEBUG") -> None:
    """Configure loguru for the project. Call once at startup."""
    logger.remove()  # Remove default handler

    # -- Console handler (INFO+) -------------------------------------------------
    logger.add(
        _stdout_utf8,
        level="INFO",
        colorize=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{line} -- {message}"
        ),
    )

    # ── File handler (DEBUG+, rotating daily, keep 30 days) ─────────────────
    logger.add(
        LOG_DIR / f"{name}.log",
        level=level,
        rotation="1 day",
        retention="30 days",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{line} — {message}"
        ),
        enqueue=True,
    )

    logger.info(f"Logger initialised — logs written to {LOG_DIR}")


# Export a pre-configured logger for import convenience
setup_logger()

__all__ = ["logger", "setup_logger"]
