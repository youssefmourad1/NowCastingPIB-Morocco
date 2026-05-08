"""
Structured logging configuration for the lamiaty package.

Features:
  - JSON-structured log records (machine-parseable for pipelines)
  - Human-readable console output with colour and pipeline stage labels
  - Per-stage timing context manager
  - Streamlit sidebar log sink (optional)
  - Log file rotation

Call setup_logging() once at startup (script, notebook, or Streamlit entry point).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

# ── Log level constants re-exported for convenience ──────────────────────────
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# ── ANSI colour codes (used in ConsoleFormatter) ─────────────────────────────
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"


# ── Formatters ────────────────────────────────────────────────────────────────

class ColourConsoleFormatter(logging.Formatter):
    """Human-readable, coloured console output.

    Format: 2026-04-01 09:12:34 | INFO     | lamiaty.data.pipeline — Stage 2: corrections
    """
    _FMT = "{colour}{bold}[{levelname:<8}]{reset} {dim}{asctime}{reset} | {name} — {message}"

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        record.colour = colour
        record.bold   = _BOLD
        record.dim    = "\033[2m"
        record.reset  = _RESET
        self._style._fmt = self._FMT
        formatted = super().format(record)
        return formatted

    def formatTime(self, record, datefmt=None):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))


class JsonFormatter(logging.Formatter):
    """JSON-structured log records for log files and machine processing.

    Each line is a self-contained JSON object:
    {"ts": "2026-04-01T09:12:34", "level": "INFO", "logger": "lamiaty.data.pipeline",
     "msg": "Stage 2: corrections", "stage": "corrections", "elapsed_ms": 12}
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts":      self.formatTime(record),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
            "file":    f"{record.filename}:{record.lineno}",
        }
        # Attach any extra fields set via logger.info(..., extra={...})
        for key in ("stage", "elapsed_ms", "rows", "columns", "series"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

    def formatTime(self, record, datefmt=None):
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))


# ── In-memory buffer for Streamlit sidebar ────────────────────────────────────

class _StreamlitLogBuffer(logging.Handler):
    """Accumulates log records in a list for display in the Streamlit sidebar."""

    MAX_RECORDS = 200

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.MAX_RECORDS:
            self.records = self.records[-self.MAX_RECORDS:]

    def get_lines(self, min_level: int = logging.DEBUG) -> list[str]:
        lines = []
        for r in self.records:
            if r.levelno >= min_level:
                colour = {"DEBUG": "⚪", "INFO": "🟢", "WARNING": "🟡",
                          "ERROR": "🔴", "CRITICAL": "🟣"}.get(r.levelname, "⚪")
                lines.append(f"{colour} `{r.name}` — {r.getMessage()}")
        return lines

    def clear(self) -> None:
        self.records.clear()


# Singleton buffer — imported by the Streamlit app
_streamlit_buffer = _StreamlitLogBuffer()


def get_streamlit_buffer() -> _StreamlitLogBuffer:
    """Return the singleton Streamlit log buffer."""
    return _streamlit_buffer


# ── Main setup function ───────────────────────────────────────────────────────

def setup_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    json_file: str | Path | None = None,
    enable_streamlit_buffer: bool = False,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
) -> None:
    """Configure root logger for the lamiaty package.

    Args:
        level: Logging level for console output (default: INFO).
        log_file: Optional path to a human-readable rotating log file.
        json_file: Optional path to a JSON-structured log file (one JSON object per line).
                   Useful for log aggregation tools.
        enable_streamlit_buffer: If True, also accumulates records in the
                                 Streamlit sidebar buffer (_streamlit_buffer).
        fmt: Fallback format string (used only if ColourConsoleFormatter fails).
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # root captures everything; handlers filter

    # Remove existing handlers to avoid duplicates on re-calls (Streamlit hot-reload)
    root.handlers.clear()

    # ── Console handler (stderr) ─────────────────────────────────────────
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(ColourConsoleFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(console)

    # ── Human-readable rotating file ────────────────────────────────────
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)

    # ── JSON structured file ─────────────────────────────────────────────
    if json_file is not None:
        json_file = Path(json_file)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        jh = logging.handlers.RotatingFileHandler(
            json_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        jh.setLevel(logging.DEBUG)
        jh.setFormatter(JsonFormatter())
        root.addHandler(jh)

    # ── Streamlit sidebar buffer ─────────────────────────────────────────
    if enable_streamlit_buffer:
        _streamlit_buffer.setLevel(logging.DEBUG)
        root.addHandler(_streamlit_buffer)

    # Quieten noisy third-party loggers
    for noisy in ["matplotlib", "PIL", "numexpr", "pyarrow", "urllib3",
                  "watchdog", "fsevents", "httpx", "asyncio"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("lamiaty").setLevel(level)


# ── Stage timing context manager ──────────────────────────────────────────────

@contextmanager
def log_stage(name: str, logger_name: str = "lamiaty") -> Generator[None, None, None]:
    """Context manager that logs stage start/end with elapsed time.

    Usage:
        with log_stage("Stage 2: corrections"):
            apply_all_corrections(...)

    Emits:
        INFO  lamiaty — ▶ Stage 2: corrections
        INFO  lamiaty — ✓ Stage 2: corrections  [elapsed: 0.12s]
        ERROR lamiaty — ✗ Stage 2: corrections FAILED after 0.03s — ValueError: ...
    """
    log = logging.getLogger(logger_name)
    log.info("▶ %s", name)
    t0 = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - t0
        log.info("✓ %s  [elapsed: %.2fs]", name, elapsed,
                 extra={"stage": name, "elapsed_ms": round(elapsed * 1000)})
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log.error(
            "✗ %s FAILED after %.2fs — %s: %s",
            name, elapsed, type(exc).__name__, exc,
            extra={"stage": name, "elapsed_ms": round(elapsed * 1000)},
            exc_info=True,
        )
        raise
