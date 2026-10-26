"""Privacy-Compliant JSON Logging Configuration.

Specification Section 2, Section 8, Rule R11:
- JSON logs.
- Log IDs, counts, and lengths, never patient free text or secrets.
"""

import logging
import re
import sys
from typing import Any, Dict
from pythonjsonlogger import jsonlogger

# Forbidden sensitive patterns to scrub
SECRET_KEY_PATTERN = re.compile(
    r"(api[-_]?key|secret|password|token|bearer|authorization)",
    re.IGNORECASE
)
PATIENT_TEXT_KEYS = {
    "patient_text",
    "raw_text",
    "clinical_narrative",
    "narrative",
    "free_text",
    "patient_narrative",
    "chief_complaint_raw",
}


class PrivacyJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that redacts secrets and patient text from log records."""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any]
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        # Standard structured fields
        log_record["timestamp"] = self.formatTime(record, self.datefmt)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name

        # Sanitize and scrub keys
        self._sanitize_dict(log_record)

    def _sanitize_dict(self, d: Dict[str, Any]) -> None:
        for k, v in list(d.items()):
            lower_k = k.lower()
            # Redact secrets
            if SECRET_KEY_PATTERN.search(lower_k):
                d[k] = "[REDACTED_SECRET]"
            # Redact patient text fields
            elif lower_k in PATIENT_TEXT_KEYS:
                if isinstance(v, str):
                    d[k] = f"[REDACTED_PATIENT_TEXT: len={len(v)}]"
                else:
                    d[k] = "[REDACTED_PATIENT_TEXT]"
            elif isinstance(v, dict):
                self._sanitize_dict(v)
            elif isinstance(v, list):
                d[k] = [
                    self._sanitize_dict(item) if isinstance(item, dict) else item
                    for item in v
                ]


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root logger with JSON formatting."""
    logger = logging.getLogger("clinexus")
    logger.setLevel(level)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = PrivacyJsonFormatter(
            fmt="%(timestamp)s %(level)s %(logger)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


# Default logger instance
logger = configure_logging()
