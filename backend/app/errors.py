"""Application errors.

Messages must be safe to show to clients:
no filesystem paths, secrets, or raw transaction input values.
"""

from __future__ import annotations

import re
from typing import Any, Optional


_ERROR_CODE_PATTERN = re.compile(
    r"^[a-z][a-z0-9_]{0,63}$"
)


class AppError(Exception):
    """Expected application error with a safe client response."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        details: Optional[Any] = None,
        headers: Optional[dict[str, str]] = None,
    ):
        if (
            isinstance(status, bool)
            or not isinstance(status, int)
            or not 400 <= status <= 599
        ):
            raise ValueError(
                "status must be an HTTP error status between 400 and 599"
            )

        if (
            not isinstance(code, str)
            or not _ERROR_CODE_PATTERN.fullmatch(code)
        ):
            raise ValueError("invalid application error code")

        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")

        if headers is not None:
            if not isinstance(headers, dict):
                raise TypeError("headers must be a dictionary")

            for key, value in headers.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise TypeError(
                        "error headers must contain string keys and values"
                    )

        super().__init__(message)

        self.status = status
        self.code = code
        self.message = message
        self.details = details
        self.headers = dict(headers or {})

    def body(
        self,
        request_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return the standard API error response."""

        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }

        if self.details is not None:
            error["details"] = self.details

        if request_id:
            error["request_id"] = request_id

        return {"error": error}


class ConfigError(RuntimeError):
    """Invalid environment configuration.

    Error messages should identify the configuration variable,
    never disclose its secret value.
    """