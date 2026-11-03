"""Application errors. Messages must be safe to show to clients (no paths, no raw input values)."""
from __future__ import annotations


class AppError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None, headers=None):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.details, self.headers = details, headers or {}

    def body(self, request_id: str | None = None) -> dict:
        err = {"code": self.code, "message": self.message}
        if self.details is not None:
            err["details"] = self.details
        if request_id:
            err["request_id"] = request_id
        return {"error": err}


class ConfigError(RuntimeError):
    """Invalid environment configuration (raised at startup, message names the variable only)."""
