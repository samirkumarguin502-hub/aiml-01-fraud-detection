"""JWT authentication and password security utilities."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any


ROLES = {"admin", "analyst"}


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def hash_password(password: str) -> str:
    if not isinstance(password, str) or len(password) < 8:
        raise ValueError("Password must contain at least 8 characters")
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return f"scrypt${_b64encode(salt)}${_b64encode(derived)}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, salt_text, hash_text = stored_hash.split("$", 2)
        if algorithm != "scrypt":
            return False
        salt = _b64decode(salt_text)
        expected = _b64decode(hash_text)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=2**14,
            r=8,
            p=1,
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, AttributeError):
        return False


def create_access_token(
    user_id: int,
    username: str,
    role: str,
    secret_key: str | None = None,
    expires_minutes: int = 60,
) -> str:
    secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "")
    if len(secret_key) < 32:
        raise ValueError("JWT_SECRET_KEY must contain at least 32 characters")
    if role not in ROLES:
        raise ValueError("Invalid user role")

    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": now,
        "exp": now + expires_minutes * 60,
        "jti": secrets.token_hex(16),
    }

    signing_input = (
        _b64encode(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64encode(json.dumps(payload, separators=(",", ":")).encode())
    )
    signature = hmac.new(
        secret_key.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return signing_input + "." + _b64encode(signature)


def decode_access_token(
    token: str,
    secret_key: str | None = None,
) -> dict[str, Any]:
    secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "")
    if len(secret_key) < 32:
        raise ValueError("JWT_SECRET_KEY must contain at least 32 characters")

    try:
        header_part, payload_part, signature_part = token.split(".")
        signing_input = header_part + "." + payload_part
        expected = hmac.new(
            secret_key.encode("utf-8"),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        supplied = _b64decode(signature_part)

        if not hmac.compare_digest(expected, supplied):
            raise ValueError("Invalid token signature")

        header = json.loads(_b64decode(header_part))
        payload = json.loads(_b64decode(payload_part))

        if header.get("alg") != "HS256" or header.get("typ") != "JWT":
            raise ValueError("Invalid token header")
        if int(payload["exp"]) <= int(time.time()):
            raise ValueError("Token expired")
        if payload.get("role") not in ROLES:
            raise ValueError("Invalid token role")
        if not payload.get("sub") or not payload.get("username"):
            raise ValueError("Invalid token payload")

        return payload
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid or expired access token") from exc


def require_role(principal: dict[str, Any], *allowed_roles: str) -> None:
    if principal.get("role") not in allowed_roles:
        raise PermissionError("Insufficient permissions")
