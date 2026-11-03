"""Replay message contract (JSON). Same field rules as the API (backend.app.validation)."""
from __future__ import annotations

import json

from backend.app.errors import AppError
from backend.app.validation import TX_FIELDS, validate_transaction

SCHEMA_VERSION = 1
MAX_MESSAGE_BYTES = 4096


class MessageError(ValueError):
    """Malformed replay message. The text never contains message content."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def encode_message(tx: dict, seq: int, fields: tuple = TX_FIELDS) -> bytes:
    clean = validate_transaction({k: tx[k] for k in fields}, fields)
    return json.dumps({"schema": SCHEMA_VERSION, "seq": seq, "tx": clean}, separators=(",", ":")).encode()


def decode_message(raw, fields: tuple = TX_FIELDS) -> tuple[int | None, dict]:
    if not isinstance(raw, (bytes, bytearray)):
        raise MessageError("not_bytes", "Message value must be bytes.")
    if len(raw) == 0 or len(raw) > MAX_MESSAGE_BYTES:
        raise MessageError("bad_size", "Message is empty or too large.")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise MessageError("not_json", "Message is not valid UTF-8 JSON.") from None
    if not isinstance(obj, dict) or obj.get("schema") != SCHEMA_VERSION:
        raise MessageError("bad_schema", f"Unsupported or missing schema version (expected {SCHEMA_VERSION}).")
    seq = obj.get("seq")
    if seq is not None and (isinstance(seq, bool) or not isinstance(seq, int)):
        raise MessageError("bad_seq", "seq must be an integer.")
    try:
        return seq, validate_transaction(obj.get("tx"), fields)
    except AppError:
        raise MessageError("invalid_transaction", "Transaction failed validation.") from None
