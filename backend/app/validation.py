"""Single source of truth for input rules (single JSON transactions and CSV rows).

Error details name fields and row numbers but never echo submitted values.
Extra JSON fields are rejected (balance columns are deliberately unsupported: they are leakage).
Extra CSV columns (e.g. the PaySim balance columns) are ignored: never parsed, stored or logged.
"""
from __future__ import annotations

import csv
import io
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from backend.app.errors import AppError

MAX_STEP = 1_000_000
MAX_AMOUNT = 1e12
TX_FIELDS = ("step", "type", "amount", "nameOrig", "nameDest")
BALANCE_FIELDS = ("oldbalanceOrg", "oldbalanceDest", "newbalanceOrig", "newbalanceDest")
ALL_FIELDS = TX_FIELDS + BALANCE_FIELDS
TYPE_PATTERN = r"[A-Za-z0-9_\-]{1,32}"
NAME_PATTERN = r"[A-Za-z0-9_\-]{1,64}"
_TYPE_RE, _NAME_RE = re.compile(TYPE_PATTERN), re.compile(NAME_PATTERN)
MAX_HEADER_COLUMNS = 200
MAX_REPORTED_ERRORS = 10


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_transaction(data: Any, fields: tuple = TX_FIELDS) -> dict:
    """`fields` = the inputs the LOADED MODEL needs (its feature contract); anything else is rejected."""
    if not isinstance(data, dict):
        raise AppError(422, "validation_error", "Request body must be a JSON object.")
    errs: list[dict] = []
    n_extra = sum(1 for k in data if k not in fields)
    if n_extra:  # count only: never echo client-supplied field names
        errs.append({"field": "(body)", "reason": f"{n_extra} unsupported field(s); this model accepts: {list(fields)}"})
    for f in fields:
        if f not in data or data[f] is None:
            errs.append({"field": f, "reason": "required"})
    out: dict = {}
    if "step" in data and data["step"] is not None:
        v = data["step"]
        if not _is_num(v) or not math.isfinite(v) or float(v) != int(v):
            errs.append({"field": "step", "reason": "must be a whole number"})
        elif not 0 <= v <= MAX_STEP:
            errs.append({"field": "step", "reason": f"must be between 0 and {MAX_STEP}"})
        else:
            out["step"] = int(v)
    if "amount" in data and data["amount"] is not None:
        v = data["amount"]
        if not _is_num(v) or not math.isfinite(v):
            errs.append({"field": "amount", "reason": "must be a finite number"})
        elif not 0 <= v <= MAX_AMOUNT:
            errs.append({"field": "amount", "reason": f"must be between 0 and {MAX_AMOUNT:g}"})
        else:
            out["amount"] = float(v)
    for f in (x for x in fields if x in BALANCE_FIELDS):
        if f in data and data[f] is not None:
            v = data[f]
            if not _is_num(v) or not math.isfinite(v) or not 0 <= v <= MAX_AMOUNT:
                errs.append({"field": f, "reason": f"must be a finite number between 0 and {MAX_AMOUNT:g}"})
            else:
                out[f] = float(v)
    for f, rx, label in (("type", _TYPE_RE, "1-32 chars of letters, digits, _ or -"),
                         ("nameOrig", _NAME_RE, "1-64 chars of letters, digits, _ or -"),
                         ("nameDest", _NAME_RE, "1-64 chars of letters, digits, _ or -")):
        if f in data and data[f] is not None:
            v = data[f]
            if not isinstance(v, str) or not rx.fullmatch(v):
                errs.append({"field": f, "reason": f"must be {label}"})
            else:
                out[f] = v
    if errs:
        raise AppError(422, "validation_error", "Invalid transaction.", details=errs)
    return out


def parse_csv(raw: bytes, max_bytes: int, max_rows: int, fields: tuple = TX_FIELDS) -> pd.DataFrame:
    """Parse + validate an uploaded CSV held in memory (bounded by max_bytes). No temp files."""
    if len(raw) == 0:
        raise AppError(400, "empty_file", "The uploaded file is empty.")
    if len(raw) > max_bytes:
        raise AppError(413, "file_too_large", f"File exceeds the {max_bytes} byte limit.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise AppError(400, "not_utf8_text", "File must be UTF-8 encoded CSV text.") from None
    if "\x00" in text:
        raise AppError(400, "not_csv", "File does not look like CSV text.")
    reader = csv.reader(io.StringIO(text), strict=True, skipinitialspace=True)
    try:
        header = [h.strip() for h in next(reader)]
    except (StopIteration, csv.Error):
        raise AppError(400, "malformed_csv", "Could not read a CSV header.") from None
    if len(header) > MAX_HEADER_COLUMNS:
        raise AppError(400, "malformed_csv", "Too many columns in header.")
    if len(set(header)) != len(header):
        raise AppError(422, "duplicate_columns", "CSV header contains duplicate column names.")
    missing = [c for c in fields if c not in header]
    if missing:
        raise AppError(422, "missing_columns", "CSV is missing required columns.",
                       details={"missing": missing, "required": list(fields)})
    idx = [header.index(c) for c in fields]  # extra columns are never read
    data: list[list[str]] = []
    lines: list[int] = []
    try:
        for rec in reader:
            if not rec:  # blank line
                continue
            if len(rec) != len(header):
                raise AppError(400, "malformed_csv", "CSV has a row with the wrong number of fields.",
                               details={"row": reader.line_num})
            if len(data) >= max_rows:
                raise AppError(413, "too_many_rows", f"CSV exceeds the {max_rows} row limit.")
            data.append([rec[i] for i in idx])
            lines.append(reader.line_num)
    except csv.Error:
        raise AppError(400, "malformed_csv", "CSV could not be parsed (bad quoting).") from None
    if not data:
        raise AppError(422, "no_rows", "CSV has a header but no data rows.")
    df = pd.DataFrame(data, columns=list(fields), dtype=str)
    df = df[list(fields)].reset_index(drop=True)
    bad: dict[str, np.ndarray] = {}
    step = pd.to_numeric(df["step"], errors="coerce")
    bad["step"] = (step.isna() | ~np.isfinite(step.fillna(0)) | (step < 0) | (step > MAX_STEP)
                   | (step.fillna(0) != np.floor(step.fillna(0)))).to_numpy()
    amount = pd.to_numeric(df["amount"], errors="coerce")
    bad["amount"] = (amount.isna() | ~np.isfinite(amount.fillna(0)) | (amount < 0)
                     | (amount > MAX_AMOUNT)).to_numpy()
    extra_num: dict[str, pd.Series] = {}
    for f in (x for x in fields if x in BALANCE_FIELDS):
        v = pd.to_numeric(df[f], errors="coerce")
        extra_num[f] = v
        bad[f] = (v.isna() | ~np.isfinite(v.fillna(0)) | (v < 0) | (v > MAX_AMOUNT)).to_numpy()
    bad["type"] = (~df["type"].str.fullmatch(TYPE_PATTERN)).to_numpy()
    bad["nameOrig"] = (~df["nameOrig"].str.fullmatch(NAME_PATTERN)).to_numpy()
    bad["nameDest"] = (~df["nameDest"].str.fullmatch(NAME_PATTERN)).to_numpy()
    any_bad = np.zeros(len(df), dtype=bool)
    for m in bad.values():
        any_bad |= m
    if any_bad.any():
        details = []
        for i in np.flatnonzero(any_bad)[:MAX_REPORTED_ERRORS]:
            details.append({"row": lines[int(i)], "fields": [f for f, m in bad.items() if m[i]]})
        raise AppError(422, "invalid_rows", f"{int(any_bad.sum())} row(s) failed validation.",
                       details={"total_invalid_rows": int(any_bad.sum()), "first": details})
    df["step"] = step.astype("int64")
    df["amount"] = amount.astype("float64")
    for f, v in extra_num.items():
        df[f] = v.astype("float64")
    return df
