"""Single source of truth for input rules (single JSON transactions and CSV rows).

Error details identify fields and row numbers but never echo submitted values.

Extra JSON fields are rejected. The loaded model's feature contract determines
which input fields are accepted.

Extra CSV columns are ignored: they are never extracted, stored or logged.
Balance fields are optional, but are validated when provided.
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

TX_FIELDS = (
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
)

BALANCE_FIELDS = (
    "oldbalanceOrg",
    "oldbalanceDest",
    "newbalanceOrig",
    "newbalanceDest",
)

ALL_FIELDS = TX_FIELDS + BALANCE_FIELDS

TYPE_PATTERN = r"[A-Za-z0-9_\-]{1,32}"
NAME_PATTERN = r"[A-Za-z0-9_\-]{1,64}"

_TYPE_RE = re.compile(TYPE_PATTERN)
_NAME_RE = re.compile(NAME_PATTERN)

MAX_HEADER_COLUMNS = 200
MAX_REPORTED_ERRORS = 10


def _is_num(value: Any) -> bool:
    """Accept native integer/float values but reject Boolean values."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    """Check numeric finiteness without allowing integer overflow to escape."""
    if not _is_num(value):
        return False

    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _validate_fields(fields: tuple) -> None:
    """Validate the internal model input contract."""
    if (
        not isinstance(fields, tuple)
        or not fields
        or any(not isinstance(field, str) for field in fields)
        or len(set(fields)) != len(fields)
        or not set(fields).issubset(ALL_FIELDS)
        or not set(TX_FIELDS).issubset(fields)
    ):
        raise ValueError("Invalid model input field contract.")


def _validate_numeric_series(
    series: pd.Series,
    field: str,
) -> np.ndarray:
    """Return a Boolean mask identifying invalid numeric CSV values."""
    values = pd.to_numeric(series, errors="coerce")

    try:
        numeric_array = values.to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        numeric_array = np.full(len(values), np.nan, dtype=np.float64)

    finite = np.isfinite(numeric_array)

    invalid = (
        values.isna().to_numpy(dtype=bool)
        | ~finite
    )

    if field == "step":
        invalid |= (
            (numeric_array < 0)
            | (numeric_array > MAX_STEP)
            | (numeric_array != np.floor(
                np.where(finite, numeric_array, 0)
            ))
        )

    else:
        invalid |= (
            (numeric_array < 0)
            | (numeric_array > MAX_AMOUNT)
        )

    return invalid


def validate_transaction(
    data: Any,
    fields: tuple = TX_FIELDS,
) -> dict:
    """Validate one JSON transaction.

    `fields` contains the raw inputs accepted by the loaded model's
    feature contract. Balance fields are accepted only when present
    in the contract, but are optional in the submitted request.
    """
    _validate_fields(fields)

    if not isinstance(data, dict):
        raise AppError(
            422,
            "validation_error",
            "Request body must be a JSON object.",
        )

    errors: list[dict] = []

    # Reject fields outside the loaded model's input contract.
    extra_count = sum(
        1 for key in data
        if key not in fields
    )

    if extra_count:
        errors.append(
            {
                "field": "(body)",
                "reason": (
                    f"{extra_count} unsupported field(s); "
                    f"this model accepts: {list(fields)}"
                ),
            }
        )

    # The five core transaction fields are always mandatory.
    for field in TX_FIELDS:
        if field not in data or data[field] is None:
            errors.append(
                {
                    "field": field,
                    "reason": "required",
                }
            )

    output: dict = {}

    # Transaction step.
    if "step" in data and data["step"] is not None:
        value = data["step"]

        if not _finite_number(value):
            errors.append(
                {
                    "field": "step",
                    "reason": "must be a finite whole number",
                }
            )
        elif value < 0 or value > MAX_STEP:
            errors.append(
                {
                    "field": "step",
                    "reason": f"must be between 0 and {MAX_STEP}",
                }
            )
        elif float(value) != int(value):
            errors.append(
                {
                    "field": "step",
                    "reason": "must be a whole number",
                }
            )
        else:
            output["step"] = int(value)

    # Transaction amount.
    if "amount" in data and data["amount"] is not None:
        value = data["amount"]

        if not _finite_number(value):
            errors.append(
                {
                    "field": "amount",
                    "reason": "must be a finite number",
                }
            )
        elif not 0 <= value <= MAX_AMOUNT:
            errors.append(
                {
                    "field": "amount",
                    "reason": f"must be between 0 and {MAX_AMOUNT:g}",
                }
            )
        else:
            output["amount"] = float(value)

    # Optional model-specific balance inputs.
    for field in BALANCE_FIELDS:
        if field not in fields:
            continue

        if field not in data or data[field] is None:
            errors.append({
                "field": field,
                "reason": "required",
            })
            continue

        value = data[field]

        if (
            not _finite_number(value)
            or not 0 <= value <= MAX_AMOUNT
        ):
            errors.append(
                {
                    "field": field,
                    "reason": (
                        "must be a finite number between "
                        f"0 and {MAX_AMOUNT:g}"
                    ),
                }
            )
        else:
            output[field] = float(value)

    # Categorical fields and entity identifiers.
    for field, pattern, label in (
        (
            "type",
            _TYPE_RE,
            "1-32 chars of letters, digits, _ or -",
        ),
        (
            "nameOrig",
            _NAME_RE,
            "1-64 chars of letters, digits, _ or -",
        ),
        (
            "nameDest",
            _NAME_RE,
            "1-64 chars of letters, digits, _ or -",
        ),
    ):
        if field in data and data[field] is not None:
            value = data[field]

            if (
                not isinstance(value, str)
                or not pattern.fullmatch(value)
            ):
                errors.append(
                    {
                        "field": field,
                        "reason": f"must be {label}",
                    }
                )
            else:
                output[field] = value

    if errors:
        raise AppError(
            422,
            "validation_error",
            "Invalid transaction.",
            details=errors,
        )

    return output


def parse_csv(
    raw: bytes,
    max_bytes: int,
    max_rows: int,
    fields: tuple = TX_FIELDS,
) -> pd.DataFrame:
    """Parse and validate an uploaded CSV entirely in memory.

    The uploaded file is bounded by max_bytes.

    Required CSV columns:
        step, type, amount, nameOrig, nameDest

    Optional CSV columns:
        Balance fields accepted by the model contract.

    Extra columns are ignored and never extracted into the DataFrame.
    """
    _validate_fields(fields)

    if not isinstance(raw, bytes):
        raise AppError(
            400,
            "not_csv",
            "Uploaded file must contain CSV bytes.",
        )

    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or max_bytes < 1
        or not isinstance(max_rows, int)
        or isinstance(max_rows, bool)
        or max_rows < 1
    ):
        raise ValueError("Invalid CSV upload limits.")

    if len(raw) == 0:
        raise AppError(
            400,
            "empty_file",
            "The uploaded file is empty.",
        )

    if len(raw) > max_bytes:
        raise AppError(
            413,
            "file_too_large",
            f"File exceeds the {max_bytes} byte limit.",
        )

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise AppError(
            400,
            "not_utf8_text",
            "File must be UTF-8 encoded CSV text.",
        ) from None

    if "\x00" in text:
        raise AppError(
            400,
            "not_csv",
            "File does not look like CSV text.",
        )

    reader = csv.reader(
        io.StringIO(text, newline=""),
        strict=True,
        skipinitialspace=True,
    )

    # Read and validate header.
    try:
        header = [
            column.strip()
            for column in next(reader)
        ]
    except (StopIteration, csv.Error):
        raise AppError(
            400,
            "malformed_csv",
            "Could not read a CSV header.",
        ) from None

    if not header or len(header) > MAX_HEADER_COLUMNS:
        raise AppError(
            400,
            "malformed_csv",
            "Too many columns in header or empty header.",
        )

    if any(not column for column in header):
        raise AppError(
            400,
            "malformed_csv",
            "CSV header contains an empty column name.",
        )

    if len(set(header)) != len(header):
        raise AppError(
            422,
            "duplicate_columns",
            "CSV header contains duplicate column names.",
        )

    # All active model contract columns are mandatory.
    missing = [
        field
        for field in fields
        if field not in header
    ]

    if missing:
        raise AppError(
            422,
            "missing_columns",
            "CSV is missing required columns.",
            details={
                "missing": missing,
                "required": list(fields),
            },
        )

    # Extract core fields plus optional balance fields that are
    # both accepted by the model contract and present in the CSV.
    selected_fields = [
        field
        for field in fields
        if field in header
    ]

    indices = [
        header.index(field)
        for field in selected_fields
    ]

    data: list[list[str]] = []
    line_numbers: list[int] = []

    try:
        for record in reader:
            if not record:
                continue

            if len(record) != len(header):
                raise AppError(
                    400,
                    "malformed_csv",
                    "CSV has a row with the wrong number of fields.",
                    details={
                        "row": reader.line_num,
                    },
                )

            if len(data) >= max_rows:
                raise AppError(
                    413,
                    "too_many_rows",
                    f"CSV exceeds the {max_rows} row limit.",
                )

            # Extract only contract-approved fields.
            data.append(
                [
                    record[index]
                    for index in indices
                ]
            )

            line_numbers.append(reader.line_num)

    except csv.Error:
        raise AppError(
            400,
            "malformed_csv",
            "CSV could not be parsed (bad quoting).",
        ) from None

    if not data:
        raise AppError(
            422,
            "no_rows",
            "CSV has a header but no data rows.",
        )

    dataframe = pd.DataFrame(
        data,
        columns=selected_fields,
        dtype=str,
    ).reset_index(drop=True)

    bad: dict[str, np.ndarray] = {}

    # Validate numeric transaction fields.
    for field in ("step", "amount"):
        bad[field] = _validate_numeric_series(
            dataframe[field],
            field,
        )

    # Validate balance fields only when supplied.
    for field in BALANCE_FIELDS:
        if field not in dataframe.columns:
            continue

        bad[field] = _validate_numeric_series(
            dataframe[field],
            field,
        )

    # Validate categorical fields and identifiers.
    for field, pattern in (
        ("type", TYPE_PATTERN),
        ("nameOrig", NAME_PATTERN),
        ("nameDest", NAME_PATTERN),
    ):
        bad[field] = (
            ~dataframe[field].str.fullmatch(pattern)
        ).to_numpy(dtype=bool)

    # Aggregate invalid rows without exposing submitted values.
    any_bad = np.zeros(
        len(dataframe),
        dtype=bool,
    )

    for mask in bad.values():
        any_bad |= mask

    if any_bad.any():
        details = []

        for index in np.flatnonzero(any_bad)[
            :MAX_REPORTED_ERRORS
        ]:
            details.append(
                {
                    "row": line_numbers[int(index)],
                    "fields": [
                        field
                        for field, mask in bad.items()
                        if mask[index]
                    ],
                }
            )

        raise AppError(
            422,
            "invalid_rows",
            f"{int(any_bad.sum())} row(s) failed validation.",
            details={
                "total_invalid_rows": int(any_bad.sum()),
                "first": details,
            },
        )

    # Convert validated numeric fields to stable numeric dtypes.
    dataframe["step"] = pd.to_numeric(
        dataframe["step"],
        errors="raise",
    ).astype("int64")

    dataframe["amount"] = pd.to_numeric(
        dataframe["amount"],
        errors="raise",
    ).astype("float64")

    for field in BALANCE_FIELDS:
        if field in dataframe.columns:
            dataframe[field] = pd.to_numeric(
                dataframe[field],
                errors="raise",
            ).astype("float64")

    return dataframe