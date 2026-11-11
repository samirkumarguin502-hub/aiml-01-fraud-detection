"""SQLite repository.

Uses stdlib sqlite3, parameterized SQL, whitelisted sorting and explicit
transactions. Every write operation commits atomically or rolls back.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.app.errors import AppError


CASE_STATUSES = (
    "open",
    "in_review",
    "confirmed_fraud",
    "false_positive",
    "closed",
)

RETRAIN_STATUSES = (
    "requested",
    "candidate_trained",
    "evaluated_passed",
    "evaluated_failed",
    "promoted",
    "rejected",
    "cancelled",
)

RETRAIN_NEXT = {
    "requested": ("candidate_trained", "cancelled"),
    "candidate_trained": (
        "evaluated_passed",
        "evaluated_failed",
        "cancelled",
    ),
    "evaluated_passed": ("promoted", "rejected"),
    "evaluated_failed": ("rejected",),
    "promoted": (),
    "rejected": (),
    "cancelled": (),
}

SORT_COLUMNS = {
    "created_utc": "t.created_utc",
    "id": "t.id",
    "score": "t.score",
    "amount": "t.amount",
    "step": "t.step",
}

BALANCE_COLUMNS = (
    "oldbalance_orig",
    "newbalance_orig",
    "oldbalance_dest",
    "newbalance_dest",
)

MONITORING_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "score",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('single','batch')),
    batch_id TEXT,
    step INTEGER NOT NULL,
    type TEXT NOT NULL,
    amount REAL NOT NULL,
    name_orig TEXT NOT NULL,
    name_dest TEXT NOT NULL,
    oldbalance_orig REAL,
    newbalance_orig REAL,
    oldbalance_dest REAL,
    newbalance_dest REAL,
    score REAL NOT NULL,
    threshold REAL NOT NULL,
    flagged INTEGER NOT NULL CHECK (flagged IN (0,1)),
    type_in_training_vocab INTEGER NOT NULL,
    model_kind TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    model_data_label TEXT
);

CREATE INDEX IF NOT EXISTS ix_tx_orig_step
    ON transactions(name_orig, step);

CREATE INDEX IF NOT EXISTS ix_tx_created
    ON transactions(created_utc);

CREATE INDEX IF NOT EXISTS ix_tx_flagged
    ON transactions(flagged);

CREATE INDEX IF NOT EXISTS ix_tx_batch
    ON transactions(batch_id);

CREATE TABLE IF NOT EXISTS cases (
    transaction_id INTEGER PRIMARY KEY
        REFERENCES transactions(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    note TEXT,
    updated_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id INTEGER NOT NULL
        REFERENCES transactions(id) ON DELETE CASCADE,
    from_status TEXT,
    to_status TEXT NOT NULL,
    created_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    code TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS retrain_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_utc TEXT NOT NULL,
    updated_utc TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    note TEXT,
    candidate TEXT,
    evaluation_json TEXT
);

CREATE TABLE IF NOT EXISTS retrain_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL
        REFERENCES retrain_requests(id) ON DELETE CASCADE,
    from_status TEXT,
    to_status TEXT NOT NULL,
    note TEXT,
    created_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    transaction_id INTEGER PRIMARY KEY
        REFERENCES transactions(id) ON DELETE CASCADE,
    label INTEGER NOT NULL CHECK (label IN (0,1)),
    comment TEXT,
    updated_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_case_events_transaction
    ON case_events(transaction_id);

CREATE INDEX IF NOT EXISTS ix_retrain_events_request
    ON retrain_events(request_id);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin','analyst')),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0,1)),
    created_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    username TEXT NOT NULL,
    role TEXT NOT NULL,
    action TEXT NOT NULL,
    endpoint TEXT,
    request_id TEXT,
    created_utc TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS ix_audit_user ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS ix_audit_created ON audit_logs(created_utc);

"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _positive_limit(value, field: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
    ):
        raise ValueError(f"{field} must be a positive integer.")
    return value


class Repository:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def _conn(self, write: bool = False):
        """Open a connection and manage one explicit transaction."""
        conn = sqlite3.connect(
            self.path,
            timeout=10.0,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")

        try:
            conn.execute(
                "BEGIN IMMEDIATE" if write else "BEGIN"
            )
            yield conn
            conn.execute("COMMIT")

        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise

        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create tables and safely migrate older transaction schemas."""
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        conn = sqlite3.connect(
            self.path,
            timeout=10.0,
        )

        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA)

            existing = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(transactions)"
                )
            }

            for column in BALANCE_COLUMNS:
                if column not in existing:
                    # Column names come exclusively from BALANCE_COLUMNS.
                    conn.execute(
                        f"ALTER TABLE transactions ADD COLUMN {column} REAL"
                    )

            conn.commit()

        except BaseException:
            conn.rollback()
            raise

        finally:
            conn.close()

    def ping(self) -> bool:
        try:
            with self._conn() as conn:
                conn.execute(
                    "SELECT 1 FROM transactions LIMIT 1"
                ).fetchall()
            return True
        except sqlite3.Error:
            return False


    # ------------------------------------------------------------------
    # Authentication and audit
    # ------------------------------------------------------------------

    def create_user(self, username: str, password_hash: str, role: str) -> dict:
        if role not in ("admin", "analyst"):
            raise ValueError("Invalid role")
        username = username.strip()
        if not username or len(username) > 80:
            raise ValueError("Invalid username")

        now = _now()
        with self._conn(write=True) as conn:
            cursor = conn.execute(
                """INSERT INTO users
                   (username, password_hash, role, is_active, created_utc)
                   VALUES (?, ?, ?, 1, ?)""",
                (username, password_hash, role, now),
            )
            user_id = int(cursor.lastrowid)

        return self.get_user(user_id)

    def get_user(self, user_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT id, username, password_hash, role,
                          is_active, created_utc
                   FROM users WHERE id=?""",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT id, username, password_hash, role,
                          is_active, created_utc
                   FROM users WHERE username=? COLLATE NOCASE""",
                (username.strip(),),
            ).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, username, role, is_active, created_utc
                   FROM users ORDER BY id"""
            ).fetchall()
        return [dict(row) for row in rows]

    def record_audit(
        self, user_id: int | None, username: str, role: str,
        action: str, endpoint: str | None = None,
        request_id: str | None = None,
    ) -> None:
        with self._conn(write=True) as conn:
            conn.execute(
                """INSERT INTO audit_logs
                   (user_id, username, role, action, endpoint,
                    request_id, created_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id, username[:80], role[:20], action[:80],
                    endpoint[:200] if endpoint else None,
                    request_id[:100] if request_id else None,
                    _now(),
                ),
            )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert_predictions(
        self,
        rows: list[dict],
        source: str,
        batch_id: str | None,
        model: dict,
    ) -> list[int]:
        if source not in ("single", "batch"):
            raise AppError(
                422,
                "validation_error",
                "Invalid prediction source.",
            )

        if not isinstance(rows, list) or not rows:
            raise AppError(
                422,
                "validation_error",
                "Prediction rows cannot be empty.",
            )

        if not isinstance(model, dict):
            raise AppError(
                422,
                "validation_error",
                "Invalid model metadata.",
            )

        required_model_fields = ("kind", "sha256")

        if any(not model.get(key) for key in required_model_fields):
            raise AppError(
                422,
                "validation_error",
                "Model metadata is incomplete.",
            )

        ids: list[int] = []
        now = _now()

        with self._conn(write=True) as conn:
            for row in rows:
                try:
                    step = int(row["step"])
                    amount = float(row["amount"])
                    score = float(row["score"])
                    threshold = float(row["threshold"])
                    flagged = int(row["flagged"])
                    type_known = bool(row["type_known"])

                    if step < 0 or amount < 0:
                        raise ValueError("Invalid transaction values.")

                    if (
                        not 0.0 <= score <= 1.0
                        or not 0.0 <= threshold <= 1.0
                    ):
                        raise ValueError("Invalid score or threshold.")

                    transaction_type = str(row["type"])
                    name_orig = str(row["nameOrig"])
                    name_dest = str(row["nameDest"])

                    if not transaction_type or not name_orig or not name_dest:
                        raise ValueError("Transaction identifiers are empty.")

                    balances = [
                        (
                            None
                            if row.get(key) is None
                            else float(row[key])
                        )
                        for key in (
                            "oldbalanceOrg",
                            "newbalanceOrig",
                            "oldbalanceDest",
                            "newbalanceDest",
                        )
                    ]

                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    raise AppError(
                        422,
                        "validation_error",
                        "Invalid prediction row.",
                    ) from exc

                if any(
                    value is not None and (
                        not pd.notna(value) or value < 0
                    )
                    for value in balances
                ):
                    raise AppError(
                        422,
                        "validation_error",
                        "Invalid balance values.",
                    )

                cursor = conn.execute(
                    """
                    INSERT INTO transactions (
                        created_utc, source, batch_id,
                        step, type, amount, name_orig, name_dest,
                        score, threshold, flagged,
                        type_in_training_vocab,
                        model_kind, model_sha256, model_data_label,
                        oldbalance_orig, newbalance_orig,
                        oldbalance_dest, newbalance_dest
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?)
                    """,
                    (
                        now,
                        source,
                        batch_id,
                        step,
                        transaction_type,
                        amount,
                        name_orig,
                        name_dest,
                        score,
                        threshold,
                        int(flagged),
                        int(type_known),
                        model["kind"],
                        model["sha256"],
                        model.get("data_label"),
                        *balances,
                    ),
                )

                transaction_id = int(cursor.lastrowid)
                ids.append(transaction_id)

                if flagged:
                    conn.execute(
                        """
                        INSERT INTO cases (
                            transaction_id, status, note, updated_utc
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (transaction_id, "open", None, now),
                    )

                    conn.execute(
                        """
                        INSERT INTO case_events (
                            transaction_id, from_status,
                            to_status, created_utc
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (transaction_id, None, "open", now),
                    )

        return ids

    def _require_tx(self, conn, tid: int) -> None:
        if conn.execute(
            "SELECT 1 FROM transactions WHERE id=?",
            (tid,),
        ).fetchone() is None:
            raise AppError(
                404,
                "not_found",
                "Transaction not found.",
            )

    def update_case(
        self,
        tid: int,
        status: str,
        note: str | None,
    ) -> dict:
        if status not in CASE_STATUSES:
            raise AppError(
                422,
                "validation_error",
                "Invalid case status.",
                details={"allowed": list(CASE_STATUSES)},
            )

        now = _now()

        with self._conn(write=True) as conn:
            self._require_tx(conn, tid)

            previous = conn.execute(
                "SELECT status FROM cases WHERE transaction_id=?",
                (tid,),
            ).fetchone()

            if previous is None:
                conn.execute(
                    """
                    INSERT INTO cases (
                        transaction_id, status, note, updated_utc
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (tid, status, note, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE cases
                    SET status=?, note=?, updated_utc=?
                    WHERE transaction_id=?
                    """,
                    (status, note, now, tid),
                )

            conn.execute(
                """
                INSERT INTO case_events (
                    transaction_id, from_status, to_status, created_utc
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    tid,
                    previous["status"] if previous else None,
                    status,
                    now,
                ),
            )

        return {
            "transaction_id": tid,
            "status": status,
            "note": note,
            "updated_utc": now,
        }

    def set_feedback(
        self,
        tid: int,
        label: int,
        comment: str | None,
    ) -> dict:
        if (
            not isinstance(label, int)
            or isinstance(label, bool)
            or label not in (0, 1)
        ):
            raise AppError(
                422,
                "validation_error",
                "label must be integer 0 or 1.",
            )

        now = _now()

        with self._conn(write=True) as conn:
            self._require_tx(conn, tid)

            conn.execute(
                """
                INSERT INTO feedback (
                    transaction_id, label, comment, updated_utc
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(transaction_id)
                DO UPDATE SET
                    label=excluded.label,
                    comment=excluded.comment,
                    updated_utc=excluded.updated_utc
                """,
                (tid, label, comment, now),
            )

        return {
            "transaction_id": tid,
            "label": label,
            "comment": comment,
            "updated_utc": now,
        }

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        result = dict(row)
        result["flagged"] = bool(result["flagged"])
        result["type_in_training_vocab"] = bool(
            result["type_in_training_vocab"]
        )
        return result

    _SELECT = (
        "SELECT t.*, "
        "c.status AS case_status, "
        "c.note AS case_note, "
        "f.label AS feedback_label "
        "FROM transactions t "
        "LEFT JOIN cases c ON c.transaction_id=t.id "
        "LEFT JOIN feedback f ON f.transaction_id=t.id"
    )

    def get_transaction(self, tid: int) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                self._SELECT + " WHERE t.id=?",
                (tid,),
            ).fetchone()

            if row is None:
                raise AppError(
                    404,
                    "not_found",
                    "Transaction not found.",
                )

            events = [
                dict(event)
                for event in conn.execute(
                    """
                    SELECT from_status, to_status, created_utc
                    FROM case_events
                    WHERE transaction_id=?
                    ORDER BY id
                    """,
                    (tid,),
                )
            ]

        result = self._row(row)
        result["case_history"] = events
        return result

    def list_transactions(
        self,
        *,
        page: int,
        page_size: int,
        sort: str = "id",
        order: str = "desc",
        flagged: bool | None = None,
        type_: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        source: str | None = None,
        case_status: str | None = None,
        batch_id: str | None = None,
    ) -> dict:
        if (
            not isinstance(page, int)
            or isinstance(page, bool)
            or page < 1
            or not isinstance(page_size, int)
            or isinstance(page_size, bool)
            or page_size < 1
        ):
            raise AppError(
                422,
                "validation_error",
                "Invalid pagination.",
            )

        if sort not in SORT_COLUMNS or order not in ("asc", "desc"):
            raise AppError(
                422,
                "validation_error",
                "Invalid sort or order.",
                details={
                    "sort_allowed": sorted(SORT_COLUMNS),
                    "order_allowed": ["asc", "desc"],
                },
            )

        if flagged is not None and not isinstance(flagged, bool):
            raise AppError(
                422,
                "validation_error",
                "flagged must be a Boolean.",
            )

        if source not in (None, "single", "batch"):
            raise AppError(
                422,
                "validation_error",
                "Invalid source.",
            )

        if case_status is not None and case_status not in CASE_STATUSES:
            raise AppError(
                422,
                "validation_error",
                "Invalid case status.",
            )

        for value, field in (
            (min_score, "min_score"),
            (max_score, "max_score"),
        ):
            if value is not None:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    raise AppError(
                        422,
                        "validation_error",
                        f"{field} must be numeric.",
                    ) from None

                if not 0.0 <= numeric <= 1.0:
                    raise AppError(
                        422,
                        "validation_error",
                        f"{field} must be between 0 and 1.",
                    )

        if (
            min_score is not None
            and max_score is not None
            and min_score > max_score
        ):
            raise AppError(
                422,
                "validation_error",
                "min_score cannot exceed max_score.",
            )

        conditions = []
        parameters = []

        filters = (
            ("t.flagged=?", None if flagged is None else int(flagged)),
            ("t.type=?", type_),
            ("t.score>=?", min_score),
            ("t.score<=?", max_score),
            ("t.source=?", source),
            ("c.status=?", case_status),
            ("t.batch_id=?", batch_id),
        )

        for condition, value in filters:
            if value is not None:
                conditions.append(condition)
                parameters.append(value)

        where = (
            " WHERE " + " AND ".join(conditions)
            if conditions
            else ""
        )

        base = (
            " FROM transactions t "
            "LEFT JOIN cases c ON c.transaction_id=t.id "
            + where
        )

        offset = (page - 1) * page_size

        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*)" + base,
                parameters,
            ).fetchone()[0]

            rows = conn.execute(
                self._SELECT
                + where
                + f" ORDER BY {SORT_COLUMNS[sort]} {order.upper()}, "
                  "t.id DESC LIMIT ? OFFSET ?",
                [*parameters, page_size, offset],
            ).fetchall()

        return {
            "items": [self._row(row) for row in rows],
            "total": int(total),
            "page": page,
            "page_size": page_size,
        }

    def history_for(
        self,
        names: list[str],
        limit: int = 50_000,
    ) -> list[dict]:
        """Return previously scored transactions for requested senders."""
        _positive_limit(limit, "limit")

        if not names:
            return []

        unique_names = list(dict.fromkeys(names))
        output: list[dict] = []

        with self._conn() as conn:
            for index in range(0, len(unique_names), 500):
                chunk = unique_names[index:index + 500]

                placeholders = ",".join("?" for _ in chunk)
                remaining = limit - len(output)

                if remaining <= 0:
                    break

                rows = conn.execute(
                    """
                    SELECT
                        step,
                        type,
                        amount,
                        name_orig AS nameOrig,
                        name_dest AS nameDest
                    FROM transactions
                    WHERE name_orig IN (
                    """
                    + placeholders
                    + """
                    )
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    [*chunk, remaining],
                )

                output.extend(
                    dict(row)
                    for row in rows
                )

        return output

    def summary(self) -> dict:
        with self._conn() as conn:
            total, flagged = conn.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(SUM(flagged), 0)
                FROM transactions
                """
            ).fetchone()

            histogram = [0] * 10

            for row in conn.execute(
                """
                SELECT
                    MIN(CAST(score * 10 AS INTEGER), 9) AS bin,
                    COUNT(*) AS count
                FROM transactions
                GROUP BY bin
                """
            ):
                bin_index = max(
                    0,
                    min(9, row["bin"]),
                )
                histogram[bin_index] += row["count"]

            by_day = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        substr(created_utc, 1, 10) AS day,
                        COUNT(*) AS transactions,
                        SUM(flagged) AS flagged
                    FROM transactions
                    GROUP BY day
                    ORDER BY day DESC
                    LIMIT 30
                    """
                )
            ][::-1]

            by_type = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        type,
                        COUNT(*) AS transactions,
                        SUM(flagged) AS flagged
                    FROM transactions
                    GROUP BY type
                    ORDER BY transactions DESC
                    """
                )
            ]

            cases = {
                row["status"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM cases
                    GROUP BY status
                    """
                )
            }

            feedback = {
                row["label"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT label, COUNT(*) AS count
                    FROM feedback
                    GROUP BY label
                    """
                )
            }

        return {
            "total_transactions": int(total),
            "flagged_transactions": int(flagged),
            "score_histogram": [
                {
                    "bin_start": index / 10,
                    "bin_end": (index + 1) / 10,
                    "count": count,
                }
                for index, count in enumerate(histogram)
            ],
            "by_day": by_day,
            "by_type": by_type,
            "case_status_counts": {
                status: cases.get(status, 0)
                for status in CASE_STATUSES
            },
            "feedback": {
                "total": sum(feedback.values()),
                "labelled_fraud": feedback.get(1, 0),
                "labelled_not_fraud": feedback.get(0, 0),
            },
        }

    # ------------------------------------------------------------------
    # Monitoring inputs
    # ------------------------------------------------------------------

    def record_rejection(
        self,
        endpoint: str,
        code: str,
    ) -> None:
        try:
            with self._conn(write=True) as conn:
                conn.execute(
                    """
                    INSERT INTO rejections (
                        created_utc, endpoint, code
                    )
                    VALUES (?, ?, ?)
                    """,
                    (_now(), endpoint[:40], code[:40]),
                )
        except sqlite3.Error:
            # Rejection monitoring must not break request handling.
            pass

    def rejection_counts(self) -> dict:
        with self._conn() as conn:
            return {
                row["code"]: row["count"]
                for row in conn.execute(
                    """
                    SELECT code, COUNT(*) AS count
                    FROM rejections
                    GROUP BY code
                    ORDER BY count DESC
                    """
                )
            }

    def monitoring_rows(
        self,
        limit: int = 5000,
    ) -> pd.DataFrame:
        _positive_limit(limit, "limit")

        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    step,
                    type,
                    amount,
                    name_orig AS nameOrig,
                    name_dest AS nameDest,
                    score,
                    oldbalance_orig AS oldbalanceOrg,
                    newbalance_orig AS newbalanceOrig,
                    oldbalance_dest AS oldbalanceDest,
                    newbalance_dest AS newbalanceDest
                FROM (
                    SELECT *
                    FROM transactions
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (limit,),
            ).fetchall()

        dataframe = pd.DataFrame(
            [dict(row) for row in rows],
            columns=MONITORING_COLUMNS,
        )

        numeric_columns = (
            "step",
            "amount",
            "score",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        )

        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

        return dataframe

    def labelled_rows(
        self,
        limit: int = 5000,
    ) -> pd.DataFrame:
        _positive_limit(limit, "limit")

        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT
                    t.score AS score,
                    t.threshold AS threshold,
                    f.label AS label
                FROM feedback f
                JOIN transactions t
                    ON t.id=f.transaction_id
                ORDER BY f.updated_utc DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        dataframe = pd.DataFrame(
            [dict(row) for row in rows],
            columns=["score", "threshold", "label"],
        )

        for column in ("score", "threshold", "label"):
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

        return dataframe

    # ------------------------------------------------------------------
    # Retraining workflow
    # ------------------------------------------------------------------

    def create_retrain(
        self,
        reason: str,
        note: str | None,
    ) -> dict:
        now = _now()

        with self._conn(write=True) as conn:
            cursor = conn.execute(
                """
                INSERT INTO retrain_requests (
                    created_utc,
                    updated_utc,
                    status,
                    reason,
                    note
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (now, now, "requested", reason, note),
            )

            request_id = int(cursor.lastrowid)

            conn.execute(
                """
                INSERT INTO retrain_events (
                    request_id,
                    from_status,
                    to_status,
                    note,
                    created_utc
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (request_id, None, "requested", note, now),
            )

        return self.get_retrain(request_id)

    def get_retrain(self, rid: int) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM retrain_requests
                WHERE id=?
                """,
                (rid,),
            ).fetchone()

            if row is None:
                raise AppError(
                    404,
                    "not_found",
                    "Retraining request not found.",
                )

            events = [
                dict(event)
                for event in conn.execute(
                    """
                    SELECT
                        from_status,
                        to_status,
                        note,
                        created_utc
                    FROM retrain_events
                    WHERE request_id=?
                    ORDER BY id
                    """,
                    (rid,),
                )
            ]

        result = dict(row)

        try:
            evaluation_json = result.pop("evaluation_json")
            result["evaluation"] = (
                json.loads(evaluation_json)
                if evaluation_json
                else None
            )
        except (ValueError, TypeError):
            result["evaluation"] = None
            result["evaluation_error"] = "Stored evaluation JSON is invalid."

        result["events"] = events
        result["allowed_next"] = list(
            RETRAIN_NEXT.get(result["status"], ())
        )

        return result

    def list_retrain(self) -> list[dict]:
        with self._conn() as conn:
            ids = [
                row["id"]
                for row in conn.execute(
                    """
                    SELECT id
                    FROM retrain_requests
                    ORDER BY id DESC
                    LIMIT 100
                    """
                )
            ]

        return [
            self.get_retrain(request_id)
            for request_id in ids
        ]

    def transition_retrain(
        self,
        rid: int,
        to: str,
        note: str | None,
        candidate: str | None = None,
        evaluation: dict | None = None,
    ) -> dict:
        now = _now()

        with self._conn(write=True) as conn:
            row = conn.execute(
                """
                SELECT status
                FROM retrain_requests
                WHERE id=?
                """,
                (rid,),
            ).fetchone()

            if row is None:
                raise AppError(
                    404,
                    "not_found",
                    "Retraining request not found.",
                )

            current_status = row["status"]

            if to not in RETRAIN_NEXT.get(current_status, ()):
                raise AppError(
                    409,
                    "invalid_transition",
                    f"Cannot move from '{current_status}' to '{to}'.",
                    details={
                        "allowed": list(
                            RETRAIN_NEXT.get(current_status, ())
                        ),
                    },
                )

            try:
                evaluation_json = (
                    json.dumps(
                        evaluation,
                        allow_nan=False,
                    )
                    if evaluation is not None
                    else None
                )
            except (TypeError, ValueError):
                raise AppError(
                    422,
                    "validation_error",
                    "Evaluation data must be valid JSON.",
                ) from None

            conn.execute(
                """
                UPDATE retrain_requests
                SET
                    status=?,
                    updated_utc=?,
                    candidate=COALESCE(?, candidate),
                    evaluation_json=COALESCE(?, evaluation_json)
                WHERE id=?
                """,
                (
                    to,
                    now,
                    candidate,
                    evaluation_json,
                    rid,
                ),
            )

            conn.execute(
                """
                INSERT INTO retrain_events (
                    request_id,
                    from_status,
                    to_status,
                    note,
                    created_utc
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    current_status,
                    to,
                    note,
                    now,
                ),
            )

        return self.get_retrain(rid)