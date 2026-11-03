"""SQLite repository (stdlib sqlite3). Parametrised SQL only; sort columns are whitelisted;
every write happens inside one explicit transaction that rolls back on error."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.app.errors import AppError

CASE_STATUSES = ("open", "in_review", "confirmed_fraud", "false_positive", "closed")
RETRAIN_STATUSES = ("requested", "candidate_trained", "evaluated_passed", "evaluated_failed",
                    "promoted", "rejected", "cancelled")
RETRAIN_NEXT = {"requested": ("candidate_trained", "cancelled"),
                "candidate_trained": ("evaluated_passed", "evaluated_failed", "cancelled"),
                "evaluated_passed": ("promoted", "rejected"),
                "evaluated_failed": ("rejected",),
                "promoted": (), "rejected": (), "cancelled": ()}
SORT_COLUMNS = {"created_utc": "t.created_utc", "id": "t.id", "score": "t.score",
                "amount": "t.amount", "step": "t.step"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_utc TEXT NOT NULL,
  source TEXT NOT NULL CHECK (source IN ('single','batch')),
  batch_id TEXT,
  step INTEGER NOT NULL, type TEXT NOT NULL, amount REAL NOT NULL,
  name_orig TEXT NOT NULL, name_dest TEXT NOT NULL,
  oldbalance_orig REAL, newbalance_orig REAL, oldbalance_dest REAL, newbalance_dest REAL,
  score REAL NOT NULL, threshold REAL NOT NULL, flagged INTEGER NOT NULL CHECK (flagged IN (0,1)),
  type_in_training_vocab INTEGER NOT NULL,
  model_kind TEXT NOT NULL, model_sha256 TEXT NOT NULL, model_data_label TEXT
);
CREATE INDEX IF NOT EXISTS ix_tx_orig_step ON transactions(name_orig, step);
CREATE INDEX IF NOT EXISTS ix_tx_created ON transactions(created_utc);
CREATE INDEX IF NOT EXISTS ix_tx_flagged ON transactions(flagged);
CREATE TABLE IF NOT EXISTS cases (
  transaction_id INTEGER PRIMARY KEY REFERENCES transactions(id) ON DELETE CASCADE,
  status TEXT NOT NULL, note TEXT, updated_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  transaction_id INTEGER NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
  from_status TEXT, to_status TEXT NOT NULL, created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rejections (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL, endpoint TEXT NOT NULL, code TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retrain_requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL, updated_utc TEXT NOT NULL,
  status TEXT NOT NULL, reason TEXT NOT NULL, note TEXT, candidate TEXT, evaluation_json TEXT
);
CREATE TABLE IF NOT EXISTS retrain_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id INTEGER NOT NULL REFERENCES retrain_requests(id) ON DELETE CASCADE,
  from_status TEXT, to_status TEXT NOT NULL, note TEXT, created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
  transaction_id INTEGER PRIMARY KEY REFERENCES transactions(id) ON DELETE CASCADE,
  label INTEGER NOT NULL CHECK (label IN (0,1)), comment TEXT, updated_utc TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Repository:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def _conn(self, write: bool = False):
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5.0)
        try:
            conn.executescript(SCHEMA)
            # migrate databases created before balance columns existed
            have = {r[1] for r in conn.execute("PRAGMA table_info(transactions)")}
            for col in ("oldbalance_orig", "newbalance_orig", "oldbalance_dest", "newbalance_dest"):
                if col not in have:
                    conn.execute(f"ALTER TABLE transactions ADD COLUMN {col} REAL")
            conn.commit()
        finally:
            conn.close()

    def ping(self) -> bool:
        try:
            with self._conn() as c:
                c.execute("SELECT 1 FROM transactions LIMIT 1").fetchall()
            return True
        except sqlite3.Error:
            return False

    # ---- writes ----
    def insert_predictions(self, rows: list[dict], source: str, batch_id: str | None,
                           model: dict) -> list[int]:
        ids: list[int] = []
        now = _now()
        with self._conn(write=True) as c:
            for r in rows:
                cur = c.execute(
                    "INSERT INTO transactions (created_utc,source,batch_id,step,type,amount,name_orig,"
                    "name_dest,score,threshold,flagged,type_in_training_vocab,model_kind,model_sha256,"
                    "model_data_label,oldbalance_orig,newbalance_orig,oldbalance_dest,newbalance_dest) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (now, source, batch_id, int(r["step"]), str(r["type"]), float(r["amount"]), str(r["nameOrig"]),
                     str(r["nameDest"]), float(r["score"]), float(r["threshold"]), int(r["flagged"]),
                     int(r["type_known"]), model["kind"], model["sha256"], model.get("data_label"),
                     *[None if r.get(k) is None else float(r[k])
                       for k in ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")]))
                tid = int(cur.lastrowid)
                ids.append(tid)
                if r["flagged"]:
                    c.execute("INSERT INTO cases (transaction_id,status,note,updated_utc) VALUES (?,?,?,?)",
                              (tid, "open", None, now))
                    c.execute("INSERT INTO case_events (transaction_id,from_status,to_status,created_utc)"
                              " VALUES (?,?,?,?)", (tid, None, "open", now))
        return ids

    def _require_tx(self, c, tid: int) -> None:
        if c.execute("SELECT 1 FROM transactions WHERE id=?", (tid,)).fetchone() is None:
            raise AppError(404, "not_found", "Transaction not found.")

    def update_case(self, tid: int, status: str, note: str | None) -> dict:
        if status not in CASE_STATUSES:
            raise AppError(422, "validation_error", "Invalid case status.",
                           details={"allowed": list(CASE_STATUSES)})
        now = _now()
        with self._conn(write=True) as c:
            self._require_tx(c, tid)
            prev = c.execute("SELECT status FROM cases WHERE transaction_id=?", (tid,)).fetchone()
            if prev is None:
                c.execute("INSERT INTO cases (transaction_id,status,note,updated_utc) VALUES (?,?,?,?)",
                          (tid, status, note, now))
            else:
                c.execute("UPDATE cases SET status=?, note=?, updated_utc=? WHERE transaction_id=?",
                          (status, note, now, tid))
            c.execute("INSERT INTO case_events (transaction_id,from_status,to_status,created_utc)"
                      " VALUES (?,?,?,?)", (tid, prev["status"] if prev else None, status, now))
        return {"transaction_id": tid, "status": status, "note": note, "updated_utc": now}

    def set_feedback(self, tid: int, label: int, comment: str | None) -> dict:
        if label not in (0, 1):
            raise AppError(422, "validation_error", "label must be 0 (not fraud) or 1 (fraud).")
        now = _now()
        with self._conn(write=True) as c:
            self._require_tx(c, tid)
            c.execute("INSERT INTO feedback (transaction_id,label,comment,updated_utc) VALUES (?,?,?,?)"
                      " ON CONFLICT(transaction_id) DO UPDATE SET label=excluded.label,"
                      " comment=excluded.comment, updated_utc=excluded.updated_utc",
                      (tid, label, comment, now))
        return {"transaction_id": tid, "label": label, "comment": comment, "updated_utc": now}

    # ---- reads ----
    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["flagged"] = bool(d["flagged"])
        d["type_in_training_vocab"] = bool(d["type_in_training_vocab"])
        return d

    _SELECT = ("SELECT t.*, c.status AS case_status, c.note AS case_note, f.label AS feedback_label "
               "FROM transactions t LEFT JOIN cases c ON c.transaction_id=t.id "
               "LEFT JOIN feedback f ON f.transaction_id=t.id")

    def get_transaction(self, tid: int) -> dict:
        with self._conn() as c:
            r = c.execute(self._SELECT + " WHERE t.id=?", (tid,)).fetchone()
            if r is None:
                raise AppError(404, "not_found", "Transaction not found.")
            events = [dict(e) for e in c.execute(
                "SELECT from_status,to_status,created_utc FROM case_events WHERE transaction_id=? ORDER BY id",
                (tid,))]
        d = self._row(r)
        d["case_history"] = events
        return d

    def list_transactions(self, *, page: int, page_size: int, sort: str = "id", order: str = "desc",
                          flagged: bool | None = None, type_: str | None = None,
                          min_score: float | None = None, max_score: float | None = None,
                          source: str | None = None, case_status: str | None = None,
                          batch_id: str | None = None) -> dict:
        if sort not in SORT_COLUMNS or order not in ("asc", "desc"):
            raise AppError(422, "validation_error", "Invalid sort or order.",
                           details={"sort_allowed": sorted(SORT_COLUMNS), "order_allowed": ["asc", "desc"]})
        where, p = [], []
        for cond, val in (("t.flagged=?", None if flagged is None else int(flagged)),
                          ("t.type=?", type_), ("t.score>=?", min_score), ("t.score<=?", max_score),
                          ("t.source=?", source), ("c.status=?", case_status), ("t.batch_id=?", batch_id)):
            if val is not None:
                where.append(cond)
                p.append(val)
        w = (" WHERE " + " AND ".join(where)) if where else ""
        base = " FROM transactions t LEFT JOIN cases c ON c.transaction_id=t.id" + w
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*)" + base, p).fetchone()[0]
            rows = c.execute(
                self._SELECT + w + f" ORDER BY {SORT_COLUMNS[sort]} {order.upper()}, t.id DESC LIMIT ? OFFSET ?",
                [*p, page_size, (page - 1) * page_size]).fetchall()
        return {"items": [self._row(r) for r in rows], "total": int(total), "page": page,
                "page_size": page_size}

    def history_for(self, names: list[str], limit: int = 50_000) -> list[dict]:
        """Previously scored transactions for these senders (used for history features)."""
        out: list[dict] = []
        names = list(dict.fromkeys(names))
        with self._conn() as c:
            for i in range(0, len(names), 500):
                chunk = names[i:i + 500]
                q = ",".join("?" * len(chunk))
                for r in c.execute("SELECT step,type,amount,name_orig AS nameOrig,name_dest AS nameDest "
                                   f"FROM transactions WHERE name_orig IN ({q}) ORDER BY id DESC LIMIT ?",
                                   [*chunk, limit - len(out)]):
                    out.append(dict(r))
                if len(out) >= limit:
                    break
        return out

    def summary(self) -> dict:
        with self._conn() as c:
            total, flagged = c.execute("SELECT COUNT(*), COALESCE(SUM(flagged),0) FROM transactions").fetchone()
            hist = [0] * 10
            for r in c.execute("SELECT MIN(CAST(score*10 AS INTEGER),9) AS b, COUNT(*) AS n "
                               "FROM transactions GROUP BY b"):
                hist[max(0, min(9, r["b"]))] += r["n"]
            by_day = [dict(r) for r in c.execute(
                "SELECT substr(created_utc,1,10) AS day, COUNT(*) AS transactions, SUM(flagged) AS flagged "
                "FROM transactions GROUP BY day ORDER BY day DESC LIMIT 30")][::-1]
            by_type = [dict(r) for r in c.execute(
                "SELECT type, COUNT(*) AS transactions, SUM(flagged) AS flagged FROM transactions "
                "GROUP BY type ORDER BY transactions DESC")]
            cases = {r["status"]: r["n"] for r in c.execute(
                "SELECT status, COUNT(*) AS n FROM cases GROUP BY status")}
            fb = {r["label"]: r["n"] for r in c.execute("SELECT label, COUNT(*) AS n FROM feedback GROUP BY label")}
        return {"total_transactions": int(total), "flagged_transactions": int(flagged),
                "score_histogram": [{"bin_start": i / 10, "bin_end": (i + 1) / 10, "count": n}
                                    for i, n in enumerate(hist)],
                "by_day": by_day, "by_type": by_type,
                "case_status_counts": {s: cases.get(s, 0) for s in CASE_STATUSES},
                "feedback": {"total": sum(fb.values()), "labelled_fraud": fb.get(1, 0),
                             "labelled_not_fraud": fb.get(0, 0)}}

    # ---- monitoring inputs ----
    def record_rejection(self, endpoint: str, code: str) -> None:
        try:
            with self._conn(write=True) as c:
                c.execute("INSERT INTO rejections (created_utc,endpoint,code) VALUES (?,?,?)",
                          (_now(), endpoint[:40], code[:40]))
        except sqlite3.Error:
            pass  # monitoring counters must never break request handling

    def rejection_counts(self) -> dict:
        with self._conn() as c:
            return {r["code"]: r["n"] for r in c.execute(
                "SELECT code, COUNT(*) AS n FROM rejections GROUP BY code ORDER BY n DESC")}

    def monitoring_rows(self, limit: int = 5000) -> pd.DataFrame:
        with self._conn() as c:
            rows = c.execute("SELECT step,type,amount,name_orig AS nameOrig,name_dest AS nameDest,score,"
                             "oldbalance_orig AS oldbalanceOrg, newbalance_orig AS newbalanceOrig, "
                             "oldbalance_dest AS oldbalanceDest, newbalance_dest AS newbalanceDest "
                             "FROM (SELECT * FROM transactions ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                             (limit,)).fetchall()
        cols = ["step", "type", "amount", "nameOrig", "nameDest", "score",
                "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
        return pd.DataFrame([dict(r) for r in rows], columns=cols).astype(
            {"step": "int64", "amount": "float64", "score": "float64", "oldbalanceOrg": "float64",
             "newbalanceOrig": "float64", "oldbalanceDest": "float64", "newbalanceDest": "float64"})

    def labelled_rows(self, limit: int = 5000) -> pd.DataFrame:
        with self._conn() as c:
            rows = c.execute("SELECT t.score AS score, t.threshold AS threshold, f.label AS label "
                             "FROM feedback f JOIN transactions t ON t.id=f.transaction_id "
                             "ORDER BY f.updated_utc DESC LIMIT ?", (limit,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows], columns=["score", "threshold", "label"]).astype(
            {"score": "float64", "threshold": "float64", "label": "int64"})

    # ---- retraining workflow (state machine; never swaps the active model) ----
    def create_retrain(self, reason: str, note: str | None) -> dict:
        now = _now()
        with self._conn(write=True) as c:
            cur = c.execute("INSERT INTO retrain_requests (created_utc,updated_utc,status,reason,note) "
                            "VALUES (?,?,?,?,?)", (now, now, "requested", reason, note))
            rid = int(cur.lastrowid)
            c.execute("INSERT INTO retrain_events (request_id,from_status,to_status,note,created_utc) "
                      "VALUES (?,?,?,?,?)", (rid, None, "requested", note, now))
        return self.get_retrain(rid)

    def get_retrain(self, rid: int) -> dict:
        import json
        with self._conn() as c:
            r = c.execute("SELECT * FROM retrain_requests WHERE id=?", (rid,)).fetchone()
            if r is None:
                raise AppError(404, "not_found", "Retraining request not found.")
            ev = [dict(e) for e in c.execute(
                "SELECT from_status,to_status,note,created_utc FROM retrain_events WHERE request_id=? ORDER BY id", (rid,))]
        d = dict(r)
        d["evaluation"] = json.loads(d.pop("evaluation_json")) if d.get("evaluation_json") else None
        d["events"] = ev
        d["allowed_next"] = list(RETRAIN_NEXT[d["status"]])
        return d

    def list_retrain(self) -> list[dict]:
        with self._conn() as c:
            ids = [r["id"] for r in c.execute("SELECT id FROM retrain_requests ORDER BY id DESC LIMIT 100")]
        return [self.get_retrain(i) for i in ids]

    def transition_retrain(self, rid: int, to: str, note: str | None, candidate: str | None = None,
                           evaluation: dict | None = None) -> dict:
        import json
        now = _now()
        with self._conn(write=True) as c:
            r = c.execute("SELECT status FROM retrain_requests WHERE id=?", (rid,)).fetchone()
            if r is None:
                raise AppError(404, "not_found", "Retraining request not found.")
            if to not in RETRAIN_NEXT[r["status"]]:
                raise AppError(409, "invalid_transition", f"Cannot move from '{r['status']}' to '{to}'.",
                               details={"allowed": list(RETRAIN_NEXT[r["status"]])})
            c.execute("UPDATE retrain_requests SET status=?, updated_utc=?, "
                      "candidate=COALESCE(?, candidate), evaluation_json=COALESCE(?, evaluation_json) WHERE id=?",
                      (to, now, candidate, json.dumps(evaluation) if evaluation is not None else None, rid))
            c.execute("INSERT INTO retrain_events (request_id,from_status,to_status,note,created_utc) "
                      "VALUES (?,?,?,?,?)", (rid, r["status"], to, note, now))
        return self.get_retrain(rid)
