"""Configuration from environment variables only. No secrets are required or logged."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from backend.app.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]


def _int(env, name: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError as e:
        raise ConfigError(f"{name} must be an integer") from e
    if not lo <= v <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}")
    return v


def parse_sqlite_url(url: str) -> Path:
    prefix = "sqlite:///"
    if not url.startswith(prefix) or len(url) == len(prefix):
        raise ConfigError("DATABASE_URL must look like sqlite:///./file.db (SQLite only)")
    return Path(url[len(prefix):]).expanduser()


@dataclass(frozen=True)
class Settings:
    app_env: str = "development"
    database_path: Path = Path("fraud_platform.db")
    model_dir: Path | None = None
    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")
    max_upload_bytes: int = 5_000_000
    max_csv_rows: int = 5_000
    max_json_bytes: int = 16_384
    max_page_size: int = 200
    rate_predict_per_min: int = 60
    rate_batch_per_min: int = 6
    rate_read_per_min: int = 240
    trust_proxy: bool = False
    log_level: str = "INFO"
    artifact_root: Path = ROOT / "ml" / "artifacts"
    outcomes_dir: Path = ROOT / "streaming_outcomes"
    min_monitor_rows: int = 100

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        app_env = env.get("APP_ENV", "development").strip().lower()
        if app_env not in ("development", "test", "production"):
            raise ConfigError("APP_ENV must be development, test or production")
        origins = tuple(o.strip() for o in env.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip())
        if "*" in origins:
            raise ConfigError("CORS_ALLOW_ORIGINS must list explicit origins; '*' is not allowed")
        model_dir = env.get("MODEL_ARTIFACT_DIR", "").strip()
        d = cls()
        return cls(
            app_env=app_env,
            database_path=parse_sqlite_url(env.get("DATABASE_URL", "sqlite:///./fraud_platform.db")),
            model_dir=Path(model_dir) if model_dir else None,
            cors_origins=origins or d.cors_origins,
            max_upload_bytes=_int(env, "MAX_UPLOAD_BYTES", d.max_upload_bytes, 1_000, 50_000_000),
            max_csv_rows=_int(env, "MAX_CSV_ROWS", d.max_csv_rows, 1, 100_000),
            max_json_bytes=_int(env, "MAX_JSON_BYTES", d.max_json_bytes, 256, 1_000_000),
            max_page_size=_int(env, "MAX_PAGE_SIZE", d.max_page_size, 1, 1000),
            rate_predict_per_min=_int(env, "RATE_LIMIT_PREDICT_PER_MIN", d.rate_predict_per_min, 1, 100_000),
            rate_batch_per_min=_int(env, "RATE_LIMIT_BATCH_PER_MIN", d.rate_batch_per_min, 1, 10_000),
            rate_read_per_min=_int(env, "RATE_LIMIT_READ_PER_MIN", d.rate_read_per_min, 1, 100_000),
            trust_proxy=env.get("TRUST_PROXY", "false").strip().lower() == "true",
            log_level=env.get("LOG_LEVEL", "INFO").strip().upper(),
            artifact_root=Path(env.get("ARTIFACT_ROOT") or d.artifact_root),
            outcomes_dir=Path(env.get("STREAM_OUTCOMES_DIR") or d.outcomes_dir),
            min_monitor_rows=_int(env, "MIN_MONITOR_ROWS", d.min_monitor_rows, 10, 1_000_000),
        )
