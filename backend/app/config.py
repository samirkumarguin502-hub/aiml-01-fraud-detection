"""Configuration from environment variables only.

No secrets are required or logged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from backend.app.errors import ConfigError


ROOT = Path(__file__).resolve().parents[2]

ALLOWED_ENVIRONMENTS = {
    "development",
    "test",
    "production",
}

ALLOWED_LOG_LEVELS = {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}


def _int(
    env,
    name: str,
    default: int,
    lo: int,
    hi: int,
) -> int:
    raw = env.get(name)

    if raw is None or raw == "":
        return default

    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ConfigError(
            f"{name} must be an integer"
        ) from None

    if not lo <= value <= hi:
        raise ConfigError(
            f"{name} must be between {lo} and {hi}"
        )

    return value


def _bool(
    env,
    name: str,
    default: bool = False,
) -> bool:
    """Parse an explicit Boolean environment variable."""
    raw = env.get(name)

    if raw is None or raw == "":
        return default

    value = str(raw).strip().lower()

    if value in ("true", "1", "yes", "on"):
        return True

    if value in ("false", "0", "no", "off"):
        return False

    raise ConfigError(
        f"{name} must be a Boolean value"
    )


def _path(
    value: str | Path,
    name: str,
) -> Path:
    """Normalize a configured filesystem path."""
    try:
        result = Path(value).expanduser()
    except (TypeError, ValueError, OSError):
        raise ConfigError(
            f"{name} must be a valid filesystem path"
        ) from None

    if not str(result):
        raise ConfigError(
            f"{name} must not be empty"
        )

    return result


def parse_sqlite_url(url: str) -> Path:
    """Parse a SQLite-only database URL.

    Supported examples:
        sqlite:///./fraud_platform.db
        sqlite:////absolute/path/fraud_platform.db
        sqlite:///C:/Projects/fraud_platform.db

    Relative paths remain relative to the application working directory.
    """
    prefix = "sqlite:///"

    if (
        not isinstance(url, str)
        or not url.startswith(prefix)
        or len(url) == len(prefix)
    ):
        raise ConfigError(
            "DATABASE_URL must look like sqlite:///./file.db (SQLite only)"
        )

    raw_path = unquote(url[len(prefix):])

    if not raw_path or "\x00" in raw_path:
        raise ConfigError(
            "DATABASE_URL must contain a valid SQLite file path"
        )

    if raw_path in (":memory:", "./:memory:"):
        return Path(":memory:")

    try:
        return Path(raw_path).expanduser()
    except (TypeError, ValueError, OSError):
        raise ConfigError(
            "DATABASE_URL must contain a valid SQLite file path"
        ) from None


@dataclass(frozen=True)
class Settings:
    app_env: str = "development"

    database_path: Path = Path("fraud_platform.db")

    model_dir: Path | None = None

    cors_origins: tuple = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

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

    api_key: str | None = None

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        """Build validated settings from environment variables."""
        env = os.environ if env is None else env

        # Environment
        app_env = str(
            env.get("APP_ENV", "development")
        ).strip().lower()

        if app_env not in ALLOWED_ENVIRONMENTS:
            raise ConfigError(
                "APP_ENV must be development, test or production"
            )

        defaults = cls()

        # CORS
        raw_origins = env.get(
            "CORS_ALLOW_ORIGINS",
            "",
        )

        origins = tuple(
            origin.strip().rstrip("/")
            for origin in raw_origins.split(",")
            if origin.strip()
        )

        if "*" in origins:
            raise ConfigError(
                "CORS_ALLOW_ORIGINS must list explicit origins; "
                "'*' is not allowed"
            )

        if any(
            not origin.startswith(("http://", "https://"))
            for origin in origins
        ):
            raise ConfigError(
                "CORS_ALLOW_ORIGINS must contain valid HTTP(S) origins"
            )

        # Paths
        database_url = env.get(
            "DATABASE_URL",
            "sqlite:///./fraud_platform.db",
        )

        database_path = parse_sqlite_url(database_url)

        model_dir_value = str(
            env.get("MODEL_ARTIFACT_DIR", "")
        ).strip()

        model_dir = (
            _path(model_dir_value, "MODEL_ARTIFACT_DIR")
            if model_dir_value
            else None
        )

        artifact_root = _path(
            env.get("ARTIFACT_ROOT") or defaults.artifact_root,
            "ARTIFACT_ROOT",
        )

        outcomes_dir = _path(
            env.get("STREAM_OUTCOMES_DIR") or defaults.outcomes_dir,
            "STREAM_OUTCOMES_DIR",
        )

        # Logging
        log_level = str(
            env.get("LOG_LEVEL", defaults.log_level)
        ).strip().upper()

        if log_level not in ALLOWED_LOG_LEVELS:
            raise ConfigError(
                "LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL"
            )

        # Proxy configuration
        trust_proxy = _bool(
            env,
            "TRUST_PROXY",
            defaults.trust_proxy,
        )

        # API key
        api_key = str(
            env.get("API_KEY", "")
        ).strip() or None

        if app_env == "production" and not api_key:
            raise ConfigError(
                "API_KEY must be configured in production"
            )

        return cls(
            app_env=app_env,

            database_path=database_path,

            model_dir=model_dir,

            cors_origins=origins or defaults.cors_origins,

            max_upload_bytes=_int(
                env,
                "MAX_UPLOAD_BYTES",
                defaults.max_upload_bytes,
                1_000,
                50_000_000,
            ),

            max_csv_rows=_int(
                env,
                "MAX_CSV_ROWS",
                defaults.max_csv_rows,
                1,
                100_000,
            ),

            max_json_bytes=_int(
                env,
                "MAX_JSON_BYTES",
                defaults.max_json_bytes,
                256,
                1_000_000,
            ),

            max_page_size=_int(
                env,
                "MAX_PAGE_SIZE",
                defaults.max_page_size,
                1,
                1_000,
            ),

            rate_predict_per_min=_int(
                env,
                "RATE_LIMIT_PREDICT_PER_MIN",
                defaults.rate_predict_per_min,
                1,
                100_000,
            ),

            rate_batch_per_min=_int(
                env,
                "RATE_LIMIT_BATCH_PER_MIN",
                defaults.rate_batch_per_min,
                1,
                10_000,
            ),

            rate_read_per_min=_int(
                env,
                "RATE_LIMIT_READ_PER_MIN",
                defaults.rate_read_per_min,
                1,
                100_000,
            ),

            trust_proxy=trust_proxy,

            log_level=log_level,

            artifact_root=artifact_root,

            outcomes_dir=outcomes_dir,

            min_monitor_rows=_int(
                env,
                "MIN_MONITOR_ROWS",
                defaults.min_monitor_rows,
                10,
                1_000_000,
            ),

            api_key=api_key,
        )