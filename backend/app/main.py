"""FastAPI binding for AIML-01 Fraud Detection Platform.

Run:
    python -m uvicorn backend.app.main:create_app --factory

Business logic remains in service.py, validation.py and db.py.
Authentication is optional for local development and enabled through API_KEY.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.app.config import Settings
from backend.app.db import Repository
from backend.app.auth import create_access_token, decode_access_token, hash_password, verify_password
from backend.app.errors import AppError
from backend.app.logging_setup import log_event, setup_logging
from backend.app.model_service import ModelService
from backend.app.ratelimit import RateLimiter
from backend.app.schemas import (
    BatchOut,
    CaseUpdateIn,
    FeedbackIn,
    LoginIn,
    UserCreateIn,
    PredictionOut,
    RetrainCreateIn,
    RetrainTransitionIn,
    TransactionIn,
)
from backend.app.service import FraudService


API = "/api/v1"

CSV_TYPES = {
    "text/csv",
    "application/csv",
    "text/plain",
    "application/vnd.ms-excel",
}

PUBLIC_PATHS = {
    f"{API}/health",
    f"{API}/ready",
}


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = settings or Settings.from_env()

    logger = setup_logging(settings.log_level)

    # Initialize persistence and model services.
    repo = Repository(settings.database_path)
    repo.init_schema()

    admin_username = os.getenv("ADMIN_USERNAME", "").strip()
    admin_password = os.getenv("ADMIN_PASSWORD", "")
    if admin_username and admin_password and not repo.get_user_by_username(admin_username):
        repo.create_user(admin_username, hash_password(admin_password), "admin")

    jwt_secret = os.getenv("JWT_SECRET_KEY", "")
    if settings.is_production and len(jwt_secret) < 32:
        raise RuntimeError("Production requires JWT_SECRET_KEY with at least 32 characters.")

    model = ModelService(settings.model_dir)
    model.load()

    service = FraudService(settings, repo, model)

    limiters = {
        "predict": RateLimiter(settings.rate_predict_per_min),
        "batch": RateLimiter(settings.rate_batch_per_min),
        "read": RateLimiter(settings.rate_read_per_min),
    }

    docs_url = None if settings.is_production else "/docs"
    openapi_url = (
        None if settings.is_production else "/openapi.json"
    )

    app = FastAPI(
        title="AIML-01 Fraud Detection API",
        version="0.3.0",
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=openapi_url,
    )

    app.state.service = service
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    # ---------------------------------------------------------
    # Request identity and rate limiting
    # ---------------------------------------------------------

    def request_id(request: Request) -> str:
        return getattr(request.state, "request_id", "")

    def client_key(request: Request) -> str:
        """Resolve client IP.

        TRUST_PROXY must only be enabled when the application is
        behind a trusted proxy that strips client-supplied forwarding
        headers and writes its own X-Forwarded-For header.
        """

        if settings.trust_proxy:
            forwarded = request.headers.get(
                "x-forwarded-for", ""
            )

            if forwarded:
                return forwarded.split(",")[0].strip()[:64]

        if request.client:
            return request.client.host

        return "unknown"

    def limit(bucket: str):
        def dependency(request: Request) -> None:
            limiter = limiters[bucket]

            allowed, retry_after = limiter.check(
                client_key(request)
            )

            if not allowed:
                raise AppError(
                    429,
                    "rate_limited",
                    "Too many requests; slow down.",
                    headers={
                        "Retry-After": str(
                            max(1, int(retry_after) + 1)
                        )
                    },
                )

        return dependency

    # ---------------------------------------------------------
    # Middleware
    # ---------------------------------------------------------

    @app.middleware("http")
    async def base_middleware(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        request.state.request_id = rid

        started = time.perf_counter()
        path = request.url.path

        # Public endpoints: health, readiness and login.
        is_public = path in PUBLIC_PATHS or path == f"{API}/auth/login"
        request.state.principal = {
            "user_id": None,
            "username": "anonymous",
            "role": "analyst",
            "auth_type": "development",
        }

        if not is_public and (settings.api_key or jwt_secret):
            authorization = request.headers.get("authorization", "")
            bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
            api_key = request.headers.get("x-api-key", "")
            principal = None

            if bearer and jwt_secret:
                try:
                    claims = decode_access_token(bearer, jwt_secret)
                    user = repo.get_user(int(claims["sub"]))
                    if user and user["is_active"] and user["role"] == claims["role"]:
                        principal = {
                            "user_id": user["id"],
                            "username": user["username"],
                            "role": user["role"],
                            "auth_type": "jwt",
                        }
                except (ValueError, TypeError):
                    principal = None

            # Legacy API key remains supported for analyst-level operations.
            if principal is None and settings.api_key:
                candidate = bearer or api_key
                if candidate and hmac.compare_digest(candidate, settings.api_key):
                    principal = {
                        "user_id": None,
                        "username": "api-key-client",
                        "role": "analyst",
                        "auth_type": "api_key",
                    }

            if principal is None:
                response = JSONResponse(
                    AppError(401, "unauthorized", "Valid authentication is required.").body(rid),
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                response.headers["X-Request-ID"] = rid
                response.headers["Cache-Control"] = "no-store"
                return response

            request.state.principal = principal

            admin_only = (
                path == f"{API}/users"
                or path.startswith(f"{API}/users/")
                or (
                    path.startswith(f"{API}/monitoring/retraining")
                    and request.method == "POST"
                )
            )
            if admin_only and principal["role"] != "admin":
                response = JSONResponse(
                    AppError(403, "forbidden", "Administrator role required.").body(rid),
                    status_code=403,
                )
                response.headers["X-Request-ID"] = rid
                response.headers["Cache-Control"] = "no-store"
                return response

        # Record authenticated activity without storing secrets or transaction data.
        principal = request.state.principal

        # Enforce declared body size for non-batch requests.
        # Batch requests have their own streaming size enforcement.
        content_length = request.headers.get(
            "content-length", ""
        )

        is_batch = path == f"{API}/predict/batch"

        if (
            request.method in {"POST", "PUT", "PATCH"}
            and not is_batch
            and content_length.isdigit()
            and int(content_length) > settings.max_json_bytes
        ):
            response = JSONResponse(
                AppError(
                    413,
                    "payload_too_large",
                    "Request body too large.",
                ).body(rid),
                status_code=413,
            )

        else:
            response = await call_next(request)

        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"

        route = request.scope.get("route")

        log_event(
            logger,
            "request",
            request_id=rid,
            method=request.method,
            route=route.path if route else "unmatched",
            status=response.status_code,
            latency_ms=round(
                (time.perf_counter() - started) * 1000,
                1,
            ),
        )

        return response

    # ---------------------------------------------------------
    # Exception handlers
    # ---------------------------------------------------------

    @app.exception_handler(AppError)
    async def handle_app_error(
        request: Request,
        exc: AppError,
    ):
        return JSONResponse(
            exc.body(request_id(request)),
            status_code=exc.status,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ):
        # Do not include submitted values in the response.
        details = [
            {
                "loc": [
                    str(item)
                    for item in error.get("loc", [])
                ],
                "reason": error.get("msg", "invalid"),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]

        error = AppError(
            422,
            "validation_error",
            "Request validation failed.",
            details=details,
        )

        return JSONResponse(
            error.body(request_id(request)),
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request,
        exc: StarletteHTTPException,
    ):
        code = {
            404: "not_found",
            405: "method_not_allowed",
        }.get(exc.status_code, "http_error")

        message = {
            404: "Resource not found.",
            405: "Method not allowed.",
        }.get(exc.status_code, "Request failed.")

        error = AppError(
            exc.status_code,
            code,
            message,
        )

        return JSONResponse(
            error.body(request_id(request)),
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request,
        exc: Exception,
    ):
        log_event(
            logger,
            "unhandled_error",
            logging.ERROR,
            request_id=request_id(request),
            exc_type=type(exc).__name__,
        )

        error = AppError(
            500,
            "internal_error",
            "Internal server error.",
        )

        return JSONResponse(
            error.body(request_id(request)),
            status_code=500,
        )

    # ---------------------------------------------------------
    # Authentication and user administration
    # ---------------------------------------------------------

    @app.post(f"{API}/auth/login", tags=["authentication"])
    def login(body: LoginIn, request: Request):
        if len(jwt_secret) < 32:
            raise AppError(503, "auth_not_configured", "JWT authentication is not configured.")

        user = repo.get_user_by_username(body.username)
        if not user or not user["is_active"] or not verify_password(body.password, user["password_hash"]):
            raise AppError(401, "invalid_credentials", "Username or password is incorrect.")

        token = create_access_token(
            user["id"], user["username"], user["role"],
            secret_key=jwt_secret, expires_minutes=60,
        )
        repo.record_audit(user["id"], user["username"], user["role"], "login", f"{API}/auth/login", request_id(request))
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": 3600,
            "user": {"id": user["id"], "username": user["username"], "role": user["role"]},
        }

    @app.post(f"{API}/users", tags=["administration"])
    def create_user(body: UserCreateIn, request: Request):
        try:
            user = repo.create_user(body.username, hash_password(body.password), body.role)
        except Exception as exc:
            import sqlite3
            if isinstance(exc, sqlite3.IntegrityError):
                raise AppError(409, "username_exists", "Username already exists.") from None
            raise
        principal = request.state.principal
        repo.record_audit(principal.get("user_id"), principal["username"], principal["role"], "user_created", f"{API}/users", request_id(request))
        return {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "is_active": bool(user["is_active"]),
            "created_utc": user["created_utc"],
        }

    @app.get(f"{API}/users", tags=["administration"])
    def list_users():
        return [
            {**user, "is_active": bool(user["is_active"])}
            for user in repo.list_users()
        ]

    # ---------------------------------------------------------
    # Health endpoints
    # ---------------------------------------------------------

    @app.get(
        f"{API}/health",
        tags=["health"],
    )
    def health():
        return service.health()

    @app.get(
        f"{API}/ready",
        tags=["health"],
    )
    def ready():
        status, body = service.readiness()

        return JSONResponse(
            body,
            status_code=status,
        )

    # ---------------------------------------------------------
    # Model endpoints
    # ---------------------------------------------------------

    @app.get(
        f"{API}/model",
        tags=["model"],
        dependencies=[Depends(limit("read"))],
    )
    def model_info():
        return service.model_info()

    # ---------------------------------------------------------
    # Single transaction prediction
    # ---------------------------------------------------------

    @app.post(
        f"{API}/predict",
        tags=["prediction"],
        response_model=PredictionOut,
        dependencies=[Depends(limit("predict"))],
    )
    def predict(tx: TransactionIn):
        payload = tx.model_dump(exclude_none=True)

        return service.predict_one(payload)

    # ---------------------------------------------------------
    # Batch CSV prediction
    # ---------------------------------------------------------

    @app.post(
        f"{API}/predict/batch",
        tags=["prediction"],
        response_model=BatchOut,
        dependencies=[Depends(limit("batch"))],
    )
    async def predict_batch(request: Request):
        """Accept raw CSV data without writing uploaded data to disk."""

        content_type = (
            request.headers.get("content-type", "")
            .split(";")[0]
            .strip()
            .lower()
        )

        if content_type not in CSV_TYPES:
            raise AppError(
                415,
                "unsupported_media_type",
                "Send the CSV as the raw body with Content-Type: text/csv.",
            )

        content_length = request.headers.get(
            "content-length", ""
        )

        if (
            content_length.isdigit()
            and int(content_length) > settings.max_upload_bytes
        ):
            raise AppError(
                413,
                "file_too_large",
                f"File exceeds the {settings.max_upload_bytes} byte limit.",
            )

        chunks = []
        total_bytes = 0

        async for chunk in request.stream():
            total_bytes += len(chunk)

            if total_bytes > settings.max_upload_bytes:
                raise AppError(
                    413,
                    "file_too_large",
                    f"File exceeds the {settings.max_upload_bytes} byte limit.",
                )

            chunks.append(chunk)

        csv_content = b"".join(chunks)

        return await run_in_threadpool(
            service.predict_batch,
            csv_content,
        )

    # ---------------------------------------------------------
    # Transaction listing and details
    # ---------------------------------------------------------

    @app.get(
        f"{API}/transactions",
        tags=["transactions"],
        dependencies=[Depends(limit("read"))],
    )
    def transactions(
        page: int = Query(1, ge=1),
        page_size: int = Query(
            25,
            ge=1,
            le=settings.max_page_size,
        ),
        sort: str = "id",
        order: str = "desc",
        flagged: Optional[bool] = None,
        type: Optional[str] = Query(
            None,
            max_length=32,
        ),
        min_score: Optional[float] = Query(
            None,
            ge=0,
            le=1,
        ),
        max_score: Optional[float] = Query(
            None,
            ge=0,
            le=1,
        ),
        source: Optional[str] = None,
        case_status: Optional[str] = None,
        batch_id: Optional[str] = Query(
            None,
            max_length=64,
        ),
    ):
        return service.list_transactions(
            page=page,
            page_size=page_size,
            sort=sort,
            order=order,
            flagged=flagged,
            type_=type,
            min_score=min_score,
            max_score=max_score,
            source=source,
            case_status=case_status,
            batch_id=batch_id,
        )

    @app.get(
        f"{API}/transactions/{{transaction_id}}",
        tags=["transactions"],
        dependencies=[Depends(limit("read"))],
    )
    def transaction(transaction_id: int):
        return service.get_transaction(transaction_id)

    # ---------------------------------------------------------
    # Case management and analyst feedback
    # ---------------------------------------------------------

    @app.patch(
        f"{API}/transactions/{{transaction_id}}/case",
        tags=["cases"],
        dependencies=[Depends(limit("read"))],
    )
    def update_case(
        transaction_id: int,
        body: CaseUpdateIn,
    ):
        return service.update_case(
            transaction_id,
            body.model_dump(),
        )

    @app.put(
        f"{API}/transactions/{{transaction_id}}/feedback",
        tags=["cases"],
        dependencies=[Depends(limit("read"))],
    )
    def feedback(
        transaction_id: int,
        body: FeedbackIn,
    ):
        return service.set_feedback(
            transaction_id,
            body.model_dump(),
        )

    # ---------------------------------------------------------
    # Alerts
    # ---------------------------------------------------------

    @app.get(
        f"{API}/alerts",
        tags=["alerts"],
        dependencies=[Depends(limit("read"))],
    )
    def alerts(
        page: int = Query(1, ge=1),
        page_size: int = Query(
            25,
            ge=1,
            le=settings.max_page_size,
        ),
        case_status: Optional[str] = None,
    ):
        return service.alerts(
            page=page,
            page_size=page_size,
            case_status=case_status,
        )

    # ---------------------------------------------------------
    # Dashboard
    # ---------------------------------------------------------

    @app.get(
        f"{API}/dashboard/summary",
        tags=["dashboard"],
        dependencies=[Depends(limit("read"))],
    )
    def dashboard():
        return service.dashboard_summary()

    # ---------------------------------------------------------
    # Monitoring and pipeline status
    # ---------------------------------------------------------

    @app.get(
        f"{API}/monitoring/status",
        tags=["monitoring"],
        dependencies=[Depends(limit("read"))],
    )
    def monitoring_status():
        return service.monitoring_status()

    @app.get(
        f"{API}/pipeline/status",
        tags=["monitoring"],
        dependencies=[Depends(limit("read"))],
    )
    def pipeline_status():
        return service.pipeline_status()

    # ---------------------------------------------------------
    # Retraining workflow
    # ---------------------------------------------------------

    @app.get(
        f"{API}/monitoring/retraining",
        tags=["monitoring"],
        dependencies=[Depends(limit("read"))],
    )
    def retrain_list():
        return service.retrain_list()

    @app.post(
        f"{API}/monitoring/retraining",
        tags=["monitoring"],
        dependencies=[Depends(limit("read"))],
    )
    def retrain_create(body: RetrainCreateIn):
        return service.retrain_create(
            body.model_dump()
        )

    @app.post(
        f"{API}/monitoring/retraining/{{request_id}}/transition",
        tags=["monitoring"],
        dependencies=[Depends(limit("read"))],
    )
    def retrain_transition(
        request_id: int,
        body: RetrainTransitionIn,
    ):
        return service.retrain_transition(
            request_id,
            body.model_dump(exclude_none=True),
        )

    return app