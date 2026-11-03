"""FastAPI binding. Run with:  python -m uvicorn backend.app.main:create_app --factory

This module is deliberately thin: all logic lives in service.py / validation.py / db.py, which are
tested without FastAPI. It requires fastapi>=0.110, pydantic>=2.5, uvicorn (requirements-api.txt).
NOTE: this file was NOT executed in the authoring sandbox (fastapi not installable there); run
tests/test_backend_http.py locally to verify it.

Authentication/authorization are NOT implemented. Bind to 127.0.0.1 (the default below).
"""
from __future__ import annotations

import logging
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
from backend.app.errors import AppError
from backend.app.logging_setup import log_event, setup_logging
from backend.app.model_service import ModelService
from backend.app.ratelimit import RateLimiter
from backend.app.schemas import (BatchOut, CaseUpdateIn, FeedbackIn, PredictionOut, RetrainCreateIn,
                                 RetrainTransitionIn, TransactionIn)
from backend.app.service import FraudService

CSV_TYPES = {"text/csv", "application/csv", "text/plain", "application/vnd.ms-excel"}
API = "/api/v1"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logger = setup_logging(settings.log_level)
    repo = Repository(settings.database_path)
    repo.init_schema()
    model = ModelService(settings.model_dir)
    model.load()
    service = FraudService(settings, repo, model)
    limiters = {"predict": RateLimiter(settings.rate_predict_per_min),
                "batch": RateLimiter(settings.rate_batch_per_min),
                "read": RateLimiter(settings.rate_read_per_min)}

    docs = None if settings.is_production else "/docs"
    app = FastAPI(title="AIML-01 Fraud Detection API", version="0.3.0", docs_url=docs,
                  redoc_url=None, openapi_url=None if settings.is_production else "/openapi.json")
    app.state.service, app.state.settings = service, settings

    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
                       allow_methods=["GET", "POST", "PUT", "PATCH"], allow_headers=["Content-Type"])

    def client_key(request: Request) -> str:
        if settings.trust_proxy:
            fwd = request.headers.get("x-forwarded-for", "")
            if fwd:
                return fwd.split(",")[0].strip()[:64]
        return request.client.host if request.client else "unknown"

    def limit(bucket: str):
        def dep(request: Request) -> None:
            ok, retry = limiters[bucket].check(client_key(request))
            if not ok:
                raise AppError(429, "rate_limited", "Too many requests; slow down.",
                               headers={"Retry-After": str(int(retry) + 1)})
        return dep

    @app.middleware("http")
    async def base_middleware(request: Request, call_next):
        rid = uuid.uuid4().hex[:12]
        request.state.request_id = rid
        start = time.perf_counter()
        cl = request.headers.get("content-length", "")
        is_batch = request.url.path == f"{API}/predict/batch"
        if request.method in ("POST", "PUT", "PATCH") and not is_batch and cl.isdigit() \
                and int(cl) > settings.max_json_bytes:
            resp = JSONResponse(AppError(413, "payload_too_large", "Request body too large.").body(rid), 413)
        else:
            resp = await call_next(request)
        resp.headers["X-Request-ID"] = rid
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "no-store"
        log_event(logger, "request", request_id=rid, method=request.method,
                  route=request.scope.get("route").path if request.scope.get("route") else "unmatched",
                  status=resp.status_code, latency_ms=round((time.perf_counter() - start) * 1000, 1))
        return resp

    def rid(request: Request) -> str:
        return getattr(request.state, "request_id", "")

    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return JSONResponse(exc.body(rid(request)), exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        details = [{"loc": [str(x) for x in e.get("loc", [])], "reason": e.get("msg", "invalid"),
                    "type": e.get("type", "")} for e in exc.errors()]  # never echo submitted input
        return JSONResponse(AppError(422, "validation_error", "Request validation failed.",
                                     details=details).body(rid(request)), 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        msg = {404: "Resource not found.", 405: "Method not allowed."}.get(exc.status_code, "Request failed.")
        return JSONResponse(AppError(exc.status_code, code, msg).body(rid(request)), exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log_event(logger, "unhandled_error", logging.ERROR, request_id=rid(request), exc_type=type(exc).__name__)
        return JSONResponse(AppError(500, "internal_error", "Internal server error.").body(rid(request)), 500)

    # ---- routes (sync handlers run in the threadpool; service calls are blocking) ----
    @app.get(f"{API}/health", tags=["health"])
    def health():
        return service.health()

    @app.get(f"{API}/ready", tags=["health"])
    def ready():
        status, body = service.readiness()
        return JSONResponse(body, status)

    @app.get(f"{API}/model", tags=["model"], dependencies=[Depends(limit("read"))])
    def model_info():
        return service.model_info()

    @app.post(f"{API}/predict", tags=["prediction"], response_model=PredictionOut,
              dependencies=[Depends(limit("predict"))])
    def predict(tx: TransactionIn):
        return service.predict_one(tx.model_dump(exclude_none=True))

    @app.post(f"{API}/predict/batch", tags=["prediction"], response_model=BatchOut,
              dependencies=[Depends(limit("batch"))])
    async def predict_batch(request: Request):
        """Body = raw CSV (Content-Type: text/csv). Streamed with a hard size cap; never written to disk."""
        ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype not in CSV_TYPES:
            raise AppError(415, "unsupported_media_type", "Send the CSV as the raw body with Content-Type: text/csv.")
        cl = request.headers.get("content-length", "")
        if cl.isdigit() and int(cl) > settings.max_upload_bytes:
            raise AppError(413, "file_too_large", f"File exceeds the {settings.max_upload_bytes} byte limit.")
        chunks, total = [], 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > settings.max_upload_bytes:
                raise AppError(413, "file_too_large", f"File exceeds the {settings.max_upload_bytes} byte limit.")
            chunks.append(chunk)
        return await run_in_threadpool(service.predict_batch, b"".join(chunks))

    @app.get(f"{API}/transactions", tags=["transactions"], dependencies=[Depends(limit("read"))])
    def transactions(page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=settings.max_page_size),
                     sort: str = "id", order: str = "desc", flagged: Optional[bool] = None,
                     type: Optional[str] = Query(None, max_length=32), min_score: Optional[float] = Query(None, ge=0, le=1),
                     max_score: Optional[float] = Query(None, ge=0, le=1), source: Optional[str] = None,
                     case_status: Optional[str] = None, batch_id: Optional[str] = Query(None, max_length=64)):
        return service.list_transactions(page=page, page_size=page_size, sort=sort, order=order,
                                         flagged=flagged, type_=type, min_score=min_score, max_score=max_score,
                                         source=source, case_status=case_status, batch_id=batch_id)

    @app.get(f"{API}/transactions/{{transaction_id}}", tags=["transactions"], dependencies=[Depends(limit("read"))])
    def transaction(transaction_id: int):
        return service.get_transaction(transaction_id)

    @app.patch(f"{API}/transactions/{{transaction_id}}/case", tags=["cases"], dependencies=[Depends(limit("read"))])
    def update_case(transaction_id: int, body: CaseUpdateIn):
        return service.update_case(transaction_id, body.model_dump())

    @app.put(f"{API}/transactions/{{transaction_id}}/feedback", tags=["cases"], dependencies=[Depends(limit("read"))])
    def feedback(transaction_id: int, body: FeedbackIn):
        return service.set_feedback(transaction_id, body.model_dump())

    @app.get(f"{API}/alerts", tags=["alerts"], dependencies=[Depends(limit("read"))])
    def alerts(page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=settings.max_page_size),
               case_status: Optional[str] = None):
        return service.alerts(page=page, page_size=page_size, case_status=case_status)

    @app.get(f"{API}/dashboard/summary", tags=["dashboard"], dependencies=[Depends(limit("read"))])
    def dashboard():
        return service.dashboard_summary()

    @app.get(f"{API}/monitoring/status", tags=["monitoring"], dependencies=[Depends(limit("read"))])
    def monitoring_status():
        return service.monitoring_status()

    @app.get(f"{API}/pipeline/status", tags=["monitoring"], dependencies=[Depends(limit("read"))])
    def pipeline_status():
        return service.pipeline_status()

    @app.get(f"{API}/monitoring/retraining", tags=["monitoring"], dependencies=[Depends(limit("read"))])
    def retrain_list():
        return service.retrain_list()

    @app.post(f"{API}/monitoring/retraining", tags=["monitoring"], dependencies=[Depends(limit("read"))])
    def retrain_create(body: RetrainCreateIn):
        return service.retrain_create(body.model_dump())

    @app.post(f"{API}/monitoring/retraining/{{request_id}}/transition", tags=["monitoring"],
              dependencies=[Depends(limit("read"))])
    def retrain_transition(request_id: int, body: RetrainTransitionIn):
        return service.retrain_transition(request_id, body.model_dump(exclude_none=True))

    return app
