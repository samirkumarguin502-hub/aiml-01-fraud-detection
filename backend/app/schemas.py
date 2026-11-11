"""Pydantic v2 request/response schemas (documentation + first-line validation).

backend/app/validation.py is the authoritative validator and re-checks every request.
Response models use extra="allow" and optional fields so a schema/dict mismatch can never turn a
correct response into a 500. NOTE: not executed in the authoring sandbox (pydantic not installed).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from backend.app.validation import MAX_AMOUNT, MAX_STEP, NAME_PATTERN, TYPE_PATTERN

CaseStatus = Literal["open", "in_review", "confirmed_fraud", "false_positive", "closed"]


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = Field(ge=0, le=MAX_STEP, description="Time step (PaySim: 1 step = 1 hour)")
    type: str = Field(pattern=f"^{TYPE_PATTERN}$")
    amount: float = Field(ge=0, le=MAX_AMOUNT, allow_inf_nan=False)
    nameOrig: str = Field(pattern=f"^{NAME_PATTERN}$")
    nameDest: str = Field(pattern=f"^{NAME_PATTERN}$")
    # Required only when the loaded model's feature set needs them (see GET /model -> input_fields); the
    # service rejects them for models that do not.
    oldbalanceOrg: Optional[float] = Field(default=None, ge=0, le=MAX_AMOUNT, allow_inf_nan=False)
    oldbalanceDest: Optional[float] = Field(default=None, ge=0, le=MAX_AMOUNT, allow_inf_nan=False)
    newbalanceOrig: Optional[float] = Field(default=None, ge=0, le=MAX_AMOUNT, allow_inf_nan=False)
    newbalanceDest: Optional[float] = Field(default=None, ge=0, le=MAX_AMOUNT, allow_inf_nan=False)


class CaseUpdateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: CaseStatus
    note: Optional[str] = Field(default=None, max_length=500)


class FeedbackIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal[0, 1] = Field(description="Analyst verdict: 1 = fraud, 0 = not fraud")
    comment: Optional[str] = Field(default=None, max_length=500)


class RetrainCreateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = Field(min_length=1, max_length=500)
    note: Optional[str] = Field(default=None, max_length=500)


class RetrainTransitionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    to: Literal["candidate_trained", "evaluate", "promoted", "rejected", "cancelled"]
    note: Optional[str] = Field(default=None, max_length=500)
    candidate: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9_\-]{1,80}$")


class _Out(BaseModel):
    model_config = ConfigDict(extra="allow")


class PredictionOut(_Out):
    transaction_id: Optional[int] = None
    score: Optional[float] = Field(default=None, description="Uncalibrated ranking score, NOT a probability")
    threshold: Optional[float] = None
    flagged: Optional[bool] = None
    score_note: Optional[str] = None
    warnings: Optional[list[str]] = None


class BatchOut(_Out):
    batch_id: Optional[str] = None
    rows: Optional[int] = None
    flagged: Optional[int] = None
    results: Optional[list[dict]] = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[object] = None
    request_id: Optional[str] = None


class ErrorOut(BaseModel):
    error: ErrorBody


class LoginIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=8, max_length=128)


class UserCreateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)
    role: Literal["admin", "analyst"]


class TokenOut(_Out):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(_Out):
    id: int
    username: str
    role: str
    is_active: bool
