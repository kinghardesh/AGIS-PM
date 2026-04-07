"""
aegis-pm / api / main.py

Aegis PM – FastAPI backend (production-grade).

Endpoints
─────────
  System
    GET  /health                         liveness probe
    GET  /stats                          alert counts by status

  Alerts – CRUD
    GET  /alerts                         list with filters + pagination
    POST /alerts                         create (called by Monitor Agent)
    GET  /alerts/{id}                    single alert

  Alerts – State transitions
    POST /alerts/{id}/approve            pending  → approved   (human HITL)
    POST /alerts/{id}/dismiss            pending  → dismissed  (human HITL)
    POST /alerts/{id}/notified           approved → notified   (Communicator Agent)
    POST /alerts/{id}/reopen             dismissed/notified → pending  (human re-triage)

  Alerts – Bulk actions
    POST /alerts/bulk/approve            approve a list of IDs at once
    POST /alerts/bulk/dismiss            dismiss a list of IDs at once

  Audit log
    GET  /alerts/{id}/history            full state-change history for one alert

Design decisions
────────────────
  - Async SQLAlchemy (asyncpg) throughout – no sync DB calls on the event loop
  - All state transitions validated server-side – can't double-approve, etc.
  - Audit trail written atomically with every state change
  - Pagination via `limit` + `offset` query params (cursor pagination can be
    added later without breaking the response shape)
  - Structured logging: every request logs method + path + status + duration
  - CORS permissive for development; tighten CORS_ORIGINS in production
"""
from __future__ import annotations

import os
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException, Depends, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from dotenv import load_dotenv
from api.security import require_agent_key, require_admin_key, rate_limit

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("aegis.api")

# ── Constants ─────────────────────────────────────────────────────────────────

VALID_STATUSES = {"pending", "approved", "dismissed", "notified"}

# State machine: which transitions are legal
_TRANSITIONS: dict[str, set[str]] = {
    "pending":   {"approved", "dismissed"},
    "approved":  {"notified", "pending"},    # reopen from approved too
    "dismissed": {"pending"},                # reopen
    "notified":  {"pending"},                # reopen
}

# ── Database ──────────────────────────────────────────────────────────────────

def _dsn() -> str:
    return (
        "postgresql+asyncpg://"
        f"{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'postgres')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ['POSTGRES_DB']}"
    )


engine       = create_async_engine(_dsn(), echo=False, pool_pre_ping=True, pool_size=10)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
metadata     = sa.MetaData()

# ── Table definitions ─────────────────────────────────────────────────────────

alerts_table = sa.Table(
    "alerts",
    metadata,
    sa.Column("id",             sa.Integer,                  primary_key=True),
    sa.Column("task_key",       sa.String(64),               nullable=False),
    sa.Column("task_summary",   sa.Text),
    sa.Column("assignee",       sa.String(255),              nullable=False),
    sa.Column("assignee_email", sa.String(255)),
    sa.Column("jira_url",       sa.Text),
    sa.Column("last_updated",   sa.DateTime(timezone=True)),
    sa.Column("detected_at",    sa.DateTime(timezone=True),  server_default=sa.func.now()),
    sa.Column("status",         sa.String(32),               server_default="pending"),
    sa.Column("slack_sent",       sa.Boolean,                  server_default="false"),
    sa.Column("slack_ts",         sa.String(64)),
    sa.Column("notes",            sa.Text),
    # Notification cooldown (migration 001)
    sa.Column("last_notified_at", sa.DateTime(timezone=True)),
    sa.Column("notify_count",     sa.Integer,    server_default="0",  default=0),
    sa.Column("cooldown_hours",   sa.Integer,    server_default="24", default=24),
)

audit_log_table = sa.Table(
    "alert_audit_log",
    metadata,
    sa.Column("id",          sa.Integer,                  primary_key=True),
    sa.Column("alert_id",    sa.Integer,                  nullable=False),
    sa.Column("from_status", sa.String(32)),
    sa.Column("to_status",   sa.String(32),               nullable=False),
    sa.Column("actor",       sa.String(64),               server_default="system"),
    sa.Column("notes",       sa.Text),
    sa.Column("created_at",  sa.DateTime(timezone=True),  server_default=sa.func.now()),
)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Aegis PM API — starting up")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)   # safety net; init.sql runs first in Docker
    yield
    log.info("Aegis PM API — shutting down")
    await engine.dispose()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Aegis PM API",
    description=(
        "HITL alert management backend for the Aegis autonomous project manager.\n\n"
        "Agents (Monitor, Communicator) and the HITL dashboard all talk through this API."
    ),
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount health router ──────────────────────────────────────────────────────

if _HEALTH_ROUTER_AVAILABLE and _health_router:
    app.include_router(_health_router)

# ── Middleware: request logging + timing ──────────────────────────────────────

@app.middleware("http")
async def _log_requests(request: Request, call_next) -> Response:
    import uuid as _uuid
    rid   = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
    start = time.perf_counter()
    response: Response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    log.info(
        "%s %s → %d  (%.1fms)  rid=%s",
        request.method,
        request.url.path,
        response.status_code,
        ms,
        rid[:8],
    )
    return response

# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )

# ── DB dependency ─────────────────────────────────────────────────────────────

async def get_db():
    async with SessionLocal() as session:
        yield session

# ══════════════════════════════════════════════════════════════════════════════
#  Pydantic schemas
# ══════════════════════════════════════════════════════════════════════════════

class AlertOut(BaseModel):
    id:             int
    task_key:       str
    task_summary:   Optional[str]
    assignee:       str
    assignee_email: Optional[str]
    jira_url:       Optional[str]
    last_updated:   Optional[datetime]
    detected_at:    datetime
    status:         str
    slack_sent:       bool
    slack_ts:         Optional[str]
    notes:            Optional[str]
    last_notified_at: Optional[datetime]
    notify_count:     int
    cooldown_hours:   int

    class Config:
        from_attributes = True


class AlertCreate(BaseModel):
    task_key:       str             = Field(..., min_length=1, max_length=64)
    task_summary:   Optional[str]  = None
    assignee:       str             = Field(..., min_length=1, max_length=255)
    assignee_email: Optional[str]  = None
    jira_url:       Optional[str]  = None
    last_updated:   Optional[datetime] = None

    @field_validator("task_key")
    @classmethod
    def task_key_upper(cls, v: str) -> str:
        return v.strip().upper()


class ActionRequest(BaseModel):
    notes: Optional[str] = Field(None, max_length=1000)
    actor: str           = Field("human", max_length=64)  # who performed the action


class BulkActionRequest(BaseModel):
    ids:   List[int]     = Field(..., min_length=1, max_length=100)
    notes: Optional[str] = Field(None, max_length=1000)
    actor: str           = Field("human", max_length=64)


class AlertStats(BaseModel):
    pending:   int
    approved:  int
    notified:  int
    dismissed: int
    total:     int


class AuditEntry(BaseModel):
    id:          int
    alert_id:    int
    from_status: Optional[str]
    to_status:   str
    actor:       str
    notes:       Optional[str]
    created_at:  datetime

    class Config:
        from_attributes = True


class PaginatedAlerts(BaseModel):
    items:  List[AlertOut]
    total:  int
    limit:  int
    offset: int


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

async def _fetch_or_404(alert_id: int, db: AsyncSession):
    result = await db.execute(
        sa.select(alerts_table).where(alerts_table.c.id == alert_id)
    )
    row = result.mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return row


async def _transition(
    alert_id:    int,
    to_status:   str,
    db:          AsyncSession,
    actor:       str = "system",
    notes:       Optional[str] = None,
    extra_values: Optional[dict] = None,
) -> dict:
    """
    Atomically transition an alert's status and write an audit log entry.

    Raises 404 if alert doesn't exist.
    Raises 400 if the transition is illegal per the state machine.
    Returns the updated alert row as a dict.
    """
    row = await _fetch_or_404(alert_id, db)
    current = row["status"]

    allowed = _TRANSITIONS.get(current, set())
    if to_status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot transition alert {alert_id} from '{current}' to '{to_status}'. "
                f"Allowed targets from '{current}': {sorted(allowed) or 'none'}"
            ),
        )

    # Update alert
    update_vals = {"status": to_status, "notes": notes}
    if extra_values:
        update_vals.update(extra_values)

    await db.execute(
        alerts_table.update()
        .where(alerts_table.c.id == alert_id)
        .values(**update_vals)
    )

    # Audit log entry
    await db.execute(
        audit_log_table.insert().values(
            alert_id=alert_id,
            from_status=current,
            to_status=to_status,
            actor=actor,
            notes=notes,
        )
    )

    await db.commit()
    log.info(
        "Alert %d: %s → %s  (actor=%s)",
        alert_id, current, to_status, actor,
    )
    return dict(await _fetch_or_404(alert_id, db))


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – System
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    summary="Liveness probe",
    tags=["System"],
)
async def health(db: AsyncSession = Depends(get_db)):
    """
    Returns 200 + DB connectivity status.
    Used by Docker Compose healthcheck and load balancers.
    """
    try:
        await db.execute(sa.text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "service": "aegis-pm-api",
        "version": "0.2.0",
        "database": "connected" if db_ok else "unreachable",
    }


@app.get(
    "/stats",
    response_model=AlertStats,
    summary="Alert counts by status",
    tags=["System"],
)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """
    Returns a summary count of alerts per status.
    Used by the HITL dashboard stat cards.
    """
    result = await db.execute(
        sa.select(
            alerts_table.c.status,
            sa.func.count().label("count"),
        ).group_by(alerts_table.c.status)
    )
    counts = {row.status: row.count for row in result}
    total = sum(counts.values())
    return AlertStats(
        pending=counts.get("pending", 0),
        approved=counts.get("approved", 0),
        notified=counts.get("notified", 0),
        dismissed=counts.get("dismissed", 0),
        total=total,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – Alerts CRUD
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/alerts",
    response_model=PaginatedAlerts,
    summary="List alerts with filters and pagination",
    tags=["Alerts"],
)
async def list_alerts(
    _auth: str = Depends(require_agent_key),
    _rl: None = Depends(rate_limit),
    # ── Filters ──────────────────────────────────────────────────────────────
    status:   Optional[str]  = Query(None, description="Filter by status: pending | approved | notified | dismissed"),
    assignee: Optional[str]  = Query(None, description="Filter by assignee name (partial match, case-insensitive)"),
    task_key: Optional[str]  = Query(None, description="Filter by Jira task key (partial match)"),
    slack_sent: Optional[bool] = Query(None, description="Filter by whether Slack was sent"),
    detected_after:  Optional[datetime] = Query(None, description="Detected at or after this datetime (ISO 8601)"),
    detected_before: Optional[datetime] = Query(None, description="Detected at or before this datetime (ISO 8601)"),
    # ── Sorting ───────────────────────────────────────────────────────────────
    order_by: Literal["detected_at", "last_updated", "status", "assignee"] = Query(
        "detected_at", description="Field to sort by"
    ),
    order_dir: Literal["asc", "desc"] = Query("desc", description="Sort direction"),
    # ── Pagination ────────────────────────────────────────────────────────────
    limit:  int = Query(50,  ge=1, le=500, description="Max items to return"),
    offset: int = Query(0,   ge=0,         description="Items to skip"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns a paginated list of alerts with optional filters.

    **Filters** can be combined freely. All filters are AND-ed together.

    **Pagination**: use `limit` + `offset`. The response includes `total`
    (unfiltered count with these filters applied) so the frontend can
    calculate page count.

    **Sorting**: default is newest first (`detected_at desc`).
    """
    if status and status not in VALID_STATUSES:
        raise HTTPException(
            400,
            detail=f"Invalid status '{status}'. Valid: {sorted(VALID_STATUSES)}",
        )

    col = alerts_table.c

    # ── Build WHERE clauses ────────────────────────────────────────────────────
    conditions = []
    if status:
        conditions.append(col.status == status)
    if assignee:
        conditions.append(col.assignee.ilike(f"%{assignee}%"))
    if task_key:
        conditions.append(col.task_key.ilike(f"%{task_key}%"))
    if slack_sent is not None:
        conditions.append(col.slack_sent == slack_sent)
    if detected_after:
        conditions.append(col.detected_at >= detected_after)
    if detected_before:
        conditions.append(col.detected_at <= detected_before)

    where_clause = sa.and_(*conditions) if conditions else sa.true()

    # ── Count total matching rows ──────────────────────────────────────────────
    count_result = await db.execute(
        sa.select(sa.func.count()).select_from(alerts_table).where(where_clause)
    )
    total = count_result.scalar_one()

    # ── Sort column ───────────────────────────────────────────────────────────
    sort_col  = getattr(col, order_by)
    sort_expr = sort_col.desc() if order_dir == "desc" else sort_col.asc()

    # ── Fetch page ────────────────────────────────────────────────────────────
    rows_result = await db.execute(
        sa.select(alerts_table)
        .where(where_clause)
        .order_by(sort_expr)
        .limit(limit)
        .offset(offset)
    )
    rows = rows_result.mappings().all()

    return PaginatedAlerts(
        items=[dict(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.post(
    "/alerts",
    response_model=AlertOut,
    status_code=201,
    summary="Create a new stale-task alert",
    tags=["Alerts"],
)
async def create_alert(
    payload: AlertCreate,
    db: AsyncSession = Depends(get_db),
    _auth: str = Depends(require_agent_key),
    _rl: None = Depends(rate_limit),
):
    """
    Called by the **Monitor Agent** when it finds a stale Jira task.

    **Idempotent**: if a `pending` alert already exists for the same
    `task_key`, the existing alert is returned without creating a duplicate.
    The HTTP status is still 201 to keep agent logic simple.
    """
    # Deduplicate: only one pending alert per task at a time
    existing_result = await db.execute(
        sa.select(alerts_table).where(
            alerts_table.c.task_key == payload.task_key,
            alerts_table.c.status == "pending",
        )
    )
    existing_row = existing_result.mappings().first()
    if existing_row:
        log.info("Duplicate suppressed – pending alert already exists for %s", payload.task_key)
        return dict(existing_row)

    ins_result = await db.execute(
        alerts_table.insert()
        .values(**payload.model_dump())
        .returning(alerts_table)
    )
    await db.commit()
    row = ins_result.mappings().first()

    # Write initial audit entry
    await db.execute(
        audit_log_table.insert().values(
            alert_id=row["id"],
            from_status=None,
            to_status="pending",
            actor="monitor_agent",
            notes=f"Detected stale task {payload.task_key}",
        )
    )
    await db.commit()

    log.info("Alert created: id=%d  task=%s  assignee=%s", row["id"], row["task_key"], row["assignee"])
    return dict(row)


@app.get(
    "/alerts/{alert_id}",
    response_model=AlertOut,
    summary="Get a single alert",
    tags=["Alerts"],
)
async def get_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Returns a single alert by ID. Raises 404 if not found."""
    row = await _fetch_or_404(alert_id, db)
    return dict(row)


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – State transitions
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/alerts/{alert_id}/approve",
    response_model=AlertOut,
    summary="Approve a pending alert",
    tags=["Alerts – Actions"],
)
async def approve_alert(
    alert_id: int,
    body: ActionRequest = ActionRequest(),
    db: AsyncSession = Depends(get_db),
    _auth: str = Depends(require_admin_key),
    _rl: None = Depends(rate_limit),
):
    """
    **Human HITL action.** Approves a `pending` alert.

    After approval the **Communicator Agent** will pick it up on its next
    30-second cycle and send a Slack message to the task assignee.

    State machine: `pending → approved`

    Returns 400 if the alert is not currently `pending`.
    """
    return await _transition(
        alert_id=alert_id,
        to_status="approved",
        db=db,
        actor=body.actor,
        notes=body.notes,
    )


@app.post(
    "/alerts/{alert_id}/dismiss",
    response_model=AlertOut,
    summary="Dismiss a pending alert",
    tags=["Alerts – Actions"],
)
async def dismiss_alert(
    alert_id: int,
    body: ActionRequest = ActionRequest(),
    db: AsyncSession = Depends(get_db),
    _auth: str = Depends(require_admin_key),
    _rl: None = Depends(rate_limit),
):
    """
    **Human HITL action.** Dismisses a `pending` alert.

    No Slack message will be sent. Use this when the alert is a false positive
    or the task already has an offline update.

    State machine: `pending → dismissed`

    Returns 400 if the alert is not currently `pending`.
    """
    return await _transition(
        alert_id=alert_id,
        to_status="dismissed",
        db=db,
        actor=body.actor,
        notes=body.notes,
    )


@app.post(
    "/alerts/{alert_id}/notified",
    response_model=AlertOut,
    summary="Mark alert as notified (Communicator Agent)",
    tags=["Alerts – Actions"],
)
async def mark_notified(
    alert_id: int,
    slack_ts: Optional[str] = Query(None, description="Slack message timestamp for threading"),
    db: AsyncSession = Depends(get_db),
):
    """
    Called by the **Communicator Agent** after a Slack message is sent.

    Sets `slack_sent=True` and records the Slack message timestamp
    (`slack_ts`) for future thread replies.

    State machine: `approved → notified`

    Returns 400 if the alert is not currently `approved`.
    """
    return await _transition(
        alert_id=alert_id,
        to_status="notified",
        db=db,
        actor="communicator_agent",
        notes="Slack notification sent",
        extra_values={"slack_sent": True, "slack_ts": slack_ts},
    )


@app.post(
    "/alerts/{alert_id}/reopen",
    response_model=AlertOut,
    summary="Reopen a dismissed or notified alert",
    tags=["Alerts – Actions"],
)
async def reopen_alert(
    alert_id: int,
    body: ActionRequest = ActionRequest(),
    db: AsyncSession = Depends(get_db),
    _auth: str = Depends(require_admin_key),
    _rl: None = Depends(rate_limit),
):
    """
    **Human HITL action.** Reopens an alert back to `pending`.

    Use when a dismissed alert needs another look, or a notified task
    is still blocked and requires another nudge.

    State machine: `dismissed | notified | approved → pending`

    Returns 400 if already `pending`.
    """
    return await _transition(
        alert_id=alert_id,
        to_status="pending",
        db=db,
        actor=body.actor,
        notes=body.notes or "Reopened",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – Bulk actions
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/alerts/bulk/approve",
    summary="Bulk approve a list of pending alerts",
    tags=["Alerts – Bulk"],
)
async def bulk_approve(
    body: BulkActionRequest,
    db: AsyncSession = Depends(get_db),
    _auth: str = Depends(require_admin_key),
    _rl: None = Depends(rate_limit),
):
    """
    Approve multiple `pending` alerts in a single request.
    Useful in the HITL dashboard when reviewing a batch.

    Returns a summary of successes and failures.
    Failures (e.g. wrong state) are reported but do not abort the batch.
    """
    return await _bulk_transition(body, to_status="approved", db=db)


@app.post(
    "/alerts/bulk/dismiss",
    summary="Bulk dismiss a list of pending alerts",
    tags=["Alerts – Bulk"],
)
async def bulk_dismiss(
    body: BulkActionRequest,
    db: AsyncSession = Depends(get_db),
    _auth: str = Depends(require_admin_key),
    _rl: None = Depends(rate_limit),
):
    """
    Dismiss multiple `pending` alerts in a single request.
    """
    return await _bulk_transition(body, to_status="dismissed", db=db)


async def _bulk_transition(
    body: BulkActionRequest,
    to_status: str,
    db: AsyncSession,
) -> dict:
    succeeded, failed = [], []
    for alert_id in body.ids:
        try:
            await _transition(
                alert_id=alert_id,
                to_status=to_status,
                db=db,
                actor=body.actor,
                notes=body.notes,
            )
            succeeded.append(alert_id)
        except HTTPException as exc:
            failed.append({"id": alert_id, "reason": exc.detail})
        except Exception as exc:
            failed.append({"id": alert_id, "reason": str(exc)})

    log.info(
        "Bulk %s: %d succeeded, %d failed",
        to_status, len(succeeded), len(failed),
    )
    return {
        "to_status":  to_status,
        "succeeded":  succeeded,
        "failed":     failed,
        "total":      len(body.ids),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – Audit log
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/alerts/{alert_id}/history",
    response_model=List[AuditEntry],
    summary="Get full state-change history for an alert",
    tags=["Audit"],
)
async def get_alert_history(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns every status transition ever made on an alert, oldest first.

    Useful for debugging, compliance, and the HITL dashboard detail view.
    """
    # Confirm the alert exists first
    await _fetch_or_404(alert_id, db)

    result = await db.execute(
        sa.select(audit_log_table)
        .where(audit_log_table.c.alert_id == alert_id)
        .order_by(audit_log_table.c.created_at.asc())
    )
    rows = result.mappings().all()
    return [dict(r) for r in rows]
