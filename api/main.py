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
from api.security import require_agent_key, require_admin_key, rate_limit, inject_request_id

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
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
    sa.Column("slack_sent",     sa.Boolean,                  server_default="false"),
    sa.Column("slack_ts",       sa.String(64)),
    sa.Column("notes",          sa.Text),
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

# ── New tables: Projects, Employees, Tasks ────────────────────────────────────

projects_table = sa.Table(
    "projects",
    metadata,
    sa.Column("id",          sa.Integer,                  primary_key=True),
    sa.Column("name",        sa.String(255),              nullable=False),
    sa.Column("description", sa.Text),
    sa.Column("prd_text",    sa.Text),
    sa.Column("status",      sa.String(32),               server_default="active"),
    sa.Column("total_tasks", sa.Integer,                  server_default="0"),
    sa.Column("completed_tasks", sa.Integer,              server_default="0"),
    sa.Column("created_at",  sa.DateTime(timezone=True),  server_default=sa.func.now()),
    sa.Column("updated_at",  sa.DateTime(timezone=True),  server_default=sa.func.now()),
)

employees_table = sa.Table(
    "employees",
    metadata,
    sa.Column("id",          sa.Integer,                  primary_key=True),
    sa.Column("name",        sa.String(255),              nullable=False),
    sa.Column("email",       sa.String(255)),
    sa.Column("role",        sa.String(128)),
    sa.Column("skills",      sa.Text),              # JSON array stored as text
    sa.Column("availability",sa.String(32),               server_default="available"),
    sa.Column("current_load",sa.Integer,                  server_default="0"),
    sa.Column("created_at",  sa.DateTime(timezone=True),  server_default=sa.func.now()),
)

tasks_table = sa.Table(
    "tasks",
    metadata,
    sa.Column("id",              sa.Integer,                  primary_key=True),
    sa.Column("project_id",      sa.Integer,                  nullable=False),
    sa.Column("title",           sa.String(500),              nullable=False),
    sa.Column("description",     sa.Text),
    sa.Column("priority",        sa.String(32),               server_default="medium"),
    sa.Column("status",          sa.String(32),               server_default="todo"),
    sa.Column("estimated_hours", sa.Float,                    server_default="0"),
    sa.Column("assigned_to",     sa.Integer),                 # employee_id
    sa.Column("assigned_name",   sa.String(255)),
    sa.Column("ai_confidence",   sa.Float),
    sa.Column("required_skills", sa.Text),             # JSON array
    sa.Column("created_at",      sa.DateTime(timezone=True),  server_default=sa.func.now()),
    sa.Column("completed_at",    sa.DateTime(timezone=True)),
)



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

# if _HEALTH_ROUTER_AVAILABLE and _health_router:
#     app.include_router(_health_router)

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
    slack_sent:     bool
    slack_ts:       Optional[str]
    notes:          Optional[str]

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


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – Analytics
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/analytics",
    summary="Full analytics data for the dashboard",
    tags=["Analytics"],
)
async def get_analytics(db: AsyncSession = Depends(get_db)):
    """
    Returns comprehensive analytics data:
    - Status distribution
    - Alerts by assignee
    - Daily alert trend (last 7 days)
    - Recent audit log activity
    - Resolution metrics
    """
    from datetime import timedelta

    now = datetime.utcnow()
    col = alerts_table.c

    # 1. Status distribution
    status_result = await db.execute(
        sa.select(col.status, sa.func.count().label("count"))
        .group_by(col.status)
    )
    status_dist = {row.status: row.count for row in status_result}

    # 2. Alerts by assignee
    assignee_result = await db.execute(
        sa.select(col.assignee, col.status, sa.func.count().label("count"))
        .group_by(col.assignee, col.status)
    )
    assignee_map = {}
    for row in assignee_result:
        if row.assignee not in assignee_map:
            assignee_map[row.assignee] = {"total": 0, "pending": 0, "approved": 0, "dismissed": 0, "notified": 0}
        assignee_map[row.assignee][row.status] = row.count
        assignee_map[row.assignee]["total"] += row.count

    assignee_breakdown = [
        {"assignee": k, **v} for k, v in sorted(assignee_map.items(), key=lambda x: -x[1]["total"])
    ]

    # 3. Daily trend (last 7 days)
    daily_trend = []
    for i in range(6, -1, -1):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        day_result = await db.execute(
            sa.select(sa.func.count()).select_from(alerts_table)
            .where(col.detected_at >= day_start, col.detected_at < day_end)
        )
        count = day_result.scalar_one()
        daily_trend.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "label": day_start.strftime("%a"),
            "count": count,
        })

    # 4. Recent audit activity (last 20 entries)
    audit_result = await db.execute(
        sa.select(audit_log_table)
        .order_by(audit_log_table.c.created_at.desc())
        .limit(20)
    )
    recent_activity = [dict(r) for r in audit_result.mappings().all()]

    # 5. Resolution metrics
    resolved_result = await db.execute(
        sa.select(sa.func.count()).select_from(alerts_table)
        .where(col.status.in_(["approved", "notified", "dismissed"]))
    )
    total_resolved = resolved_result.scalar_one()

    total_result = await db.execute(
        sa.select(sa.func.count()).select_from(alerts_table)
    )
    total_all = total_result.scalar_one()

    # Slack sent count
    slack_result = await db.execute(
        sa.select(sa.func.count()).select_from(alerts_table)
        .where(col.slack_sent == True)
    )
    slack_sent = slack_result.scalar_one()

    return {
        "status_distribution": status_dist,
        "assignee_breakdown": assignee_breakdown,
        "daily_trend": daily_trend,
        "recent_activity": recent_activity,
        "metrics": {
            "total_alerts": total_all,
            "total_resolved": total_resolved,
            "resolution_rate": round(total_resolved / max(total_all, 1) * 100, 1),
            "slack_notifications_sent": slack_sent,
            "pending": status_dist.get("pending", 0),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – Employees
# ══════════════════════════════════════════════════════════════════════════════

class EmployeeCreate(BaseModel):
    name:         str  = Field(..., min_length=1, max_length=255)
    email:        Optional[str] = None
    role:         Optional[str] = None
    skills:       List[str]     = Field(default_factory=list)
    availability: str           = "available"

class EmployeeOut(BaseModel):
    id:           int
    name:         str
    email:        Optional[str]
    role:         Optional[str]
    skills:       str      # JSON string
    availability: str
    current_load: int
    created_at:   datetime
    class Config:
        from_attributes = True


@app.post("/employees", status_code=201, summary="Add employee", tags=["Employees"])
async def create_employee(payload: EmployeeCreate, db: AsyncSession = Depends(get_db)):
    import json
    result = await db.execute(
        employees_table.insert()
        .values(name=payload.name, email=payload.email, role=payload.role,
                skills=json.dumps(payload.skills), availability=payload.availability)
        .returning(employees_table)
    )
    await db.commit()
    row = result.mappings().first()
    log.info("Employee created: %s (%s)", row["name"], row["role"])
    return dict(row)


@app.get("/employees", summary="List employees", tags=["Employees"])
async def list_employees(db: AsyncSession = Depends(get_db)):
    import json
    result = await db.execute(sa.select(employees_table).order_by(employees_table.c.name))
    rows = result.mappings().all()
    emp_list = []
    for r in rows:
        d = dict(r)
        try:
            d["skills_list"] = json.loads(d["skills"]) if d["skills"] else []
        except:
            d["skills_list"] = []
        emp_list.append(d)
    return emp_list


@app.put("/employees/{emp_id}", summary="Update employee", tags=["Employees"])
async def update_employee(emp_id: int, payload: EmployeeCreate, db: AsyncSession = Depends(get_db)):
    import json
    result = await db.execute(sa.select(employees_table).where(employees_table.c.id == emp_id))
    if not result.mappings().first():
        raise HTTPException(404, f"Employee {emp_id} not found")
    await db.execute(
        employees_table.update().where(employees_table.c.id == emp_id)
        .values(name=payload.name, email=payload.email, role=payload.role,
                skills=json.dumps(payload.skills), availability=payload.availability)
    )
    await db.commit()
    result2 = await db.execute(sa.select(employees_table).where(employees_table.c.id == emp_id))
    return dict(result2.mappings().first())


@app.delete("/employees/{emp_id}", summary="Delete employee", tags=["Employees"])
async def delete_employee(emp_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(sa.select(employees_table).where(employees_table.c.id == emp_id))
    if not result.mappings().first():
        raise HTTPException(404, f"Employee {emp_id} not found")
    await db.execute(employees_table.delete().where(employees_table.c.id == emp_id))
    await db.commit()
    return {"deleted": emp_id}


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – Projects
# ══════════════════════════════════════════════════════════════════════════════

class ProjectCreate(BaseModel):
    name:        str           = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    prd_text:    Optional[str] = None


@app.post("/projects", status_code=201, summary="Create project", tags=["Projects"])
async def create_project(payload: ProjectCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        projects_table.insert()
        .values(name=payload.name, description=payload.description, prd_text=payload.prd_text)
        .returning(projects_table)
    )
    await db.commit()
    row = result.mappings().first()
    log.info("Project created: %s", row["name"])
    return dict(row)


@app.get("/projects", summary="List projects", tags=["Projects"])
async def list_projects(db: AsyncSession = Depends(get_db)):
    result = await db.execute(sa.select(projects_table).order_by(projects_table.c.created_at.desc()))
    projects = []
    for r in result.mappings().all():
        d = dict(r)
        # get task counts
        task_result = await db.execute(
            sa.select(tasks_table.c.status, sa.func.count().label("c"))
            .where(tasks_table.c.project_id == d["id"])
            .group_by(tasks_table.c.status)
        )
        status_counts = {row.status: row.c for row in task_result}
        d["task_stats"] = status_counts
        d["total_tasks"] = sum(status_counts.values())
        d["completed_tasks"] = status_counts.get("done", 0)
        d["progress"] = round(d["completed_tasks"] / max(d["total_tasks"], 1) * 100, 1)
        projects.append(d)
    return projects


@app.get("/projects/{project_id}", summary="Get project detail", tags=["Projects"])
async def get_project(project_id: int, db: AsyncSession = Depends(get_db)):
    import json
    result = await db.execute(sa.select(projects_table).where(projects_table.c.id == project_id))
    row = result.mappings().first()
    if not row:
        raise HTTPException(404, f"Project {project_id} not found")
    d = dict(row)
    # Get tasks
    task_result = await db.execute(
        sa.select(tasks_table).where(tasks_table.c.project_id == project_id)
        .order_by(tasks_table.c.priority.desc(), tasks_table.c.created_at)
    )
    tasks = []
    for t in task_result.mappings().all():
        td = dict(t)
        try:
            td["required_skills_list"] = json.loads(td["required_skills"]) if td["required_skills"] else []
        except:
            td["required_skills_list"] = []
        tasks.append(td)
    d["tasks"] = tasks

    status_counts = {}
    for t in tasks:
        s = t["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    d["task_stats"] = status_counts
    d["total_tasks"] = len(tasks)
    d["completed_tasks"] = status_counts.get("done", 0)
    d["progress"] = round(d["completed_tasks"] / max(d["total_tasks"], 1) * 100, 1)
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – AI PRD Parsing
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/projects/{project_id}/parse", summary="AI parse PRD into tasks", tags=["AI"])
async def parse_prd(project_id: int, db: AsyncSession = Depends(get_db)):
    """
    Uses OpenAI to parse the project's PRD text into structured tasks.
    Each task gets a title, description, priority, estimated hours, and required skills.
    """
    import json
    import httpx

    # Get the project
    result = await db.execute(sa.select(projects_table).where(projects_table.c.id == project_id))
    project = result.mappings().first()
    if not project:
        raise HTTPException(404, f"Project {project_id} not found")
    if not project["prd_text"]:
        raise HTTPException(400, "No PRD text uploaded for this project")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-..."):
        # Fallback: generate mock tasks from PRD keywords
        log.warning("No valid OpenAI key — using rule-based task extraction")
        tasks = _fallback_parse_prd(project["prd_text"])
    else:
        prompt = f"""You are a project management AI. Analyze this Product Requirements Document (PRD) and break it down into actionable development tasks.

For each task, provide:
- title: Clear, concise task title
- description: Detailed description of what needs to be done
- priority: "high", "medium", or "low"
- estimated_hours: Estimated hours to complete (number)
- required_skills: Array of skill tags needed (e.g. ["python", "react", "devops", "ml", "backend", "frontend", "testing", "database", "api", "ui/ux"])

Return a JSON array of tasks. Respond with ONLY valid JSON, no markdown.

PRD:
{project["prd_text"][:4000]}"""

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 3000,
                    }
                )
                if resp.status_code != 200:
                    log.error("OpenAI error: %s", resp.text)
                    tasks = _fallback_parse_prd(project["prd_text"])
                else:
                    content = resp.json()["choices"][0]["message"]["content"]
                    # Clean markdown if present
                    if content.strip().startswith("```"):
                        content = content.strip().split("\n", 1)[1].rsplit("```", 1)[0]
                    tasks = json.loads(content)
        except Exception as e:
            log.error("OpenAI parsing failed: %s", e)
            tasks = _fallback_parse_prd(project["prd_text"])

    # Insert tasks into DB
    created_tasks = []
    for t in tasks:
        ins = await db.execute(
            tasks_table.insert().values(
                project_id=project_id,
                title=t.get("title", "Untitled Task"),
                description=t.get("description", ""),
                priority=t.get("priority", "medium"),
                estimated_hours=t.get("estimated_hours", 4),
                required_skills=json.dumps(t.get("required_skills", [])),
            ).returning(tasks_table)
        )
        created_tasks.append(dict(ins.mappings().first()))

    # Update project task count
    await db.execute(
        projects_table.update().where(projects_table.c.id == project_id)
        .values(total_tasks=len(created_tasks))
    )
    await db.commit()

    log.info("Parsed PRD for project %d: %d tasks created", project_id, len(created_tasks))
    return {"project_id": project_id, "tasks_created": len(created_tasks), "tasks": created_tasks}


def _fallback_parse_prd(prd_text: str) -> list:
    """Simple rule-based fallback when OpenAI is not available."""
    import re
    lines = prd_text.strip().split("\n")
    tasks = []
    skill_keywords = {
        "api": ["api", "endpoint", "rest", "graphql"],
        "frontend": ["ui", "interface", "dashboard", "page", "component", "react", "css"],
        "backend": ["server", "logic", "middleware", "service", "handler"],
        "database": ["database", "schema", "migration", "table", "query", "sql"],
        "testing": ["test", "coverage", "qa", "validation"],
        "devops": ["deploy", "ci/cd", "docker", "kubernetes", "pipeline"],
        "ml": ["model", "ai", "machine learning", "prediction", "training"],
        "python": ["python", "flask", "django", "fastapi"],
        "react": ["react", "next.js", "component", "jsx"],
    }

    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue
        # Extract headings or bullet points as task candidates
        if line.startswith(("#", "-", "*", "•")) or re.match(r'^\d+\.', line):
            title = re.sub(r'^[#\-*•\d.]+\s*', '', line).strip()
            if len(title) < 5:
                continue
            # Detect skills from text
            skills = []
            lower = title.lower()
            for skill, keywords in skill_keywords.items():
                if any(kw in lower for kw in keywords):
                    skills.append(skill)
            if not skills:
                skills = ["backend"]  # default

            tasks.append({
                "title": title[:200],
                "description": f"Task derived from PRD: {title}",
                "priority": "high" if any(w in lower for w in ["critical", "must", "important", "core"]) else "medium",
                "estimated_hours": 8,
                "required_skills": skills,
            })

    if not tasks:
        # Create at least a few generic tasks
        tasks = [
            {"title": "Review and finalize requirements", "description": "Review the PRD and clarify requirements", "priority": "high", "estimated_hours": 4, "required_skills": ["backend"]},
            {"title": "Design system architecture", "description": "Create architecture diagram and tech stack decisions", "priority": "high", "estimated_hours": 8, "required_skills": ["backend", "devops"]},
            {"title": "Set up development environment", "description": "Configure repos, CI/CD, and dev tools", "priority": "medium", "estimated_hours": 4, "required_skills": ["devops"]},
        ]
    return tasks[:20]  # Cap at 20 tasks


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – AI Task Assignment
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/projects/{project_id}/assign-all", summary="AI assign all unassigned tasks", tags=["AI"])
async def ai_assign_all(project_id: int, db: AsyncSession = Depends(get_db)):
    """
    For each unassigned task in the project, use AI to match the best employee
    based on skill overlap, current workload, and availability.
    """
    import json

    # Get unassigned tasks
    task_result = await db.execute(
        sa.select(tasks_table).where(
            tasks_table.c.project_id == project_id,
            tasks_table.c.assigned_to.is_(None),
        )
    )
    unassigned = [dict(r) for r in task_result.mappings().all()]
    if not unassigned:
        return {"message": "No unassigned tasks", "assignments": []}

    # Get available employees
    emp_result = await db.execute(
        sa.select(employees_table).where(employees_table.c.availability == "available")
    )
    employees = []
    for r in emp_result.mappings().all():
        d = dict(r)
        try:
            d["skills_list"] = json.loads(d["skills"]) if d["skills"] else []
        except:
            d["skills_list"] = []
        employees.append(d)

    if not employees:
        raise HTTPException(400, "No available employees. Add employees first.")

    assignments = []
    for task in unassigned:
        try:
            task_skills = json.loads(task["required_skills"]) if task["required_skills"] else []
        except:
            task_skills = []

        # Score each employee
        best_emp = None
        best_score = -1
        for emp in employees:
            emp_skills = emp["skills_list"]
            # Skill overlap score (0–1)
            if task_skills:
                overlap = len(set(s.lower() for s in task_skills) & set(s.lower() for s in emp_skills))
                skill_score = overlap / len(task_skills)
            else:
                skill_score = 0.5

            # Workload penalty (fewer tasks = better)
            load_score = max(0, 1 - (emp["current_load"] * 0.15))

            # Combined score
            score = (skill_score * 0.7) + (load_score * 0.3)
            if score > best_score:
                best_score = score
                best_emp = emp

        if best_emp:
            confidence = round(best_score * 100, 1)
            await db.execute(
                tasks_table.update().where(tasks_table.c.id == task["id"])
                .values(assigned_to=best_emp["id"], assigned_name=best_emp["name"], ai_confidence=confidence)
            )
            # Increment load
            await db.execute(
                employees_table.update().where(employees_table.c.id == best_emp["id"])
                .values(current_load=employees_table.c.current_load + 1)
            )
            best_emp["current_load"] += 1  # update in-memory too

            assignments.append({
                "task_id": task["id"],
                "task_title": task["title"],
                "assigned_to": best_emp["name"],
                "employee_id": best_emp["id"],
                "confidence": confidence,
                "matched_skills": list(set(s.lower() for s in (json.loads(task["required_skills"]) if task["required_skills"] else [])) & set(s.lower() for s in best_emp["skills_list"])),
            })

    await db.commit()
    log.info("AI assigned %d tasks in project %d", len(assignments), project_id)
    return {"project_id": project_id, "assignments": assignments, "total_assigned": len(assignments)}


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – Tasks
# ══════════════════════════════════════════════════════════════════════════════

class TaskStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(todo|in_progress|done)$")

@app.get("/projects/{project_id}/tasks", summary="Get project tasks", tags=["Tasks"])
async def list_project_tasks(project_id: int, db: AsyncSession = Depends(get_db)):
    import json
    result = await db.execute(
        sa.select(tasks_table).where(tasks_table.c.project_id == project_id)
        .order_by(
            sa.case(
                (tasks_table.c.priority == "high", 1),
                (tasks_table.c.priority == "medium", 2),
                (tasks_table.c.priority == "low", 3),
                else_=4
            ),
            tasks_table.c.created_at
        )
    )
    tasks = []
    for r in result.mappings().all():
        d = dict(r)
        try:
            d["required_skills_list"] = json.loads(d["required_skills"]) if d["required_skills"] else []
        except:
            d["required_skills_list"] = []
        tasks.append(d)
    return tasks


@app.put("/tasks/{task_id}/status", summary="Update task status", tags=["Tasks"])
async def update_task_status(task_id: int, body: TaskStatusUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(sa.select(tasks_table).where(tasks_table.c.id == task_id))
    task = result.mappings().first()
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")

    update_vals = {"status": body.status}
    if body.status == "done":
        update_vals["completed_at"] = datetime.utcnow()

    await db.execute(tasks_table.update().where(tasks_table.c.id == task_id).values(**update_vals))

    # Update project completed count
    if body.status == "done" or task["status"] == "done":
        proj_id = task["project_id"]
        done_result = await db.execute(
            sa.select(sa.func.count()).select_from(tasks_table)
            .where(tasks_table.c.project_id == proj_id, tasks_table.c.status == "done")
        )
        done_count = done_result.scalar_one()
        await db.execute(
            projects_table.update().where(projects_table.c.id == proj_id)
            .values(completed_tasks=done_count)
        )

    await db.commit()
    updated = await db.execute(sa.select(tasks_table).where(tasks_table.c.id == task_id))
    return dict(updated.mappings().first())


# ══════════════════════════════════════════════════════════════════════════════
#  Routes – Project Analytics
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/projects/{project_id}/analytics", summary="Project analytics", tags=["Analytics"])
async def project_analytics(project_id: int, db: AsyncSession = Depends(get_db)):
    import json
    # Get project
    proj_result = await db.execute(sa.select(projects_table).where(projects_table.c.id == project_id))
    project = proj_result.mappings().first()
    if not project:
        raise HTTPException(404, f"Project {project_id} not found")

    # Task stats
    task_result = await db.execute(
        sa.select(tasks_table).where(tasks_table.c.project_id == project_id)
    )
    tasks = [dict(r) for r in task_result.mappings().all()]
    total = len(tasks)
    done = sum(1 for t in tasks if t["status"] == "done")
    in_progress = sum(1 for t in tasks if t["status"] == "in_progress")
    todo = sum(1 for t in tasks if t["status"] == "todo")

    # Priority breakdown
    high = sum(1 for t in tasks if t["priority"] == "high")
    medium = sum(1 for t in tasks if t["priority"] == "medium")
    low = sum(1 for t in tasks if t["priority"] == "low")

    # Workload by assignee
    workload = {}
    for t in tasks:
        name = t["assigned_name"] or "Unassigned"
        if name not in workload:
            workload[name] = {"total": 0, "done": 0, "in_progress": 0, "todo": 0, "hours": 0}
        workload[name]["total"] += 1
        workload[name][t["status"]] = workload[name].get(t["status"], 0) + 1
        workload[name]["hours"] += t["estimated_hours"] or 0

    # Total estimated hours
    total_hours = sum(t["estimated_hours"] or 0 for t in tasks)
    completed_hours = sum(t["estimated_hours"] or 0 for t in tasks if t["status"] == "done")

    return {
        "project": {"id": project["id"], "name": project["name"], "status": project["status"]},
        "progress": round(done / max(total, 1) * 100, 1),
        "task_summary": {
            "total": total, "done": done, "in_progress": in_progress, "todo": todo,
        },
        "priority_breakdown": {"high": high, "medium": medium, "low": low},
        "workload": [{"assignee": k, **v} for k, v in sorted(workload.items(), key=lambda x: -x[1]["total"])],
        "hours": {"total_estimated": round(total_hours, 1), "completed": round(completed_hours, 1)},
    }
