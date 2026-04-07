"""
aegis-pm / agents / health / monitor.py

Agent health monitoring system.

Features
────────
  AgentHealthTracker  – records last-run timestamp, last-error, consecutive
                        failures; stored in-process (survives restarts via DB)
  /agents/status      – HTTP endpoint polled by dashboard + load balancers
  Slack dead-man       – fires a Slack alert when an agent misses N consecutive
                        cycles or fails with an exception
  DB heartbeat table  – `agent_heartbeats` persisted to Postgres so you can
                        query "when did monitor_agent last succeed?" from SQL

Usage
─────
  # In runner.py, wrap each agent call:
  from agents.health.monitor import health, AgentName

  async def run_monitor():
      with health.track(AgentName.MONITOR):
          summary = await loop.run_in_executor(None, MonitorAgent().run_once)
          health.record_success(AgentName.MONITOR, stale_found=summary["stale_found"])

  # Mount the status endpoint in FastAPI:
  from agents.health.monitor import router as health_router
  app.include_router(health_router)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

import httpx
import psycopg2
from fastapi import APIRouter

log = logging.getLogger("aegis.health")

SLACK_WEBHOOK_URL    = os.getenv("SLACK_WEBHOOK_URL", "")
FAILURE_ALERT_AFTER  = int(os.getenv("HEALTH_ALERT_AFTER_FAILURES", "2"))
STALE_THRESHOLD_SECS = int(os.getenv("HEALTH_STALE_SECS", "720"))  # 12 min default

DB_PARAMS: Dict[str, str] = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     os.getenv("POSTGRES_PORT", "5432"),
    "dbname":   os.getenv("POSTGRES_DB", "aegispm"),
    "user":     os.getenv("POSTGRES_USER", "aegis"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}


class AgentName(str, Enum):
    MONITOR      = "monitor_agent"
    COMMUNICATOR = "communicator_agent"
    GROUP_CHAT   = "group_chat"


@dataclass
class AgentStatus:
    name:                str
    last_success_at:     Optional[datetime]  = None
    last_failure_at:     Optional[datetime]  = None
    last_error:          Optional[str]       = None
    consecutive_failures: int                = 0
    total_cycles:        int                 = 0
    total_failures:      int                 = 0
    last_metadata:       Dict[str, Any]      = field(default_factory=dict)
    alert_sent:          bool                = False   # avoid Slack spam

    @property
    def is_healthy(self) -> bool:
        if self.consecutive_failures >= FAILURE_ALERT_AFTER:
            return False
        if self.last_success_at is None and self.total_cycles > 1:
            return False
        if self.last_success_at:
            age = (datetime.now(timezone.utc) - self.last_success_at).total_seconds()
            if age > STALE_THRESHOLD_SECS:
                return False
        return True

    @property
    def status_str(self) -> str:
        if self.consecutive_failures >= FAILURE_ALERT_AFTER:
            return "critical"
        if not self.is_healthy:
            return "stale"
        return "healthy"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":                 self.name,
            "status":               self.status_str,
            "is_healthy":           self.is_healthy,
            "last_success_at":      self.last_success_at.isoformat() if self.last_success_at else None,
            "last_failure_at":      self.last_failure_at.isoformat() if self.last_failure_at else None,
            "last_error":           self.last_error,
            "consecutive_failures": self.consecutive_failures,
            "total_cycles":         self.total_cycles,
            "total_failures":       self.total_failures,
            "last_metadata":        self.last_metadata,
        }


class AgentHealthTracker:
    """
    Thread-safe in-process health tracker with Postgres persistence
    and Slack alerting.
    """

    def __init__(self) -> None:
        self._statuses: Dict[str, AgentStatus] = {
            name.value: AgentStatus(name=name.value)
            for name in AgentName
        }
        self._lock   = threading.Lock()
        self._ensure_table()

    # ── Public API ────────────────────────────────────────────────────────────

    @contextmanager
    def track(self, agent: AgentName):
        """
        Context manager that automatically records success or failure.

        Usage:
            with health.track(AgentName.MONITOR):
                agent.run_once()
        """
        try:
            yield
            self.record_success(agent)
        except Exception as exc:
            self.record_failure(agent, str(exc))
            raise

    def record_success(
        self,
        agent:    AgentName,
        **metadata,
    ) -> None:
        name = agent.value
        now  = datetime.now(timezone.utc)
        with self._lock:
            s = self._statuses[name]
            s.last_success_at      = now
            s.consecutive_failures = 0
            s.total_cycles        += 1
            s.last_metadata        = metadata
            s.alert_sent           = False   # reset so next failure re-alerts
        log.debug("Health: %s SUCCESS  cycles=%d  meta=%s", name, self._statuses[name].total_cycles, metadata)
        self._persist_heartbeat(name, "success", metadata)

    def record_failure(
        self,
        agent: AgentName,
        error: str,
    ) -> None:
        name = agent.value
        now  = datetime.now(timezone.utc)
        with self._lock:
            s = self._statuses[name]
            s.last_failure_at       = now
            s.last_error            = error[:500]
            s.consecutive_failures += 1
            s.total_cycles         += 1
            s.total_failures       += 1
            should_alert = (
                s.consecutive_failures >= FAILURE_ALERT_AFTER
                and not s.alert_sent
            )
            if should_alert:
                s.alert_sent = True

        log.error(
            "Health: %s FAILURE  consecutive=%d  error=%s",
            name, self._statuses[name].consecutive_failures, error[:120],
        )
        self._persist_heartbeat(name, "failure", {"error": error})

        if should_alert:
            self._send_slack_alert(name, error, self._statuses[name].consecutive_failures)

    def get_status(self, agent: AgentName) -> Dict[str, Any]:
        with self._lock:
            return self._statuses[agent.value].to_dict()

    def get_all_statuses(self) -> Dict[str, Any]:
        with self._lock:
            statuses = {k: v.to_dict() for k, v in self._statuses.items()}
        overall_healthy = all(v["is_healthy"] for v in statuses.values())
        return {
            "overall": "healthy" if overall_healthy else "degraded",
            "agents":  statuses,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Slack alert ───────────────────────────────────────────────────────────

    def _send_slack_alert(self, agent_name: str, error: str, failures: int) -> None:
        if not SLACK_WEBHOOK_URL:
            log.warning("Health: Slack alert skipped – SLACK_WEBHOOK_URL not set")
            return

        message = {
            "text": f"⚠️ Aegis PM – Agent failure alert",
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": "⚠️  Aegis PM – Agent Alert", "emoji": True},
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Agent:*\n`{agent_name}`"},
                        {"type": "mrkdwn", "text": f"*Consecutive failures:*\n{failures}"},
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*Error:*\n```{error[:400]}```\n\n"
                            f"The agent will keep retrying. Check container logs:\n"
                            f"`docker compose logs -f agents`"
                        ),
                    },
                },
            ],
        }

        try:
            r = httpx.post(SLACK_WEBHOOK_URL, json=message, timeout=5.0)
            r.raise_for_status()
            log.info("Health: Slack alert sent for %s", agent_name)
        except Exception as exc:
            log.error("Health: Failed to send Slack alert: %s", exc)

    # ── DB persistence ────────────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        """Create agent_heartbeats table if it doesn't exist."""
        sql = """
            CREATE TABLE IF NOT EXISTS agent_heartbeats (
                id          SERIAL PRIMARY KEY,
                agent_name  VARCHAR(64)  NOT NULL,
                outcome     VARCHAR(16)  NOT NULL,   -- 'success' | 'failure'
                metadata    JSONB,
                recorded_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_heartbeats_agent
                ON agent_heartbeats (agent_name, recorded_at DESC);
        """
        try:
            conn = psycopg2.connect(**DB_PARAMS)
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            conn.close()
        except Exception as exc:
            log.warning("Health: Could not create heartbeats table: %s", exc)

    def _persist_heartbeat(
        self,
        agent_name: str,
        outcome:    str,
        metadata:   Dict,
    ) -> None:
        import json as _json
        try:
            conn = psycopg2.connect(**DB_PARAMS)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO agent_heartbeats (agent_name, outcome, metadata) VALUES (%s, %s, %s)",
                    (agent_name, outcome, _json.dumps(metadata)),
                )
            conn.commit()
            conn.close()
        except Exception as exc:
            # Never let DB write failures break the agent loop
            log.debug("Health: heartbeat persist failed: %s", exc)


# ── Singleton ─────────────────────────────────────────────────────────────────

health = AgentHealthTracker()


# ── FastAPI router ────────────────────────────────────────────────────────────

router = APIRouter(prefix="/agents", tags=["Health"])


@router.get("/status", summary="Agent health status")
def agent_status():
    """
    Returns health status for all agents.
    Used by the HITL dashboard and external monitoring (UptimeRobot, etc.).

    Responds with HTTP 200 if all agents healthy, 503 if any are degraded.
    """
    from fastapi.responses import JSONResponse
    data = health.get_all_statuses()
    code = 200 if data["overall"] == "healthy" else 503
    return JSONResponse(content=data, status_code=code)


@router.get("/status/{agent_name}", summary="Single agent health")
def single_agent_status(agent_name: str):
    from fastapi.responses import JSONResponse
    try:
        agent = AgentName(agent_name)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(404, f"Unknown agent: {agent_name}")

    data = health.get_status(agent)
    code = 200 if data["is_healthy"] else 503
    return JSONResponse(content=data, status_code=code)
