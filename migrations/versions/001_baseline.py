"""Baseline schema – create all tables and add cooldown columns

Revision ID: 001_baseline
Revises:
Create Date: 2025-01-01 00:00:00.000000

This is the single source-of-truth migration for a fresh database.
It is idempotent: safe to run against a DB that was bootstrapped by
init.sql directly (all DDL uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS).
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# ── Revision identifiers ──────────────────────────────────────────────────────

revision = "001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── alerts ────────────────────────────────────────────────────────────────
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS alerts (
            id              SERIAL PRIMARY KEY,
            task_key        VARCHAR(64)  NOT NULL,
            task_summary    TEXT,
            assignee        VARCHAR(255) NOT NULL,
            assignee_email  VARCHAR(255),
            jira_url        TEXT,
            last_updated    TIMESTAMPTZ,
            detected_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            status          VARCHAR(32)  NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'approved', 'dismissed', 'notified')),
            slack_sent      BOOLEAN      NOT NULL DEFAULT FALSE,
            slack_ts        VARCHAR(64),
            notes           TEXT
        )
    """))

    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS idx_alerts_status   ON alerts (status)"
    ))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS idx_alerts_task_key ON alerts (task_key)"
    ))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS idx_alerts_detected ON alerts (detected_at DESC)"
    ))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS idx_alerts_assignee ON alerts (assignee)"
    ))
    op.execute(sa.text("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_alerts_pending_unique
            ON alerts (task_key)
            WHERE status = 'pending'
    """))

    # ── alert_audit_log ───────────────────────────────────────────────────────
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS alert_audit_log (
            id          SERIAL PRIMARY KEY,
            alert_id    INTEGER     NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
            from_status VARCHAR(32),
            to_status   VARCHAR(32) NOT NULL,
            actor       VARCHAR(64) NOT NULL DEFAULT 'system',
            notes       TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """))

    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS idx_audit_alert_id ON alert_audit_log (alert_id)"
    ))
    op.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS idx_audit_created  ON alert_audit_log (created_at DESC)"
    ))

    # ── cooldown columns (ADD COLUMN IF NOT EXISTS is safe on existing DBs) ──
    op.execute(sa.text(
        "ALTER TABLE alerts ADD COLUMN IF NOT EXISTS last_notified_at TIMESTAMPTZ"
    ))
    op.execute(sa.text(
        "ALTER TABLE alerts ADD COLUMN IF NOT EXISTS notify_count INTEGER NOT NULL DEFAULT 0"
    ))
    op.execute(sa.text(
        "ALTER TABLE alerts ADD COLUMN IF NOT EXISTS cooldown_hours INTEGER NOT NULL DEFAULT 24"
    ))

    op.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_alerts_last_notified
            ON alerts (last_notified_at DESC)
            WHERE last_notified_at IS NOT NULL
    """))

    # ── agent_heartbeats ──────────────────────────────────────────────────────
    op.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS agent_heartbeats (
            id          SERIAL PRIMARY KEY,
            agent_name  VARCHAR(64)  NOT NULL,
            outcome     VARCHAR(16)  NOT NULL,
            metadata    JSONB,
            recorded_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
        )
    """))

    op.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_heartbeats_agent
            ON agent_heartbeats (agent_name, recorded_at DESC)
    """))


def downgrade() -> None:
    op.execute(sa.text("DROP TABLE IF EXISTS agent_heartbeats"))
    op.execute(sa.text("DROP TABLE IF EXISTS alert_audit_log"))
    op.execute(sa.text("DROP TABLE IF EXISTS alerts"))
