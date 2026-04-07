"""Add cooldown columns and agent_heartbeats table

Revision ID: 001_baseline
Revises:
Create Date: 2025-01-01 00:00:00.000000

This is the baseline migration. It creates all tables that init.sql
also creates, using IF NOT EXISTS guards so it is safe to run against
a DB that was bootstrapped by init.sql directly.
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
    # ── alerts: add cooldown columns if missing ───────────────────────────────
    conn = op.get_bind()

    # Check which columns already exist (safe for repeated runs)
    existing = {
        row[0] for row in conn.execute(
            sa.text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'alerts'"
            )
        )
    }

    if "last_notified_at" not in existing:
        op.add_column("alerts", sa.Column(
            "last_notified_at", sa.DateTime(timezone=True), nullable=True
        ))

    if "notify_count" not in existing:
        op.add_column("alerts", sa.Column(
            "notify_count", sa.Integer(), nullable=False, server_default="0"
        ))

    if "cooldown_hours" not in existing:
        op.add_column("alerts", sa.Column(
            "cooldown_hours", sa.Integer(), nullable=False, server_default="24"
        ))

    # ── agent_heartbeats table ────────────────────────────────────────────────
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

    op.execute(sa.text("""
        CREATE INDEX IF NOT EXISTS idx_alerts_last_notified
            ON alerts (last_notified_at DESC)
            WHERE last_notified_at IS NOT NULL
    """))


def downgrade() -> None:
    # Remove cooldown columns
    op.drop_column("alerts", "cooldown_hours")
    op.drop_column("alerts", "notify_count")
    op.drop_column("alerts", "last_notified_at")

    op.execute(sa.text("DROP TABLE IF EXISTS agent_heartbeats"))
