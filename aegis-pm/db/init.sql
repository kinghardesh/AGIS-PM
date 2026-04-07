-- ============================================================
--  Aegis PM – Database Schema
--  Runs automatically on first postgres container start
-- ============================================================

CREATE TABLE IF NOT EXISTS alerts (
    id            SERIAL PRIMARY KEY,
    task_key      VARCHAR(64)  NOT NULL,          -- e.g. "ENG-42"
    task_summary  TEXT,                            -- Jira issue title
    assignee      VARCHAR(255) NOT NULL,           -- Jira display name
    assignee_email VARCHAR(255),                   -- for Slack DM targeting
    jira_url      TEXT,                            -- direct link to issue
    last_updated  TIMESTAMPTZ,                     -- last activity on the issue
    detected_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    status        VARCHAR(32)  NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'approved', 'dismissed', 'notified')),
    slack_sent    BOOLEAN      NOT NULL DEFAULT FALSE,
    slack_ts      VARCHAR(64),                     -- Slack message timestamp (for threading)
    notes         TEXT,                            -- optional human notes on approve/dismiss
    -- Notification cooldown (prevents re-notifying within cooldown_hours)
    last_notified_at  TIMESTAMPTZ,                 -- when Slack was last sent for this task
    notify_count      INTEGER      NOT NULL DEFAULT 0,   -- total notifications sent
    cooldown_hours    INTEGER      NOT NULL DEFAULT 24   -- hours before re-notification allowed
);

-- Index for fast lookups by status (dashboard queries)
CREATE INDEX IF NOT EXISTS idx_alerts_status     ON alerts (status);
CREATE INDEX IF NOT EXISTS idx_alerts_task_key   ON alerts (task_key);
CREATE INDEX IF NOT EXISTS idx_alerts_detected   ON alerts (detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_assignee   ON alerts (assignee);

-- Prevent duplicate pending alerts for the same task
CREATE UNIQUE INDEX IF NOT EXISTS idx_alerts_pending_unique
    ON alerts (task_key)
    WHERE status = 'pending';

-- ============================================================
--  Audit log – every state transition is recorded here
-- ============================================================

CREATE TABLE IF NOT EXISTS alert_audit_log (
    id           SERIAL PRIMARY KEY,
    alert_id     INTEGER      NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    from_status  VARCHAR(32),                       -- NULL on initial creation
    to_status    VARCHAR(32)  NOT NULL,
    actor        VARCHAR(64)  NOT NULL DEFAULT 'system',  -- 'human' | 'monitor_agent' | 'communicator_agent'
    notes        TEXT,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_alert_id  ON alert_audit_log (alert_id);
CREATE INDEX IF NOT EXISTS idx_audit_created   ON alert_audit_log (created_at DESC);

-- ============================================================
--  Notification cooldown  (item 4)
--  Prevents the Monitor Agent from re-notifying the same task
--  assignee more than once per NOTIFY_COOLDOWN_HOURS hours.
-- ============================================================

-- Cooldown columns (last_notified_at, notify_count, cooldown_hours) are
-- declared in the CREATE TABLE above. The index below is created separately
-- since conditional indexes can't be in CREATE TABLE.
CREATE INDEX IF NOT EXISTS idx_alerts_last_notified
    ON alerts (last_notified_at DESC)
    WHERE last_notified_at IS NOT NULL;

-- ============================================================
--  Agent heartbeats  (item 3 – created by health/monitor.py
--  but also declared here so init.sql is the single source)
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_heartbeats (
    id          SERIAL PRIMARY KEY,
    agent_name  VARCHAR(64)  NOT NULL,
    outcome     VARCHAR(16)  NOT NULL,
    metadata    JSONB,
    recorded_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_heartbeats_agent
    ON agent_heartbeats (agent_name, recorded_at DESC);

-- ============================================================
--  Alembic migration tracking  (item 2)
-- ============================================================

CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
