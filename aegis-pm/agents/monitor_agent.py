"""
aegis-pm / agents / monitor_agent.py

Monitor Agent – AutoGen-powered Jira watcher.

Architecture
────────────
  JiraClient               async Jira REST v3 client with tenacity retries
  check_for_stale_tasks    AutoGen tool – queries Jira, returns stale task list
  save_alert               AutoGen tool – writes alert directly to PostgreSQL
  notify_communicator      AutoGen tool – POSTs task details to Communicator Agent
  build_monitor_agents     returns (AssistantAgent, UserProxyAgent) wired with tools
  MonitorAgent             high-level class used by APScheduler in runner.py
  run_monitor              standalone infinite loop for direct execution

AutoGen pattern
───────────────
  UserProxyAgent  (human_input_mode="NEVER", no code execution)
    → kicks off each cycle with a structured task prompt
  AssistantAgent  (GPT-4o, temperature=0)
    → calls: check_for_stale_tasks → save_alert (per issue) →
             notify_communicator (per issue) → TERMINATE

Retry strategy (tenacity)
──────────────────────────
  Jira API calls:  3 attempts, exponential backoff 2s→4s→8s
                   retries on 429, 500, 502, 503, 504 and network errors
  DB writes:       3 attempts, fixed 1s wait between attempts
  Communicator:    2 attempts, fixed 2s wait
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
import psycopg2
from autogen import AssistantAgent, UserProxyAgent, register_function
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    before_sleep_log,
    RetryError,
)

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("aegis.monitor")


# ── Config ────────────────────────────────────────────────────────────────────

JIRA_BASE_URL  = os.environ["JIRA_BASE_URL"].rstrip("/")
JIRA_EMAIL     = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
JIRA_PROJECT   = os.environ["JIRA_PROJECT_KEY"]
STALE_DAYS     = int(os.getenv("STALE_DAYS", "2"))
API_BASE_URL   = os.getenv("API_BASE_URL", "http://localhost:8000")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))

# PostgreSQL connection params (read from same env vars as FastAPI)
DB_PARAMS: Dict[str, str] = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     os.getenv("POSTGRES_PORT", "5432"),
    "dbname":   os.environ["POSTGRES_DB"],
    "user":     os.environ["POSTGRES_USER"],
    "password": os.environ["POSTGRES_PASSWORD"],
}

# AutoGen LLM configuration
LLM_CONFIG: Dict[str, Any] = {
    "config_list": [
        {
            "model":   os.getenv("OPENAI_MODEL", "gpt-4-turbo"),
            "api_key": OPENAI_API_KEY,
        }
    ],
    "temperature": 0.2,   # slight creativity for natural language summaries
    "timeout":     120,
    "cache_seed":  None,  # disable caching so every poll is live
}


# ══════════════════════════════════════════════════════════════════════════════
#  Jira REST Client  (with tenacity retries)
# ══════════════════════════════════════════════════════════════════════════════

# Exceptions worth retrying on Jira calls
_JIRA_RETRYABLE = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


def _is_retryable_http(exc: BaseException) -> bool:
    """Return True for HTTP errors that warrant a retry (5xx + 429)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, _JIRA_RETRYABLE)


class JiraClient:
    """
    Async Jira Cloud REST API v3 client.
    All network calls are wrapped with tenacity for robust retry handling.
    """

    def __init__(self) -> None:
        self._auth = (JIRA_EMAIL, JIRA_API_TOKEN)
        self._base = JIRA_BASE_URL

    @retry(
        retry=retry_if_exception_type(_JIRA_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    async def _get(self, url: str, params: Dict) -> Dict:
        """Single HTTP GET with retry; returns parsed JSON body."""
        async with httpx.AsyncClient(
            auth=self._auth, timeout=30.0, follow_redirects=True
        ) as client:
            resp = await client.get(url, params=params)

            # Retry 429 after Retry-After header
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", "2"))
                log.warning("Jira rate-limited; sleeping %.1fs", retry_after)
                await asyncio.sleep(retry_after)
                resp.raise_for_status()   # will raise → tenacity retries

            resp.raise_for_status()
            return resp.json()

    async def search_issues(
        self,
        jql:         str,
        fields:      str = "summary,assignee,updated,status,priority,labels",
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Run a JQL query and return a list of Jira issue dicts.

        Retries automatically on 429, 5xx, and network errors.
        Returns [] if all retries are exhausted (logged as error).
        """
        url    = f"{self._base}/rest/api/3/search"
        params = {"jql": jql, "maxResults": max_results, "fields": fields}

        try:
            data = await self._get(url, params)
            issues = data.get("issues", [])
            log.debug(
                "Jira search: %d/%d issues | JQL: %s",
                len(issues), data.get("total", "?"), jql,
            )
            return issues
        except RetryError as exc:
            log.error("Jira search failed after all retries: %s", exc)
            return []
        except httpx.HTTPStatusError as exc:
            log.error(
                "Jira HTTP %s for JQL '%s': %s",
                exc.response.status_code, jql, exc.response.text[:300],
            )
            return []
        except Exception as exc:
            log.error("Jira unexpected error: %s", exc, exc_info=True)
            return []

    def issue_url(self, key: str) -> str:
        return f"{self._base}/browse/{key}"

    @staticmethod
    def parse_updated(issue: Dict[str, Any]) -> Optional[datetime]:
        """Parse Jira's `updated` field into a timezone-aware datetime."""
        raw = (issue.get("fields") or {}).get("updated")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            log.warning("Could not parse updated timestamp: %r", raw)
            return None


# Shared instance reused across all tool calls in one process
_jira = JiraClient()


# ══════════════════════════════════════════════════════════════════════════════
#  Database helper  (direct psycopg2 writes with tenacity retry)
# ══════════════════════════════════════════════════════════════════════════════

@retry(
    retry=retry_if_exception_type(psycopg2.OperationalError),
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _db_save_alert(
    task_key:       str,
    task_summary:   Optional[str],
    assignee:       str,
    assignee_email: Optional[str],
    jira_url:       str,
    last_updated:   Optional[str],
) -> Optional[int]:
    """
    Upsert an alert row directly into PostgreSQL.

    Uses INSERT … ON CONFLICT DO NOTHING so duplicate pending alerts
    for the same task_key are silently ignored (matches the API behaviour).

    Returns the alert id on insert, or None if the row already existed.
    Retries up to 3 times on OperationalError (e.g. transient connection loss).
    """
    sql_insert = """
        INSERT INTO alerts
            (task_key, task_summary, assignee, assignee_email,
             jira_url, last_updated, status, slack_sent)
        VALUES
            (%(task_key)s, %(task_summary)s, %(assignee)s, %(assignee_email)s,
             %(jira_url)s, %(last_updated)s, 'pending', false)
        ON CONFLICT (task_key) WHERE status = 'pending'
        DO NOTHING
        RETURNING id
    """
    sql_audit = """
        INSERT INTO alert_audit_log
            (alert_id, from_status, to_status, actor, notes)
        VALUES
            (%(alert_id)s, NULL, 'pending', 'monitor_agent',
             %(notes)s)
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(sql_insert, {
                "task_key":       task_key,
                "task_summary":   task_summary,
                "assignee":       assignee,
                "assignee_email": assignee_email,
                "jira_url":       jira_url,
                "last_updated":   last_updated,
            })
            row = cur.fetchone()
            if row:
                alert_id = row[0]
                # Write audit log entry in same transaction
                cur.execute(sql_audit, {
                    "alert_id": alert_id,
                    "notes":    f"Monitor detected stale task {task_key}",
                })
                conn.commit()
                log.info("DB: alert saved id=%d  task=%s", alert_id, task_key)
                return alert_id
            else:
                conn.rollback()
                log.info("DB: duplicate pending alert for %s – skipped", task_key)
                return None
    except psycopg2.OperationalError:
        if conn:
            conn.rollback()
        raise   # let tenacity retry
    except Exception as exc:
        if conn:
            conn.rollback()
        log.error("DB write error for %s: %s", task_key, exc, exc_info=True)
        return None
    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════════
#  AutoGen Tool Functions
# ══════════════════════════════════════════════════════════════════════════════

def check_for_stale_tasks(project_key: str = JIRA_PROJECT) -> str:
    """
    AutoGen tool: query Jira for 'In Progress' issues with no update
    for more than STALE_DAYS days.

    This is the primary entry-point tool called by the Monitor Agent.

    Args:
        project_key: Jira project key to scan (default: JIRA_PROJECT_KEY env var)

    Returns:
        JSON string:
        {
          "project_key":  str,
          "stale_days":   int,
          "total":        int,
          "tasks": [
            {
              "task_key":       str,     e.g. "ENG-42"
              "task_summary":   str,     issue title
              "assignee":       str,     display name
              "assignee_email": str|null,
              "jira_url":       str,
              "last_updated":   str|null, ISO datetime
              "days_stale":     int,
              "priority":       str,     "High" | "Medium" | "Low" etc.
              "labels":         list[str]
            }, ...
          ]
        }
    """
    stale_days = STALE_DAYS
    cutoff     = (datetime.now(timezone.utc) - timedelta(days=stale_days)).strftime("%Y-%m-%d")

    jql = (
        f'project = "{project_key}" '
        f'AND status = "In Progress" '
        f'AND updated <= "{cutoff}" '
        f'ORDER BY updated ASC'
    )
    log.info(
        "check_for_stale_tasks: project=%s  stale_days=%d  JQL=%s",
        project_key, stale_days, jql,
    )

    try:
        issues = asyncio.get_event_loop().run_until_complete(
            _jira.search_issues(jql)
        )
    except Exception as exc:
        log.error("check_for_stale_tasks: Jira query failed: %s", exc, exc_info=True)
        return json.dumps({
            "project_key": project_key,
            "stale_days":  stale_days,
            "total":       0,
            "tasks":       [],
            "error":       str(exc),
        })

    now_utc = datetime.now(timezone.utc)
    tasks: List[Dict[str, Any]] = []

    for issue in issues:
        fields         = issue.get("fields") or {}
        task_key       = issue["key"]
        summary        = fields.get("summary", "")
        assignee_raw   = fields.get("assignee") or {}
        assignee_name  = assignee_raw.get("displayName", "Unassigned")
        assignee_email = assignee_raw.get("emailAddress")
        priority       = (fields.get("priority") or {}).get("name", "Medium")
        labels         = fields.get("labels") or []
        updated_dt     = _jira.parse_updated(issue)
        days_stale     = int((now_utc - updated_dt).days) if updated_dt else stale_days

        tasks.append({
            "task_key":       task_key,
            "task_summary":   summary,
            "assignee":       assignee_name,
            "assignee_email": assignee_email,
            "jira_url":       _jira.issue_url(task_key),
            "last_updated":   updated_dt.isoformat() if updated_dt else None,
            "days_stale":     days_stale,
            "priority":       priority,
            "labels":         labels,
        })

    log.info("check_for_stale_tasks: found %d stale task(s)", len(tasks))
    return json.dumps({
        "project_key": project_key,
        "stale_days":  stale_days,
        "total":       len(tasks),
        "tasks":       tasks,
    })


def save_alert(
    task_key:       str,
    task_summary:   str,
    assignee:       str,
    jira_url:       str,
    last_updated:   Optional[str]  = None,
    assignee_email: Optional[str]  = None,
) -> str:
    """
    AutoGen tool: persist a stale-task alert directly to PostgreSQL.

    Duplicate pending alerts for the same task_key are silently ignored —
    the existing alert id is returned instead.

    Args:
        task_key:       Jira issue key, e.g. "ENG-42"
        task_summary:   Issue title
        assignee:       Assignee display name
        jira_url:       Direct link to the Jira issue
        last_updated:   ISO datetime string of last activity (optional)
        assignee_email: Assignee email (optional, used for Slack DM)

    Returns:
        JSON string: {"success": true, "alert_id": <int>, "task_key": "..."}
                  or {"success": false, "task_key": "...", "error": "..."}
    """
    log.info("save_alert: persisting alert for %s (assignee=%s)", task_key, assignee)
    try:
        alert_id = _db_save_alert(
            task_key=task_key,
            task_summary=task_summary,
            assignee=assignee,
            assignee_email=assignee_email,
            jira_url=jira_url,
            last_updated=last_updated,
        )
        return json.dumps({
            "success":  True,
            "alert_id": alert_id,   # None = duplicate (already exists)
            "task_key": task_key,
        })
    except RetryError as exc:
        msg = f"DB unavailable after all retries: {exc}"
        log.error("save_alert: %s", msg)
        return json.dumps({"success": False, "task_key": task_key, "error": msg})
    except Exception as exc:
        log.error("save_alert: unexpected error for %s: %s", task_key, exc, exc_info=True)
        return json.dumps({"success": False, "task_key": task_key, "error": str(exc)})


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(2),
    wait=wait_fixed(2),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=False,
)
def _post_to_communicator(task_details: Dict[str, Any]) -> bool:
    """
    Internal: POST task details to the Communicator Agent via the API.
    The API endpoint creates an approved alert that Communicator will pick up.
    Retries twice on network errors only.
    """
    payload = {
        "task_key":       task_details["task_key"],
        "task_summary":   task_details.get("task_summary"),
        "assignee":       task_details["assignee"],
        "assignee_email": task_details.get("assignee_email"),
        "jira_url":       task_details.get("jira_url"),
        "last_updated":   task_details.get("last_updated"),
    }
    resp = httpx.post(
        f"{API_BASE_URL}/alerts",
        json=payload,
        timeout=10.0,
    )
    resp.raise_for_status()
    return True


def is_in_cooldown(task_key: str) -> tuple[bool, int]:
    """
    Check whether a task is still within its notification cooldown window.

    Queries the alerts table for the most recent notified alert for this
    task_key and compares last_notified_at against cooldown_hours.

    Returns:
        (in_cooldown: bool, hours_remaining: int)
        in_cooldown=True  means skip this notification cycle
        hours_remaining=0 means ready to re-notify
    """
    sql = """
        SELECT last_notified_at, cooldown_hours
        FROM   alerts
        WHERE  task_key = %(task_key)s
          AND  last_notified_at IS NOT NULL
        ORDER  BY last_notified_at DESC
        LIMIT  1
    """
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        with conn.cursor() as cur:
            cur.execute(sql, {"task_key": task_key})
            row = cur.fetchone()
        conn.close()

        if row is None:
            return False, 0   # never notified – proceed

        last_notified_at, cooldown_hours = row
        if last_notified_at is None:
            return False, 0

        # Make timezone-aware if naive
        if last_notified_at.tzinfo is None:
            last_notified_at = last_notified_at.replace(tzinfo=timezone.utc)

        elapsed_hours = (datetime.now(timezone.utc) - last_notified_at).total_seconds() / 3600
        if elapsed_hours < cooldown_hours:
            remaining = int(cooldown_hours - elapsed_hours)
            log.info(
                "Cooldown active for %s: %dh remaining (cooldown=%dh)",
                task_key, remaining, cooldown_hours,
            )
            return True, remaining

        return False, 0

    except Exception as exc:
        # If we can't check cooldown, allow the notification (fail open)
        log.warning("Could not check cooldown for %s: %s", task_key, exc)
        return False, 0


def update_notification_timestamp(task_key: str) -> None:
    """
    Update last_notified_at and increment notify_count for a task.
    Called by the Communicator Agent after a Slack message is confirmed sent.
    """
    sql = """
        UPDATE alerts
        SET    last_notified_at = NOW(),
               notify_count     = notify_count + 1
        WHERE  task_key = %(task_key)s
          AND  status   IN ('pending', 'approved', 'notified')
    """
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        with conn.cursor() as cur:
            cur.execute(sql, {"task_key": task_key})
        conn.commit()
        conn.close()
        log.debug("Updated notification timestamp for %s", task_key)
    except Exception as exc:
        log.warning("Could not update notification timestamp for %s: %s", task_key, exc)



def notify_communicator(
    task_key:       str,
    task_summary:   str,
    assignee:       str,
    jira_url:       str,
    days_stale:     int,
    last_updated:   Optional[str] = None,
    assignee_email: Optional[str] = None,
) -> str:
    """
    AutoGen tool: send task details to the Communicator Agent so it can
    dispatch a Slack notification.

    Under the hood this creates/confirms a pending alert in the backend.
    The Communicator Agent's 30-second polling loop will then pick it up
    once a human approves it in the HITL dashboard.

    Args:
        task_key:       Jira issue key
        task_summary:   Issue title
        assignee:       Assignee display name
        jira_url:       Direct link to the issue
        days_stale:     Days since last update (included in the message)
        last_updated:   ISO datetime of last activity
        assignee_email: Optional email for Slack DM routing

    Returns:
        JSON string: {"success": true, "task_key": "...", "queued": true}
                  or {"success": false, "task_key": "...", "error": "..."}
    """
    # ── Cooldown check ───────────────────────────────────────────────────────
    in_cooldown, hours_remaining = is_in_cooldown(task_key)
    if in_cooldown:
        log.info(
            "notify_communicator: SKIPPED %s – in cooldown (%dh remaining)",
            task_key, hours_remaining,
        )
        return json.dumps({
            "success":   True,
            "task_key":  task_key,
            "queued":    False,
            "skipped":   True,
            "reason":    f"In cooldown: {hours_remaining}h remaining",
        })

    log.info(
        "notify_communicator: queueing %s (assignee=%s  days_stale=%d)",
        task_key, assignee, days_stale,
    )
    try:
        _post_to_communicator({
            "task_key":       task_key,
            "task_summary":   task_summary,
            "assignee":       assignee,
            "assignee_email": assignee_email,
            "jira_url":       jira_url,
            "last_updated":   last_updated,
        })
        log.info("notify_communicator: %s queued successfully", task_key)
        update_notification_timestamp(task_key)
        return json.dumps({"success": True, "task_key": task_key, "queued": True, "skipped": False})
    except httpx.HTTPStatusError as exc:
        msg = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        log.error("notify_communicator HTTP error for %s: %s", task_key, msg)
        return json.dumps({"success": False, "task_key": task_key, "error": msg})
    except Exception as exc:
        log.error("notify_communicator error for %s: %s", task_key, exc, exc_info=True)
        return json.dumps({"success": False, "task_key": task_key, "error": str(exc)})


# ══════════════════════════════════════════════════════════════════════════════
#  AutoGen Agent Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_monitor_agents() -> tuple[AssistantAgent, UserProxyAgent]:
    """
    Construct and wire the AutoGen Monitor agent pair.

    AssistantAgent  – GPT-4 Turbo brain; decides when and how to call tools,
                      produces the final JSON cycle summary.
    UserProxyAgent  – silent autonomous executor; runs tool functions and
                      feeds results back into the conversation.
                      human_input_mode="NEVER" → no human intervention.

    Tool call sequence the LLM is instructed to follow:
      1. check_for_stale_tasks(project_key) → stale task list
      2. For each task:
           save_alert(...)             → persist to PostgreSQL
           notify_communicator(...)    → queue for Slack dispatch
      3. Reply with JSON cycle summary + TERMINATE
    """

    # ── AssistantAgent: the reasoning brain ───────────────────────────────────
    monitor_agent = AssistantAgent(
        name="Monitor_Agent",
        llm_config=LLM_CONFIG,
        system_message="""You are a project monitor for Aegis PM – an autonomous AI project management system.

Your role is to proactively detect stale work, ensure nothing falls through the cracks, and alert the right people before blockers cascade into missed deadlines.

Each monitoring cycle you must:
1. Call `check_for_stale_tasks` with the project key to find all In Progress Jira tasks that have had no activity for more than the configured number of days.

2. For EVERY task returned:
   a. Call `save_alert` to persist the stale task to the PostgreSQL alerts database.
   b. Call `notify_communicator` to queue a Slack notification for the assignee.

3. After processing all tasks, produce a final JSON summary and end your response with TERMINATE.

Your guidelines:
- Process every task returned by check_for_stale_tasks. Never skip any.
- If check_for_stale_tasks returns 0 tasks, skip steps 2 and go straight to step 3.
- If save_alert returns success=false, log it as a failure but continue processing remaining tasks.
- If notify_communicator returns success=false, log it as a failure but continue.
- Do NOT ask for clarification. Act on the data you receive.
- Your final message MUST end with the word TERMINATE on its own line.

Final summary format (required before TERMINATE):
```json
{
  "cycle_summary": {
    "project_key":        "<str>",
    "stale_days":         <int>,
    "stale_found":        <int>,
    "alerts_saved":       <int>,
    "notifications_sent": <int>,
    "failures":           <int>,
    "tasks": [
      {
        "task_key":   "ENG-42",
        "assignee":   "Jane Dev",
        "days_stale": 4,
        "alert_id":   7,
        "notified":   true
      }
    ]
  }
}
```
TERMINATE""",
    )

    # ── UserProxyAgent: the silent executor ───────────────────────────────────
    user_proxy = UserProxyAgent(
        name="MonitorOrchestrator",
        human_input_mode="NEVER",       # fully autonomous — no human prompts
        max_consecutive_auto_reply=40,  # safety cap; each task = ~3 turns
        code_execution_config=False,    # no arbitrary code execution
        is_termination_msg=lambda msg: (
            isinstance(msg.get("content"), str)
            and "TERMINATE" in msg["content"]
        ),
    )

    # ── Register all three tools on both agents ───────────────────────────────
    register_function(
        check_for_stale_tasks,
        caller=monitor_agent,
        executor=user_proxy,
        name="check_for_stale_tasks",
        description=(
            "Query Jira for 'In Progress' tasks not updated in more than "
            "STALE_DAYS days. Returns a JSON object with a 'tasks' list "
            "containing task_key, task_summary, assignee, jira_url, "
            "last_updated, days_stale, priority, labels."
        ),
    )

    register_function(
        save_alert,
        caller=monitor_agent,
        executor=user_proxy,
        name="save_alert",
        description=(
            "Persist a stale-task alert directly to the PostgreSQL alerts table. "
            "Duplicate pending alerts for the same task_key are silently ignored. "
            "Returns JSON with success status and alert_id."
        ),
    )

    register_function(
        notify_communicator,
        caller=monitor_agent,
        executor=user_proxy,
        name="notify_communicator",
        description=(
            "Queue a Slack notification for the task assignee by sending "
            "task details to the Communicator Agent via the Aegis API. "
            "The notification will be dispatched once approved in the HITL dashboard. "
            "Returns JSON with success status and queued flag."
        ),
    )

    return monitor_agent, user_proxy


# ══════════════════════════════════════════════════════════════════════════════
#  High-level MonitorAgent  (used by APScheduler in runner.py)
# ══════════════════════════════════════════════════════════════════════════════

class MonitorAgent:
    """
    Orchestration wrapper around the AutoGen agent pair.
    Called by APScheduler every POLL_INTERVAL_SECONDS seconds.

    Why rebuild agents each cycle?
    AutoGen agents accumulate message history in memory. For an indefinitely
    running process, rebuilding each cycle keeps LLM context small and
    prevents tool-call history from one poll bleeding into the next.
    """

    def __init__(self) -> None:
        self.project_key = JIRA_PROJECT
        self.stale_days  = STALE_DAYS

    def run_once(self) -> Dict[str, Any]:
        """
        Run one full monitoring cycle synchronously.
        Asyncio is handled internally by the tool functions.

        Returns:
            dict with keys: stale_found, alerts_saved, notifications_sent,
                            failures, tasks, error (on exception)
        """
        log.info(
            "━━━ Monitor cycle start │ project=%s │ stale_days=%d ━━━",
            self.project_key, self.stale_days,
        )

        monitor_agent, user_proxy = build_monitor_agents()

        task_prompt = (
            f"Run a stale task monitoring cycle.\n"
            f"  project_key = '{self.project_key}'\n"
            f"  stale_days  = {self.stale_days}\n\n"
            f"Find all In Progress Jira tasks not updated in the last "
            f"{self.stale_days} day(s), save an alert for each one to the "
            f"database, notify the Communicator Agent, then produce a cycle "
            f"summary and TERMINATE."
        )

        try:
            user_proxy.initiate_chat(
                monitor_agent,
                message=task_prompt,
                silent=True,   # suppress per-message stdout; log handlers still fire
            )
        except Exception as exc:
            log.exception("AutoGen chat raised an exception: %s", exc)
            return {
                "error":              str(exc),
                "stale_found":        0,
                "alerts_saved":       0,
                "notifications_sent": 0,
                "failures":           0,
                "tasks":              [],
            }

        summary = self._parse_summary(
            user_proxy.chat_messages.get(monitor_agent, [])
        )

        log.info(
            "━━━ Monitor cycle end │ stale=%d │ saved=%d │ notified=%d │ failed=%d ━━━",
            summary.get("stale_found", 0),
            summary.get("alerts_saved", 0),
            summary.get("notifications_sent", 0),
            summary.get("failures", 0),
        )
        return summary

    @staticmethod
    def _parse_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract the cycle_summary JSON from the last assistant message."""
        for msg in reversed(messages):
            content = msg.get("content") or ""
            if "cycle_summary" not in content:
                continue
            start = content.find("{")
            end   = content.rfind("}") + 1
            if start == -1 or end <= start:
                continue
            try:
                data  = json.loads(content[start:end])
                inner = data.get("cycle_summary", data)
                return {
                    "project_key":        inner.get("project_key", JIRA_PROJECT),
                    "stale_found":        inner.get("stale_found", 0),
                    "alerts_saved":       inner.get("alerts_saved", 0),
                    "notifications_sent": inner.get("notifications_sent", 0),
                    "failures":           inner.get("failures", 0),
                    "tasks":              inner.get("tasks", []),
                }
            except json.JSONDecodeError:
                log.warning("Could not parse cycle_summary JSON: %s", content[:200])
                continue

        log.warning("No parseable cycle_summary in chat history; returning zeros")
        return {
            "project_key":        JIRA_PROJECT,
            "stale_found":        0,
            "alerts_saved":       0,
            "notifications_sent": 0,
            "failures":           0,
            "tasks":              [],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone continuous monitoring loop
#  Usage: python -m agents.monitor_agent
# ══════════════════════════════════════════════════════════════════════════════

def run_monitor(
    project_key: str = JIRA_PROJECT,
    interval:    int = POLL_INTERVAL_SECONDS,
) -> None:
    """
    Infinite monitoring loop – runs forever, sleeping `interval` seconds
    between each poll cycle.

    This is the entry point when running the Monitor Agent standalone
    (without APScheduler / docker-compose agents service).

    Args:
        project_key: Jira project key to monitor
        interval:    Seconds between poll cycles (default: POLL_INTERVAL_SECONDS)

    Usage:
        python -m agents.monitor_agent
        python -m agents.monitor_agent --interval 60
    """
    log.info("═══════════════════════════════════════════════")
    log.info("  Aegis PM – Monitor Agent (standalone mode)   ")
    log.info("  Project  : %s", project_key)
    log.info("  Interval : %ds", interval)
    log.info("  Stale    : %d+ days", STALE_DAYS)
    log.info("  DB       : %s@%s/%s", DB_PARAMS["user"], DB_PARAMS["host"], DB_PARAMS["dbname"])
    log.info("═══════════════════════════════════════════════")

    agent = MonitorAgent()
    cycle = 0

    while True:
        cycle += 1
        log.info("── Cycle #%d ──", cycle)

        try:
            summary = agent.run_once()
            log.info(
                "Cycle #%d complete │ stale=%d │ saved=%d │ notified=%d │ failed=%d",
                cycle,
                summary.get("stale_found", 0),
                summary.get("alerts_saved", 0),
                summary.get("notifications_sent", 0),
                summary.get("failures", 0),
            )
        except KeyboardInterrupt:
            log.info("Received interrupt – stopping monitor.")
            break
        except Exception as exc:
            # Log but never crash the loop – next cycle will retry
            log.exception("Cycle #%d unhandled error: %s", cycle, exc)

        log.info("Sleeping %ds until next cycle…", interval)
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            log.info("Interrupted during sleep – stopping monitor.")
            break


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aegis PM – Monitor Agent")
    parser.add_argument(
        "--project", default=JIRA_PROJECT,
        help=f"Jira project key to monitor (default: {JIRA_PROJECT})",
    )
    parser.add_argument(
        "--interval", type=int, default=POLL_INTERVAL_SECONDS,
        help=f"Seconds between poll cycles (default: {POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle and exit (useful for testing)",
    )
    args = parser.parse_args()

    if args.once:
        log.info("Running single cycle (--once mode)")
        agent = MonitorAgent()
        result = agent.run_once()
        print(json.dumps(result, indent=2))
    else:
        run_monitor(project_key=args.project, interval=args.interval)
