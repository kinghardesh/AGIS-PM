"""
aegis-pm / agents / communicator_agent.py

Communicator Agent – AutoGen-powered Slack notifier.

Architecture
────────────
  SlackClient             – Slack Incoming Webhook sender with retry + Block Kit builder
  fetch_approved_alerts() – AutoGen tool: GET /alerts?status=approved from FastAPI
  send_slack_notification()– AutoGen tool: POST rich Block Kit message to Slack
  mark_alert_notified()   – AutoGen tool: POST /alerts/{id}/notified to FastAPI
  build_communicator_agents() – returns (AssistantAgent, UserProxyAgent) wired with tools
  CommunicatorAgent       – high-level class called by APScheduler in runner.py

AutoGen pattern (mirrors MonitorAgent)
────────────────────────────────────────
  UserProxyAgent  (NEVER, no code exec)
      → "Find all approved alerts and send Slack messages for each one"
  AssistantAgent  (GPT-4o)
      → fetch_approved_alerts → for each: send_slack_notification → mark_alert_notified
      → returns JSON cycle summary then TERMINATE
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from autogen import AssistantAgent, UserProxyAgent, register_function
from dotenv import load_dotenv

# Import cooldown updater – called after confirmed Slack send so cooldown resets correctly
try:
    from agents.monitor_agent import update_notification_timestamp as _update_cooldown
except ImportError:
    def _update_cooldown(task_key: str) -> None:  # type: ignore
        pass

load_dotenv()
log = logging.getLogger("aegis.communicator")


# ── Env / Config ──────────────────────────────────────────────────────────────

SLACK_WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]
API_BASE_URL      = os.getenv("API_BASE_URL", "http://localhost:8000")
STALE_DAYS        = int(os.getenv("STALE_DAYS", "2"))
OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]

AUTOGEN_LLM_CONFIG: Dict[str, Any] = {
    "config_list": [
        {
            "model": "gpt-4o",
            "api_key": OPENAI_API_KEY,
        }
    ],
    "temperature": 0,
    "timeout": 120,
    "cache_seed": None,
}

# Retry settings for Slack webhook calls
_SLACK_MAX_RETRIES  = 3
_SLACK_RETRY_DELAY  = 2.0   # seconds between retries (doubles on each attempt)


# ══════════════════════════════════════════════════════════════════════════════
#  Slack Client
# ══════════════════════════════════════════════════════════════════════════════

class SlackClient:
    """
    Sends Slack messages via Incoming Webhook.

    Features:
    - Exponential-backoff retries (up to _SLACK_MAX_RETRIES)
    - Rich Block Kit message builder for stale-task alerts
    - Compact fallback text for notification previews
    """

    def __init__(self, webhook_url: str = SLACK_WEBHOOK_URL) -> None:
        self._webhook = webhook_url

    # ── Sending ───────────────────────────────────────────────────────────────

    async def send(self, payload: Dict[str, Any]) -> bool:
        """
        POST a Block Kit payload to the Slack webhook.
        Retries up to _SLACK_MAX_RETRIES times with exponential backoff.
        Returns True on success, False if all retries are exhausted.
        """
        delay = _SLACK_RETRY_DELAY
        last_error: Optional[Exception] = None

        for attempt in range(1, _SLACK_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.post(self._webhook, json=payload)
                    if resp.status_code == 200:
                        log.info("Slack message sent (attempt %d)", attempt)
                        return True
                    # 429 = rate limited → always retry
                    if resp.status_code == 429:
                        retry_after = float(resp.headers.get("Retry-After", delay))
                        log.warning("Slack rate-limited; retrying in %.1fs", retry_after)
                        await asyncio.sleep(retry_after)
                        continue
                    # 4xx (not 429) = bad payload, no point retrying
                    resp.raise_for_status()
            except httpx.TimeoutException as exc:
                last_error = exc
                log.warning("Slack timeout on attempt %d/%d", attempt, _SLACK_MAX_RETRIES)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                log.error(
                    "Slack HTTP %s on attempt %d: %s",
                    exc.response.status_code, attempt, exc.response.text[:200],
                )
                if 400 <= exc.response.status_code < 500:
                    break   # 4xx: don't retry
            except Exception as exc:
                last_error = exc
                log.error("Slack unexpected error on attempt %d: %s", attempt, exc)

            if attempt < _SLACK_MAX_RETRIES:
                await asyncio.sleep(delay)
                delay *= 2  # exponential backoff

        log.error("Slack send failed after %d attempt(s). Last error: %s",
                  _SLACK_MAX_RETRIES, last_error)
        return False

    # ── Message builder ───────────────────────────────────────────────────────

    def build_stale_task_message(
        self,
        alert: Dict[str, Any],
        days_stale: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build a rich Slack Block Kit message for a stale-task alert.

        Layout:
        ┌─────────────────────────────────────────┐
        │  ⚠️  Stale task alert – Aegis PM         │  ← header
        ├─────────────────────────────────────────┤
        │  Task: ENG-42  │  Assignee: Jane Dev     │  ← 2-col fields
        │  Summary: ...  │  Last updated: ...      │
        ├─────────────────────────────────────────┤
        │  Hi Jane 👋 – ENG-42 hasn't been ...    │  ← body text
        │  Please add an update or flag blockers.  │
        │  👉 Open in Jira                         │
        ├─────────────────────────────────────────┤
        │  [ View in Jira ↗ ]                     │  ← action button
        └─────────────────────────────────────────┘
        """
        task_key     = alert["task_key"]
        summary      = (alert.get("task_summary") or "No summary provided").strip()
        assignee     = alert.get("assignee") or "Unassigned"
        jira_url     = alert.get("jira_url") or "#"
        last_updated = alert.get("last_updated") or "Unknown"
        alert_id     = alert.get("id", "?")

        # Human-readable date
        if isinstance(last_updated, str) and "T" in last_updated:
            last_updated = last_updated.split("T")[0]

        stale_text = (
            f"{days_stale} day(s)" if days_stale else f"{STALE_DAYS}+ day(s)"
        )

        return {
            # Fallback text for notifications / accessibility
            "text": f"⚠️ Stale task {task_key} assigned to {assignee} – no update in {stale_text}",
            "blocks": [
                # ── Header ────────────────────────────────────────────────────
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "⚠️  Stale task alert  –  Aegis PM",
                        "emoji": True,
                    },
                },
                # ── Task details (2-column) ───────────────────────────────────
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Task*\n<{jira_url}|{task_key}>",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Assignee*\n{assignee}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Summary*\n{summary[:120]}{'…' if len(summary) > 120 else ''}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Last updated*\n{last_updated}",
                        },
                    ],
                },
                {"type": "divider"},
                # ── Body message ──────────────────────────────────────────────
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"Hi *{assignee}* 👋\n\n"
                            f"Your task *<{jira_url}|{task_key}>* has had no updates "
                            f"for *{stale_text}*.\n\n"
                            f"Could you please do one of the following?\n"
                            f"• Add a status comment on the Jira ticket\n"
                            f"• Update the ticket's progress\n"
                            f"• Flag any blockers so the team can help\n\n"
                            f"_Alert #{alert_id} raised by Aegis PM_"
                        ),
                    },
                },
                # ── Action button ─────────────────────────────────────────────
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Open in Jira ↗",
                                "emoji": False,
                            },
                            "url": jira_url,
                            "style": "primary",
                            "action_id": f"open_jira_{task_key}",
                        },
                    ],
                },
                # ── Context footer ────────────────────────────────────────────
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"Aegis PM  •  Automated stale-task monitor  •  "
                                f"Alert #{alert_id}"
                            ),
                        }
                    ],
                },
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Tool Functions  (registered into AutoGen agents)
# ══════════════════════════════════════════════════════════════════════════════

_slack = SlackClient()


def fetch_approved_alerts() -> str:
    """
    AutoGen tool: fetch all alerts with status='approved' from the Aegis API.

    Returns:
        JSON string:
          {
            "total": <int>,
            "alerts": [
              {"id": int, "task_key": str, "task_summary": str,
               "assignee": str, "assignee_email": str|null,
               "jira_url": str, "last_updated": str|null,
               "slack_sent": bool, ...},
              ...
            ]
          }
        or {"error": "...", "total": 0, "alerts": []} on failure.
    """
    async def _get() -> List[Dict]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{API_BASE_URL}/alerts",
                params={"status": "approved"},
            )
            resp.raise_for_status()
            return resp.json()

    try:
        alerts = asyncio.get_event_loop().run_until_complete(_get())
        log.info("fetch_approved_alerts: %d alert(s) ready to notify", len(alerts))
        return json.dumps({"total": len(alerts), "alerts": alerts})
    except httpx.HTTPStatusError as exc:
        msg = f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
        log.error("fetch_approved_alerts HTTP error: %s", msg)
        return json.dumps({"error": msg, "total": 0, "alerts": []})
    except Exception as exc:
        log.error("fetch_approved_alerts error: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc), "total": 0, "alerts": []})


def send_slack_notification(
    alert_id: int,
    task_key: str,
    task_summary: str,
    assignee: str,
    jira_url: str,
    last_updated: Optional[str],
    days_stale: Optional[int] = None,
    assignee_email: Optional[str] = None,
) -> str:
    """
    AutoGen tool: build and send a Slack Block Kit message for a stale task.

    Args:
        alert_id:       The Aegis alert ID (for logging and footer)
        task_key:       Jira issue key, e.g. "ENG-42"
        task_summary:   Short title of the Jira issue
        assignee:       Display name of the assignee
        jira_url:       Direct link to the Jira issue
        last_updated:   ISO datetime of last activity (shown as date in message)
        days_stale:     How many days without update (shown in message body)
        assignee_email: Not used for webhook sends, reserved for DM future support

    Returns:
        JSON string: {"success": true, "alert_id": <int>}
                  or {"success": false, "alert_id": <int>, "error": "..."}
    """
    alert_dict = {
        "id":           alert_id,
        "task_key":     task_key,
        "task_summary": task_summary,
        "assignee":     assignee,
        "jira_url":     jira_url,
        "last_updated": last_updated,
    }

    payload = _slack.build_stale_task_message(alert_dict, days_stale=days_stale)

    try:
        success = asyncio.get_event_loop().run_until_complete(
            _slack.send(payload)
        )
        if success:
            log.info("Slack sent OK for alert %d (%s → %s)", alert_id, task_key, assignee)
            return json.dumps({"success": True, "alert_id": alert_id})
        else:
            return json.dumps({
                "success": False,
                "alert_id": alert_id,
                "error": f"Slack send failed after {_SLACK_MAX_RETRIES} attempts",
            })
    except Exception as exc:
        log.error("send_slack_notification error for alert %d: %s", alert_id, exc, exc_info=True)
        return json.dumps({"success": False, "alert_id": alert_id, "error": str(exc)})


def mark_alert_notified(alert_id: int) -> str:
    """
    AutoGen tool: mark an alert as 'notified' in the Aegis API.
    Call this ONLY after send_slack_notification returns success=true.

    Args:
        alert_id: The Aegis alert ID to mark as notified

    Returns:
        JSON string: {"success": true, "alert_id": <int>, "status": "notified"}
                  or {"success": false, "alert_id": <int>, "error": "..."}
    """
    async def _patch() -> Dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{API_BASE_URL}/alerts/{alert_id}/notified")
            resp.raise_for_status()
            return resp.json()

    try:
        updated = asyncio.get_event_loop().run_until_complete(_patch())
        log.info("Alert %d marked as notified (status=%s)", alert_id, updated.get("status"))
        # Update cooldown timestamp so monitor won't re-notify for NOTIFY_COOLDOWN_HOURS
        task_key = updated.get("task_key", "")
        if task_key:
            _update_cooldown(task_key)
        return json.dumps({
            "success": True,
            "alert_id": alert_id,
            "status": updated.get("status"),
        })
    except httpx.HTTPStatusError as exc:
        msg = f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
        log.error("mark_alert_notified HTTP error for alert %d: %s", alert_id, msg)
        return json.dumps({"success": False, "alert_id": alert_id, "error": msg})
    except Exception as exc:
        log.error("mark_alert_notified error for alert %d: %s", alert_id, exc, exc_info=True)
        return json.dumps({"success": False, "alert_id": alert_id, "error": str(exc)})


# ══════════════════════════════════════════════════════════════════════════════
#  AutoGen Agent Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_communicator_agents() -> tuple[AssistantAgent, UserProxyAgent]:
    """
    Build and return the AutoGen (AssistantAgent, UserProxyAgent) pair
    for the Communicator Agent, with all three tools registered.

    Tool call sequence the LLM follows per cycle:
      1. fetch_approved_alerts()           → get list of approved alerts
      2. send_slack_notification(...)      → for each alert
      3. mark_alert_notified(alert_id)     → only if step 2 succeeded
      repeat 2–3 for every alert
      4. reply with JSON summary + TERMINATE
    """

    # ── AssistantAgent ────────────────────────────────────────────────────────
    assistant = AssistantAgent(
        name="CommunicatorAssistant",
        llm_config=AUTOGEN_LLM_CONFIG,
        system_message="""You are the Communicator Agent for Aegis PM – an autonomous project management system.

Your job each cycle:
1. Call `fetch_approved_alerts` to get all alerts waiting to be sent.
2. For EACH alert in the result:
   a. Call `send_slack_notification` with the alert's details.
   b. If it returns success=true, call `mark_alert_notified` with the alert_id.
   c. If it returns success=false, record the failure but continue to the next alert.
3. After processing every alert, produce a final JSON summary and end with TERMINATE.

Rules:
- Never skip an alert. Process all of them.
- Never call mark_alert_notified unless send_slack_notification succeeded.
- If fetch_approved_alerts returns 0 alerts, immediately reply:
    {"cycle_summary": {"approved_found": 0, "notified": 0, "failed": 0, "alerts": []}}
    TERMINATE
- Do NOT ask clarifying questions. Act on the data you receive.
- Your final message MUST end with TERMINATE on its own line.

Final message format (required):
```json
{
  "cycle_summary": {
    "approved_found": <int>,
    "notified": <int>,
    "failed": <int>,
    "alerts": [
      {"alert_id": <int>, "task_key": "...", "assignee": "...", "slack_sent": true|false},
      ...
    ]
  }
}
```
TERMINATE""",
    )

    # ── UserProxyAgent ────────────────────────────────────────────────────────
    user_proxy = UserProxyAgent(
        name="CommunicatorOrchestrator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=40,   # allow for large batches
        code_execution_config=False,
        is_termination_msg=lambda msg: (
            isinstance(msg.get("content"), str)
            and "TERMINATE" in msg["content"]
        ),
    )

    # ── Register tools ────────────────────────────────────────────────────────
    register_function(
        fetch_approved_alerts,
        caller=assistant,
        executor=user_proxy,
        name="fetch_approved_alerts",
        description=(
            "Fetch all alerts with status='approved' from the Aegis PM backend. "
            "Returns a JSON list of alert objects ready to be notified."
        ),
    )

    register_function(
        send_slack_notification,
        caller=assistant,
        executor=user_proxy,
        name="send_slack_notification",
        description=(
            "Send a rich Slack Block Kit message to the team channel notifying "
            "the assignee of a stale Jira task. Returns success/failure JSON."
        ),
    )

    register_function(
        mark_alert_notified,
        caller=assistant,
        executor=user_proxy,
        name="mark_alert_notified",
        description=(
            "Mark an alert as 'notified' in the Aegis PM backend after a "
            "Slack message has been successfully sent. Only call on success."
        ),
    )

    return assistant, user_proxy


# ══════════════════════════════════════════════════════════════════════════════
#  High-level CommunicatorAgent  (called by runner.py via APScheduler)
# ══════════════════════════════════════════════════════════════════════════════

class CommunicatorAgent:
    """
    Thin orchestration wrapper around the AutoGen agent pair.
    Mirrors the MonitorAgent pattern exactly.

    APScheduler calls run_once() every 30 seconds.
    Each call:
      1. Builds a fresh agent pair (clean state)
      2. Initiates the AutoGen conversation with the task prompt
      3. Parses and returns the cycle summary dict
    """

    def run_once(self) -> Dict[str, Any]:
        """
        Run one full communicator cycle. Synchronous.

        Returns cycle summary:
          {
            "approved_found": int,
            "notified": int,
            "failed": int,
            "alerts": [...]
          }
        """
        log.info("━━━ Communicator cycle start ━━━")

        assistant, user_proxy = build_communicator_agents()

        task_prompt = (
            "Run a notification cycle.\n"
            "Fetch all approved alerts, send a Slack message for each one, "
            "mark each successful send as notified, then summarise and TERMINATE."
        )

        try:
            user_proxy.initiate_chat(
                assistant,
                message=task_prompt,
                silent=True,
            )
        except Exception as exc:
            log.exception("CommunicatorAgent AutoGen chat crashed: %s", exc)
            return {
                "error": str(exc),
                "approved_found": 0,
                "notified": 0,
                "failed": 0,
                "alerts": [],
            }

        summary = self._parse_summary(
            user_proxy.chat_messages.get(assistant, [])
        )

        if summary.get("notified", 0) or summary.get("failed", 0):
            log.info(
                "━━━ Communicator cycle end │ approved=%d │ notified=%d │ failed=%d ━━━",
                summary.get("approved_found", 0),
                summary.get("notified", 0),
                summary.get("failed", 0),
            )
        else:
            log.debug("━━━ Communicator cycle end │ no approved alerts ━━━")

        return summary

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse the cycle_summary JSON from the last assistant message."""
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
                    "approved_found": inner.get("approved_found", 0),
                    "notified":       inner.get("notified", 0),
                    "failed":         inner.get("failed", 0),
                    "alerts":         inner.get("alerts", []),
                }
            except json.JSONDecodeError:
                log.warning("Could not parse cycle_summary from: %s", content[:200])
                continue

        return {"approved_found": 0, "notified": 0, "failed": 0, "alerts": []}
