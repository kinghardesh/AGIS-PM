"""
aegis-pm / agents / group_chat.py

AutoGen GroupChat orchestrator for Aegis PM.

Wires the Monitor and Communicator agents into a shared GroupChat so they can
collaborate, pass context between each other, and be supervised by a single
HITL admin agent — all in one conversation.

Architecture
────────────
  MonitorAssistant      – finds stale Jira tasks, registers alerts
  CommunicatorAssistant – sends Slack messages for approved alerts
  AegisAdmin            – UserProxyAgent; initiates tasks, approves decisions,
                          receives final summaries (human_input_mode configurable)
  AegisSupervisor       – AssistantAgent; orchestrates the flow between agents,
                          decides who acts next, synthesises the final report

GroupChat flow
──────────────
  1. AegisAdmin kicks off with a "run full cycle" message
  2. AegisSupervisor delegates to MonitorAssistant → polls Jira
  3. MonitorAssistant reports stale tasks found
  4. AegisSupervisor delegates to CommunicatorAssistant → sends Slack
  5. CommunicatorAssistant reports notifications sent
  6. AegisSupervisor produces final report → AegisAdmin
  7. AegisAdmin sees the report; conversation ends (TERMINATE)

Usage
─────
  # Run a full automated cycle (no human input)
  from agents.group_chat import run_full_cycle
  report = run_full_cycle()
  print(report)

  # Run with human approval gate (pauses before Slack send)
  from agents.group_chat import run_full_cycle
  report = run_full_cycle(require_human_approval=True)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from autogen import (
    AssistantAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    register_function,
)
from dotenv import load_dotenv

from agents.monitor_agent import (
    check_for_stale_tasks,
    save_alert,
    notify_communicator,
)
from agents.communicator_agent import (
    fetch_approved_alerts,
    send_slack_notification,
    mark_alert_notified,
)

load_dotenv()
log = logging.getLogger("aegis.groupchat")

# ── Config ────────────────────────────────────────────────────────────────────

JIRA_PROJECT   = os.environ["JIRA_PROJECT_KEY"]
STALE_DAYS     = int(os.getenv("STALE_DAYS", "2"))
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

_LLM_CONFIG: Dict[str, Any] = {
    "config_list": [{"model": "gpt-4o", "api_key": OPENAI_API_KEY}],
    "temperature": 0,
    "timeout": 120,
    "cache_seed": None,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Agent definitions
# ══════════════════════════════════════════════════════════════════════════════

def _make_monitor_assistant() -> AssistantAgent:
    return AssistantAgent(
        name="MonitorAssistant",
        llm_config=_LLM_CONFIG,
        system_message="""You are the Monitor Agent for Aegis PM.

When AegisSupervisor asks you to check for stale tasks:
1. Call `check_for_stale_tasks` with the project key to find stale In Progress tasks.
2. For each task returned, call `save_alert` to persist it to the database.
3. For each task, also call `notify_communicator` to queue a Slack notification.
4. Report back to AegisSupervisor with a concise JSON summary:
   {"monitor_result": {"stale_found": <int>, "alerts_saved": <int>, "notifications_queued": <int>, "tasks": [...]}}

Rules:
- Only speak when AegisSupervisor addresses you.
- Do not send Slack messages. That is CommunicatorAssistant's job.
- Do not use the word TERMINATE. Only AegisSupervisor may end the chat.
""",
    )


def _make_communicator_assistant() -> AssistantAgent:
    return AssistantAgent(
        name="CommunicatorAssistant",
        llm_config=_LLM_CONFIG,
        system_message="""You are the Communicator Agent for Aegis PM.

When AegisSupervisor asks you to send notifications:
1. Call `fetch_approved_alerts` to get alerts ready to send.
2. For each approved alert:
   a. Call `send_slack_notification`.
   b. If success=true, call `mark_alert_notified`.
3. Report back to AegisSupervisor with a concise JSON summary:
   {"communicator_result": {"approved_found": <int>, "notified": <int>, "failed": <int>}}

Rules:
- Only speak when AegisSupervisor addresses you.
- Do not poll Jira. That is MonitorAssistant's job.
- Do not use the word TERMINATE. Only AegisSupervisor may end the chat.
""",
    )


def _make_supervisor() -> AssistantAgent:
    return AssistantAgent(
        name="AegisSupervisor",
        llm_config=_LLM_CONFIG,
        system_message="""You are the Aegis PM Supervisor – the orchestrator of the project management pipeline.

Your role is to coordinate the other agents in order and produce a final report.

Standard full-cycle flow:
1. Ask MonitorAssistant to check for stale Jira tasks.
2. Wait for MonitorAssistant's monitor_result JSON.
3. Ask CommunicatorAssistant to send Slack notifications for all approved alerts.
4. Wait for CommunicatorAssistant's communicator_result JSON.
5. Synthesise both results into a final report for AegisAdmin:

Final report format:
```json
{
  "aegis_report": {
    "cycle": "full",
    "monitor":       { "stale_found": <int>, "alerts_registered": <int> },
    "communicator":  { "approved_found": <int>, "notified": <int>, "failed": <int> },
    "status": "completed" | "partial_failure",
    "summary": "<one sentence>"
  }
}
```

After delivering the report, end your message with TERMINATE.

Rules:
- Always address agents by name (MonitorAssistant, CommunicatorAssistant).
- Never call tools yourself. Delegate all tool use to the specialist agents.
- If an agent reports an error, include it in the report and still TERMINATE.
""",
    )


def _make_admin(require_human_approval: bool = False) -> UserProxyAgent:
    """
    AegisAdmin is the entry-point UserProxyAgent.

    require_human_approval=False → fully autonomous (human_input_mode="NEVER")
    require_human_approval=True  → pauses for human input after MonitorAssistant
                                   reports (human_input_mode="ALWAYS")
    """
    return UserProxyAgent(
        name="AegisAdmin",
        human_input_mode="ALWAYS" if require_human_approval else "NEVER",
        max_consecutive_auto_reply=50,
        code_execution_config=False,
        is_termination_msg=lambda msg: (
            isinstance(msg.get("content"), str)
            and "TERMINATE" in msg["content"]
        ),
        system_message=(
            "You are the Aegis PM administrator. "
            "You initiate cycles and receive final reports. "
            "When you see a final report, acknowledge it and stop."
            if not require_human_approval else
            "You are the Aegis PM administrator. "
            "After MonitorAssistant reports stale tasks, you may review and "
            "approve or modify before CommunicatorAssistant sends messages."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Tool registration helper
# ══════════════════════════════════════════════════════════════════════════════

def _register_all_tools(
    monitor: AssistantAgent,
    communicator: AssistantAgent,
    admin: UserProxyAgent,
) -> None:
    """
    Register all tool functions on the correct caller/executor pairs.

    In GroupChat, tool results are executed by the UserProxyAgent (admin),
    but each tool is *callable* only by its designated AssistantAgent.
    This keeps the tool namespaces cleanly separated.
    """
    # ── Monitor tools ─────────────────────────────────────────────────────────
    register_function(
        check_for_stale_tasks,
        caller=monitor,
        executor=admin,
        name="check_for_stale_tasks",
        description=(
            "Query Jira for 'In Progress' issues not updated in the configured "
            "number of days. Returns JSON with a 'tasks' list."
        ),
    )
    register_function(
        save_alert,
        caller=monitor,
        executor=admin,
        name="save_alert",
        description=(
            "Persist a stale-task alert directly to PostgreSQL. "
            "Duplicate pending alerts are silently ignored."
        ),
    )
    register_function(
        notify_communicator,
        caller=monitor,
        executor=admin,
        name="notify_communicator",
        description=(
            "Queue a Slack notification for the task assignee by sending "
            "task details to the Communicator Agent via the Aegis API."
        ),
    )

    # ── Communicator tools ────────────────────────────────────────────────────
    register_function(
        fetch_approved_alerts,
        caller=communicator,
        executor=admin,
        name="fetch_approved_alerts",
        description=(
            "Fetch all alerts with status='approved' from the Aegis PM backend."
        ),
    )
    register_function(
        send_slack_notification,
        caller=communicator,
        executor=admin,
        name="send_slack_notification",
        description=(
            "Send a Slack Block Kit message notifying a task assignee of a stale task."
        ),
    )
    register_function(
        mark_alert_notified,
        caller=communicator,
        executor=admin,
        name="mark_alert_notified",
        description=(
            "Mark an alert as notified in the Aegis PM backend after Slack send."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  GroupChat factory
# ══════════════════════════════════════════════════════════════════════════════

def build_group_chat(
    require_human_approval: bool = False,
) -> tuple[UserProxyAgent, GroupChatManager]:
    """
    Assemble the full Aegis PM GroupChat.

    Returns:
        (admin, manager) – call admin.initiate_chat(manager, message=...)
    """
    monitor      = _make_monitor_assistant()
    communicator = _make_communicator_assistant()
    supervisor   = _make_supervisor()
    admin        = _make_admin(require_human_approval=require_human_approval)

    _register_all_tools(monitor, communicator, admin)

    group_chat = GroupChat(
        agents=[admin, supervisor, monitor, communicator],
        messages=[],
        max_round=60,                    # safety ceiling on total turns
        speaker_selection_method="auto", # let the LLM decide who speaks next
        allow_repeat_speaker=True,       # supervisor may address same agent twice
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=_LLM_CONFIG,
        is_termination_msg=lambda msg: (
            isinstance(msg.get("content"), str)
            and "TERMINATE" in msg["content"]
        ),
    )

    return admin, manager


# ══════════════════════════════════════════════════════════════════════════════
#  High-level entry points
# ══════════════════════════════════════════════════════════════════════════════

def run_full_cycle(
    project_key: str = JIRA_PROJECT,
    stale_days: int = STALE_DAYS,
    require_human_approval: bool = False,
) -> Dict[str, Any]:
    """
    Run a complete Monitor → Communicate cycle via GroupChat.

    Args:
        project_key:            Jira project to scan (default: JIRA_PROJECT_KEY env)
        stale_days:             Staleness threshold in days (default: STALE_DAYS env)
        require_human_approval: If True, pauses for human review after Monitor step

    Returns:
        Parsed aegis_report dict, or {"error": "..."} on failure.
    """
    log.info(
        "GroupChat: starting full cycle │ project=%s │ stale_days=%d │ hitl=%s",
        project_key, stale_days, require_human_approval,
    )

    admin, manager = build_group_chat(require_human_approval=require_human_approval)

    kick_off = (
        f"Run a full Aegis PM cycle:\n"
        f"  1. Check project '{project_key}' for tasks stale >= {stale_days} day(s)\n"
        f"  2. Send Slack notifications for all approved alerts\n"
        f"  3. Report back with the full cycle summary\n"
    )

    try:
        admin.initiate_chat(manager, message=kick_off, silent=False)
    except Exception as exc:
        log.exception("GroupChat crashed: %s", exc)
        return {"error": str(exc)}

    return _extract_report(admin.chat_messages.get(manager, []))


def run_monitor_only(
    project_key: str = JIRA_PROJECT,
    stale_days: int = STALE_DAYS,
) -> Dict[str, Any]:
    """
    Run only the Monitor step via GroupChat (useful for testing/partial runs).
    """
    log.info("GroupChat: monitor-only run │ project=%s", project_key)

    admin, manager = build_group_chat(require_human_approval=False)

    kick_off = (
        f"Run ONLY the monitoring step:\n"
        f"  Ask MonitorAssistant to check project '{project_key}' "
        f"for tasks stale >= {stale_days} day(s) and register alerts.\n"
        f"  Do NOT involve CommunicatorAssistant. Report results and TERMINATE."
    )

    try:
        admin.initiate_chat(manager, message=kick_off, silent=False)
    except Exception as exc:
        log.exception("GroupChat (monitor-only) crashed: %s", exc)
        return {"error": str(exc)}

    return _extract_report(admin.chat_messages.get(manager, []))


def run_communicate_only() -> Dict[str, Any]:
    """
    Run only the Communicator step (send pending approved alerts).
    """
    log.info("GroupChat: communicate-only run")

    admin, manager = build_group_chat(require_human_approval=False)

    kick_off = (
        "Run ONLY the notification step:\n"
        "  Ask CommunicatorAssistant to send Slack messages for all "
        "approved alerts and mark them notified.\n"
        "  Do NOT involve MonitorAssistant. Report results and TERMINATE."
    )

    try:
        admin.initiate_chat(manager, message=kick_off, silent=False)
    except Exception as exc:
        log.exception("GroupChat (communicate-only) crashed: %s", exc)
        return {"error": str(exc)}

    return _extract_report(admin.chat_messages.get(manager, []))


# ══════════════════════════════════════════════════════════════════════════════
#  Parse helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_report(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Walk chat history backwards to find the aegis_report JSON block.
    Falls back to a minimal dict if not found.
    """
    for msg in reversed(messages):
        content = msg.get("content") or ""
        if "aegis_report" not in content:
            continue
        start = content.find("{")
        end   = content.rfind("}") + 1
        if start == -1 or end <= start:
            continue
        try:
            data = json.loads(content[start:end])
            report = data.get("aegis_report", data)
            log.info(
                "GroupChat report: %s",
                report.get("summary", json.dumps(report)),
            )
            return report
        except json.JSONDecodeError:
            log.warning("Could not parse aegis_report JSON")
            continue

    log.warning("No aegis_report found in GroupChat messages")
    return {
        "cycle": "unknown",
        "status": "no_report",
        "summary": "GroupChat completed but no structured report was produced.",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point  (python -m agents.group_chat)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "monitor":
        result = run_monitor_only()
    elif mode == "communicate":
        result = run_communicate_only()
    elif mode == "hitl":
        result = run_full_cycle(require_human_approval=True)
    else:
        result = run_full_cycle()

    print("\n" + "═" * 60)
    print("  AEGIS PM – CYCLE REPORT")
    print("═" * 60)
    print(json.dumps(result, indent=2))
