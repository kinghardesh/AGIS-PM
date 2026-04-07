"""
aegis-pm / tests / test_spec_and_cooldown.py

Tests for Spec Interpreter agent and notification cooldown.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.asyncio


# ══════════════════════════════════════════════════════════════════════════════
#  Spec Interpreter
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_PRD = """
## Feature: User Authentication

We need to build a secure login system for our web app.

### Requirements
1. Users can register with email and password
2. Users can log in and receive a JWT token
3. Passwords must be hashed with bcrypt
4. Failed login attempts should be rate-limited (5 attempts per 15 min)
5. JWT tokens expire after 24 hours
6. Add unit tests for all auth functions
"""

MOCK_PARSED_TASKS = {
    "project_title": "User Authentication",
    "total_tasks": 4,
    "tasks": [
        {
            "summary":              "Implement user registration endpoint",
            "description":          "Create POST /register endpoint that accepts email and password.",
            "issue_type":           "Story",
            "priority":             "High",
            "story_points":         3,
            "acceptance_criteria":  ["Given valid email/password, when POST /register, then user created"],
            "labels":               ["backend", "auth"],
            "depends_on_index":     None,
        },
        {
            "summary":              "Implement JWT login endpoint",
            "description":          "Create POST /login endpoint that returns a signed JWT.",
            "issue_type":           "Story",
            "priority":             "High",
            "story_points":         3,
            "acceptance_criteria":  ["Given valid credentials, when POST /login, then JWT returned"],
            "labels":               ["backend", "auth"],
            "depends_on_index":     0,
        },
        {
            "summary":              "Add bcrypt password hashing",
            "description":          "Hash passwords using bcrypt before storing in DB.",
            "issue_type":           "Task",
            "priority":             "High",
            "story_points":         2,
            "acceptance_criteria":  ["Passwords never stored in plaintext"],
            "labels":               ["backend", "security"],
            "depends_on_index":     None,
        },
        {
            "summary":              "Write unit tests for auth functions",
            "description":          "Cover registration, login, token generation, and password hashing.",
            "issue_type":           "Task",
            "priority":             "Medium",
            "story_points":         2,
            "acceptance_criteria":  ["Coverage >= 90% for auth module"],
            "labels":               ["testing"],
            "depends_on_index":     1,
        },
    ],
}


class TestParseSpecification:

    def test_parse_returns_json_string(self):
        from agents.spec_interpreter import parse_specification

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(MOCK_PARSED_TASKS)

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            result = parse_specification(SAMPLE_PRD)

        data = json.loads(result)
        assert "tasks" in data
        assert data["total_tasks"] == 4

    def test_parse_task_fields_present(self):
        from agents.spec_interpreter import parse_specification

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(MOCK_PARSED_TASKS)

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            data = json.loads(parse_specification(SAMPLE_PRD))

        task = data["tasks"][0]
        for field in ["summary", "description", "issue_type", "priority",
                      "story_points", "acceptance_criteria", "labels"]:
            assert field in task, f"Missing field: {field}"

    def test_parse_summary_capped_at_255_chars(self):
        from agents.spec_interpreter import parse_specification

        long_summary = "A" * 300
        tasks_with_long = dict(MOCK_PARSED_TASKS)
        tasks_with_long["tasks"] = [
            {**MOCK_PARSED_TASKS["tasks"][0], "summary": long_summary}
        ]
        tasks_with_long["total_tasks"] = 1

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(tasks_with_long)

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            data = json.loads(parse_specification(SAMPLE_PRD))

        assert len(data["tasks"][0]["summary"]) <= 255

    def test_parse_handles_openai_error(self):
        from agents.spec_interpreter import parse_specification

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = Exception("OpenAI down")
            result = json.loads(parse_specification(SAMPLE_PRD))

        assert "error" in result
        assert result.get("tasks") == []


class TestCreateJiraTask:

    def test_create_task_returns_issue_key(self):
        from agents.spec_interpreter import create_jira_task

        with patch("agents.spec_interpreter._jira_creator.create_issue", return_value="ENG-42"):
            result = json.loads(create_jira_task(
                summary="Implement login endpoint",
                description="Create POST /login returning JWT.",
                issue_type="Story",
                priority="High",
                story_points=3,
            ))

        assert result["success"]   is True
        assert result["issue_key"] == "ENG-42"
        assert "jira_url" in result

    def test_create_task_appends_acceptance_criteria(self):
        from agents.spec_interpreter import create_jira_task

        calls = []
        def mock_create(summary, description, **kwargs):
            calls.append({"description": description})
            return "ENG-1"

        with patch("agents.spec_interpreter._jira_creator.create_issue", side_effect=mock_create):
            create_jira_task(
                summary="Test task",
                description="Base description.",
                acceptance_criteria=["Given X, when Y, then Z"],
            )

        assert "Acceptance Criteria" in calls[0]["description"]
        assert "Given X" in calls[0]["description"]

    def test_create_task_handles_http_error(self):
        import httpx
        from agents.spec_interpreter import create_jira_task

        err = httpx.HTTPStatusError(
            "400", request=MagicMock(),
            response=MagicMock(status_code=400, text="Bad Request"),
        )
        with patch("agents.spec_interpreter._jira_creator.create_issue", side_effect=err):
            result = json.loads(create_jira_task(
                summary="Will fail",
                description="x",
            ))

        assert result["success"] is False
        assert "400" in result["error"]


class TestSpecInterpreterAgentParseSummary:

    def test_parse_valid_summary(self):
        from agents.spec_interpreter import SpecInterpreterAgent

        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "spec_summary": {
                    "project_title":  "Auth System",
                    "tasks_parsed":   4,
                    "tasks_created":  3,
                    "tasks_failed":   1,
                    "created_issues": [
                        {"index": 0, "issue_key": "ENG-1", "summary": "A", "jira_url": "http://x"},
                        {"index": 1, "issue_key": "ENG-2", "summary": "B", "jira_url": "http://y"},
                        {"index": 2, "issue_key": "ENG-3", "summary": "C", "jira_url": "http://z"},
                    ],
                    "failed_tasks": [{"index": 3, "summary": "D", "error": "Timeout"}],
                }
            }) + "\nTERMINATE",
        }]

        result = SpecInterpreterAgent._parse_summary(messages)
        assert result["tasks_created"]  == 3
        assert result["tasks_failed"]   == 1
        assert result["issue_keys"]     == ["ENG-1", "ENG-2", "ENG-3"]

    def test_parse_empty_messages_returns_zeros(self):
        from agents.spec_interpreter import SpecInterpreterAgent
        result = SpecInterpreterAgent._parse_summary([])
        assert result["tasks_parsed"]  == 0
        assert result["tasks_created"] == 0
        assert result["issue_keys"]    == []


# ══════════════════════════════════════════════════════════════════════════════
#  Notification cooldown
# ══════════════════════════════════════════════════════════════════════════════

class TestNotificationCooldown:

    def test_no_prior_notification_allows_send(self):
        from agents.monitor_agent import is_in_cooldown

        # Simulate no row in DB (never notified)
        with patch("agents.monitor_agent.psycopg2.connect") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = None
            mock_conn.return_value.__enter__ = MagicMock(return_value=mock_conn.return_value)
            mock_conn.return_value.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.return_value.cursor.return_value.__exit__ = MagicMock(return_value=False)

            in_cooldown, remaining = is_in_cooldown("ENG-1")

        assert in_cooldown is False
        assert remaining    == 0

    def test_recent_notification_blocks_send(self):
        from agents.monitor_agent import is_in_cooldown
        from datetime import datetime, timezone, timedelta

        recent = datetime.now(timezone.utc) - timedelta(hours=2)  # 2h ago
        cooldown_hours = 24

        with patch("agents.monitor_agent.psycopg2.connect") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = (recent, cooldown_hours)
            mock_conn.return_value.__enter__ = MagicMock(return_value=mock_conn.return_value)
            mock_conn.return_value.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.return_value.cursor.return_value.__exit__ = MagicMock(return_value=False)

            in_cooldown, remaining = is_in_cooldown("ENG-2")

        assert in_cooldown is True
        assert remaining    >= 21   # ~22 hours remaining

    def test_expired_cooldown_allows_send(self):
        from agents.monitor_agent import is_in_cooldown
        from datetime import datetime, timezone, timedelta

        old = datetime.now(timezone.utc) - timedelta(hours=25)  # 25h ago > 24h cooldown
        cooldown_hours = 24

        with patch("agents.monitor_agent.psycopg2.connect") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = (old, cooldown_hours)
            mock_conn.return_value.__enter__ = MagicMock(return_value=mock_conn.return_value)
            mock_conn.return_value.cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
            mock_conn.return_value.cursor.return_value.__exit__ = MagicMock(return_value=False)

            in_cooldown, remaining = is_in_cooldown("ENG-3")

        assert in_cooldown is False
        assert remaining    == 0

    def test_db_error_fails_open(self):
        """If DB is unreachable, cooldown check should fail open (allow send)."""
        from agents.monitor_agent import is_in_cooldown
        import psycopg2

        with patch("agents.monitor_agent.psycopg2.connect",
                   side_effect=psycopg2.OperationalError("DB down")):
            in_cooldown, remaining = is_in_cooldown("ENG-4")

        # Fail open: allow the notification
        assert in_cooldown is False
