"""
aegis-pm / tests / test_monitor_agent.py

Unit tests for the Monitor Agent.

Strategy: mock Jira HTTP calls and DB writes.
No real Jira / Postgres needed.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.asyncio


class TestJiraClient:

    async def test_search_issues_returns_list(self, mock_stale_jira_issues):
        from agents.monitor_agent import JiraClient

        with patch("httpx.AsyncClient") as mock_http:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"issues": mock_stale_jira_issues, "total": 2}
            mock_resp.raise_for_status = MagicMock()

            mock_http.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(get=AsyncMock(return_value=mock_resp))
            )
            mock_http.return_value.__aexit__ = AsyncMock(return_value=False)

            client = JiraClient()
            result = await client.search_issues('project = "TEST"')
            assert len(result) == 2
            assert result[0]["key"] == "TEST-10"

    async def test_search_issues_returns_empty_on_timeout(self):
        import httpx
        from agents.monitor_agent import JiraClient

        with patch("httpx.AsyncClient") as mock_http:
            mock_http.return_value.__aenter__ = AsyncMock(
                return_value=MagicMock(
                    get=AsyncMock(side_effect=httpx.TimeoutException("timeout"))
                )
            )
            mock_http.return_value.__aexit__ = AsyncMock(return_value=False)

            client = JiraClient()
            result = await client.search_issues('project = "TEST"')
            assert result == []

    def test_parse_updated_valid(self):
        from agents.monitor_agent import JiraClient
        issue = {"fields": {"updated": "2024-03-15T10:22:33.000+0000"}}
        dt = JiraClient.parse_updated(issue)
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 3

    def test_parse_updated_missing(self):
        from agents.monitor_agent import JiraClient
        assert JiraClient.parse_updated({"fields": {}}) is None
        assert JiraClient.parse_updated({}) is None

    def test_issue_url(self):
        from agents.monitor_agent import JiraClient
        c = JiraClient()
        url = c.issue_url("ENG-42")
        assert "ENG-42" in url
        assert url.startswith("https://")


class TestCheckForStaleTasks:

    def test_returns_json_string(self, mock_stale_jira_issues):
        from agents.monitor_agent import check_for_stale_tasks, _jira

        with patch.object(_jira, "search_issues", new=AsyncMock(return_value=mock_stale_jira_issues)):
            result = check_for_stale_tasks("TEST")

        data = json.loads(result)
        assert "tasks" in data
        assert data["total"] == 2

    def test_task_fields_present(self, mock_stale_jira_issues):
        from agents.monitor_agent import check_for_stale_tasks, _jira

        with patch.object(_jira, "search_issues", new=AsyncMock(return_value=mock_stale_jira_issues)):
            result = json.loads(check_for_stale_tasks("TEST"))

        task = result["tasks"][0]
        required = {"task_key", "task_summary", "assignee", "assignee_email",
                    "jira_url", "last_updated", "days_stale", "priority", "labels"}
        assert required.issubset(task.keys())

    def test_assignees_mapped_correctly(self, mock_stale_jira_issues):
        from agents.monitor_agent import check_for_stale_tasks, _jira

        with patch.object(_jira, "search_issues", new=AsyncMock(return_value=mock_stale_jira_issues)):
            result = json.loads(check_for_stale_tasks("TEST"))

        assignees = [t["assignee"] for t in result["tasks"]]
        assert "Alice Smith" in assignees
        assert "Bob Jones"   in assignees

    def test_unassigned_task_handled(self):
        from agents.monitor_agent import check_for_stale_tasks, _jira

        issues = [{
            "key": "TEST-99",
            "fields": {
                "summary":  "Orphaned task",
                "updated":  "2024-01-01T00:00:00.000+0000",
                "status":   {"name": "In Progress"},
                "assignee": None,   # unassigned
                "priority": {"name": "Low"},
                "labels":   [],
            },
        }]
        with patch.object(_jira, "search_issues", new=AsyncMock(return_value=issues)):
            result = json.loads(check_for_stale_tasks("TEST"))

        assert result["tasks"][0]["assignee"] == "Unassigned"
        assert result["tasks"][0]["assignee_email"] is None

    def test_empty_project_returns_zero(self):
        from agents.monitor_agent import check_for_stale_tasks, _jira

        with patch.object(_jira, "search_issues", new=AsyncMock(return_value=[])):
            result = json.loads(check_for_stale_tasks("TEST"))

        assert result["total"] == 0
        assert result["tasks"] == []

    def test_jira_failure_returns_error_json(self):
        from agents.monitor_agent import check_for_stale_tasks, _jira

        with patch.object(_jira, "search_issues", new=AsyncMock(side_effect=Exception("Jira down"))):
            result = json.loads(check_for_stale_tasks("TEST"))

        assert "error" in result
        assert result["total"] == 0


class TestSaveAlert:

    def test_saves_to_db_returns_id(self):
        from agents.monitor_agent import save_alert

        with patch("agents.monitor_agent._db_save_alert", return_value=42) as mock_db:
            result = json.loads(save_alert(
                task_key="TEST-1",
                task_summary="Fix bug",
                assignee="Alice",
                jira_url="https://jira.example.com/TEST-1",
            ))

        assert result["success"]  is True
        assert result["alert_id"] == 42
        assert result["task_key"] == "TEST-1"
        mock_db.assert_called_once()

    def test_duplicate_returns_none_id(self):
        from agents.monitor_agent import save_alert

        # _db_save_alert returns None when duplicate is suppressed
        with patch("agents.monitor_agent._db_save_alert", return_value=None):
            result = json.loads(save_alert(
                task_key="TEST-DUP",
                task_summary="Dup",
                assignee="Bob",
                jira_url="https://jira.example.com/TEST-DUP",
            ))

        assert result["success"]  is True
        assert result["alert_id"] is None

    def test_db_error_returns_failure_json(self):
        from agents.monitor_agent import save_alert
        from tenacity import RetryError

        with patch("agents.monitor_agent._db_save_alert", side_effect=RetryError(None)):
            result = json.loads(save_alert(
                task_key="TEST-ERR",
                task_summary="Error task",
                assignee="Charlie",
                jira_url="https://jira.example.com/TEST-ERR",
            ))

        assert result["success"] is False
        assert "error" in result


class TestNotifyCommunicator:

    def test_queues_successfully(self):
        from agents.monitor_agent import notify_communicator

        with patch("agents.monitor_agent._post_to_communicator", return_value=True):
            result = json.loads(notify_communicator(
                task_key="TEST-1",
                task_summary="Fix",
                assignee="Alice",
                jira_url="https://jira.example.com/TEST-1",
                days_stale=4,
            ))

        assert result["success"] is True
        assert result["queued"]  is True

    def test_http_error_returns_failure(self):
        import httpx
        from agents.monitor_agent import notify_communicator

        err = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock(
                status_code=500, text="Server error"
            )
        )
        with patch("agents.monitor_agent._post_to_communicator", side_effect=err):
            result = json.loads(notify_communicator(
                task_key="TEST-FAIL",
                task_summary="x",
                assignee="x",
                jira_url="x",
                days_stale=3,
            ))

        assert result["success"] is False
        assert "500" in result["error"]


class TestMonitorAgentParsesSummary:

    def test_parse_valid_summary(self):
        from agents.monitor_agent import MonitorAgent

        messages = [{
            "role": "assistant",
            "content": json.dumps({
                "cycle_summary": {
                    "project_key":        "ENG",
                    "stale_found":        3,
                    "alerts_saved":       3,
                    "notifications_sent": 2,
                    "failures":           1,
                    "tasks":              [],
                }
            }) + "\nTERMINATE",
        }]
        summary = MonitorAgent._parse_summary(messages)
        assert summary["stale_found"]        == 3
        assert summary["alerts_saved"]       == 3
        assert summary["notifications_sent"] == 2
        assert summary["failures"]           == 1

    def test_parse_empty_messages_returns_zeros(self):
        from agents.monitor_agent import MonitorAgent
        summary = MonitorAgent._parse_summary([])
        assert summary["stale_found"] == 0

    def test_parse_malformed_json_returns_zeros(self):
        from agents.monitor_agent import MonitorAgent
        messages = [{"role": "assistant", "content": "{cycle_summary: BROKEN}"}]
        summary = MonitorAgent._parse_summary(messages)
        assert summary["stale_found"] == 0
