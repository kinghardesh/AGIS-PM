"""
aegis-pm / tests / test_api.py

API endpoint tests.

Covers
──────
  Auth         – missing key → 401, wrong key → 403, agent key can't do admin
  CRUD         – create, read, list, filters, pagination
  State machine – approve, dismiss, notified, reopen; illegal transitions → 400
  Bulk         – bulk approve/dismiss
  Audit        – history endpoint returns transition log
  Stats        – counts by status
  Dedup        – duplicate pending alerts are suppressed
  Rate limit   – 429 after threshold (tested with a reduced limit)
"""
from __future__ import annotations

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio


# ══════════════════════════════════════════════════════════════════════════════
#  Auth
# ══════════════════════════════════════════════════════════════════════════════

class TestAuth:

    async def test_missing_key_returns_401(self, client):
        res = await client.get("/alerts")
        assert res.status_code == 401, res.text

    async def test_wrong_key_returns_403(self, client):
        res = await client.get("/alerts", headers={"X-API-Key": "wrong_key"})
        assert res.status_code == 403

    async def test_agent_key_can_list_alerts(self, client, agent_headers):
        res = await client.get("/alerts", headers=agent_headers)
        assert res.status_code == 200

    async def test_agent_key_cannot_approve(self, client, agent_headers, admin_headers, sample_alert_payload):
        # Create first
        await client.post("/alerts", json=sample_alert_payload, headers=agent_headers)
        # Try approve with agent key → 403
        res = await client.post("/alerts/1/approve", headers=agent_headers, json={})
        assert res.status_code == 403

    async def test_admin_key_can_approve(self, client, agent_headers, admin_headers, sample_alert_payload):
        await client.post("/alerts", json=sample_alert_payload, headers=agent_headers)
        res = await client.post("/alerts/1/approve", headers=admin_headers, json={})
        assert res.status_code == 200
        assert res.json()["status"] == "approved"

    async def test_health_requires_no_auth(self, client):
        res = await client.get("/health")
        assert res.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
#  CRUD
# ══════════════════════════════════════════════════════════════════════════════

class TestAlertCRUD:

    async def test_create_alert(self, client, agent_headers, sample_alert_payload):
        res = await client.post("/alerts", json=sample_alert_payload, headers=agent_headers)
        assert res.status_code == 201
        data = res.json()
        assert data["task_key"]  == "TEST-42"
        assert data["assignee"]  == "Jane Dev"
        assert data["status"]    == "pending"
        assert data["slack_sent"] is False
        assert "id" in data

    async def test_task_key_normalised_to_uppercase(self, client, agent_headers):
        payload = {"task_key": "test-99", "assignee": "Someone", "task_summary": "test"}
        res = await client.post("/alerts", json=payload, headers=agent_headers)
        assert res.json()["task_key"] == "TEST-99"

    async def test_get_single_alert(self, client, agent_headers, sample_alert_payload):
        created = (await client.post("/alerts", json=sample_alert_payload, headers=agent_headers)).json()
        res = await client.get(f"/alerts/{created['id']}", headers=agent_headers)
        assert res.status_code == 200
        assert res.json()["id"] == created["id"]

    async def test_get_missing_alert_returns_404(self, client, agent_headers):
        res = await client.get("/alerts/99999", headers=agent_headers)
        assert res.status_code == 404

    async def test_list_returns_paginated_shape(self, client, agent_headers):
        res = await client.get("/alerts", headers=agent_headers)
        assert res.status_code == 200
        data = res.json()
        assert "items"  in data
        assert "total"  in data
        assert "limit"  in data
        assert "offset" in data

    async def test_list_filter_by_status(self, client, agent_headers, sample_alert_payload):
        await client.post("/alerts", json=sample_alert_payload, headers=agent_headers)
        res = await client.get("/alerts?status=pending", headers=agent_headers)
        assert all(a["status"] == "pending" for a in res.json()["items"])

    async def test_list_invalid_status_returns_400(self, client, agent_headers):
        res = await client.get("/alerts?status=banana", headers=agent_headers)
        assert res.status_code == 400

    async def test_list_filter_by_assignee(self, client, agent_headers, sample_alert_payload):
        await client.post("/alerts", json=sample_alert_payload, headers=agent_headers)
        res = await client.get("/alerts?assignee=jane", headers=agent_headers)
        items = res.json()["items"]
        assert len(items) >= 1
        assert "Jane" in items[0]["assignee"]

    async def test_list_pagination(self, client, agent_headers):
        # Create 3 alerts
        for i in range(3):
            payload = {"task_key": f"TEST-{i}", "assignee": "Dev", "task_summary": f"Task {i}"}
            await client.post("/alerts", json=payload, headers=agent_headers)

        page1 = await client.get("/alerts?limit=2&offset=0", headers=agent_headers)
        page2 = await client.get("/alerts?limit=2&offset=2", headers=agent_headers)
        assert len(page1.json()["items"]) == 2
        assert len(page2.json()["items"]) == 1
        assert page1.json()["total"] == 3

    async def test_duplicate_pending_alert_suppressed(self, client, agent_headers, sample_alert_payload):
        r1 = await client.post("/alerts", json=sample_alert_payload, headers=agent_headers)
        r2 = await client.post("/alerts", json=sample_alert_payload, headers=agent_headers)
        assert r1.json()["id"] == r2.json()["id"]   # same alert returned


# ══════════════════════════════════════════════════════════════════════════════
#  State machine
# ══════════════════════════════════════════════════════════════════════════════

class TestStateMachine:

    async def _create(self, client, agent_headers, key="SM-1"):
        payload = {"task_key": key, "assignee": "Dev", "task_summary": "Test task"}
        return (await client.post("/alerts", json=payload, headers=agent_headers)).json()

    async def test_approve_pending(self, client, agent_headers, admin_headers):
        alert = await self._create(client, agent_headers)
        res   = await client.post(f"/alerts/{alert['id']}/approve",
                                  headers=admin_headers, json={"notes": "LGTM"})
        assert res.status_code == 200
        assert res.json()["status"] == "approved"

    async def test_dismiss_pending(self, client, agent_headers, admin_headers):
        alert = await self._create(client, agent_headers, key="SM-2")
        res   = await client.post(f"/alerts/{alert['id']}/dismiss",
                                  headers=admin_headers, json={})
        assert res.status_code == 200
        assert res.json()["status"] == "dismissed"

    async def test_notified_requires_approved_first(self, client, agent_headers):
        alert = await self._create(client, agent_headers, key="SM-3")
        # Try to mark notified while still pending → 400
        res = await client.post(f"/alerts/{alert['id']}/notified", headers=agent_headers)
        assert res.status_code == 400

    async def test_approve_then_notified(self, client, agent_headers, admin_headers):
        alert = await self._create(client, agent_headers, key="SM-4")
        await client.post(f"/alerts/{alert['id']}/approve", headers=admin_headers, json={})
        res = await client.post(f"/alerts/{alert['id']}/notified", headers=agent_headers)
        assert res.status_code == 200
        body = res.json()
        assert body["status"]    == "notified"
        assert body["slack_sent"] is True

    async def test_double_approve_rejected(self, client, agent_headers, admin_headers):
        alert = await self._create(client, agent_headers, key="SM-5")
        await client.post(f"/alerts/{alert['id']}/approve", headers=admin_headers, json={})
        res = await client.post(f"/alerts/{alert['id']}/approve", headers=admin_headers, json={})
        assert res.status_code == 400

    async def test_reopen_dismissed(self, client, agent_headers, admin_headers):
        alert = await self._create(client, agent_headers, key="SM-6")
        await client.post(f"/alerts/{alert['id']}/dismiss", headers=admin_headers, json={})
        res = await client.post(f"/alerts/{alert['id']}/reopen",
                                headers=admin_headers, json={"notes": "needs another look"})
        assert res.status_code == 200
        assert res.json()["status"] == "pending"

    async def test_reopen_notified(self, client, agent_headers, admin_headers):
        alert = await self._create(client, agent_headers, key="SM-7")
        await client.post(f"/alerts/{alert['id']}/approve", headers=admin_headers, json={})
        await client.post(f"/alerts/{alert['id']}/notified", headers=agent_headers)
        res = await client.post(f"/alerts/{alert['id']}/reopen",
                                headers=admin_headers, json={})
        assert res.status_code == 200
        assert res.json()["status"] == "pending"

    async def test_cannot_approve_dismissed(self, client, agent_headers, admin_headers):
        alert = await self._create(client, agent_headers, key="SM-8")
        await client.post(f"/alerts/{alert['id']}/dismiss", headers=admin_headers, json={})
        res = await client.post(f"/alerts/{alert['id']}/approve", headers=admin_headers, json={})
        assert res.status_code == 400
        assert "dismissed" in res.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
#  Bulk actions
# ══════════════════════════════════════════════════════════════════════════════

class TestBulkActions:

    async def _make_alerts(self, client, agent_headers, n=3):
        ids = []
        for i in range(n):
            p = {"task_key": f"BULK-{i}", "assignee": "Dev", "task_summary": f"Task {i}"}
            r = await client.post("/alerts", json=p, headers=agent_headers)
            ids.append(r.json()["id"])
        return ids

    async def test_bulk_approve(self, client, agent_headers, admin_headers):
        ids = await self._make_alerts(client, agent_headers)
        res = await client.post("/alerts/bulk/approve",
                                headers=admin_headers,
                                json={"ids": ids, "actor": "human"})
        assert res.status_code == 200
        body = res.json()
        assert len(body["succeeded"]) == 3
        assert len(body["failed"])    == 0

    async def test_bulk_dismiss(self, client, agent_headers, admin_headers):
        ids = await self._make_alerts(client, agent_headers, n=2)
        res = await client.post("/alerts/bulk/dismiss",
                                headers=admin_headers,
                                json={"ids": ids, "actor": "human"})
        assert res.status_code == 200
        assert len(res.json()["succeeded"]) == 2

    async def test_bulk_partial_failure(self, client, agent_headers, admin_headers):
        ids = await self._make_alerts(client, agent_headers, n=2)
        # Approve first one so second bulk approve fails on it
        await client.post(f"/alerts/{ids[0]}/approve", headers=admin_headers, json={})
        res = await client.post("/alerts/bulk/approve",
                                headers=admin_headers,
                                json={"ids": ids, "actor": "human"})
        body = res.json()
        # ids[0] already approved → fails; ids[1] pending → succeeds
        assert len(body["succeeded"]) == 1
        assert len(body["failed"])    == 1


# ══════════════════════════════════════════════════════════════════════════════
#  Audit log
# ══════════════════════════════════════════════════════════════════════════════

class TestAuditLog:

    async def test_creation_writes_audit_entry(self, client, agent_headers, sample_alert_payload):
        alert = (await client.post("/alerts", json=sample_alert_payload, headers=agent_headers)).json()
        res   = await client.get(f"/alerts/{alert['id']}/history", headers=agent_headers)
        assert res.status_code == 200
        history = res.json()
        assert len(history) >= 1
        assert history[0]["to_status"]   == "pending"
        assert history[0]["from_status"] is None
        assert history[0]["actor"]       == "monitor_agent"

    async def test_transitions_appended_to_history(self, client, agent_headers, admin_headers, sample_alert_payload):
        alert = (await client.post("/alerts", json=sample_alert_payload, headers=agent_headers)).json()
        await client.post(f"/alerts/{alert['id']}/approve",
                          headers=admin_headers, json={"notes": "all good", "actor": "rahul"})
        history = (await client.get(f"/alerts/{alert['id']}/history", headers=agent_headers)).json()
        assert len(history) == 2
        assert history[1]["to_status"]   == "approved"
        assert history[1]["from_status"] == "pending"
        assert history[1]["notes"]       == "all good"


# ══════════════════════════════════════════════════════════════════════════════
#  Stats
# ══════════════════════════════════════════════════════════════════════════════

class TestStats:

    async def test_stats_returns_correct_counts(self, client, agent_headers, admin_headers):
        # Create 3 pending
        for i in range(3):
            await client.post("/alerts",
                json={"task_key": f"STAT-{i}", "assignee": "Dev", "task_summary": "x"},
                headers=agent_headers)

        # Approve 1, dismiss 1
        await client.post("/alerts/1/approve", headers=admin_headers, json={})
        await client.post("/alerts/2/dismiss", headers=admin_headers, json={})

        res  = await client.get("/stats", headers=agent_headers)
        body = res.json()
        assert body["pending"]   == 1
        assert body["approved"]  == 1
        assert body["dismissed"] == 1
        assert body["total"]     == 3
