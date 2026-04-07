"""
aegis-pm / tests / conftest.py

Shared pytest fixtures for all test modules.

Fixtures
────────
  client           AsyncClient wired to the FastAPI app with mocked DB
  db_session       In-memory SQLite async session for unit tests
  mock_jira        Monkeypatched JiraClient that returns fixture data
  sample_alert     A dict matching a freshly-created alert row
  agent_headers    HTTP headers with AGENT_API_KEY
  admin_headers    HTTP headers with ADMIN_API_KEY
"""
from __future__ import annotations

import os
import pytest
import pytest_asyncio

# ── Force test env vars before any app import ─────────────────────────────────
os.environ.setdefault("POSTGRES_USER",     "aegis_test")
os.environ.setdefault("POSTGRES_PASSWORD", "test_password")
os.environ.setdefault("POSTGRES_DB",       "aegispm_test")
os.environ.setdefault("POSTGRES_HOST",     "localhost")
os.environ.setdefault("JIRA_BASE_URL",     "https://test.atlassian.net")
os.environ.setdefault("JIRA_EMAIL",        "test@example.com")
os.environ.setdefault("JIRA_API_TOKEN",    "test_jira_token_000000000000000")
os.environ.setdefault("JIRA_PROJECT_KEY",  "TEST")
os.environ.setdefault("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
os.environ.setdefault("OPENAI_API_KEY",    "sk-test-0000000000000000000000000")
os.environ.setdefault("AGENT_API_KEY",     "agent_test_key_abcdef1234567890abcdef")
os.environ.setdefault("ADMIN_API_KEY",     "admin_test_key_xyz9876543210fedcba00")
os.environ.setdefault("AEGIS_ENFORCE_AUTH","true")
os.environ.setdefault("LOG_LEVEL",         "WARNING")   # keep test output clean

from httpx import AsyncClient, ASGITransport
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from api.main import app, metadata, get_db


# ── In-memory SQLite for tests (no real Postgres needed) ─────────────────────

TEST_DSN = "sqlite+aiosqlite:///:memory:"

@pytest_asyncio.fixture(scope="function")
async def test_engine():
    engine = create_async_engine(TEST_DSN, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(test_engine):
    Session = async_sessionmaker(test_engine, expire_on_commit=False)
    async with Session() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def client(test_engine):
    """
    FastAPI AsyncClient with the DB overridden to use in-memory SQLite.
    All routes are available. Auth headers must be provided per-test.
    """
    Session = async_sessionmaker(test_engine, expire_on_commit=False)

    async def _override_db():
        async with Session() as session:
            yield session

    app.dependency_overrides[get_db] = _override_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Auth headers ──────────────────────────────────────────────────────────────

@pytest.fixture
def agent_headers():
    return {"X-API-Key": os.environ["AGENT_API_KEY"]}


@pytest.fixture
def admin_headers():
    return {"X-API-Key": os.environ["ADMIN_API_KEY"]}


# ── Sample data ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_alert_payload():
    return {
        "task_key":       "TEST-42",
        "task_summary":   "Fix login regression",
        "assignee":       "Jane Dev",
        "assignee_email": "jane@example.com",
        "jira_url":       "https://test.atlassian.net/browse/TEST-42",
        "last_updated":   "2024-01-01T10:00:00+00:00",
    }


@pytest.fixture
def mock_stale_jira_issues():
    """Realistic Jira API response with 2 stale issues."""
    return [
        {
            "key": "TEST-10",
            "fields": {
                "summary":  "Implement OAuth flow",
                "updated":  "2024-01-01T08:00:00.000+0000",
                "status":   {"name": "In Progress"},
                "assignee": {
                    "displayName":  "Alice Smith",
                    "emailAddress": "alice@example.com",
                },
                "priority": {"name": "High"},
                "labels":   ["backend", "auth"],
            },
        },
        {
            "key": "TEST-11",
            "fields": {
                "summary":  "Refactor DB layer",
                "updated":  "2024-01-02T09:00:00.000+0000",
                "status":   {"name": "In Progress"},
                "assignee": {
                    "displayName":  "Bob Jones",
                    "emailAddress": "bob@example.com",
                },
                "priority": {"name": "Medium"},
                "labels":   [],
            },
        },
    ]
