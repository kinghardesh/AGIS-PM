# Aegis PM

Autonomous multi-agent project management. Six specialised AI agents built with Microsoft AutoGen collaborate across Jira, Slack, and GitHub to manage projects end-to-end.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Monitor Agent  ──▶  Communicator Agent                         │
│  (Jira poll)         (Slack notify)                             │
│       │                    │                                    │
│  Spec Interpreter     GroupChat Orchestrator                    │
│  (PRD → tasks)        (optional unified mode)                   │
│       └──────────────────┬─────────────────────┘               │
│                  FastAPI Backend  (REST + API keys)             │
│                       │                                         │
│          ┌────────────┼──────────────┐                          │
│      PostgreSQL   HITL Dashboard  Agent Health                  │
│    (alerts + audit)  (nginx + auth)   (/agents/status)         │
└─────────────────────────────────────────────────────────────────┘
```

### Agent Status

| Agent | Status | Role |
|---|---|---|
| Monitor | ✅ | Polls Jira every 5 min, finds stale tasks |
| Communicator | ✅ | Sends Slack notifications for approved alerts |
| Spec Interpreter | ✅ | Parses PRDs → creates Jira tasks via LLM |
| Planner & Scheduler | 🔜 | Timelines, dependencies, Gantt charts |
| Resource Manager | 🔜 | Matches tasks to humans or AI agents |
| HITL Supervisor | ✅ | Human approval via dashboard |

---

## Quick Start

**Prerequisites:** Docker + Compose 24+, Jira Cloud, Slack workspace, OpenAI key.

```bash
# 1. Configure
cp .env.example .env
# Fill in JIRA_*, SLACK_WEBHOOK_URL, OPENAI_API_KEY, AGENT_API_KEY, ADMIN_API_KEY

# 2. Generate dashboard password
bash scripts/gen_htpasswd.sh --interactive

# 3. Start
docker compose up --build
```

| URL | Purpose |
|---|---|
| http://localhost:3000 | HITL Dashboard (Basic Auth) |
| http://localhost:8000/docs | API docs (Admin key required) |
| http://localhost:8000/agents/status | Agent health |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `STALE_DAYS` | `2` | Days without Jira update = stale |
| `POLL_INTERVAL_SECONDS` | `300` | Jira poll interval |
| `NOTIFY_COOLDOWN_HOURS` | `24` | Hours before re-notifying same task |
| `AEGIS_MODE` | `agents` | `agents` or `groupchat` |
| `OPENAI_MODEL` | `gpt-4-turbo` | Model for AutoGen agents |
| `AEGIS_ENFORCE_AUTH` | `false` | Require API keys (`true` for prod) |
| `HEALTH_ALERT_AFTER_FAILURES` | `2` | Failures before Slack health alert |
| `RATE_LIMIT_REQUESTS` | `120` | Max requests per IP per minute |

---

## API Reference

All endpoints require `X-API-Key` when `AEGIS_ENFORCE_AUTH=true`.

| Key | Access level |
|---|---|
| `AGENT_API_KEY` | Read + create alerts, mark notified |
| `ADMIN_API_KEY` | All agent access + approve, dismiss, reopen, bulk |

```
GET  /health                    Liveness probe
GET  /stats                     Alert counts by status
GET  /agents/status             Agent health (no auth)

GET  /alerts                    List (filters: status, assignee, task_key, dates, pagination)
POST /alerts                    Create  [Agent]
GET  /alerts/{id}               Single alert
GET  /alerts/{id}/history       Audit trail

POST /alerts/{id}/approve       pending → approved    [Admin]
POST /alerts/{id}/dismiss       pending → dismissed   [Admin]
POST /alerts/{id}/notified      approved → notified   [Agent]
POST /alerts/{id}/reopen        any → pending         [Admin]
POST /alerts/bulk/approve       Bulk approve          [Admin]
POST /alerts/bulk/dismiss       Bulk dismiss          [Admin]
```

---

## Tests

```bash
pytest tests/ -v                              # all tests
pytest tests/test_api.py -v                   # API endpoints
pytest tests/test_monitor_agent.py -v         # Monitor unit tests
pytest tests/test_spec_and_cooldown.py -v     # Spec + cooldown
pytest tests/ --cov=api --cov=agents          # with coverage
```

No real Jira or Postgres needed — tests use in-memory SQLite and mocked HTTP.

---

## Migrations

```bash
alembic upgrade head          # apply all pending
alembic current               # show version
alembic downgrade -1          # roll back one step
alembic revision --autogenerate -m "add column"  # generate from model changes
```

---

## Spec Interpreter

```bash
# Preview tasks without creating in Jira
python -m agents.spec_interpreter --file docs/prd.md --dry-run

# Create tasks
python -m agents.spec_interpreter --file docs/prd.md

# Programmatic
from agents.spec_interpreter import SpecInterpreterAgent
result = SpecInterpreterAgent().run_from_file("prd.md")
print(result["issue_keys"])   # ['ENG-101', 'ENG-102', ...]
```

---

## Project Structure

```
aegis-pm/
├── agents/
│   ├── monitor_agent.py        Jira poll + tenacity + cooldown
│   ├── communicator_agent.py   Slack Block Kit
│   ├── spec_interpreter.py     PRD → Jira tasks
│   ├── group_chat.py           AutoGen GroupChat orchestrator
│   ├── runner.py               APScheduler entry point
│   └── health/monitor.py       Health tracking + dead-man Slack alert
├── api/
│   ├── main.py                 FastAPI (all endpoints + state machine)
│   └── security.py             API key auth + rate limiting
├── migrations/
│   ├── env.py                  Alembic env (reads .env)
│   └── versions/001_baseline.py
├── db/init.sql                 Schema bootstrap
├── frontend/index.html         HITL Dashboard
├── nginx/                      nginx config + Basic Auth
├── tests/                      pytest suite (no real infra needed)
├── scripts/
│   ├── gen_htpasswd.sh         Generate dashboard credentials
│   └── test_api.sh             Shell smoke tests
├── docker-compose.yml
├── docker-compose.override.yml Dev overrides
├── Dockerfile.api              Multi-stage (dev + prod)
├── Dockerfile.agents
├── alembic.ini
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Security Checklist

- [ ] `AEGIS_ENFORCE_AUTH=true` in production
- [ ] Unique 64-char `AGENT_API_KEY` and `ADMIN_API_KEY`
- [ ] `nginx/.htpasswd` generated (never committed — in `.gitignore`)
- [ ] `CORS_ORIGINS` set to your actual domain
- [ ] `.env` not committed (in `.gitignore`)
- [ ] Strong `POSTGRES_PASSWORD`
