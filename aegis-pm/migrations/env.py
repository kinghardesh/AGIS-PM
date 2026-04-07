"""
aegis-pm / migrations / env.py

Alembic migration environment.
Reads DB connection from the same environment variables as the app,
so you never have to maintain a separate DB URL for migrations.

Usage
─────
  # Apply all pending migrations
  alembic upgrade head

  # Roll back one migration
  alembic downgrade -1

  # Auto-generate a new migration from SQLAlchemy model changes
  alembic revision --autogenerate -m "add snooze_until to alerts"

  # Show current migration version
  alembic current

  # Show pending migrations
  alembic history --verbose
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

load_dotenv()

# ── Alembic Config object ─────────────────────────────────────────────────────

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── Build DSN from env vars (sync psycopg2 URL for Alembic) ──────────────────

def _dsn() -> str:
    return (
        "postgresql+psycopg2://"
        f"{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ['POSTGRES_DB']}"
    )


config.set_main_option("sqlalchemy.url", _dsn())

# ── Import target metadata so autogenerate can diff against it ────────────────
# We import the SQLAlchemy metadata from the FastAPI app so Alembic knows
# exactly what tables and columns should exist.

from api.main import metadata as target_metadata   # noqa: E402

# ── Run migrations ────────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations without a live DB connection (generates SQL scripts).
    Useful for reviewing changes before applying.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations against a live DB connection.
    Used by `alembic upgrade head`.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
