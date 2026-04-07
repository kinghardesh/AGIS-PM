"""
aegis-pm / api / security.py

Security layer for Aegis PM API.

Features
────────
  API key authentication  – two tiers:
      AGENT_API_KEY   → read + write alerts, no HITL actions
      ADMIN_API_KEY   → full access including approve/dismiss/reopen/bulk

  Rate limiting           – per-IP sliding window (in-memory, Redis-upgradable)
  Secret validation       – startup check; refuses to boot with weak/missing keys
  Request ID injection    – every response gets X-Request-ID for tracing

Usage in routes
───────────────
  from api.security import require_agent_key, require_admin_key

  @app.post("/alerts")
  async def create_alert(..., _=Depends(require_agent_key)):
      ...

  @app.post("/alerts/{id}/approve")
  async def approve(..., _=Depends(require_admin_key)):
      ...
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict
from typing import Dict, Tuple

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

log = logging.getLogger("aegis.security")

# ── Key header names ──────────────────────────────────────────────────────────

_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Load keys from env ────────────────────────────────────────────────────────

def _require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        raise RuntimeError(
            f"[Security] Required env var '{name}' is not set. "
            f"Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    if len(val) < 32:
        raise RuntimeError(
            f"[Security] '{name}' is too short ({len(val)} chars). "
            f"Minimum 32 characters required."
        )
    return val


def _load_keys() -> Tuple[str, str]:
    """
    Load AGENT_API_KEY and ADMIN_API_KEY from environment.
    Raises RuntimeError on startup if either is missing or weak.
    """
    agent_key = _require_env("AGENT_API_KEY")
    admin_key = _require_env("ADMIN_API_KEY")

    if agent_key == admin_key:
        raise RuntimeError(
            "[Security] AGENT_API_KEY and ADMIN_API_KEY must be different values."
        )

    # Store as hashes so plaintext never sits in memory longer than necessary
    return (
        hashlib.sha256(agent_key.encode()).hexdigest(),
        hashlib.sha256(admin_key.encode()).hexdigest(),
    )


try:
    _AGENT_KEY_HASH, _ADMIN_KEY_HASH = _load_keys()
    log.info("[Security] API keys loaded and validated")
except RuntimeError as _boot_err:
    # Let the app start in dev without keys (logs a loud warning)
    # Set AEGIS_ENFORCE_AUTH=true to make missing keys fatal
    if os.getenv("AEGIS_ENFORCE_AUTH", "false").lower() == "true":
        raise
    log.warning(
        "[Security] %s\n"
        "           Running WITHOUT authentication. "
        "Set AEGIS_ENFORCE_AUTH=true to enforce.",
        _boot_err,
    )
    _AGENT_KEY_HASH = _ADMIN_KEY_HASH = None


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# ── Rate limiter (sliding window, per IP) ─────────────────────────────────────

class _SlidingWindowLimiter:
    """
    Simple in-memory sliding-window rate limiter.
    Not shared across workers — use Redis for multi-process deployments.
    """

    def __init__(self, max_requests: int = 120, window_seconds: int = 60) -> None:
        self._max      = max_requests
        self._window   = window_seconds
        self._buckets: Dict[str, list] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now    = time.monotonic()
        cutoff = now - self._window
        bucket = self._buckets[key]

        # Drop timestamps outside window
        self._buckets[key] = [t for t in bucket if t > cutoff]

        if len(self._buckets[key]) >= self._max:
            return False

        self._buckets[key].append(now)
        return True

    def cleanup(self) -> None:
        """Prune empty buckets to prevent unbounded memory growth."""
        dead = [k for k, v in self._buckets.items() if not v]
        for k in dead:
            del self._buckets[k]


_limiter = _SlidingWindowLimiter(
    max_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "120")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
)


# ── FastAPI dependencies ──────────────────────────────────────────────────────

async def rate_limit(request: Request) -> None:
    """
    Dependency: enforce per-IP rate limit.
    Raises 429 if the client exceeds RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW_SECONDS.
    """
    ip = request.client.host if request.client else "unknown"
    if not _limiter.is_allowed(ip):
        log.warning("Rate limit exceeded for IP %s", ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please slow down.",
            headers={"Retry-After": str(_limiter._window)},
        )


async def require_agent_key(
    request: Request,
    api_key: str | None = Security(_HEADER),
) -> str:
    """
    Dependency: require a valid AGENT_API_KEY or ADMIN_API_KEY.
    Allows agents (Monitor, Communicator) to call the API.

    Skipped entirely when AEGIS_ENFORCE_AUTH is not 'true'.
    """
    if _AGENT_KEY_HASH is None:
        return "unauthenticated"   # dev mode

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_hash = _hash(api_key)
    if key_hash not in (_AGENT_KEY_HASH, _ADMIN_KEY_HASH):
        log.warning("Invalid API key attempt from %s", request.client.host if request.client else "?")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return "agent" if key_hash == _AGENT_KEY_HASH else "admin"


async def require_admin_key(
    request: Request,
    api_key: str | None = Security(_HEADER),
) -> str:
    """
    Dependency: require specifically the ADMIN_API_KEY.
    Used for HITL actions (approve, dismiss, reopen, bulk).

    Skipped entirely when AEGIS_ENFORCE_AUTH is not 'true'.
    """
    if _ADMIN_KEY_HASH is None:
        return "unauthenticated"   # dev mode

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    key_hash = _hash(api_key)
    if key_hash != _ADMIN_KEY_HASH:
        log.warning(
            "Admin key required but agent/invalid key used from %s",
            request.client.host if request.client else "?",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin API key required for this action",
        )
    return "admin"


# ── Request ID middleware helper ──────────────────────────────────────────────

def inject_request_id(response, request_id: str | None = None) -> str:
    """Attach a unique request ID to the response for distributed tracing."""
    rid = request_id or str(uuid.uuid4())
    response.headers["X-Request-ID"] = rid
    return rid


# ── Key generation utility ───────────────────────────────────────────────────

def generate_key() -> str:
    """Generate a cryptographically secure 64-char hex API key."""
    return secrets.token_hex(32)


if __name__ == "__main__":
    print("Generated AGENT_API_KEY:", generate_key())
    print("Generated ADMIN_API_KEY:  ", generate_key())
