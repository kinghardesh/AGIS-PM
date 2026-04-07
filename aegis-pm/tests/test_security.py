"""
aegis-pm / tests / test_security.py

Tests for the security module.
"""
from __future__ import annotations

import os
import pytest

pytestmark = pytest.mark.asyncio


class TestKeyValidation:

    def test_generate_key_is_64_chars(self):
        from api.security import generate_key
        key = generate_key()
        assert len(key) == 64
        assert key.isalnum() or all(c in "0123456789abcdef" for c in key)

    def test_hashing_is_deterministic(self):
        from api.security import _hash
        assert _hash("same_key") == _hash("same_key")

    def test_different_keys_have_different_hashes(self):
        from api.security import _hash
        assert _hash("key_a") != _hash("key_b")


class TestRateLimiter:

    def test_allows_requests_under_limit(self):
        from api.security import _SlidingWindowLimiter
        limiter = _SlidingWindowLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.is_allowed("192.168.1.1") is True

    def test_blocks_requests_over_limit(self):
        from api.security import _SlidingWindowLimiter
        limiter = _SlidingWindowLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.is_allowed("10.0.0.1")
        # 4th request should be blocked
        assert limiter.is_allowed("10.0.0.1") is False

    def test_different_ips_have_separate_buckets(self):
        from api.security import _SlidingWindowLimiter
        limiter = _SlidingWindowLimiter(max_requests=2, window_seconds=60)
        limiter.is_allowed("1.1.1.1")
        limiter.is_allowed("1.1.1.1")
        # 1.1.1.1 is exhausted, 2.2.2.2 should still pass
        assert limiter.is_allowed("1.1.1.1") is False
        assert limiter.is_allowed("2.2.2.2") is True


class TestAuthEndpoints:

    async def test_rate_limit_raises_429(self, client, agent_headers):
        """Simulate many rapid requests to trigger rate limiting."""
        from api.security import _limiter

        # Temporarily lower the limit for testing
        original_max = _limiter._max
        _limiter._max = 3
        try:
            for _ in range(3):
                await client.get("/health")
            # 4th request should 429
            res = await client.get("/health")
            assert res.status_code == 429
        finally:
            _limiter._max = original_max
            # Reset bucket
            ip = "testclient"
            _limiter._buckets.pop(ip, None)

    async def test_request_id_header_in_response(self, client, agent_headers):
        res = await client.get("/health")
        assert "x-request-id" in res.headers or "X-Request-ID" in res.headers
