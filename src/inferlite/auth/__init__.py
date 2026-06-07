"""Multi-tenant auth and policy module (Phase 4)."""

from inferlite.auth.deps import AuthService, auth_dependency
from inferlite.auth.models import APIKeyRecord
from inferlite.auth.rate_limit import PerKeyRateLimiter
from inferlite.auth.store import APIKeyStore

__all__ = ["AuthService", "auth_dependency", "APIKeyRecord", "PerKeyRateLimiter", "APIKeyStore"]
