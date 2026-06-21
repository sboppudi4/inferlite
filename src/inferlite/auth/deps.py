from __future__ import annotations

from collections.abc import Callable

from fastapi import Header, HTTPException

from inferlite.auth.models import APIKeyRecord
from inferlite.auth.rate_limit import PerKeyRateLimiter
from inferlite.auth.store import APIKeyStore


class AuthService:
    def __init__(self, store: APIKeyStore, limiter: PerKeyRateLimiter) -> None:
        self.store = store
        self.limiter = limiter

    def authorize(self, authorization: str | None) -> APIKeyRecord:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        api_key = authorization.removeprefix("Bearer ").strip()
        record = self.store.get_by_key(api_key)
        if record is None or not record.enabled:
            raise HTTPException(status_code=401, detail="invalid API key")
        if not self.limiter.allow(key_id=record.key_id, rpm=record.requests_per_minute):
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        return record


def auth_dependency(service: AuthService) -> Callable[[str | None], APIKeyRecord]:
    def _dep(authorization: str | None = Header(default=None)) -> APIKeyRecord:
        return service.authorize(authorization)

    return _dep

