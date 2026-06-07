from __future__ import annotations

from dataclasses import dataclass

from inferlite.scheduler import Priority


@dataclass(frozen=True)
class APIKeyRecord:
    key_id: str
    api_key: str
    tier: str
    requests_per_minute: int
    enabled: bool

    @property
    def priority(self) -> Priority:
        if self.tier == "paid":
            return Priority.PAID
        if self.tier == "batch":
            return Priority.BATCH
        return Priority.FREE

