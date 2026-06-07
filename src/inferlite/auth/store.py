from __future__ import annotations

import secrets
import sqlite3
import uuid
from pathlib import Path

from inferlite.auth.models import APIKeyRecord


class APIKeyStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    api_key TEXT UNIQUE NOT NULL,
                    tier TEXT NOT NULL,
                    requests_per_minute INTEGER NOT NULL,
                    enabled INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    def create_key(self, tier: str, requests_per_minute: int) -> APIKeyRecord:
        key_id = f"key_{uuid.uuid4().hex[:10]}"
        api_key = f"il-{secrets.token_urlsafe(24)}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (key_id, api_key, tier, requests_per_minute, enabled)
                VALUES (?, ?, ?, ?, 1)
                """,
                (key_id, api_key, tier, requests_per_minute),
            )
            conn.commit()
        return APIKeyRecord(
            key_id=key_id,
            api_key=api_key,
            tier=tier,
            requests_per_minute=requests_per_minute,
            enabled=True,
        )

    def get_by_key(self, api_key: str) -> APIKeyRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT key_id, api_key, tier, requests_per_minute, enabled
                FROM api_keys WHERE api_key = ?
                """,
                (api_key,),
            ).fetchone()
        if row is None:
            return None
        return APIKeyRecord(
            key_id=str(row[0]),
            api_key=str(row[1]),
            tier=str(row[2]),
            requests_per_minute=int(row[3]),
            enabled=bool(int(row[4])),
        )

    def list_keys(self) -> list[APIKeyRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key_id, api_key, tier, requests_per_minute, enabled FROM api_keys"
            ).fetchall()
        return [
            APIKeyRecord(
                key_id=str(r[0]),
                api_key=str(r[1]),
                tier=str(r[2]),
                requests_per_minute=int(r[3]),
                enabled=bool(int(r[4])),
            )
            for r in rows
        ]

