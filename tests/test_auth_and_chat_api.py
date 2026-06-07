from __future__ import annotations

from fastapi.testclient import TestClient

import inferlite.api.app as app_module
from inferlite.engine.model_runner import BatchGenerationResult


class _FakeBatchRunner:
    async def generate_batch(
        self,
        *,
        model_name: str,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> BatchGenerationResult:
        _ = (model_name, max_new_tokens, temperature, top_p)
        return BatchGenerationResult(
            texts=[f"generated:{prompts[0]}"],
            prompt_tokens=4,
            completion_tokens=3,
        )


def _create_key(client: TestClient, tier: str = "free", rpm: int = 100) -> str:
    response = client.post(
        "/admin/keys",
        headers={"x-admin-secret": "inferlite-admin"},
        json={"tier": tier, "requests_per_minute": rpm},
    )
    assert response.status_code == 200
    return response.json()["api_key"]


def test_requires_api_key() -> None:
    client = TestClient(app_module.app)
    response = client.post(
        "/v1/completions",
        json={"model": "gpt2", "prompt": "hi", "max_tokens": 4},
    )
    assert response.status_code == 401


def test_chat_completion_with_api_key() -> None:
    app_module.runner = _FakeBatchRunner()
    client = TestClient(app_module.app)
    key = _create_key(client, tier="paid")
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "gpt2",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
    )
    assert response.status_code == 200
    assert "choices" in response.json()


def test_rate_limit() -> None:
    app_module.runner = _FakeBatchRunner()
    client = TestClient(app_module.app)
    key = _create_key(client, tier="free", rpm=1)
    payload = {"model": "gpt2", "prompt": "hello", "max_tokens": 4}
    first = client.post("/v1/completions", headers={"Authorization": f"Bearer {key}"}, json=payload)
    second = client.post("/v1/completions", headers={"Authorization": f"Bearer {key}"}, json=payload)
    assert first.status_code == 200
    assert second.status_code == 429

