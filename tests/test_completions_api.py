from __future__ import annotations

from fastapi.testclient import TestClient

import inferlite.api.app as app_module
from inferlite.engine.model_runner import BatchGenerationResult
from inferlite.engine.naive_runner import GenerationResult


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
            texts=[f" -> out:{p}" for p in prompts],
            prompt_tokens=10,
            completion_tokens=6,
        )


class _FakeNaiveRunner:
    async def generate(
        self,
        *,
        model_name: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerationResult:
        _ = (model_name, max_new_tokens, temperature, top_p)
        return GenerationResult(text=f" -> naive:{prompt}", prompt_tokens=3, completion_tokens=2)


def _create_key(client: TestClient) -> str:
    response = client.post(
        "/admin/keys",
        headers={"x-admin-secret": "inferlite-admin"},
        json={"tier": "paid", "requests_per_minute": 500},
    )
    assert response.status_code == 200
    return response.json()["api_key"]


def test_completions_supports_static_batching() -> None:
    app_module.runner = _FakeBatchRunner()
    client = TestClient(app_module.app)
    key = _create_key(client)
    response = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": "sshleifer/tiny-gpt2",
            "prompt": ["hello", "world"],
            "max_tokens": 8,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["choices"]) == 2
    assert body["choices"][0]["text"] == " -> out:hello"
    assert body["choices"][1]["text"] == " -> out:world"


def test_completions_rejects_empty_prompt() -> None:
    client = TestClient(app_module.app)
    key = _create_key(client)
    response = client.post(
        "/v1/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "sshleifer/tiny-gpt2", "prompt": "   ", "max_tokens": 8},
    )
    assert response.status_code == 400


def test_baseline_endpoint_works() -> None:
    app_module.naive_runner = _FakeNaiveRunner()
    client = TestClient(app_module.app)
    key = _create_key(client)
    response = client.post(
        "/v1/completions/baseline",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "sshleifer/tiny-gpt2", "prompt": "hello", "max_tokens": 8},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"] == " -> naive:hello"
