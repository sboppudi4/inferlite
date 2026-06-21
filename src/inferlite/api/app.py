from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from inferlite.api.schemas import (
    AdminCreateKeyRequest,
    AdminKeyResponse,
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
)
from inferlite.auth import (
    APIKeyRecord,
    APIKeyStore,
    AuthService,
    PerKeyRateLimiter,
    auth_dependency,
)
from inferlite.config import settings
from inferlite.engine.model_runner import HuggingFaceModelRunner
from inferlite.engine.naive_runner import NaivePipelineRunner
from inferlite.observability.logging import configure_logging
from inferlite.observability.metrics import (
    BATCH_SIZE,
    DECODE_TOKENS_TOTAL,
    KV_CACHE_UTILIZATION,
    PREFILL_TOKENS_TOTAL,
    QUEUE_DEPTH,
    REQUEST_LATENCY_SECONDS,
    TPOT_SECONDS,
    TTFT_SECONDS,
)

configure_logging(settings.log_level)
logger = structlog.get_logger("inferlite.api")

app = FastAPI(title="InferLite", version="0.1.0")
runner = HuggingFaceModelRunner()
naive_runner = NaivePipelineRunner()
auth_service = AuthService(APIKeyStore(settings.sqlite_path), PerKeyRateLimiter())
require_api_key = auth_dependency(auth_service)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(details={"phase": "5", "runner": "hf-static-batching+metrics"})


@app.get("/metrics")
async def metrics() -> StreamingResponse:
    # Placeholders until scheduler/cache wiring in later phases updates them live.
    QUEUE_DEPTH.set(0)
    KV_CACHE_UTILIZATION.set(0.0)
    return StreamingResponse(
        iter([generate_latest()]),
        media_type=CONTENT_TYPE_LATEST,
    )


def _normalize_prompts(prompt: str | list[str]) -> list[str]:
    prompts = [prompt] if isinstance(prompt, str) else prompt
    non_empty = [p for p in prompts if p.strip()]
    if not non_empty:
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    return non_empty


def _chat_to_prompt(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for msg in messages:
        if msg.role and msg.content:
            parts.append(f"{msg.role}: {msg.content}")
    parts.append("assistant:")
    return "\n".join(parts)


async def _stream_text_completion(
    *,
    model: str,
    text: str,
    endpoint: str,
    tier: str,
    started_at: float,
) -> AsyncGenerator[str, None]:
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    tokens = text.split()
    first = True
    for token in tokens:
        now = time.perf_counter()
        if first:
            TTFT_SECONDS.labels(endpoint=endpoint, tier=tier).observe(max(0.0, now - started_at))
            first = False
        payload = {
            "id": completion_id,
            "object": "text_completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": token + " ", "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload)}\n\n"
    total = max(0.0, time.perf_counter() - started_at)
    out_tokens = max(1, len(tokens))
    TPOT_SECONDS.labels(endpoint=endpoint, tier=tier).observe(total / out_tokens)
    yield "data: [DONE]\n\n"


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(
    payload: CompletionRequest, tenant: APIKeyRecord = Depends(require_api_key)
) -> CompletionResponse | StreamingResponse:
    started = time.perf_counter()
    trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    prompts = _normalize_prompts(payload.prompt)

    logger.info(
        "completion.request",
        trace_id=trace_id,
        tenant_id=tenant.key_id,
        tenant_tier=tenant.tier,
        request_priority=int(tenant.priority),
        model=payload.model,
        batch_size=len(prompts),
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
    )
    BATCH_SIZE.labels(endpoint="completions").observe(len(prompts))
    result = await runner.generate_batch(
        model_name=payload.model,
        prompts=prompts,
        max_new_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
    )
    PREFILL_TOKENS_TOTAL.inc(result.prompt_tokens)
    DECODE_TOKENS_TOTAL.inc(result.completion_tokens)
    elapsed = max(0.0, time.perf_counter() - started)
    REQUEST_LATENCY_SECONDS.labels(endpoint="completions", tier=tenant.tier).observe(elapsed)
    TTFT_SECONDS.labels(endpoint="completions", tier=tenant.tier).observe(elapsed)
    out_tokens = max(1, result.completion_tokens)
    TPOT_SECONDS.labels(endpoint="completions", tier=tenant.tier).observe(elapsed / out_tokens)

    if payload.stream:
        return StreamingResponse(
            _stream_text_completion(
                model=payload.model,
                text=result.texts[0],
                endpoint="completions",
                tier=tenant.tier,
                started_at=started,
            ),
            media_type="text/event-stream",
        )
    return CompletionResponse.from_generations(
        model=payload.model,
        generated=result.texts,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


@app.post("/v1/completions/baseline", response_model=CompletionResponse)
async def baseline_completions(
    payload: CompletionRequest, tenant: APIKeyRecord = Depends(require_api_key)
) -> CompletionResponse:
    _ = tenant
    prompts = _normalize_prompts(payload.prompt)
    first_prompt = prompts[0]
    result = await naive_runner.generate(
        model_name=payload.model,
        prompt=first_prompt,
        max_new_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
    )
    return CompletionResponse.from_generations(
        model=payload.model,
        generated=[result.text],
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


@app.post("/v1/chat/completions", response_model=CompletionResponse)
async def chat_completions(
    payload: ChatCompletionRequest, tenant: APIKeyRecord = Depends(require_api_key)
) -> CompletionResponse | StreamingResponse:
    started = time.perf_counter()
    trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    prompt = _chat_to_prompt(payload.messages)
    BATCH_SIZE.labels(endpoint="chat_completions").observe(1)
    result = await runner.generate_batch(
        model_name=payload.model,
        prompts=[prompt],
        max_new_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
    )
    logger.info(
        "chat.request",
        trace_id=trace_id,
        tenant_id=tenant.key_id,
        tenant_tier=tenant.tier,
        request_priority=int(tenant.priority),
        model=payload.model,
        max_tokens=payload.max_tokens,
    )
    PREFILL_TOKENS_TOTAL.inc(result.prompt_tokens)
    DECODE_TOKENS_TOTAL.inc(result.completion_tokens)
    elapsed = max(0.0, time.perf_counter() - started)
    REQUEST_LATENCY_SECONDS.labels(endpoint="chat_completions", tier=tenant.tier).observe(elapsed)
    TTFT_SECONDS.labels(endpoint="chat_completions", tier=tenant.tier).observe(elapsed)
    out_tokens = max(1, result.completion_tokens)
    TPOT_SECONDS.labels(endpoint="chat_completions", tier=tenant.tier).observe(elapsed / out_tokens)
    if payload.stream:
        return StreamingResponse(
            _stream_text_completion(
                model=payload.model,
                text=result.texts[0],
                endpoint="chat_completions",
                tier=tenant.tier,
                started_at=started,
            ),
            media_type="text/event-stream",
        )
    return CompletionResponse.from_generations(
        model=payload.model,
        generated=result.texts,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


@app.post("/admin/keys", response_model=AdminKeyResponse)
async def admin_create_key(
    payload: AdminCreateKeyRequest,
    x_admin_secret: str | None = Header(default=None),
) -> AdminKeyResponse:
    if x_admin_secret != settings.admin_bootstrap_secret:
        raise HTTPException(status_code=401, detail="invalid admin secret")
    record = auth_service.store.create_key(
        tier=payload.tier, requests_per_minute=payload.requests_per_minute
    )
    return AdminKeyResponse(
        key_id=record.key_id,
        api_key=record.api_key,
        tier=record.tier,
        requests_per_minute=record.requests_per_minute,
        enabled=record.enabled,
    )


@app.get("/admin/keys", response_model=list[AdminKeyResponse])
async def admin_list_keys(
    x_admin_secret: str | None = Header(default=None),
) -> list[AdminKeyResponse]:
    if x_admin_secret != settings.admin_bootstrap_secret:
        raise HTTPException(status_code=401, detail="invalid admin secret")
    return [
        AdminKeyResponse(
            key_id=r.key_id,
            api_key=r.api_key,
            tier=r.tier,
            requests_per_minute=r.requests_per_minute,
            enabled=r.enabled,
        )
        for r in auth_service.store.list_keys()
    ]
