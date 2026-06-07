from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str = Field(default="gpt2")
    prompt: str | list[str]
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage

    @classmethod
    def from_generations(
        cls, model: str, generated: list[str], prompt_tokens: int, completion_tokens: int
    ) -> "CompletionResponse":
        return cls(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model,
            choices=[CompletionChoice(text=text, index=i) for i, text in enumerate(generated)],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


class HealthResponse(BaseModel):
    status: str = "ok"
    details: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt2")
    messages: list[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False


class AdminCreateKeyRequest(BaseModel):
    tier: str = Field(default="free")
    requests_per_minute: int = Field(default=30, ge=1, le=10_000)


class AdminKeyResponse(BaseModel):
    key_id: str
    api_key: str
    tier: str
    requests_per_minute: int
    enabled: bool
