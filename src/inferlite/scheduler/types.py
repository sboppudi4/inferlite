from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Priority(IntEnum):
    BATCH = 0
    FREE = 1
    PAID = 2


@dataclass
class GenerationRequest:
    request_id: str
    prompt: str
    max_new_tokens: int
    priority: Priority = Priority.FREE
    arrived_at: float = 0.0
    generated_tokens: int = 0
    finished: bool = False
    # Remaining decode budget for this request.
    remaining_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.remaining_tokens = self.max_new_tokens


@dataclass
class ScheduledBatch:
    step_id: int
    request_ids: list[str]
    batch_size: int

