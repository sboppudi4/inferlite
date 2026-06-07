from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkloadConfig:
    name: str
    duration_s: int
    qps: float
    model: str
    prompt_min: int
    prompt_max: int
    output_min: int
    output_max: int
    paid_ratio: float
    free_ratio: float
    batch_ratio: float
    burst_enabled: bool = False
    burst_every_s: int = 0
    burst_qps: float = 0.0
    burst_duration_s: int = 0


def load_config(path: str) -> WorkloadConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    burst = data.get("burst", {})
    return WorkloadConfig(
        name=str(data["name"]),
        duration_s=int(data["duration_s"]),
        qps=float(data["qps"]),
        model=str(data["model"]),
        prompt_min=int(data["prompt_length_tokens"]["min"]),
        prompt_max=int(data["prompt_length_tokens"]["max"]),
        output_min=int(data["output_length_tokens"]["min"]),
        output_max=int(data["output_length_tokens"]["max"]),
        paid_ratio=float(data["priority_mix"]["paid"]),
        free_ratio=float(data["priority_mix"]["free"]),
        batch_ratio=float(data["priority_mix"]["batch"]),
        burst_enabled=bool(burst.get("enabled", False)),
        burst_every_s=int(burst.get("burst_every_s", 0)),
        burst_qps=float(burst.get("burst_qps", 0.0)),
        burst_duration_s=int(burst.get("burst_duration_s", 0)),
    )


def _choose_tier(cfg: WorkloadConfig) -> str:
    x = random.random()
    if x < cfg.paid_ratio:
        return "paid"
    if x < cfg.paid_ratio + cfg.free_ratio:
        return "free"
    return "batch"


def _prompt_from_len(token_len: int) -> str:
    # Rough token approximation for repeatable synthetic load.
    return " ".join(["inferlite"] * token_len)


def generate_requests(cfg: WorkloadConfig, seed: int = 7) -> list[dict[str, object]]:
    random.seed(seed)
    requests: list[dict[str, object]] = []
    start = time.time()
    t = 0.0
    request_idx = 0
    while t < cfg.duration_s:
        second = int(t)
        effective_qps = cfg.qps
        if (
            cfg.burst_enabled
            and cfg.burst_every_s > 0
            and second % cfg.burst_every_s < cfg.burst_duration_s
        ):
            effective_qps = cfg.burst_qps

        # Generate N requests for this second using deterministic interval spacing.
        n = max(1, int(round(effective_qps)))
        for j in range(n):
            prompt_len = random.randint(cfg.prompt_min, cfg.prompt_max)
            out_len = random.randint(cfg.output_min, cfg.output_max)
            requests.append(
                {
                    "request_id": f"req-{request_idx}",
                    "arrival_offset_s": t + (j / n),
                    "model": cfg.model,
                    "prompt": _prompt_from_len(prompt_len),
                    "max_tokens": out_len,
                    "tier": _choose_tier(cfg),
                    "created_at_unix": start,
                }
            )
            request_idx += 1
        t += 1.0
    return requests

