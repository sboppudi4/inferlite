from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class NaivePipelineRunner:
    """Phase 0 baseline runner.

    This intentionally uses Hugging Face pipeline() and generates one request at a time.
    It is our "before" system for measuring scheduler/cache improvements in later phases.
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def _get_pipeline(self, model_name: str) -> Any:
        if model_name in self._pipelines:
            return self._pipelines[model_name]
        async with self._lock:
            if model_name in self._pipelines:
                return self._pipelines[model_name]
            from transformers import pipeline

            pipe = await asyncio.to_thread(
                pipeline,
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device_map="auto",
            )
            self._pipelines[model_name] = pipe
            return pipe

    async def generate(
        self,
        *,
        model_name: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerationResult:
        pipe = await self._get_pipeline(model_name)
        outputs = await asyncio.to_thread(
            pipe,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            do_sample=temperature > 0.0,
            return_full_text=True,
        )
        generated_full = str(outputs[0]["generated_text"])
        generated_suffix = generated_full[len(prompt) :]

        tokenizer = pipe.tokenizer
        prompt_tokens = len(tokenizer(prompt)["input_ids"])
        full_tokens = len(tokenizer(generated_full)["input_ids"])
        completion_tokens = max(0, full_tokens - prompt_tokens)
        return GenerationResult(
            text=generated_suffix,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
