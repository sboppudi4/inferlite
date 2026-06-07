from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class BatchGenerationResult:
    texts: list[str]
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _ModelBundle:
    model: Any
    tokenizer: Any
    device: Any


class HuggingFaceModelRunner:
    """Phase 1 model runner with static request batching."""

    def __init__(self) -> None:
        self._models: dict[str, _ModelBundle] = {}
        self._lock = asyncio.Lock()

    async def _load_bundle(self, model_name: str) -> _ModelBundle:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = await asyncio.to_thread(AutoModelForCausalLM.from_pretrained, model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        return _ModelBundle(model=model, tokenizer=tokenizer, device=device)

    async def _get_bundle(self, model_name: str) -> _ModelBundle:
        if model_name in self._models:
            return self._models[model_name]
        async with self._lock:
            if model_name in self._models:
                return self._models[model_name]
            bundle = await self._load_bundle(model_name)
            self._models[model_name] = bundle
            return bundle

    async def generate_batch(
        self,
        *,
        model_name: str,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> BatchGenerationResult:
        import torch

        bundle = await self._get_bundle(model_name)
        tokenizer = bundle.tokenizer
        model = bundle.model
        device = bundle.device

        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        def _run_generate() -> torch.Tensor:
            with torch.inference_mode():
                return model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        generated_ids = await asyncio.to_thread(_run_generate)

        texts: list[str] = []
        prompt_token_total = 0
        completion_token_total = 0
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        for i, prompt_len in enumerate(prompt_lengths):
            output_ids = generated_ids[i]
            new_token_ids = output_ids[int(prompt_len) :]
            text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
            texts.append(text)
            prompt_token_total += int(prompt_len)
            completion_token_total += int(new_token_ids.shape[0])

        return BatchGenerationResult(
            texts=texts,
            prompt_tokens=prompt_token_total,
            completion_tokens=completion_token_total,
        )
