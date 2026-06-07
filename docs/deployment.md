# Deployment Guide

This guide covers practical deployment paths for InferLite, from a zero-cost CPU demo to a
single-GPU benchmark host.

## Option 0: CPU demo (no GPU, free)

InferLite defaults to `gpt2`, which runs comfortably on CPU. This is the fastest way to see the API,
auth, and metrics working end to end — good enough for a portfolio demo, not for throughput numbers.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn inferlite.api.app:app --host 0.0.0.0 --port 8000
```

```bash
# mint a key, then call the OpenAI-compatible endpoint
KEY=$(curl -s -X POST http://localhost:8000/admin/keys \
  -H "x-admin-secret: inferlite-admin" -H "Content-Type: application/json" \
  -d '{"tier":"paid","requests_per_minute":500}' | python -c "import sys,json;print(json.load(sys.stdin)['api_key'])")

curl -s -X POST http://localhost:8000/v1/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"gpt2","prompt":"Continuous batching means","max_tokens":32}'

curl -s http://localhost:8000/metrics | head
```

### Hosting the CPU demo for free

The same process runs on free/low-cost tiers. Two easy paths:

- **Render / Railway / Fly.io (Docker):** point the platform at this repo's `Dockerfile` (swap the
  CUDA base for `python:3.11-slim` on CPU-only tiers), expose port `8000`, and set
  `INFERLITE_ADMIN_BOOTSTRAP_SECRET` to a real secret. The `/healthz` endpoint works as the health
  check.
- **Hugging Face Spaces (Docker SDK):** add a `Dockerfile`-based Space, expose `8000`, keep the
  model at `gpt2` (or `distilgpt2` for an even smaller footprint).

Keep the default model small on free tiers — a 7B model will not fit. Use a GPU host (below) only
when you actually want throughput numbers.

---

## Single-GPU deployment

The options below target reproducible benchmarking on a single GPU.

## Minimum deployment target

- 1x NVIDIA GPU (24 GB+ preferred for 7B-class models)
- Python 3.11+
- CUDA runtime compatible with installed `torch`

## Option A: Docker on a GPU VM

1. Start a GPU VM (AWS/GCP/Azure/RunPod/any provider).
2. Install Docker + NVIDIA Container Toolkit.
3. Clone InferLite and run:

```bash
docker compose up --build
```

4. Health check:

```bash
curl http://<host>:8000/healthz
```

## Option B: Process mode (no container)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn inferlite.api.app:app --host 0.0.0.0 --port 8000
```

## Tenant/bootstrap setup

Create an initial API key:

```bash
curl -X POST http://<host>:8000/admin/keys \
  -H "x-admin-secret: inferlite-admin" \
  -H "Content-Type: application/json" \
  -d '{"tier":"paid","requests_per_minute":500}'
```

## Metrics/Grafana

- Prometheus scrape endpoint: `GET /metrics`
- Starter dashboard JSON: `deploy/grafana/dashboard.json`

## Production hardening checklist

- Replace bootstrap admin secret with environment-backed secret management.
- Put API behind TLS + gateway (Nginx/Envoy/API Gateway).
- Enforce request body limits and auth audit logging.
- Add persistent Postgres key store if running multi-instance.
- Add model load warmup and startup probes.
- Pin model revisions for benchmark reproducibility.

## Cost/scope note

InferLite is single-GPU first by design. Keep deployment small and stable for reproducible
benchmarking before considering multi-GPU extensions.
