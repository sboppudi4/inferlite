# Deployment Guide

This guide covers practical deployment paths for InferLite on a single GPU.

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
