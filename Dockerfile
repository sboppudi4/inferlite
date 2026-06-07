FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests

RUN python3 -m pip install --upgrade pip && python3 -m pip install -e ".[dev]"

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "inferlite.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
