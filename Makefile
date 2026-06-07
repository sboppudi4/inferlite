# InferLite convenience targets.
# On Windows without `make`, run the underlying commands directly (shown in each recipe).

.PHONY: help install test lint type check serve bench-kv bench-cpu bench report docker

help:
	@echo "install   - editable install with dev extras"
	@echo "test      - run pytest with coverage"
	@echo "lint      - ruff lint"
	@echo "type      - mypy type check"
	@echo "check     - lint + type + test"
	@echo "serve     - run the API on :8000"
	@echo "bench-kv  - KV-cache fragmentation benchmark (no GPU required)"
	@echo "bench-cpu - inferlite vs naive on distilgpt2, CPU (needs running server + API_KEY=...)"
	@echo "bench     - full inferlite/naive/vllm matrix (needs API_KEY=...)"
	@echo "report    - turn benchmark CSV into charts + markdown"
	@echo "docker    - build and run via docker compose"

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests benchmarks

type:
	mypy src

check: lint type test

serve:
	uvicorn inferlite.api.app:app --host 0.0.0.0 --port 8000

# Reproducible, CPU-only. Writes benchmarks/results/kv_cache_memory_comparison.csv
bench-kv:
	python benchmarks/scripts/kv_cache_memory_benchmark.py --requests 512 --max-seq-len 256 --seed 7

# CPU-only smoke comparison on distilgpt2. Requires a running InferLite server. Pass API_KEY=...
# Warm each endpoint once before timing for a fair steady-state measurement.
bench-cpu:
	python benchmarks/scripts/run_benchmark.py --backend inferlite --api-key $(API_KEY) \
		--workload benchmarks/configs/cpu_smoke_workload.json --out benchmarks/results/cpu_summary.csv
	python benchmarks/scripts/run_benchmark.py --backend naive --api-key $(API_KEY) \
		--workload benchmarks/configs/cpu_smoke_workload.json --out benchmarks/results/cpu_summary.csv
	python benchmarks/scripts/plot_results.py --csv benchmarks/results/cpu_summary.csv --out-dir benchmarks/results

# Requires a running InferLite (and vLLM on :8001 for the comparison). Pass API_KEY=...
bench:
	python benchmarks/scripts/run_matrix.py --api-key $(API_KEY) \
		--inferlite-url http://localhost:8000 --vllm-url http://localhost:8001

report:
	python benchmarks/scripts/plot_results.py \
		--csv benchmarks/results/benchmark_summary.csv --out-dir benchmarks/results

docker:
	docker compose up --build
