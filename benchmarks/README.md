# Benchmarks

InferLite benchmark harness compares:

- `inferlite` endpoint (`/v1/completions`)
- `naive` baseline endpoint (`/v1/completions/baseline`)
- `vllm` OpenAI-compatible endpoint (`/v1/completions`)

## Files

- `configs/default_workload.json`: steady mixed load
- `configs/bursty_workload.json`: burst-heavy workload for scheduler fairness analysis
- `scripts/load_generator.py`: synthetic request/event generation
- `scripts/run_benchmark.py`: executes one backend/workload run and appends CSV row
- `scripts/run_matrix.py`: executes inferlite + naive + vllm across workload list
- `scripts/plot_results.py`: turns CSV into PNG charts + markdown report

## Usage

1. Create an API key from InferLite admin endpoint:

```bash
curl -X POST http://localhost:8000/admin/keys \
  -H "x-admin-secret: inferlite-admin" \
  -H "Content-Type: application/json" \
  -d '{"tier":"paid","requests_per_minute":500}'
```

2. Run each backend against the same workload:

```bash
python benchmarks/scripts/run_benchmark.py --backend inferlite --api-key <KEY> --workload benchmarks/configs/default_workload.json
python benchmarks/scripts/run_benchmark.py --backend naive --api-key <KEY> --workload benchmarks/configs/default_workload.json
python benchmarks/scripts/run_benchmark.py --backend vllm --base-url http://localhost:8001 --api-key <KEY> --workload benchmarks/configs/default_workload.json
```

Or run the full matrix in one command:

```bash
python benchmarks/scripts/run_matrix.py --api-key <KEY> --inferlite-url http://localhost:8000 --vllm-url http://localhost:8001
```

3. Generate report artifacts:

```bash
python benchmarks/scripts/plot_results.py --csv benchmarks/results/benchmark_summary.csv --out-dir benchmarks/results
```

Outputs:

- `benchmarks/results/benchmark_summary.csv`
- `benchmarks/results/throughput_tokens_per_s.png`
- `benchmarks/results/latency_p95_s.png`
- `benchmarks/results/ttft_p95_s.png`
- `benchmarks/results/benchmark_report.md`

## Important benchmark notes

- Current TTFT for non-streaming runs is an approximation inside the harness.
- Use identical model, quantization, GPU class, and prompt/output distributions for fair runs.
- Report both wins and losses; the point is architectural understanding, not headline throughput.
