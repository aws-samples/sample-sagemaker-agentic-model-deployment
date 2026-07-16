---
name: sagemaker-benchmark
description: >
  Run a managed performance benchmark against a deployed Amazon SageMaker AI
  endpoint using SageMaker AI inference benchmarking (part of optimized GenAI
  inference recommendations; NVIDIA AIPerf under the hood). Measures TTFT, ITL,
  request-latency percentiles, and throughput. Use when the user asks to
  benchmark / load-test / measure the performance of a live endpoint.
license: MIT-0
compatibility: Requires AWS credentials with Amazon SageMaker AI access and Python with boto3>=1.43 (the create_ai_workload_config / create_ai_benchmark_job APIs); bundled scripts run on the local machine.
metadata:
  author: sample-sagemaker-agentic-model-deployment
  version: "2.0"
---

# sagemaker-benchmark

Managed performance benchmarking with **Amazon SageMaker AI inference benchmarking**
(the benchmark capability within optimized GenAI inference recommendations). SageMaker
runs **NVIDIA AIPerf** on managed compute and writes AIPerf results to S3 — no
hand-built load generator, no self-managed AIPerf.

> This is the baseline benchmark (pre-optimization). Optimization (speculative
> decoding / EAGLE 3, quantization, kernel tuning) is the recommendation/optimization
> path (`create_ai_recommendation_job` / `create_optimization_job`) and is out of scope
> for the baseline measurement.

## Scope
- Benchmark an **already-deployed** endpoint (from `sagemaker-deploy`).
- Produces AIPerf metrics: TTFT, ITL, P50/P90/P99 request latency, output-token
  throughput, requests/sec → written to S3.

## APIs (verified present in boto3 1.43.24)
`create_ai_workload_config` → `create_ai_benchmark_job` → poll `describe_ai_benchmark_job`.

## Defaults

| Field | Value |
|---|---|
| Target | standard endpoint, or endpoint plus Inference Component |
| Workload | `aiperf`, the bundled `datasets/sharegpt-curated.jsonl` (real sharegpt prompts; see note below) |
| Profile | ~500 input / ~256 output tokens, concurrency 10, 300 requests |
| Output | `s3://<sagemaker-default-bucket>/benchmark-output/` (auto-detected; never hardcoded) |
| Role | the SageMaker execution role (must trust `sagemaker.amazonaws.com`) |

**Why a bundled dataset and not `public_dataset: "sharegpt"`?** The raw feed derives each
request's output budget from the dataset's recorded answer lengths, and a fixed handful of
turns carry budgets of only 1–3 tokens. A reasoning model cannot emit visible content within
1–3 tokens (the budget is consumed before the answer channel opens), so those requests score
invalid and trip AIPerf's ~1% validity gate — deterministically, every run, regardless of
`output_tokens_mean` (per-request budgets ignore it) or `min_tokens` (capped per-request).
`datasets/sharegpt-curated.jsonl` is 500 real sharegpt prompts with output budgets ≥32
tokens: same realistic traffic, but the gate measures the endpoint, not a dataset artifact.
It is the default for **any** model; `--public-dataset` restores the raw feed.

## Workflow

### Step 1: Define the workload config
Stage the dataset file in S3 (any folder; the URI must end with `/`), then:
```python
workload_spec = {
    "benchmark": {"type": "aiperf"},
    "parameters": {
        "custom_dataset_type": "single_turn",   # one independent request per JSONL line
        "input_file": "/opt/ml/input/data/datasets/sharegpt-curated.jsonl",
        "prompt_input_tokens_mean": 500, "prompt_input_tokens_stddev": 10,
        "output_tokens_mean": 256, "output_tokens_stddev": 16,
        # Model-agnostic by default. Add only if a model needs it (see reasoning-model note):
        #   "extra_inputs": "reasoning_effort:low",
        "concurrency": 10, "request_count": 300,
    },
}
client.create_ai_workload_config(
    AIWorkloadConfigName=config_name,
    AIWorkloadConfigs={"WorkloadSpec": {"Inline": json.dumps(workload_spec)}},
    DatasetConfig={"InputDataConfig": [{
        "ChannelName": "datasets",   # mounted at /opt/ml/input/data/datasets/
        "DataSource": {"S3DataSource": {"S3Uri": dataset_s3_folder}},  # ends with /
    }]},
)
```
Each dataset line is `{"text": "<prompt>", "output_length": <int>}`.

### Step 2: Launch the benchmark job
```python
endpoint_target = {"Identifier": endpoint_name}
if ic_name:
    endpoint_target["InferenceComponents"] = [{"Identifier": ic_name}]
client.create_ai_benchmark_job(
    AIBenchmarkJobName=job_name,
    BenchmarkTarget={"Endpoint": endpoint_target},
    OutputConfig={"S3OutputLocation": S3_OUTPUT},
    AIWorkloadConfigIdentifier=config_name,
    RoleArn=role,   # role must trust sagemaker.amazonaws.com
)
```

### Step 3: Poll to completion
Poll `describe_ai_benchmark_job(AIBenchmarkJobName=job_name)["AIBenchmarkJobStatus"]`
every 30s until `Completed | Failed | Stopped`.

### Step 4: Read the results from S3
The job writes one tarball per run to `<S3OutputLocation>/output/output.tar.gz`.
Extracted, the bundle looks like this:

```
output/
├── profile_export_aiperf.json   # aggregated metrics — parse THIS for the numbers
├── profile_export_aiperf.csv    # the same aggregates as CSV (spreadsheet-friendly)
├── profile_export.jsonl         # raw per-request records
├── inputs.json                  # the prompts AIPerf sent during the run
├── outputs.json                 # what the model answered
├── benchmark_summary.txt        # completion summary
├── failure_reason.txt           # present only when the validity gate tripped
├── plot_generation.log          # plot generation log
├── plots/
│   ├── ttft_timeline.png        # TTFT per request over the run
│   ├── ttft_over_time.png       # TTFT aggregated over the run duration
│   └── summary.txt              # list of generated plots
└── logs/
    └── aiperf.log               # full AIPerf execution log
```

The bundle serves two audiences: **an agent** parses `profile_export_aiperf.json`
(each metric is `{"unit", "avg", "p1"…"p99", "min", "max"}`), and **a human** opens
the PNG plots, the CSV, and the raw logs. Headline keys to surface:
`output_token_throughput`, `time_to_first_token`, `inter_token_latency`,
`request_latency`, `request_throughput`, plus `request_count` / `error_request_count`
for validity.

### Step 5: Report the results
Run `scripts/benchmark_results.py` (read-only) to fetch the bundle, print the file
tree with annotations, and surface the headline numbers rather than reporting
only an S3 path:

```
python scripts/benchmark_results.py                  # latest standalone job
python scripts/benchmark_results.py --job JOB_NAME   # a specific job
```

A job marked `Failed` by AIPerf's ~1% validity gate still has a complete bundle —
the reader detects that case, says so, and reports the metrics over the valid
requests.

### Inspect the bundled sample
A real run's complete bundle ships with this skill in `sample-output/` (GPT-OSS-20B
baseline on `ml.g6.16xlarge` with **218 tok/s** output throughput; see
`sample-output/README.md`). Present it without any AWS call:

```
python scripts/benchmark_results.py --local sample-output
```

Use it to inspect the benchmark result schema without launching a job.

## Reference implementation
`scripts/benchmark.py` implements this contract (dry-run by default, `--run` to launch).
It stages the bundled dataset to S3 automatically; `--dataset-file` swaps in your own,
`--public-dataset` uses the raw sharegpt feed instead.
Region / role / output bucket are auto-detected (`scripts/config.py`).
`scripts/benchmark_results.py` presents the finished job's results (Step 4–5).
`scripts/cloudwatch_metrics.py` reads the matching endpoint observability (invocations,
concurrency, latency) after the run.

## Reasoning-model output shape
Some reasoning models can spend a small output budget entirely in a reasoning
channel and return `content: null`, which AIPerf scores as an invalid result.
The benchmark fails if the invalid rate exceeds approximately 1%.
- The **bundled curated dataset (the default) already resolves this** — its per-request
  output budgets are ≥32 tokens, enough for the answer channel to open.
- `extra_inputs: "reasoning_effort:low"` additionally keeps the reasoning channel short
  (useful belt-and-suspenders for reasoning models).
- Knobs that do **not** fix it on the raw public feed: `output_tokens_mean` (per-request
  budgets come from the dataset, not the mean) and `min_tokens` (capped by the
  per-request `max_completion_tokens`).
Do **not** set `ignore_eos:true` for reasoning models — let the model stop naturally.

## Pre-reqs / guards
- The endpoint must be `InService`. When targeting an Inference Component, the
  component must also be `InService`.
- `RoleArn` must trust `sagemaker.amazonaws.com` (the benchmark service assumes it).
- Regional availability varies. Check the SageMaker AI documentation for the
  deployment Region.
- Pricing: no extra service fee for the benchmark capability itself; you pay for the
  managed compute the benchmark runs on (see the SageMaker pricing page).
