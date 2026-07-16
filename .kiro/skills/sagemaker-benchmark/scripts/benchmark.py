#!/usr/bin/env python3
"""Benchmark a deployed SageMaker AI endpoint with the managed benchmark service.

This is the reference implementation of the ``sagemaker-benchmark`` SKILL.md contract.

What "managed benchmark" means
------------------------------
We do NOT write a load generator. SageMaker AI inference benchmarking (part of the
optimized GenAI inference recommendations feature) runs **NVIDIA AIPerf** for us on
SageMaker-managed compute, drives a realistic load against the endpoint, and writes
the standard AIPerf metrics to S3: time-to-first-token (TTFT), inter-token latency
(ITL), request-latency percentiles, and output-token throughput.

The three public APIs, in order:
    create_ai_workload_config  ->  create_ai_benchmark_job  ->  describe_ai_benchmark_job

This produces the baseline later compared against an optimized configuration.
Benchmark first, optimize second, and benchmark again. When the job finishes,
``benchmark_results.py`` fetches the output bundle from S3 and presents it.

Model-agnostic
--------------
Nothing here is specific to a particular model. The workload is realistic chat traffic
drawn from the public sharegpt dataset. One optional knob, ``--extra-inputs``, lets you
pass provider-specific request fields when a model needs them (example below) — but it
defaults to empty, so the same command benchmarks any deployed endpoint.

The bundled dataset (and why it is the default)
-----------------------------------------------
By default we benchmark with ``datasets/sharegpt-curated.jsonl`` — 500 real sharegpt
prompts bundled with this skill — uploaded to S3 and attached to the workload config.
Why not just ``public_dataset: sharegpt``? The raw feed derives each request's output
budget from the dataset's recorded answer lengths, and a fixed handful of turns carry
budgets of only 1–3 tokens. A reasoning model structurally cannot place visible content
in so few tokens (the budget is consumed before the answer channel opens), so those
requests score invalid and AIPerf's ~1% validity gate fails the whole job — same 10
sessions, every run, regardless of ``output_tokens_mean`` (per-request budgets ignore
it) or ``min_tokens`` (capped by the per-request ``max_completion_tokens``). The curated
file is the same traffic with output budgets ≥32 tokens, so the gate measures the
endpoint rather than a dataset artifact — for any model. ``--public-dataset`` restores
the raw feed; ``--dataset-file`` swaps in your own JSONL ({"text", "output_length"}).

Usage:
    python scripts/benchmark.py --endpoint NAME                         # standard endpoint
    python scripts/benchmark.py --endpoint NAME --ic IC_NAME --run     # IC endpoint
    # Optional, only if your model uses extra request fields:
    python scripts/benchmark.py --endpoint NAME --ic IC_NAME \\
        --extra-inputs "reasoning_effort:low" --run
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import boto3

import config  # region / role / bucket — auto-detected, nothing hardcoded

# The curated sharegpt slice bundled with the skill (see module docstring).
DEFAULT_DATASET = pathlib.Path(__file__).resolve().parent.parent / "datasets" / "sharegpt-curated.jsonl"


# ---------------------------------------------------------------------------
# The workload. These numbers describe a realistic chat load against the endpoint:
# ~500-token prompts from the public sharegpt dataset, a fixed output budget, and a
# chosen concurrency. Override these defaults for a heavier or lighter run.
# ---------------------------------------------------------------------------
def workload_spec(concurrency: int, request_count: int, out_tokens: int,
                  extra_inputs: str, dataset_file: str | None,
                  tokenizer: str = "") -> dict:
    params = {
        "prompt_input_tokens_mean": 500,            # ~500-token inputs…
        "prompt_input_tokens_stddev": 10,           # …with a little natural variation
        "output_tokens_mean": out_tokens,           # how many tokens to ask the model to generate
        "output_tokens_stddev": 16,
        "concurrency": concurrency,                 # simultaneous in-flight requests
        "request_count": request_count,             # total requests in the run
    }
    if dataset_file:
        # The dataset channel is mounted by SageMaker at /opt/ml/input/data/datasets/.
        # "single_turn" = AIPerf's one-independent-request-per-line type, matching our
        # {"text", "output_length"} records. (Not "generic" — AIPerf rejects that.)
        params["custom_dataset_type"] = "single_turn"
        params["input_file"] = f"/opt/ml/input/data/datasets/{dataset_file}"
    else:
        params["public_dataset"] = "sharegpt"       # the raw feed (see docstring caveat)
    # Optional, model-specific request fields. Left empty by default so the benchmark
    # is model-agnostic. Example: a reasoning model that otherwise spends its whole
    # output budget "thinking" and returns empty visible text can be nudged with
    #   --extra-inputs "reasoning_effort:low"
    # so the answer actually gets written (which keeps AIPerf's validity rate high).
    if extra_inputs:
        params["extra_inputs"] = extra_inputs
    # AIPerf needs a tokenizer to count tokens. It is normally auto-detected from the
    # endpoint's model artifact, but an endpoint deployed from a Model Package (e.g. a
    # sagemaker-optimize recommendation) reports its model as a local path AIPerf can't
    # map to a tokenizer — auto-detection then fails with "No tokenizer available".
    # Pass a HuggingFace model/tokenizer ID here to resolve it explicitly. Model-agnostic:
    # empty by default, so endpoints with a detectable tokenizer are unaffected.
    if tokenizer:
        params["tokenizer"] = tokenizer
    return {"benchmark": {"type": "aiperf"}, "parameters": params}


def main() -> int:
    ap = argparse.ArgumentParser(description="Managed benchmark of a SageMaker AI endpoint.")
    ap.add_argument("--endpoint", required=True, help="endpoint name from deploy.py")
    ap.add_argument("--ic", help="inference component name; omit for a standard endpoint")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="simultaneous in-flight requests (default: 10)")
    ap.add_argument("--requests", type=int, default=300,
                    help="total requests in the run (default: 300)")
    ap.add_argument("--out-tokens", type=int, default=256,
                    help="mean output tokens to generate per request (default: 256)")
    ap.add_argument("--extra-inputs", default="",
                    help='optional model-specific request fields, space-separated '
                         '(e.g. "reasoning_effort:low"); empty by default')
    ap.add_argument("--tokenizer", default="",
                    help="HuggingFace model/tokenizer ID for AIPerf token counting "
                         "(e.g. openai/gpt-oss-20b). Only needed when the endpoint's "
                         "tokenizer can't be auto-detected, such as an endpoint deployed "
                         "from a Model Package; empty by default")
    ap.add_argument("--dataset-file", default=str(DEFAULT_DATASET),
                    help="JSONL dataset to benchmark with ({'text','output_length'} per "
                         "line); default: the curated sharegpt slice bundled with the skill")
    ap.add_argument("--public-dataset", action="store_true",
                    help="use the raw public sharegpt feed instead of a dataset file "
                         "(note: its 1-3-token output budgets fail reasoning models "
                         "against AIPerf's validity gate)")
    ap.add_argument("--run", action="store_true",
                    help="actually launch the job; without this it's a dry run")
    args = ap.parse_args()

    region = config.region()
    sess = boto3.session.Session(region_name=region)
    role = config.execution_role_arn(sess)
    bucket = config.bucket(sess)
    # Results land under the SageMaker default bucket so the role can already write there.
    s3_output = f"s3://{bucket}/benchmark-output/"

    dataset_path = None if args.public_dataset else pathlib.Path(args.dataset_file)
    if dataset_path and not dataset_path.is_file():
        raise SystemExit(f"dataset file not found: {dataset_path}")

    spec = workload_spec(args.concurrency, args.requests, args.out_tokens,
                         args.extra_inputs, dataset_path.name if dataset_path else None,
                         args.tokenizer)
    stamp = time.strftime("%y%m%d-%H%M%S")
    config_name = f"wl-{stamp}"     # the reusable workload definition
    job_name = f"bench-{stamp}"     # this specific run
    dataset_s3 = f"s3://{bucket}/benchmark-datasets/{stamp}/"  # folder URI (must end with /)

    print("=== BENCHMARK PLAN (sagemaker-benchmark contract) ===")
    print(f"  region     : {region}")
    print(f"  endpoint   : {args.endpoint}")
    print(f"  ic         : {args.ic or '(standard endpoint)'}")
    print(f"  dataset    : {dataset_path if dataset_path else 'public sharegpt feed'}")
    print(f"  workload   : {json.dumps(spec['parameters'])}")
    print(f"  output     : {s3_output}")
    print(f"  job        : {job_name}")
    if not args.run:
        print("\nDRY RUN — add --run to launch the managed benchmark (this is billable).")
        return 0

    sm = sess.client("sagemaker")

    # 1) Define the workload. We pass it inline as JSON; SageMaker stores it as a
    #    named, reusable config that benchmark jobs reference. With a dataset file we
    #    first stage it in S3 and attach it as an input channel SageMaker mounts for
    #    the AIPerf driver.
    cfg_kwargs = {
        "AIWorkloadConfigName": config_name,
        "AIWorkloadConfigs": {"WorkloadSpec": {"Inline": json.dumps(spec)}},
    }
    if dataset_path:
        key = f"benchmark-datasets/{stamp}/{dataset_path.name}"
        sess.client("s3").upload_file(str(dataset_path), bucket, key)
        print(f"Dataset staged: s3://{bucket}/{key}")
        cfg_kwargs["DatasetConfig"] = {
            "InputDataConfig": [{
                "ChannelName": "datasets",
                "DataSource": {"S3DataSource": {"S3Uri": dataset_s3}},
            }]
        }
    sm.create_ai_workload_config(**cfg_kwargs)
    print("WorkloadConfig:", config_name)

    # 2) Launch the benchmark against the endpoint and optional component. SageMaker
    #    spins up the AIPerf driver on managed compute — we don't manage any of it.
    #    RoleArn is the role the managed benchmark service assumes on your behalf; its
    #    trust policy must allow the sagemaker.amazonaws.com principal (the endpoint's
    #    execution role already does).
    endpoint_target = {"Identifier": args.endpoint}
    if args.ic:
        endpoint_target["InferenceComponents"] = [{"Identifier": args.ic}]
    r = sm.create_ai_benchmark_job(
        AIBenchmarkJobName=job_name,
        BenchmarkTarget={"Endpoint": endpoint_target},
        OutputConfig={"S3OutputLocation": s3_output},
        AIWorkloadConfigIdentifier=config_name,
        RoleArn=role)
    print("BenchmarkJob:", r["AIBenchmarkJobArn"])

    # 3) Poll until the job finishes. The job is fully managed by SageMaker.
    print("Polling (every 30s)…")
    running = ("InProgress", "Pending", "Starting", "Stopping")
    while True:
        d = sm.describe_ai_benchmark_job(AIBenchmarkJobName=job_name)
        status = d["AIBenchmarkJobStatus"]
        print(f"  status: {status}")
        if status == "Completed":
            print("Results in S3:", d["OutputConfig"]["S3OutputLocation"])
            print(f"Show them: python benchmark_results.py --job {job_name}")
            break
        if status not in running:
            # Failed / Stopped, or any unexpected terminal state. One thing to know:
            # AIPerf enforces a ~1% result-validity gate. A model that sometimes returns
            # empty visible text (e.g. a reasoning model that spends its whole budget
            # "thinking") can trip that gate even though the endpoint returned 200 for
            # every request and the metrics over the valid requests are sound. If you
            # hit this, pass an appropriate --extra-inputs for that model. The output
            # bundle in S3 is still complete — benchmark_results.py can read it.
            print("FailureReason:", d.get("FailureReason", "(none reported)"))
            print(f"If the validity gate tripped, the results bundle is still complete: "
                  f"python benchmark_results.py --job {job_name}")
            break
        time.sleep(30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
