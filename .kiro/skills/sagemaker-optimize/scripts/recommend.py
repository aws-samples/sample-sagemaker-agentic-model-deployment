#!/usr/bin/env python3
"""Find an optimized serving config for a model with SageMaker AI Recommendation.

This is the reference implementation of the ``sagemaker-optimize`` SKILL.md contract.

What a recommendation job does
------------------------------
You give SageMaker a model (weights in S3) and a workload (what your traffic looks
like). It then **searches serving configurations on managed compute** and returns the
best one it found: the instance type, the number of model copies, the container image
and environment, and — crucially — the **ExpectedPerformance** (throughput, latency).
You never hand-tune vLLM flags or stand up a sweep harness yourself.

Two depths, one API (`create_ai_recommendation_job`):
  * OptimizeModel=False  -> config search only: best instance + serving knobs for the
                            model as-is.
  * OptimizeModel=True   -> deep optimization: SageMaker may also apply speculative
                            decoding (EAGLE 3), quantization, and kernel tuning, and
                            register the optimized artifact as a Model Package. This is
                            long-running and uses large instances such as p5en.

The output (a recommended config, or a Model Package for the optimized path) is then
deployed by scripts/deploy_recommendation.py, and you benchmark again to show the
before/after.

Usage:
    python scripts/recommend.py --model-id MODEL --model-s3 S3_URI
    python scripts/recommend.py --model-id MODEL --model-s3 S3_URI \\
        --instance ml.g6.24xlarge --run
    python scripts/recommend.py --model-id MODEL --model-s3 S3_URI \\
        --optimize --instance ml.p5en.48xlarge \\
        --dataset-s3 s3://<bucket>/datasets/ --run                # deep optimize (slow/$$$)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import boto3

import config  # region / account / role / bucket — auto-detected, nothing hardcoded

# Curated prompts in OpenAI Chat format — the same 500 prompts the benchmark skill uses,
# converted because the recommendation service only accepts ShareGPT / OpenAI formats.
DEFAULT_DATASET = (pathlib.Path(__file__).resolve().parent.parent
                   / "datasets" / "sharegpt-curated-openai.jsonl")


def workload_spec(concurrency: int, out_tokens: int, in_tokens: int,
                  custom_input_file: str | None) -> dict:
    """The workload profile the recommendation job optimizes *for*.

    With no custom dataset we use the public sharegpt dataset (realistic chat). With a
    custom dataset (``--dataset-s3``) we point AIPerf at a JSONL file staged in S3 — use
    this when your real traffic differs from sharegpt (longer prompts, specific format).
    """
    params = {
        "concurrency": concurrency,
        "prompt_input_tokens_mean": in_tokens,
        "prompt_input_tokens_stddev": 10,
        "output_tokens_mean": out_tokens,
        "output_tokens_stddev": 10,
    }
    if custom_input_file:
        # A custom dataset is mounted by SageMaker at /opt/ml/input/data/datasets/.
        # NOTE the format asymmetry with the benchmark service: the recommendation
        # service validates the dataset FILES themselves and accepts only ShareGPT /
        # OpenAI Chat Completions / OpenAI Completions records — the benchmark-style
        # AIPerf single_turn lines ({"text", ...}) are rejected ("unrecognized format").
        # Use the bundled datasets/sharegpt-curated-openai.jsonl (the benchmark's
        # curated prompts converted to OpenAI Chat format). "generic" + input_file is
        # the shape the reference aws-samples recommendation notebook uses.
        params["custom_dataset_type"] = "generic"
        params["input_file"] = f"/opt/ml/input/data/datasets/{custom_input_file}"
    else:
        params["public_dataset"] = "sharegpt"
    return {"benchmark": {"type": "aiperf"}, "parameters": params}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a SageMaker AI Recommendation / optimization job.")
    # --- What to optimize (model-agnostic) ---
    ap.add_argument("--model-id", required=True,
                    help="friendly name used for the job and config names")
    ap.add_argument("--model-s3", required=True,
                    help="S3 URI of the model weights")
    # --- Where it searches, and how deep ---
    ap.add_argument("--instance", default="ml.g6.24xlarge",
                    help="candidate instance type to search on (default: ml.g6.24xlarge)")
    ap.add_argument("--optimize", action="store_true",
                    help="deep optimization using a large, usually capacity-reserved "
                         "instance; this can run for hours")
    ap.add_argument("--framework", default=None,
                    help="serving framework: VLLM (default for config search) or LMI "
                         "(default for --optimize). Override if needed.")
    ap.add_argument("--metric", default="throughput", choices=["throughput", "latency"],
                    help="what to optimize for (default: throughput)")
    # --- Workload profile ---
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--in-tokens", type=int, default=500)
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--dataset-s3", default=None,
                    help="optional S3 folder (must end with /) holding a custom JSONL dataset")
    ap.add_argument("--dataset-file", default=None,
                    help="custom dataset: a local JSONL path to stage automatically (e.g. the "
                         "bundled datasets/sharegpt-curated-openai.jsonl), or — together with "
                         "--dataset-s3 — the filename inside that folder. Records must be "
                         "ShareGPT or OpenAI Chat/Completions format (the recommendation "
                         "service rejects other shapes); default: the public sharegpt feed")
    # --- Capacity reservation (deep-optimize on scarce instances) ---
    ap.add_argument("--reservation-arn", action="append", default=[],
                    help="ML reservation / training-plan ARN to run on (repeatable). "
                         "Required in practice for p5en-class instances.")
    # --- Safety ---
    ap.add_argument("--run", action="store_true",
                    help="actually launch the job; without this it's a dry run")
    args = ap.parse_args()

    region = config.region()
    sess = boto3.session.Session(region_name=region)
    role = config.execution_role_arn(sess)
    bucket = config.bucket(sess)
    model_s3 = args.model_s3
    s3_output = f"s3://{bucket}/recommendations/"

    # Framework default depends on depth: the deep-optimize path uses LMI (which carries
    # the speculative-decoding / quantization toolchain); plain config search uses vLLM.
    framework = args.framework or ("LMI" if args.optimize else "VLLM")

    stamp = time.strftime("%y%m%d-%H%M%S")
    config_name = f"rec-cfg-{args.model_id}-{stamp}"
    job_name = f"rec-job-{args.model_id}-{stamp}"

    # A --dataset-file without --dataset-s3 is a local JSONL we stage to S3 ourselves
    # (mirrors benchmark.py). With --dataset-s3 it's just the filename in that folder.
    dataset_local = None
    dataset_name = None
    if args.dataset_file and not args.dataset_s3:
        dataset_local = pathlib.Path(args.dataset_file)
        if not dataset_local.is_file():
            raise SystemExit(f"dataset file not found: {dataset_local}")
        dataset_name = dataset_local.name
        args.dataset_s3 = f"s3://{bucket}/recommendation-datasets/{stamp}/"
    elif args.dataset_file:
        dataset_name = pathlib.Path(args.dataset_file).name

    spec = workload_spec(args.concurrency, args.out_tokens, args.in_tokens, dataset_name)

    print("=== RECOMMENDATION PLAN (sagemaker-optimize contract) ===")
    print(f"  region     : {region}")
    print(f"  account    : {config.account_id(sess)}")
    print(f"  model-id   : {args.model_id}")
    print(f"  weights    : {model_s3}")
    print(f"  instance   : {args.instance}")
    print(f"  optimize   : {args.optimize}  (EAGLE3 / quant / kernels)" if args.optimize
          else f"  optimize   : {args.optimize}  (config search only)")
    print(f"  framework  : {framework}")
    print(f"  metric     : {args.metric}")
    print(f"  workload   : {json.dumps(spec['parameters'])}")
    print(f"  output     : {s3_output}")
    print(f"  job        : {job_name}")
    if args.optimize and not args.reservation_arn:
        print("  WARNING: --optimize on a large instance usually requires a capacity "
              "reservation (--reservation-arn). Without one the job may fail to get capacity.")
    if not args.run:
        print("\nDRY RUN — add --run to launch the job (this is billable; --optimize is slow/$$$).")
        return 0

    sm = sess.client("sagemaker")

    # 1) Workload config — same building block as the benchmark. If a custom dataset was
    #    given, attach it as an input channel SageMaker will mount for the AIPerf driver.
    create_cfg_kwargs = {
        "AIWorkloadConfigName": config_name,
        "AIWorkloadConfigs": {"WorkloadSpec": {"Inline": json.dumps(spec)}},
    }
    if args.dataset_s3:
        if dataset_local:
            key = f"recommendation-datasets/{stamp}/{dataset_name}"
            sess.client("s3").upload_file(str(dataset_local), bucket, key)
            print(f"Dataset staged: s3://{bucket}/{key}")
        create_cfg_kwargs["DatasetConfig"] = {
            "InputDataConfig": [{
                "ChannelName": "datasets",
                # Must be a *folder* path ending with /, not a single file.
                "DataSource": {"S3DataSource": {"S3Uri": args.dataset_s3}},
            }]
        }
    sm.create_ai_workload_config(**create_cfg_kwargs)
    print("WorkloadConfig:", config_name)

    # 2) The recommendation job. ModelSource points at the raw weights; PerformanceTarget
    #    says what to optimize for; ComputeSpec says which instance(s) to search on;
    #    OptimizeModel toggles the deep (EAGLE3/quant/kernel) path.
    job_kwargs = {
        "AIRecommendationJobName": job_name,
        "ModelSource": {"S3": {"S3Uri": model_s3}},
        "OutputConfig": {"S3OutputLocation": s3_output},
        "AIWorkloadConfigIdentifier": config_name,
        "RoleArn": role,
        "PerformanceTarget": {"Constraints": [{"Metric": args.metric}]},
        "InferenceSpecification": {"Framework": framework},
        "OptimizeModel": args.optimize,
        "ComputeSpec": {"InstanceTypes": [args.instance]},
    }
    # Attach a capacity reservation if provided (needed for scarce p5en-class instances).
    if args.reservation_arn:
        job_kwargs["ComputeSpec"]["CapacityReservationConfig"] = {
            "CapacityReservationPreference": "capacity-reservations-only",
            "MlReservationArns": args.reservation_arn,
        }
    r = sm.create_ai_recommendation_job(**job_kwargs)
    print("RecommendationJob:", r["AIRecommendationJobArn"])

    # 3) Poll. A config search can take minutes; deep optimization can take hours.
    print("Polling (every 30s)…")
    running = ("InProgress", "Pending", "Starting", "Stopping")
    while True:
        d = sm.describe_ai_recommendation_job(AIRecommendationJobName=job_name)
        status = d["AIRecommendationJobStatus"]
        print(f"  status: {status}")
        if status not in running:
            break
        time.sleep(30)

    # 4) Show what came back: the recommended config + expected performance, and (for the
    #    optimized path) the Model Package ARN that deploy_recommendation.py will deploy.
    if status == "Completed":
        recs = d.get("Recommendations", [])
        print(f"\n{len(recs)} recommendation(s):")
        for i, rec in enumerate(recs, 1):
            print(f"\n--- Recommendation {i} ---")
            print(f"  {rec.get('RecommendationDescription', '')}")
            for inst in rec.get("InstanceDetails", []):
                print(f"  instance: {inst['InstanceType']} x{inst.get('InstanceCount', 1)}")
            for opt in rec.get("OptimizationDetails", []):
                print(f"  optimization applied: {opt['OptimizationType']}")
            for perf in rec.get("ExpectedPerformance", []):
                print(f"  {perf['Metric']}: {perf['Value']} {perf.get('Unit', '')}")
            md = rec.get("ModelDetails", {})
            if md.get("ModelPackageArn"):
                print(f"  ModelPackageArn: {md['ModelPackageArn']}")
        print("\nNext: deploy the top recommendation with scripts/deploy_recommendation.py, "
              "then benchmark again for the before/after.")
    else:
        print("FailureReason:", d.get("FailureReason", "(none reported)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
