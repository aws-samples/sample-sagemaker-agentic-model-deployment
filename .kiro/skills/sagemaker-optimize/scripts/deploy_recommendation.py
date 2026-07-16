#!/usr/bin/env python3
"""Deploy the result of a SageMaker AI Recommendation job as a live endpoint.

Part of the ``sagemaker-optimize`` SKILL.md contract. scripts/recommend.py
produced an optimized serving config and, on the deep path, a Model Package
containing the optimized artifact. This script deploys the top recommendation
as a real-time endpoint for a like-for-like benchmark comparison.

Why this is a separate step from a plain deploy
-----------------------------------------------
A normal deploy (scripts/deploy.py) builds the model from raw weights + a container we
choose. Here, the recommendation already decided the best instance, copy count, container
image, and environment — and on the optimized path it baked a Model Package. So we deploy
**from the recommendation's own answer** rather than re-deciding anything: CreateModel
references the Model Package, and the endpoint config uses the recommended instance.

    describe_ai_recommendation_job  ->  CreateModel(from ModelPackage)
                                    ->  CreateEndpointConfig(recommended instance)
                                    ->  CreateEndpoint  ->  smoke test

Usage:
    python scripts/deploy_recommendation.py --rec-job rec-job-...            # dry run
    python scripts/deploy_recommendation.py --rec-job rec-job-... --deploy   # create (billable)
"""
import argparse
import json
import sys
import time

import boto3

import config  # region / account / role / bucket — auto-detected, nothing hardcoded


def main() -> int:
    ap = argparse.ArgumentParser(description="Deploy a completed recommendation job's result.")
    ap.add_argument("--rec-job", required=True,
                    help="name of a Completed recommendation job (from recommend.py)")
    ap.add_argument("--rank", type=int, default=0,
                    help="which recommendation to deploy (0 = top-ranked)")
    ap.add_argument("--deploy", action="store_true",
                    help="actually create resources; without this it's a dry run")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="model-download / health-check timeout in seconds")
    args = ap.parse_args()

    region = config.region()
    sess = boto3.session.Session(region_name=region)
    role = config.execution_role_arn(sess)
    sm = sess.client("sagemaker")
    sm_rt = sess.client("sagemaker-runtime")

    # 1) Read the recommendation job and pick the chosen recommendation. Fail with a clear
    #    message rather than a stack trace if the name is wrong.
    try:
        d = sm.describe_ai_recommendation_job(AIRecommendationJobName=args.rec_job)
    except sm.exceptions.ResourceNotFound:
        # Wrong or mistyped job name. Return a clean message, not a stack trace.
        print(f"Recommendation job {args.rec_job!r} not found.")
        print("List your jobs with: aws sagemaker list-ai-recommendation-jobs")
        return 1
    # Any other ClientError (throttling, permissions, …) is a different problem — let it
    # surface rather than mislabel it as "job not found".
    status = d["AIRecommendationJobStatus"]
    if status != "Completed":
        print(f"Recommendation job is {status}, not Completed — cannot deploy yet.")
        return 1
    recs = d.get("Recommendations", [])
    if not recs:
        print("No recommendations returned by the job.")
        return 1
    rec = recs[args.rank]

    # The deployment configuration the recommendation chose for us.
    dc = rec.get("DeploymentConfiguration", {})
    instance_type = dc.get("InstanceType")
    instance_count = dc.get("InstanceCount", 1)
    env_vars = dc.get("EnvironmentVariables", {}) or {}
    model_pkg = rec.get("ModelDetails", {}).get("ModelPackageArn")

    stamp = time.strftime("%y%m%d-%H%M%S")
    name = f"rec-{stamp}"          # endpoint / model / config share this stem
    cfg_name = f"{name}-config"

    print("=== DEPLOY-RECOMMENDATION PLAN (sagemaker-optimize contract) ===")
    print(f"  region        : {region}")
    print(f"  rec job       : {args.rec_job}  (rank {args.rank})")
    print(f"  instance      : {instance_type} x{instance_count}")
    print(f"  model package : {model_pkg or '(none — config-only recommendation)'}")
    print(f"  env overrides : {json.dumps(env_vars)}")
    for opt in rec.get("OptimizationDetails", []):
        print(f"  optimization  : {opt['OptimizationType']}")
    for perf in rec.get("ExpectedPerformance", []):
        print(f"  expected      : {perf['Metric']} = {perf['Value']} {perf.get('Unit', '')}")
    print(f"  endpoint      : {name}")
    if not args.deploy:
        print("\nDRY RUN — add --deploy to create the endpoint (this is billable).")
        return 0

    if not model_pkg:
        # A config-only recommendation (OptimizeModel=False) returns instance + env but
        # no Model Package. In that case you deploy the original weights with scripts/
        # deploy.py using the recommended --instance / --env, rather than from here.
        print("This recommendation has no Model Package (config-only). Deploy the original "
              "weights with scripts/deploy.py using the recommended instance + env above.")
        return 1

    if not instance_type:
        # Defensive: a Model Package with no recommended instance can't build an endpoint
        # config. Fail with a clear message rather than a raw boto3 ValidationException.
        print("Recommendation has a Model Package but no InstanceType in its "
              "DeploymentConfiguration — cannot build an endpoint config. Inspect the job "
              "with: aws sagemaker describe-ai-recommendation-job --ai-recommendation-job-name "
              f"{args.rec_job}")
        return 1

    t0 = time.time()

    # 2) CreateModel from the Model Package. When deploying a Model Package you reference
    #    it by ARN in Containers[] and leave PrimaryContainer unset — the package already
    #    carries the optimized artifact, image, and serving config.
    sm.create_model(
        ModelName=name, ExecutionRoleArn=role,
        Containers=[{"ModelPackageName": model_pkg}])
    print("Model created (from Model Package):", name)

    # 3) CreateEndpointConfig on the recommended instance. Generous startup timeouts:
    #    an optimized artifact can be large and take time to download + warm up.
    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic", "ModelName": name,
            "InstanceType": instance_type, "InitialInstanceCount": instance_count,
            "ModelDataDownloadTimeoutInSeconds": args.timeout,
            "ContainerStartupHealthCheckTimeoutInSeconds": min(args.timeout, 1200)}])

    # 4) CreateEndpoint — provision and poll to InService.
    sm.create_endpoint(EndpointName=name, EndpointConfigName=cfg_name)
    print("Endpoint creating…")
    while True:
        e = sm.describe_endpoint(EndpointName=name)
        st = e["EndpointStatus"]
        print(f"  endpoint: {st}  (+{int(time.time() - t0)}s)")
        if st == "InService":
            break
        if st == "Failed":
            print("  FAILED:", e.get("FailureReason"))
            return 1
        time.sleep(30)
    print(f"TIMING optimized_endpoint_inservice_sec={int(time.time() - t0)}")

    print("\n=== DEPLOYED (optimized) ===")
    print(f"ENDPOINT_NAME={name}")

    # 5) Smoke test (this deploy path uses the endpoint directly, not a separate IC).
    payload = {"messages": [{"role": "user", "content": "Reply with exactly: pong"}],
               "max_tokens": 32}
    try:
        res = sm_rt.invoke_endpoint(EndpointName=name,
                                    Body=json.dumps(payload), ContentType="application/json")
        print("SMOKE TEST:", json.dumps(json.loads(res["Body"].read()))[:400])
    except Exception as e:
        print(f"SMOKE TEST WARN (endpoint is InService — continuing): {e}")

    print("Next: benchmark this endpoint and compare to the baseline (before/after). "
          "Tear down with scripts/teardown.py when done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
