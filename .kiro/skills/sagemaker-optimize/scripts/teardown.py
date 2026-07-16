#!/usr/bin/env python3
"""Delete an endpoint and everything attached to it. Endpoints bill while InService.

Run this the moment you are done. SageMaker keeps a real GPU instance running for a
real-time endpoint, so the cost clock runs until you delete it. We tear down in the
reverse order of creation:

    InferenceComponent  ->  Endpoint  ->  EndpointConfig  ->  Model

Each delete is best-effort: if a resource is already gone we keep going, so this is
safe to run twice.

Usage:
    python scripts/teardown.py --endpoint NAME                     # dry run: list what would go
    python scripts/teardown.py --endpoint NAME --yes               # actually delete
"""
import argparse
import time

import boto3

import config


def main() -> int:
    ap = argparse.ArgumentParser(description="Tear down a SageMaker AI endpoint and its resources.")
    ap.add_argument("--endpoint", required=True, help="endpoint name to delete")
    ap.add_argument("--yes", action="store_true", help="confirm deletion (without it, dry run)")
    args = ap.parse_args()

    sess = boto3.session.Session(region_name=config.region())
    sm = sess.client("sagemaker")
    ep = args.endpoint

    # Discover the inference components on this endpoint (there can be more than one).
    ics = [c["InferenceComponentName"]
           for c in sm.list_inference_components(EndpointNameEquals=ep).get("InferenceComponents", [])]

    # The endpoint config and model in this repo share the endpoint's name, but the
    # config may also point at a differently-named model — resolve it to be safe.
    model_names = []
    try:
        cfg_name = sm.describe_endpoint(EndpointName=ep)["EndpointConfigName"]
        for v in sm.describe_endpoint_config(EndpointConfigName=cfg_name).get("ProductionVariants", []):
            if v.get("ModelName"):
                model_names.append(v["ModelName"])
    except sm.exceptions.ClientError:
        cfg_name = ep  # fall back to the convention used by deploy.py

    print("=== TEARDOWN PLAN ===")
    print(f"  endpoint          : {ep}   <- this is what holds the GPU and bills")
    print(f"  inference comps   : {ics or '(none)'}")
    print(f"  endpoint config   : {cfg_name}")
    print(f"  models            : {model_names or [ep]}")
    if not args.yes:
        print("\nDRY RUN — add --yes to delete these (stops billing).")
        return 0

    # 1) Inference components first — the endpoint can't be deleted while ICs exist.
    for ic in ics:
        try:
            sm.delete_inference_component(InferenceComponentName=ic)
            print("deleted IC:", ic)
        except sm.exceptions.ClientError as e:
            print("  (skip IC)", ic, "-", e.response["Error"]["Code"])
    # Wait for the ICs to actually disappear before deleting the endpoint — but with a
    # deadline. If an IC gets stuck in Deleting (e.g. an unhealthy instance), we don't
    # want teardown to hang forever while the endpoint keeps billing; we warn
    # and move on (deleting the endpoint will clean up the IC anyway).
    deadline = time.time() + 600  # 10 minutes
    for ic in ics:
        while time.time() < deadline:
            try:
                sm.describe_inference_component(InferenceComponentName=ic)
                time.sleep(10)
            except sm.exceptions.ClientError:
                break
        else:
            print(f"  WARN: IC {ic} still deleting after 10 min — proceeding anyway")

    # 2) Endpoint, 3) config, 4) model(s). Best-effort each.
    for label, fn in [
        ("endpoint", lambda: sm.delete_endpoint(EndpointName=ep)),
        ("endpoint-config", lambda: sm.delete_endpoint_config(EndpointConfigName=cfg_name)),
    ]:
        try:
            fn()
            print(f"deleted {label}:", ep if label == "endpoint" else cfg_name)
        except sm.exceptions.ClientError as e:
            print(f"  (skip {label})", e.response["Error"]["Code"])

    for m in (model_names or [ep]):
        try:
            sm.delete_model(ModelName=m)
            print("deleted model:", m)
        except sm.exceptions.ClientError as e:
            print("  (skip model)", m, "-", e.response["Error"]["Code"])

    print("\nTeardown complete — billing stopped for this endpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
