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
from botocore.exceptions import ClientError

import config


def wait_until_deleted(describe, name_key: str, name: str, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            describe(**{name_key: name})
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {
                "ResourceNotFound",
                "ValidationException",
            }:
                return True
            raise
        time.sleep(10)
    return False


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
    for ic in ics:
        try:
            model_name = sm.describe_inference_component(
                InferenceComponentName=ic
            ).get("Specification", {}).get("ModelName")
            if model_name:
                model_names.append(model_name)
        except ClientError:
            pass
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
    for ic in ics:
        if not wait_until_deleted(
            sm.describe_inference_component,
            "InferenceComponentName",
            ic,
            600,
        ):
            print(f"  WARN: IC {ic} still deleting after 10 min - proceeding anyway")

    # 2) Delete the endpoint and wait before deleting resources it references.
    try:
        sm.delete_endpoint(EndpointName=ep)
        print("deleting endpoint:", ep)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {
            "ResourceNotFound",
            "ValidationException",
        }:
            raise
        print("  (skip endpoint)", exc.response["Error"]["Code"])
    if not wait_until_deleted(sm.describe_endpoint, "EndpointName", ep, 1800):
        print(f"ERROR: endpoint {ep} is still present after 30 minutes")
        return 1
    print("deleted endpoint:", ep)

    # 3) Endpoint configuration, 4) model(s).
    try:
        sm.delete_endpoint_config(EndpointConfigName=cfg_name)
        print("deleted endpoint-config:", cfg_name)
    except ClientError as exc:
        print("  (skip endpoint-config)", exc.response["Error"]["Code"])

    for m in dict.fromkeys(model_names or [ep]):
        try:
            sm.delete_model(ModelName=m)
            print("deleted model:", m)
        except ClientError as exc:
            print("  (skip model)", m, "-", exc.response["Error"]["Code"])

    print("\nTeardown complete - endpoint compute is deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
