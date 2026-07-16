#!/usr/bin/env python3
"""Deploy an S3-staged open-weight model to a SageMaker AI real-time endpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

import config


INSTANCE_GPUS = {
    "ml.g5.12xlarge": 4,
    "ml.g5.48xlarge": 8,
    "ml.g6.16xlarge": 1,
    "ml.g6.12xlarge": 4,
    "ml.g6.24xlarge": 4,
    "ml.g6.48xlarge": 8,
    "ml.g6e.16xlarge": 1,
    "ml.g6e.12xlarge": 4,
    "ml.g6e.24xlarge": 4,
    "ml.g6e.48xlarge": 8,
    "ml.g7e.2xlarge": 1,
    "ml.g7e.4xlarge": 1,
    "ml.g7e.8xlarge": 1,
    "ml.g7e.12xlarge": 2,
    "ml.g7e.24xlarge": 4,
    "ml.g7e.48xlarge": 8,
    "ml.p4d.24xlarge": 8,
    "ml.p5.48xlarge": 8,
    "ml.p5e.48xlarge": 8,
    "ml.p5en.48xlarge": 8,
}

_DLC_TAGS = {
    "vllm": re.compile(
        r"^(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?"
        r"-gpu-py312-cu(\d+)-ubuntu[\d.]+-sagemaker$"
    ),
    "sglang": re.compile(
        r"^(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?"
        r"-gpu-py312-cu(\d+)-ubuntu[\d.]+-sagemaker$"
    ),
}


class DeploymentFailure(RuntimeError):
    """A SageMaker resource reached Failed state."""


def gpus_for(instance: str) -> int:
    try:
        return INSTANCE_GPUS[instance]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GPU count for {instance!r}. Pass --num-gpu explicitly."
        ) from exc


def cuda_compatible(instance: str, cuda_version: int) -> bool:
    if instance.startswith(("ml.g7e.", "ml.p6")):
        return cuda_version >= 130
    if instance.startswith(("ml.g5.", "ml.p4")):
        return cuda_version == 128
    return cuda_version >= 129


def latest_dlc(engine: str, region: str, instance: str) -> str:
    """Resolve the highest compatible semantic version from the AWS DLC ECR repo."""
    pattern = _DLC_TAGS[engine]
    ecr = boto3.client("ecr", region_name=region)
    candidates = []
    for page in ecr.get_paginator("describe_images").paginate(
        registryId="763104351884",
        repositoryName=engine,
    ):
        for image in page["imageDetails"]:
            for tag in image.get("imageTags", []):
                match = pattern.match(tag)
                if not match:
                    continue
                cuda_version = int(match.group(5))
                if not cuda_compatible(instance, cuda_version):
                    continue
                version = tuple(int(match.group(i) or 0) for i in (1, 2, 3, 4))
                candidates.append((version, cuda_version, tag))
    if not candidates:
        raise RuntimeError(
            f"No compatible SageMaker {engine} DLC found for {instance} in {region}"
        )
    _, _, tag = max(candidates)
    return (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/"
        f"{engine}:{tag}"
    )


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an S3 URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/") + "/"


def inspect_s3_model(
    s3,
    model_s3: str,
    require_manifest: bool = True,
) -> dict:
    bucket, prefix = parse_s3_uri(model_s3)
    files = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        files.extend(page.get("Contents", []))
    relative = [item["Key"][len(prefix):] for item in files]
    missing = [
        name for name in ("config.json",)
        if name not in relative
    ]
    weight_files = [
        name for name in relative
        if name.endswith((".safetensors", ".bin", ".gguf"))
    ]
    manifest_name = ".hf-model-manifest.json"
    manifest = None
    if manifest_name in relative:
        response = s3.get_object(Bucket=bucket, Key=prefix + manifest_name)
        try:
            manifest = json.loads(response["Body"].read())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"Invalid staging manifest at {model_s3}{manifest_name}"
            ) from exc
        manifest_files = manifest.get("files") if isinstance(manifest, dict) else None
        if (
            not isinstance(manifest_files, list)
            or not manifest_files
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("path"), str)
                or not isinstance(item.get("size"), int)
                for item in manifest_files
            )
        ):
            raise RuntimeError(
                f"Invalid staging manifest at {model_s3}{manifest_name}"
            )
        inventory = {
            item["Key"][len(prefix):]: item["Size"]
            for item in files
        }
        invalid = [
            item["path"]
            for item in manifest_files
            if inventory.get(item["path"]) != item["size"]
        ]
        if invalid:
            preview = ", ".join(invalid[:5])
            raise RuntimeError(
                f"Incomplete model at {model_s3}: {len(invalid)} manifest files "
                f"are missing or have the wrong size ({preview})"
            )

    if require_manifest and manifest is None:
        missing.append(manifest_name)
    if missing or not weight_files:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if not weight_files:
            details.append("no supported weight files")
        raise RuntimeError(
            f"Incomplete model at {model_s3}: {'; '.join(details)}. "
            "Run stage_model.py first."
        )
    return {
        "files": len(files),
        "weight_files": len(weight_files),
        "bytes": sum(item["Size"] for item in files),
        "has_manifest": manifest is not None,
        "revision": manifest.get("revision") if manifest else None,
        "source_model": manifest.get("repo_id") if manifest else None,
    }


def quota_for_instance(session, instance: str) -> float | None:
    quota_name = f"{instance} for endpoint usage"
    client = session.client("service-quotas")
    for page in client.get_paginator("list_service_quotas").paginate(
        ServiceCode="sagemaker"
    ):
        for quota in page["Quotas"]:
            if quota["QuotaName"] == quota_name:
                return float(quota["Value"])
    return None


def build_env(
    engine: str,
    num_gpu: int,
    max_model_len: int | None,
    max_num_seqs: int | None,
    extra_env: dict,
) -> dict:
    if engine == "vllm":
        env = {
            "SM_VLLM_MODEL": "/opt/ml/model",
            "SM_VLLM_TENSOR_PARALLEL_SIZE": str(num_gpu),
        }
        if max_num_seqs is not None:
            env["SM_VLLM_MAX_NUM_SEQS"] = str(max_num_seqs)
        if max_model_len is not None:
            env["SM_VLLM_MAX_MODEL_LEN"] = str(max_model_len)
    else:
        env = {
            "SM_SGLANG_MODEL_PATH": "/opt/ml/model",
            "SM_SGLANG_TP": str(num_gpu),
        }
        if max_model_len is not None:
            env["SM_SGLANG_CONTEXT_LENGTH"] = str(max_model_len)
    for key, value in extra_env.items():
        if isinstance(value, bool):
            # The SageMaker DLC entrypoints translate SM_VLLM_* and
            # SM_SGLANG_* variables directly into CLI arguments. Boolean
            # options are presence-only flags, so "true" would become an
            # invalid extra positional value.
            if not value:
                continue
            normalized = ""
        elif isinstance(value, (dict, list)):
            normalized = json.dumps(value)
        elif value is None:
            raise ValueError(f"--env value for {key} cannot be null")
        else:
            normalized = str(value)
        if key in env and normalized != env[key]:
            raise ValueError(
                f"--env cannot override derived setting {key}={env[key]!r}"
            )
        env[key] = normalized
    return env


def redacted_env(env: dict) -> dict:
    secret_terms = ("TOKEN", "PASSWORD", "SECRET", "KEY")
    return {
        key: ("***" if any(term in key.upper() for term in secret_terms) else value)
        for key, value in env.items()
    }


def resource_name(stem: str, engine: str, attempt: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9-]+", "-", stem).strip("-")
    suffix = f"-{engine}-{time.strftime('%y%m%d-%H%M%S')}"
    if attempt > 1:
        suffix += f"-a{attempt}"
    return (clean[: 63 - len(suffix)] + suffix).strip("-")


def wait_for_status(
    describe,
    name_key: str,
    name: str,
    status_key: str,
    label: str,
    deadline_seconds: int,
) -> dict:
    started = time.time()
    while True:
        description = describe(**{name_key: name})
        status = description[status_key]
        print(f"  {label}: {status} (+{int(time.time() - started)}s)")
        if status == "InService":
            return description
        if status == "Failed":
            reason = description.get("FailureReason", "no failure reason returned")
            raise DeploymentFailure(f"{label} {name} failed: {reason}")
        if time.time() - started > deadline_seconds:
            raise DeploymentFailure(
                f"{label} {name} did not reach InService within {deadline_seconds}s"
            )
        time.sleep(30)


def production_variant(
    name: str,
    instance: str,
    timeout: int,
    ami: str | None,
    capacity_reservation_arn: str | None,
) -> dict:
    variant = {
        "VariantName": "v1",
        "InstanceType": instance,
        "InitialInstanceCount": 1,
        "ModelDataDownloadTimeoutInSeconds": timeout,
        "ContainerStartupHealthCheckTimeoutInSeconds": timeout,
    }
    if name:
        variant["ModelName"] = name
    if ami:
        variant["InferenceAmiVersion"] = ami
    if capacity_reservation_arn:
        variant["CapacityReservationConfig"] = {
            "CapacityReservationPreference": "capacity-reservations-only",
            "MlReservationArn": capacity_reservation_arn,
        }
    return variant


def delete_failed_resources(
    sm,
    endpoint: str,
    endpoint_config: str,
    model: str,
    inference_component: str | None = None,
) -> None:
    if inference_component:
        try:
            sm.delete_inference_component(
                InferenceComponentName=inference_component
            )
            deadline = time.time() + 900
            while time.time() < deadline:
                try:
                    sm.describe_inference_component(
                        InferenceComponentName=inference_component
                    )
                    time.sleep(15)
                except ClientError as exc:
                    if exc.response["Error"]["Code"] == "ValidationException":
                        break
                    raise
        except ClientError:
            pass
    try:
        sm.delete_endpoint(EndpointName=endpoint)
        deadline = time.time() + 900
        while time.time() < deadline:
            try:
                sm.describe_endpoint(EndpointName=endpoint)
                time.sleep(15)
            except ClientError as exc:
                if exc.response["Error"]["Code"] == "ValidationException":
                    break
                raise
    except ClientError:
        pass
    for delete, kwargs in (
        (sm.delete_endpoint_config, {"EndpointConfigName": endpoint_config}),
        (sm.delete_model, {"ModelName": model}),
    ):
        try:
            delete(**kwargs)
        except ClientError:
            pass


def deploy_once(
    sm,
    role: str,
    name: str,
    ic_name: str,
    image: str,
    env: dict,
    model_s3: str,
    instance: str,
    num_gpu: int,
    deployment_mode: str,
    min_memory_mb: int,
    timeout: int,
    wait_timeout: int,
    ami: str | None,
    capacity_reservation_arn: str | None,
) -> tuple[str, str | None]:
    tags = [
        {"Key": "Project", "Value": "sample-sagemaker-agentic-model-deployment"},
        {"Key": "ServingEngine", "Value": image.split("/")[-1].split(":")[0]},
    ]
    sm.create_model(
        ModelName=name,
        ExecutionRoleArn=role,
        PrimaryContainer={
            "Image": image,
            "Environment": env,
            "ModelDataSource": {
                "S3DataSource": {
                    "S3Uri": model_s3,
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                },
            },
        },
        Tags=tags,
    )
    print("Model created:", name)

    standard = deployment_mode == "standard"
    variant = production_variant(
        name if standard else "",
        instance,
        timeout,
        ami,
        capacity_reservation_arn,
    )
    endpoint_config_args = {
        "EndpointConfigName": name,
        "ProductionVariants": [variant],
        "Tags": tags,
    }
    if not standard:
        endpoint_config_args["ExecutionRoleArn"] = role
    sm.create_endpoint_config(**endpoint_config_args)
    sm.create_endpoint(
        EndpointName=name,
        EndpointConfigName=name,
        Tags=tags,
    )
    print("Endpoint creating...")
    wait_for_status(
        sm.describe_endpoint,
        "EndpointName",
        name,
        "EndpointStatus",
        "endpoint",
        wait_timeout,
    )
    if standard:
        return name, None

    sm.create_inference_component(
        InferenceComponentName=ic_name,
        EndpointName=name,
        VariantName="v1",
        Specification={
            "ModelName": name,
            "StartupParameters": {
                "ModelDataDownloadTimeoutInSeconds": timeout,
                "ContainerStartupHealthCheckTimeoutInSeconds": timeout,
            },
            "ComputeResourceRequirements": {
                "MinMemoryRequiredInMb": min_memory_mb,
                "NumberOfAcceleratorDevicesRequired": num_gpu,
            },
        },
        RuntimeConfig={"CopyCount": 1},
        Tags=tags,
    )
    print("Inference component creating...")
    wait_for_status(
        sm.describe_inference_component,
        "InferenceComponentName",
        ic_name,
        "InferenceComponentStatus",
        "inference component",
        wait_timeout,
    )
    return name, ic_name


def smoke_test(runtime, endpoint: str, ic_name: str | None, extra_payload: dict) -> None:
    payload = {
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "max_tokens": 128,
        **extra_payload,
    }
    kwargs = {
        "EndpointName": endpoint,
        "Body": json.dumps(payload),
        "ContentType": "application/json",
    }
    if ic_name:
        kwargs["InferenceComponentName"] = ic_name
    response = runtime.invoke_endpoint(**kwargs)
    body = json.loads(response["Body"].read())
    message = body.get("choices", [{}])[0].get("message", {})
    answer = (
        message.get("content")
        or message.get("reasoning")
        or message.get("reasoning_content")
    )
    if not answer:
        raise RuntimeError(f"Smoke test returned no text: {json.dumps(body)[:1000]}")
    print("SMOKE_TEST_RESPONSE=", json.dumps(body)[:1000])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deploy an S3-staged open-weight model to SageMaker AI."
    )
    parser.add_argument("--engine", choices=("vllm", "sglang"), default="vllm")
    parser.add_argument(
        "--model-id",
        required=True,
        help="friendly SageMaker resource-name stem",
    )
    parser.add_argument(
        "--model-s3",
        required=True,
        help="S3 prefix containing Hugging Face model files",
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--fallback-instance")
    parser.add_argument("--num-gpu", type=int)
    parser.add_argument(
        "--deployment-mode",
        choices=("standard", "inference-component"),
        default="standard",
    )
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument(
        "--min-memory-mb",
        type=int,
        help="required host memory for an Inference Component",
    )
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--wait-timeout", type=int, default=3 * 60 * 60)
    parser.add_argument("--inference-ami-version")
    parser.add_argument("--capacity-reservation-arn")
    parser.add_argument("--image", help="explicit DLC image URI instead of live resolution")
    parser.add_argument(
        "--env",
        default="{}",
        help="extra engine environment variables as a JSON object",
    )
    parser.add_argument(
        "--smoke-inputs",
        default="{}",
        help="additional fields for the OpenAI-compatible smoke-test request",
    )
    parser.add_argument(
        "--smoke-read-timeout",
        type=int,
        default=900,
        help="runtime response timeout in seconds",
    )
    parser.add_argument("--skip-s3-check", action="store_true")
    parser.add_argument(
        "--allow-unverified-s3",
        action="store_true",
        help="allow a model prefix without a stage_model.py manifest",
    )
    parser.add_argument("--no-smoke-test", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    args = parser.parse_args()

    region = config.region()
    session = boto3.session.Session(region_name=region)
    role = config.execution_role_arn(session)
    engine = args.engine
    model_id = args.model_id
    model_s3 = args.model_s3
    instance = args.instance
    deployment_mode = args.deployment_mode
    max_model_len = args.max_model_len
    max_num_seqs = args.max_num_seqs
    min_memory_mb = args.min_memory_mb
    timeout = args.timeout
    ami = args.inference_ami_version
    try:
        extra_env = json.loads(args.env)
    except json.JSONDecodeError as exc:
        parser.error(f"--env is not valid JSON: {exc}")
    if not isinstance(extra_env, dict):
        parser.error("--env must be a JSON object")
    try:
        smoke_inputs = json.loads(args.smoke_inputs)
    except json.JSONDecodeError as exc:
        parser.error(f"--smoke-inputs is not valid JSON: {exc}")
    if not isinstance(smoke_inputs, dict):
        parser.error("--smoke-inputs must be a JSON object")
    if deployment_mode == "inference-component" and min_memory_mb is None:
        parser.error("--min-memory-mb is required for --deployment-mode inference-component")
    if args.num_gpu is not None and args.num_gpu < 1:
        parser.error("--num-gpu must be at least 1")
    if max_model_len is not None and max_model_len < 1:
        parser.error("--max-model-len must be at least 1")
    if max_num_seqs is not None and max_num_seqs < 1:
        parser.error("--max-num-seqs must be at least 1")

    instances = [instance]
    if args.fallback_instance and args.fallback_instance != instance:
        instances.append(args.fallback_instance)

    s3_summary = None
    if not args.skip_s3_check:
        s3_summary = inspect_s3_model(
            session.client("s3"),
            model_s3,
            require_manifest=not args.allow_unverified_s3,
        )

    print("=== DEPLOY PLAN ===")
    print(f"  region       : {region}")
    print(f"  model id     : {model_id}")
    print(f"  weights      : {model_s3}")
    if s3_summary:
        print(
            f"  S3 contents  : {s3_summary['files']} files, "
            f"{s3_summary['bytes'] / (1024 ** 3):,.2f} GiB, "
            f"manifest={'yes' if s3_summary['has_manifest'] else 'no'}"
        )
        if s3_summary["source_model"]:
            print(
                f"  source       : {s3_summary['source_model']}"
                f"@{s3_summary['revision']}"
            )
    print(f"  engine       : {engine}")
    print(f"  deploy mode  : {deployment_mode}")
    print(f"  instances    : {', '.join(instances)}")
    print(f"  context cap  : {max_model_len or '(engine/model default)'}")
    if not args.deploy:
        preview_gpu = args.num_gpu or gpus_for(instance)
        image = args.image or latest_dlc(engine, region, instance)
        env = build_env(
            engine,
            preview_gpu,
            max_model_len,
            max_num_seqs,
            extra_env,
        )
        print(f"  container    : {image}")
        print(f"  env          : {json.dumps(redacted_env(env), sort_keys=True)}")
        print("\nDRY RUN - add --deploy to create the billable endpoint.")
        return 0

    sm = session.client("sagemaker")
    runtime = session.client(
        "sagemaker-runtime",
        config=Config(
            connect_timeout=10,
            read_timeout=args.smoke_read_timeout,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )
    started = time.time()
    last_error = None
    for attempt, candidate in enumerate(instances, start=1):
        num_gpu = args.num_gpu or gpus_for(candidate)
        quota = quota_for_instance(session, candidate)
        if quota is not None and quota < 1:
            last_error = RuntimeError(
                f"Endpoint quota for {candidate} is {quota:g}; request at least 1 first"
            )
            print(f"SKIP {candidate}: {last_error}")
            continue
        image = args.image or latest_dlc(engine, region, candidate)
        env = build_env(
            engine,
            num_gpu,
            max_model_len,
            max_num_seqs,
            extra_env,
        )
        name = resource_name(model_id, engine, attempt)
        ic_name = f"ic-{name}"[:63]
        print(f"\n=== DEPLOY ATTEMPT {attempt}/{len(instances)} ===")
        print(f"  endpoint  : {name}")
        print(f"  instance  : {candidate} ({num_gpu} GPUs, quota={quota})")
        print(f"  container : {image}")
        print(f"  env       : {json.dumps(redacted_env(env), sort_keys=True)}")
        try:
            endpoint, ic = deploy_once(
                sm,
                role,
                name,
                ic_name,
                image,
                env,
                model_s3,
                candidate,
                num_gpu,
                deployment_mode,
                min_memory_mb or 0,
                timeout,
                args.wait_timeout,
                ami,
                args.capacity_reservation_arn,
            )
            print("\n=== DEPLOYED ===")
            print(f"ENDPOINT_NAME={endpoint}")
            print(f"IC_NAME={ic or ''}")
            print(f"TIMING total_deploy_sec={int(time.time() - started)}")
            if not args.no_smoke_test:
                try:
                    smoke_test(runtime, endpoint, ic, smoke_inputs)
                    print("SMOKE_TEST=passed")
                except Exception as exc:
                    print(f"SMOKE_TEST=failed ({type(exc).__name__}: {exc})")
                    print(
                        "The endpoint remains InService. Run smoke_test.py for diagnosis "
                        "or teardown.py to stop billing."
                    )
                    return 1
            return 0
        except (ClientError, DeploymentFailure, RuntimeError) as exc:
            last_error = exc
            print(f"DEPLOY FAILED on {candidate}: {exc}")
            delete_failed_resources(
                sm,
                name,
                name,
                name,
                ic_name if deployment_mode == "inference-component" else None,
            )
            capacity_error = "capacity" in str(exc).lower()
            if attempt < len(instances) and capacity_error:
                print("Retrying on the configured fallback instance.")
                continue
            break

    print(f"ERROR: no deployment succeeded: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ClientError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
