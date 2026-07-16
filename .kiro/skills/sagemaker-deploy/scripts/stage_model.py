#!/usr/bin/env python3
"""Stream a Hugging Face model into S3 without storing weights on the caller's disk.

The controller starts a small SageMaker Processing job. The worker in that managed
job reads each Hugging Face file as an HTTP stream and writes it with an S3 multipart
upload. Completed objects are checked by source revision, blob ID, and size, so a
stopped job can resume without copying finished shards again.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

import config


MIB = 1024 * 1024
GIB = 1024 * MIB
MANIFEST_NAME = ".hf-model-manifest.json"
USER_AGENT = "sample-sagemaker-agentic-model-deployment/1.0"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an S3 URI, got {uri!r}")
    prefix = parsed.path.lstrip("/")
    return parsed.netloc, prefix.rstrip("/") + "/"


def human_bytes(value: int) -> str:
    if value >= GIB:
        return f"{value / GIB:,.2f} GiB"
    if value >= MIB:
        return f"{value / MIB:,.2f} MiB"
    return f"{value:,} bytes"


def default_destination(repo_id: str) -> str:
    """Build an account-neutral destination from the complete repository ID."""
    stem = repo_id.lower().replace("/", "--")
    stem = "".join(char if char.isalnum() or char in ".-" else "-" for char in stem)
    stem = stem.strip(".-")
    return f"s3://{config.bucket()}/models/{stem}/"


def hf_api_url(repo_id: str, revision: str | None = None) -> str:
    repo = urllib.parse.quote(repo_id, safe="/")
    if revision:
        rev = urllib.parse.quote(revision, safe="")
        return f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
    return f"https://huggingface.co/api/models/{repo}?blobs=true"


def request_headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def validate_https_url(url: str, require_huggingface_host: bool = False) -> None:
    """Reject local, credential-bearing, and non-HTTPS request targets."""
    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid URL port in {url!r}") from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise ValueError(f"Only credential-free HTTPS URLs are allowed, got {url!r}")
    if require_huggingface_host and parsed.hostname != "huggingface.co":
        raise ValueError(
            f"Initial model requests must target huggingface.co, got {url!r}"
        )


class HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Permit Hugging Face CDN redirects without allowing unsafe URL schemes."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_hf_url(url: str, timeout: int):
    validate_https_url(url, require_huggingface_host=True)
    request = urllib.request.Request(url, headers=request_headers())
    opener = urllib.request.build_opener(HttpsOnlyRedirectHandler())
    return opener.open(request, timeout=timeout)


def load_hf_token(secret_id: str) -> None:
    """Load a Hugging Face token from Secrets Manager without exposing its value."""
    response = boto3.client(
        "secretsmanager",
        region_name=config.region(),
    ).get_secret_value(SecretId=secret_id)
    secret = response.get("SecretString")
    if not secret:
        raise ValueError(f"Secret {secret_id!r} does not contain a SecretString")
    try:
        parsed = json.loads(secret)
    except json.JSONDecodeError:
        token = secret
    else:
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Secret {secret_id!r} must be a token string or JSON object"
            )
        token = (
            parsed.get("HF_TOKEN")
            or parsed.get("token")
            or parsed.get("access_token")
        )
    if not token or not isinstance(token, str):
        raise ValueError(
            f"Secret {secret_id!r} does not contain HF_TOKEN, token, or access_token"
        )
    os.environ["HF_TOKEN"] = token


def read_json(url: str) -> dict:
    with open_hf_url(url, timeout=60) as response:
        return json.load(response)


def model_manifest(repo_id: str, revision: str | None = None) -> dict:
    info = read_json(hf_api_url(repo_id, revision))
    resolved_revision = info["sha"]
    files = []
    for item in info.get("siblings", []):
        size = item.get("size")
        lfs = item.get("lfs") or {}
        blob_id = item.get("blobId") or lfs.get("oid") or ""
        if size is None:
            size = lfs.get("size", 0)
        files.append({
            "path": item["rfilename"],
            "size": int(size or 0),
            "blob_id": blob_id,
        })
    if not files:
        raise RuntimeError(f"Hugging Face returned no files for {repo_id}@{resolved_revision}")
    return {
        "schema_version": 1,
        "source": "huggingface",
        "repo_id": repo_id,
        "revision": resolved_revision,
        "files": sorted(files, key=lambda item: item["path"]),
    }


def object_matches(s3, bucket: str, key: str, source: dict) -> bool:
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    metadata = head.get("Metadata", {})
    return (
        head["ContentLength"] == source["size"]
        and metadata.get("hf-revision") == source["revision"]
        and metadata.get("hf-blob-id", "") == source["blob_id"]
    )


def destination_status(s3, destination: str, manifest: dict) -> tuple[list[dict], int]:
    bucket, prefix = parse_s3_uri(destination)
    missing = []
    present_bytes = 0
    for item in manifest["files"]:
        source = {
            **item,
            "revision": manifest["revision"],
        }
        if object_matches(s3, bucket, prefix + item["path"], source):
            present_bytes += item["size"]
        else:
            missing.append(item)
    return missing, present_bytes


def manifest_matches(s3, destination: str, manifest: dict) -> bool:
    bucket, prefix = parse_s3_uri(destination)
    try:
        response = s3.get_object(Bucket=bucket, Key=prefix + MANIFEST_NAME)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    try:
        stored = json.loads(response["Body"].read())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    return all(
        stored.get(key) == manifest.get(key)
        for key in ("schema_version", "source", "repo_id", "revision", "files")
    )


def write_manifest(s3, destination: str, manifest: dict) -> None:
    bucket, prefix = parse_s3_uri(destination)
    completed = {
        **manifest,
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_bytes": sum(item["size"] for item in manifest["files"]),
    }
    s3.put_object(
        Bucket=bucket,
        Key=prefix + MANIFEST_NAME,
        Body=json.dumps(completed, indent=2).encode(),
        ContentType="application/json",
    )


def resolve_processing_image(region: str, instance_type: str) -> str:
    try:
        from sagemaker.core import image_uris
    except ImportError:
        try:
            from sagemaker import image_uris
        except ImportError as exc:
            raise RuntimeError(
                "The SageMaker Python SDK is required to resolve the managed "
                "Processing image. Install the repository requirements."
            ) from exc
    return image_uris.retrieve(
        "sklearn",
        region,
        version="1.2-1",
        image_scope="training",
        instance_type=instance_type,
        py_version="py3",
    )


def processing_job_name(repo_id: str) -> str:
    stem = repo_id.rsplit("/", 1)[-1].lower()
    stem = "".join(c if c.isalnum() else "-" for c in stem).strip("-")
    return f"stage-{stem[:34]}-{time.strftime('%y%m%d-%H%M%S')}"


def start_processing_job(args, manifest: dict, destination: str) -> tuple[object, str]:
    region = config.region()
    sess = boto3.session.Session(region_name=region)
    sm = sess.client("sagemaker")
    s3 = sess.client("s3")
    role = config.execution_role_arn(sess)
    bucket = config.bucket(sess)
    job_name = processing_job_name(manifest["repo_id"])
    code_prefix = f"model-staging/code/{job_name}/"
    source_dir = Path(__file__).resolve().parent
    for filename in ("stage_model.py", "config.py"):
        s3.upload_file(str(source_dir / filename), bucket, code_prefix + filename)
    code_s3 = f"s3://{bucket}/{code_prefix}"
    image = args.processing_image or resolve_processing_image(
        region, args.processing_instance
    )

    worker_args = [
        "--worker",
        "--hf-model-id", manifest["repo_id"],
        "--revision", manifest["revision"],
        "--destination", destination,
        "--workers", str(args.workers),
        "--retries", str(args.retries),
    ]
    if args.only:
        worker_args.extend(["--only", args.only])
    if args.hf_token_secret_id:
        worker_args.extend(["--hf-token-secret-id", args.hf_token_secret_id])

    print("=== SAGEMAKER PROCESSING PLAN ===")
    print(f"  job         : {job_name}")
    print(f"  role        : {role}")
    print(f"  image       : {image}")
    print(f"  instance    : {args.processing_instance}")
    print(f"  code        : {code_s3}")
    print(f"  destination : {destination}")
    print(f"  workers     : {args.workers}")

    sm.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn=role,
        AppSpecification={
            "ImageUri": image,
            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/input/code/stage_model.py",
            ],
            "ContainerArguments": worker_args,
        },
        ProcessingInputs=[{
            "InputName": "code",
            "S3Input": {
                "S3Uri": code_s3,
                "LocalPath": "/opt/ml/processing/input/code",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        }],
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": args.processing_instance,
                "VolumeSizeInGB": args.volume_size,
            },
        },
        StoppingCondition={"MaxRuntimeInSeconds": args.max_runtime},
        Tags=[
            {"Key": "Project", "Value": "sample-sagemaker-agentic-model-deployment"},
            {"Key": "Purpose", "Value": "model-staging"},
        ],
    )
    print(f"PROCESSING_JOB_NAME={job_name}")
    return sm, job_name


def wait_for_processing_job(sm, job_name: str) -> dict:
    started = time.time()
    terminal = {"Completed", "Failed", "Stopped"}
    while True:
        description = sm.describe_processing_job(ProcessingJobName=job_name)
        status = description["ProcessingJobStatus"]
        print(f"  processing: {status} (+{int(time.time() - started)}s)")
        if status in terminal:
            if status != "Completed":
                reason = description.get("FailureReason", "no failure reason returned")
                raise RuntimeError(f"Processing job {status}: {reason}")
            return description
        time.sleep(30)


def source_url(repo_id: str, revision: str, path: str) -> str:
    repo = urllib.parse.quote(repo_id, safe="/")
    rev = urllib.parse.quote(revision, safe="")
    filename = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/{repo}/resolve/{rev}/{filename}?download=true"


def upload_one(
    s3,
    bucket: str,
    prefix: str,
    repo_id: str,
    revision: str,
    item: dict,
    retries: int,
) -> str:
    key = prefix + item["path"]
    source = {**item, "revision": revision}
    if object_matches(s3, bucket, key, source):
        print(f"SKIP {item['path']} ({human_bytes(item['size'])})")
        return "skipped"

    transfer = TransferConfig(
        multipart_threshold=64 * MIB,
        multipart_chunksize=64 * MIB,
        max_concurrency=1,
        use_threads=False,
    )
    url = source_url(repo_id, revision, item["path"])
    for attempt in range(1, retries + 1):
        try:
            print(
                f"START {item['path']} ({human_bytes(item['size'])}, "
                f"attempt {attempt}/{retries})"
            )
            with open_hf_url(url, timeout=120) as response:
                s3.upload_fileobj(
                    response,
                    bucket,
                    key,
                    ExtraArgs={
                        "Metadata": {
                            "hf-repo-id": repo_id,
                            "hf-revision": revision,
                            "hf-blob-id": item["blob_id"],
                        },
                    },
                    Config=transfer,
                )
            if not object_matches(s3, bucket, key, source):
                raise RuntimeError("uploaded object did not pass size/metadata verification")
            print(f"DONE  {item['path']} ({human_bytes(item['size'])})")
            return "uploaded"
        except Exception as exc:
            print(f"RETRY {item['path']}: {type(exc).__name__}: {exc}")
            if attempt == retries:
                raise
            time.sleep(min(60, 5 * attempt))
    raise AssertionError("unreachable")


def run_worker(args) -> int:
    if args.hf_token_secret_id:
        load_hf_token(args.hf_token_secret_id)
    manifest = model_manifest(args.hf_model_id, args.revision)
    if args.only:
        manifest["files"] = [
            item for item in manifest["files"] if item["path"] == args.only
        ]
        if not manifest["files"]:
            raise ValueError(f"File {args.only!r} is not present in the model repository")
    bucket, prefix = parse_s3_uri(args.destination)
    s3 = boto3.client("s3", region_name=config.region())
    total = sum(item["size"] for item in manifest["files"])
    print(
        f"Streaming {len(manifest['files'])} files ({human_bytes(total)}) from "
        f"{manifest['repo_id']}@{manifest['revision']} to {args.destination}"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                upload_one,
                s3,
                bucket,
                prefix,
                manifest["repo_id"],
                manifest["revision"],
                item,
                args.retries,
            )
            for item in manifest["files"]
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    if not args.only:
        write_manifest(s3, args.destination, manifest)
    print(f"STAGED_MODEL_S3={args.destination}")
    print(f"STAGED_REVISION={manifest['revision']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream Hugging Face model files directly into S3 with SageMaker Processing."
    )
    parser.add_argument(
        "--hf-model-id",
        required=True,
        help="Hugging Face repository, for example organization/model-name",
    )
    parser.add_argument("--revision", help="branch, tag, or immutable commit SHA")
    parser.add_argument("--destination", help="destination S3 prefix")
    parser.add_argument(
        "--hf-token-secret-id",
        help="Secrets Manager ID containing a gated-model Hugging Face token",
    )
    parser.add_argument("--processing-instance", default="ml.m5.4xlarge")
    parser.add_argument("--processing-image")
    parser.add_argument("--volume-size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-runtime", type=int, default=12 * 60 * 60)
    parser.add_argument("--only", help=argparse.SUPPRESS)
    parser.add_argument("--run", action="store_true", help="start the Processing job")
    parser.add_argument("--no-wait", action="store_true", help="return after job submission")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")

    if args.worker:
        if not args.hf_model_id or not args.revision or not args.destination:
            parser.error("--worker requires --hf-model-id, --revision, and --destination")
        return run_worker(args)

    if args.hf_token_secret_id:
        load_hf_token(args.hf_token_secret_id)
    elif args.run and os.environ.get("HF_TOKEN"):
        parser.error(
            "--run cannot forward a local HF_TOKEN securely; store it in "
            "Secrets Manager and pass --hf-token-secret-id"
        )

    repo_id = args.hf_model_id
    destination = args.destination or default_destination(repo_id)
    manifest = model_manifest(repo_id, args.revision)
    if args.only:
        manifest["files"] = [
            item for item in manifest["files"] if item["path"] == args.only
        ]
        if not manifest["files"]:
            parser.error(f"--only file {args.only!r} is not present in the model repository")
    total = sum(item["size"] for item in manifest["files"])
    sess = boto3.session.Session(region_name=config.region())
    s3 = sess.client("s3")
    missing, present_bytes = destination_status(s3, destination, manifest)
    complete_manifest = manifest_matches(s3, destination, manifest)

    print("=== MODEL STAGING PLAN ===")
    print(f"  source      : {repo_id}@{manifest['revision']}")
    print(f"  destination : {destination}")
    print(f"  files       : {len(manifest['files'])}")
    print(f"  source size : {human_bytes(total)}")
    print(f"  present     : {human_bytes(present_bytes)}")
    print(f"  remaining   : {len(missing)} files, {human_bytes(sum(x['size'] for x in missing))}")
    print(f"  manifest    : {'verified' if complete_manifest else 'missing or stale'}")
    print("  local disk  : model weights are never written to the caller's disk")
    if not missing:
        if not complete_manifest:
            if not args.run:
                print(
                    "\nWeights are complete. Add --run to write the verified manifest."
                )
                return 0
            write_manifest(s3, destination, manifest)
            print("\nDestination is complete; wrote the verified manifest.")
        else:
            print("\nDestination is complete; no Processing job is needed.")
        print(f"STAGED_MODEL_S3={destination}")
        print(f"STAGED_REVISION={manifest['revision']}")
        return 0
    if not args.run:
        print("\nDRY RUN - add --run to start the managed SageMaker Processing job.")
        return 0

    sm, job_name = start_processing_job(args, manifest, destination)
    if args.no_wait:
        return 0
    wait_for_processing_job(sm, job_name)
    missing_after, present_after = destination_status(
        sess.client("s3"), destination, manifest
    )
    if missing_after:
        print(
            f"ERROR: Processing completed but {len(missing_after)} files failed verification.",
            file=sys.stderr,
        )
        return 1
    print(f"Verified {len(manifest['files'])} files ({human_bytes(present_after)}) in S3.")
    print(f"STAGED_MODEL_S3={destination}")
    print(f"STAGED_REVISION={manifest['revision']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ClientError, RuntimeError, ValueError, urllib.error.URLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
