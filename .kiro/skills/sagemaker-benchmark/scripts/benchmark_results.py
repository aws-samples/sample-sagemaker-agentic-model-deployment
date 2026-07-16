#!/usr/bin/env python3
"""Read and present the results of a managed SageMaker AI benchmark job.

This is the final beat of the ``sagemaker-benchmark`` contract: after the job
finishes, pull the AIPerf output bundle from S3, show **what landed there** (the
full file tree), and surface the headline numbers — so "how fast is it?" gets
answered in the terminal instead of by digging through S3 by hand.

What a benchmark job writes to S3 (one folder per job, then one tarball):

    <S3OutputLocation>/output/output.tar.gz   ── extracted ──▶
    output/
    ├── profile_export_aiperf.json   # aggregated metrics — what this script reads
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

The bundle serves two audiences at once: an agent parses
``profile_export_aiperf.json`` for the numbers, while a human gets the PNG plots,
the CSV, and the raw logs for their own analysis.

Good to know: AIPerf enforces a ~1% result-validity gate. A job can be marked
``Failed`` by that gate and *still* have written the complete bundle — the metrics
over the valid requests are sound. This reader handles that case and says so.

Usage:
    python benchmark_results.py                       # latest standalone benchmark job
    python benchmark_results.py --job bench-NAME      # a specific job by name
    python benchmark_results.py --s3 s3://…/prefix/   # read an output prefix directly
    python benchmark_results.py --local DIR           # an already-extracted bundle, e.g.
                                                      #   --local ../sample-output  (bundled
                                                      #   real run — no AWS calls, no waiting)

Read-only: this downloads from S3 (or reads a local folder) and prints. Nothing billable.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import tarfile


# What each file in the bundle is for, printed next to the tree.
FILE_NOTES = {
    "profile_export_aiperf.json": "aggregated metrics (the table below reads this)",
    "profile_export_aiperf.csv": "same aggregates as CSV",
    "profile_export.jsonl": "raw per-request records",
    "inputs.json": "the prompts AIPerf sent",
    "outputs.json": "what the model answered",
    "benchmark_summary.txt": "completion summary",
    "failure_reason.txt": "present only when the validity gate tripped",
    "MANIFEST.txt": "index of all files with sizes",
    "plot_generation.log": "plot generation log",
    "plots/ttft_timeline.png": "TTFT per request over the run",
    "plots/ttft_over_time.png": "TTFT aggregated over the run duration",
    "plots/summary.txt": "list of generated plots",
    "plots/aiperf_plot.log": "plot generation trace",
    "logs/aiperf.log": "full AIPerf execution log",
}

# The headline metrics inside profile_export_aiperf.json. Each metric is an object
# like {"unit": "ms", "avg": …, "p50": …, "p90": …, "p99": …}.
HEADLINES = [
    ("output_token_throughput", "Output token throughput", ("avg",)),
    ("output_token_throughput_per_user", "Per-user token throughput", ("avg", "p50")),
    ("time_to_first_token", "Time to first token (TTFT)", ("avg", "p50", "p90", "p99")),
    ("inter_token_latency", "Inter-token latency (ITL)", ("avg", "p50", "p90", "p99")),
    ("request_latency", "Request latency", ("p50", "p90", "p99")),
    ("request_throughput", "Request throughput", ("avg",)),
]


def resolve_job(sm, name: str | None) -> dict:
    """Describe the requested job, or the most recent standalone benchmark job.

    Jobs named ``ai-rec-*`` are internal candidate runs of a recommendation
    job, not standalone user benchmarks, so the latest-job default skips them.
    """
    if not name:
        jobs = sm.list_ai_benchmark_jobs()["AIBenchmarkJobs"]
        ours = [j for j in jobs if not j["AIBenchmarkJobName"].startswith("ai-rec-")]
        if not ours:
            raise SystemExit("No benchmark jobs found in this account/region.")
        name = ours[0]["AIBenchmarkJobName"]  # the API lists newest first
    return sm.describe_ai_benchmark_job(AIBenchmarkJobName=name)


def fetch_bundle(s3, s3_prefix: str, workdir: pathlib.Path) -> pathlib.Path:
    """Download <prefix>…/output/output.tar.gz and extract it under workdir."""
    bucket_name, key_prefix = s3_prefix.replace("s3://", "").split("/", 1)
    listed = s3.list_objects_v2(Bucket=bucket_name, Prefix=key_prefix)
    keys = [o["Key"] for o in listed.get("Contents", [])]
    tar_keys = [k for k in keys if k.endswith("output.tar.gz")]
    if not tar_keys:
        raise SystemExit(f"No output.tar.gz under {s3_prefix} — did the job write output yet?")
    workdir.mkdir(parents=True, exist_ok=True)
    tar_path = workdir / "output.tar.gz"
    s3.download_file(bucket_name, tar_keys[0], str(tar_path))
    with tarfile.open(tar_path) as tar:
        try:
            tar.extractall(workdir, filter="data")  # safe extraction (Python ≥3.12)
        except TypeError:
            tar.extractall(workdir)
    return workdir


def print_tree(workdir: pathlib.Path) -> None:
    print("\n=== WHAT THE JOB WROTE (the AIPerf bundle) ===")
    files = sorted(p for p in workdir.rglob("*") if p.is_file())
    for p in files:
        rel = p.relative_to(workdir).as_posix()
        if rel == "output.tar.gz" or rel.startswith("ray_tmp"):
            continue
        note = FILE_NOTES.get(rel, "")
        size = p.stat().st_size
        print(f"  {rel:<34} {size:>12,} B  {('# ' + note) if note else ''}")


def fmt(value: float, unit: str) -> str:
    if unit == "ms":
        return f"{value:,.0f} ms" if value >= 100 else f"{value:.1f} ms"
    return f"{value:,.1f} {unit}"


def print_metrics(workdir: pathlib.Path) -> None:
    metrics_file = workdir / "profile_export_aiperf.json"
    if not metrics_file.exists():
        raise SystemExit("Bundle has no profile_export_aiperf.json — cannot summarize.")
    metrics = json.loads(metrics_file.read_text())

    print("\n=== HEADLINE NUMBERS (profile_export_aiperf.json) ===")
    for key, label, stats in HEADLINES:
        m = metrics.get(key)
        if not m:
            continue
        parts = [f"{s} {fmt(m[s], m['unit'])}" for s in stats if s in m]
        print(f"  {label:<28} {'  ·  '.join(parts)}")

    total = int(metrics.get("request_count", {}).get("avg", 0))
    errors = int(metrics.get("error_request_count", {}).get("avg", 0))
    duration = metrics.get("benchmark_duration", {}).get("avg")
    line = f"  {'Requests':<28} {total} total, {errors} invalid"
    if duration:
        line += f", over {duration:,.0f} s"
    print(line)

    plots = sorted((workdir / "plots").glob("*.png")) if (workdir / "plots").exists() else []
    if plots:
        print("\nPlots for humans (open them — TTFT over the whole run):")
        for p in plots:
            print(f"  {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Present a benchmark job's results from S3.")
    ap.add_argument("--job", help="benchmark job name (default: the latest standalone job)")
    ap.add_argument("--s3", help="read this S3 output prefix directly instead of a job name")
    ap.add_argument("--local", help="read an already-extracted bundle folder "
                                    "(e.g. the skill's bundled sample-output) — no AWS calls")
    ap.add_argument("--out", default="bench-results",
                    help="local folder to keep the bundle in (default: ./bench-results)")
    args = ap.parse_args()

    if args.local:
        workdir = pathlib.Path(args.local)
        if not workdir.is_dir():
            raise SystemExit(f"{workdir} is not a directory.")
        print(f"local bundle: {workdir}")
        print_tree(workdir)
        print_metrics(workdir)
        return 0

    import boto3
    import config  # region / role / bucket — auto-detected, nothing hardcoded

    sess = boto3.session.Session(region_name=config.region())
    s3 = sess.client("s3")

    if args.s3:
        prefix, label, status, failure = args.s3, args.s3.rstrip("/").rsplit("/", 1)[-1], None, None
    else:
        job = resolve_job(sess.client("sagemaker"), args.job)
        label = job["AIBenchmarkJobName"]
        status = job["AIBenchmarkJobStatus"]
        failure = job.get("FailureReason", "")
        prefix = job["OutputConfig"]["S3OutputLocation"]
        print(f"job    : {label}")
        print(f"status : {status}")
        print(f"output : {prefix}")

    # A "Failed" verdict from the validity gate still comes with a complete bundle —
    # explain rather than bail, because the per-request metrics are sound.
    if status == "Failed":
        if "error rate" in (failure or ""):
            print(f"\nNOTE: AIPerf's ~1% validity gate marked this run Failed "
                  f"({failure.split('AlgorithmError: ')[-1].splitlines()[0]})")
            print("The bundle below is complete; metrics cover the valid requests.")
        else:
            print(f"\nNOTE: job Failed: {failure or '(no reason reported)'} — "
                  "attempting to read whatever output exists.")

    workdir = fetch_bundle(s3, prefix, pathlib.Path(args.out) / label)
    print_tree(workdir)
    print_metrics(workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
