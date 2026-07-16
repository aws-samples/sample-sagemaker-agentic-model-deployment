#!/usr/bin/env python3
"""Read endpoint observability from CloudWatch after a benchmark run.

Every SageMaker AI endpoint publishes operational metrics to CloudWatch. This
script pulls the headline metrics that show whether generated load reached the
endpoint:

    Invocations               how many requests were served
    InvocationsPerCopy        load per inference-component copy (the scaling view)
    ConcurrentRequestsPerCopy in-flight requests per copy (the concurrency view)
    ModelLatency              server-side latency (microseconds)
    OverheadLatency           SageMaker routing overhead (microseconds)

Dimension note (matters for Inference Component endpoints):
    For an IC endpoint, the metrics that reliably carry data are published on the
    **InferenceComponentName** dimension. (Invocations on EndpointName+VariantName
    also requires InstanceId, so a name-only query there comes back empty.) So when
    you pass --ic, we query on the IC dimension — that's where the numbers live.

Usage:
    python scripts/cloudwatch_metrics.py --endpoint NAME --ic IC_NAME
    python scripts/cloudwatch_metrics.py --endpoint NAME --ic IC_NAME --minutes 60
"""
import argparse
from datetime import datetime, timedelta, timezone

import boto3

import config


def _get(cw, metric, dims, start, end, stat):
    """Fetch one metric as a list of (timestamp, value) sorted by time."""
    r = cw.get_metric_statistics(
        Namespace="AWS/SageMaker", MetricName=metric, Dimensions=dims,
        StartTime=start, EndTime=end, Period=60, Statistics=[stat])
    pts = sorted(r["Datapoints"], key=lambda d: d["Timestamp"])
    return [(p["Timestamp"], p[stat]) for p in pts]


def main() -> int:
    ap = argparse.ArgumentParser(description="Show CloudWatch metrics for an endpoint.")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--ic", help="inference component name (recommended for IC endpoints)")
    ap.add_argument("--variant", default="v1", help="production variant name (default: v1)")
    ap.add_argument("--minutes", type=int, default=30, help="look-back window")
    args = ap.parse_args()

    sess = boto3.session.Session(region_name=config.region())
    cw = sess.client("cloudwatch")
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=args.minutes)

    # Prefer the inference-component dimension — that's where IC-endpoint data lives.
    # Fall back to endpoint+variant only if no IC was given.
    if args.ic:
        dims = [{"Name": "InferenceComponentName", "Value": args.ic}]
        scope = f"inference component {args.ic}"
    else:
        dims = [{"Name": "EndpointName", "Value": args.endpoint},
                {"Name": "VariantName", "Value": args.variant}]
        scope = f"endpoint {args.endpoint} / {args.variant}"

    print(f"=== CloudWatch — {scope} (last {args.minutes} min) ===\n")

    invocations = _get(cw, "Invocations", dims, start, end, "Sum")
    total = sum(v for _, v in invocations)
    print(f"Invocations (total over window): {int(total)}")
    for ts, v in invocations[-10:]:
        print(f"  {ts:%H:%M}  {int(v)}")
    if not invocations:
        print("  (no datapoints yet — metrics lag a minute or two after a run)")

    lat = _get(cw, "ModelLatency", dims, start, end, "Average")
    if lat:
        # CloudWatch publishes these latency metrics in MICROSECONDS; divide by 1000 for ms.
        avg_ms = sum(v for _, v in lat) / len(lat) / 1000.0
        print(f"\nModelLatency (avg): {avg_ms:.0f} ms")

    overhead = _get(cw, "OverheadLatency", dims, start, end, "Average")
    if overhead:
        avg_ms = sum(v for _, v in overhead) / len(overhead) / 1000.0  # microseconds -> ms
        print(f"OverheadLatency (avg): {avg_ms:.0f} ms")

    # Concurrency story: peak in-flight requests per copy during the run.
    concurrent = _get(cw, "ConcurrentRequestsPerCopy", dims, start, end, "Maximum")
    if concurrent:
        print(f"ConcurrentRequestsPerCopy (peak): {max(v for _, v in concurrent):.0f}")

    print("\nTip: open the endpoint in the SageMaker console -> Monitor for the full graphs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
