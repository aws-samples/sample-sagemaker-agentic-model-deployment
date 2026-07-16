#!/usr/bin/env python3
"""Send one OpenAI-compatible chat request to a SageMaker endpoint."""

from __future__ import annotations

import argparse
import json

import boto3
from botocore.config import Config

import config


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test a SageMaker AI endpoint.")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--ic",
        help="inference component name; omit for a standard endpoint",
    )
    parser.add_argument(
        "--prompt",
        default="In one sentence, what is Amazon SageMaker AI?",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=900,
        help="runtime response timeout in seconds",
    )
    parser.add_argument(
        "--extra-inputs",
        default="{}",
        help='additional request fields as JSON, for example '
        '\'{"chat_template_kwargs":{"enable_thinking":false}}\'',
    )
    args = parser.parse_args()

    try:
        extra_inputs = json.loads(args.extra_inputs)
    except json.JSONDecodeError as exc:
        parser.error(f"--extra-inputs is not valid JSON: {exc}")
    if not isinstance(extra_inputs, dict):
        parser.error("--extra-inputs must be a JSON object")

    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        **extra_inputs,
    }
    invoke_args = {
        "EndpointName": args.endpoint,
        "Body": json.dumps(payload),
        "ContentType": "application/json",
    }
    if args.ic:
        invoke_args["InferenceComponentName"] = args.ic

    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=config.region(),
        config=Config(
            connect_timeout=10,
            read_timeout=args.read_timeout,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )
    print(f"PROMPT: {args.prompt}\n")
    response = runtime.invoke_endpoint(**invoke_args)
    body = json.loads(response["Body"].read())
    message = body.get("choices", [{}])[0].get("message", {})
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    if reasoning:
        print("REASONING:", reasoning[:1000])
    answer = message.get("content")
    print("ANSWER   :", answer)
    if body.get("usage"):
        print("USAGE    :", json.dumps(body["usage"]))
    if not (answer or reasoning):
        print("ERROR    : response contained no generated text")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
