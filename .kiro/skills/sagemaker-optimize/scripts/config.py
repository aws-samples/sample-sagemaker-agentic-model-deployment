#!/usr/bin/env python3
"""Shared AWS context for the deploy + benchmark scripts.

Why this file exists
--------------------
Everything the agent does needs four pieces of context: which Region, which AWS
account, which IAM execution role SageMaker should assume, and which S3 bucket to
stage weights / write results into. We deliberately **discover** all four from the
live environment instead of hardcoding them.

The same repository, with nothing edited, resolves to the active AWS
environment. That makes the workflow reproducible rather than account-specific.

Resolution order (first hit wins), so it works in SageMaker Studio AND on a laptop:
  region : AWS_REGION / AWS_DEFAULT_REGION env -> boto3 session -> "us-west-2"
  account: STS get_caller_identity (always available once credentials are set)
  role   : SAGEMAKER_ROLE_ARN env -> the SageMaker SDK's notion of the attached role
           (auto in Studio) -> the SageMaker execution role the CURRENT credentials are
           assuming (Studio terminals) -> a uniquely matching SageMaker execution role
  bucket : SAGEMAKER_BUCKET env -> sagemaker default bucket (sagemaker-<region>-<acct>)
"""
from __future__ import annotations

import os
import boto3

# us-west-2 is a broadly supported default for the GPU families and managed
# inference APIs used by this sample. Environment or session configuration wins.
DEFAULT_REGION = "us-west-2"


def region() -> str:
    """The AWS Region to operate in. Env override wins; otherwise the session default."""
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or boto3.session.Session().region_name
        or DEFAULT_REGION
    )


def account_id(sess: boto3.session.Session | None = None) -> str:
    """The 12-digit AWS account ID, read straight from the caller's credentials.

    We never hardcode this — STS reports whoever is actually running the script,
    so the repo carries no account-specific values into version control.
    """
    sess = sess or boto3.session.Session(region_name=region())
    return sess.client("sts").get_caller_identity()["Account"]


def execution_role_arn(sess: boto3.session.Session | None = None) -> str:
    """The IAM role SageMaker assumes to pull weights from S3 and run the benchmark.

    The role must trust ``sagemaker.amazonaws.com`` and be allowed to read the model
    bucket. In SageMaker Studio this is automatic; elsewhere set SAGEMAKER_ROLE_ARN.
    """
    # 1) Explicit override - the most predictable path outside SageMaker Studio.
    if os.environ.get("SAGEMAKER_ROLE_ARN"):
        return os.environ["SAGEMAKER_ROLE_ARN"]

    sess = sess or boto3.session.Session(region_name=region())

    # 2) Inside SageMaker Studio / a notebook, the SDK already knows the attached role.
    #    Outside Studio this can resolve to the *caller's* identity, which may be an IAM
    #    user ARN — SageMaker cannot assume a user, only a role. So we only trust this
    #    path if it actually returns a role ARN; otherwise we fall through to step 3.
    try:
        import sagemaker  # imported lazily so plain boto3 users don't need the SDK

        arn = sagemaker.session.Session(boto_session=sess).get_caller_identity_arn()
        if ":role/" in arn:
            return arn
    except Exception:
        pass

    # 3) If the *current credentials* are an assumed SageMaker execution role (Studio
    #    terminals, SageMaker jobs), use exactly that role — never guess a different one.
    #    The IAM scan below can pick an older, differently-permissioned role that merely
    #    matches by name.
    try:
        import re

        sts_arn = sess.client("sts").get_caller_identity()["Arn"]
        m = re.match(r"arn:aws:sts::(\d+):assumed-role/([^/]+)/", sts_arn)
        if m and "sagemaker" in m.group(2).lower():
            account, role_name = m.group(1), m.group(2)
            try:
                # get_role recovers the full ARN incl. its path (e.g. /service-role/).
                return sess.client("iam").get_role(RoleName=role_name)["Role"]["Arn"]
            except Exception:
                return f"arn:aws:iam::{account}:role/{role_name}"
    except Exception:
        pass

    # 4) Last resort outside Studio: use a matching role only when it is unique.
    #    Silently choosing one of several roles can select the wrong S3 or KMS policy.
    iam = sess.client("iam")
    candidates = []
    for page in iam.get_paginator("list_roles").paginate():
        for r in page["Roles"]:
            name = r["RoleName"]
            if "SageMaker" in name and ("ExecutionRole" in name or "execution" in name.lower()):
                candidates.append(r)
    if len(candidates) == 1:
        return candidates[0]["Arn"]
    if candidates:
        names = ", ".join(sorted(r["RoleName"] for r in candidates))
        raise RuntimeError(
            "Multiple SageMaker execution roles matched automatic discovery: "
            f"{names}. Set SAGEMAKER_ROLE_ARN to the intended role ARN and re-run."
        )
    raise RuntimeError(
        "Could not resolve a SageMaker execution role. "
        "Set SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/<role> and re-run."
    )


def bucket(sess: boto3.session.Session | None = None) -> str:
    """The S3 bucket for model weights and benchmark output.

    Defaults to SageMaker's conventional bucket for the account/region
    (``sagemaker-<region>-<account>``), which the execution role can already access.
    """
    if os.environ.get("SAGEMAKER_BUCKET"):
        return os.environ["SAGEMAKER_BUCKET"]
    sess = sess or boto3.session.Session(region_name=region())
    return f"sagemaker-{region()}-{account_id(sess)}"


def summary() -> dict:
    """Resolve the full context once and return it as a dict (handy for printing)."""
    sess = boto3.session.Session(region_name=region())
    return {
        "region": region(),
        "account_id": account_id(sess),
        "execution_role_arn": execution_role_arn(sess),
        "bucket": bucket(sess),
    }


if __name__ == "__main__":
    # `python scripts/config.py` prints the resolved context so you can sanity-check
    # the environment before deploying anything billable.
    import json

    print(json.dumps(summary(), indent=2))
