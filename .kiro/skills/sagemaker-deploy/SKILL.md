---
name: sagemaker-deploy
description: >
  Stage any supported open-weight Hugging Face model directly into Amazon S3
  and deploy it to an Amazon SageMaker AI real-time endpoint with a compatible
  vLLM or SGLang Deep Learning Container. Use for model hosting, endpoint
  creation, direct-to-S3 transfer, smoke testing, and deployment cleanup.
license: MIT-0
compatibility: Requires Python 3.10+, boto3, the SageMaker Python SDK, AWS credentials, a SageMaker execution role, S3 access, and endpoint quota for the selected GPU instance.
metadata:
  author: sample-sagemaker-agentic-model-deployment
  version: "2.0"
---

# SageMaker deploy

This skill is a model-agnostic deployment contract. The scripts do not select
behavior by model name. The agent researches the requested checkpoint, derives
its serving requirements, and supplies those decisions as explicit arguments.

## Agent owns the model plan

The bundled scripts are generic execution primitives, not a model catalog or a
policy engine. The agent must interpret current primary sources and choose the
checkpoint, engine, image, hardware, topology, context cap, and request shape.
A newly supported model should normally require different arguments, not a code
change.

Do not grow the scripts with model profiles or model-name branches. Add code
only for a reusable SageMaker capability. If no compatible checkpoint fits a
single SageMaker real-time endpoint, choose another official quantization or
move to the appropriate SageMaker multi-node path, such as HyperPod, instead of
encoding a one-model exception in `deploy.py`.

## Model runtime lookup

Before deriving serving arguments from scratch, look up a known-good configuration for the
requested checkpoint in the AWS SageMaker GenAI hosting examples:

<https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models>

Find the requested model or an architecturally similar one and adopt its engine, image
family, instance and topology, context cap, and required `SM_VLLM_*` / `SM_SGLANG_*` options
as the starting point. This does not change the model-neutral contract: any configuration
found is supplied through generic CLI arguments (`--image`, `--instance`, `--env`,
`--smoke-inputs`, `--capacity-reservation-arn`), never added to the scripts. Use the model
publisher's model card for anything the examples do not cover.

Always re-validate against current reality before deploying: resolve the newest compatible
DLC (reuse an example's pinned image only if it still exists in ECR), confirm the
checkpoint's required GPU microarchitecture and quantization support, check endpoint quota,
and size context and concurrency to the measured KV-cache budget.

If no matching example exists, or the example is stale (superseded DLC tag, deprecated flag,
or an older engine version than the checkpoint needs), do not block: fall back to
first-principles research from the model card and proceed. Treat examples as a starting
point, not an authority. A dated example never overrides current DLC, quota, or hardware
facts.

## Required sequence

1. Run the **Model runtime lookup** above, then identify the exact model repository,
   license, checkpoint format, engine support, minimum engine version, weight size,
   and native context length. A borrowed config is a starting point to validate, not
   a substitute for these facts.
2. Choose a checkpoint and instance whose aggregate GPU memory can hold weights,
   non-quantized layers, runtime buffers, and the intended KV cache.
3. Resolve the repository to an immutable commit and stream it into S3 with
   `scripts/stage_model.py`. Do not create a workstation model snapshot.
4. Select vLLM or SGLang, an endpoint pattern, a context cap, tensor parallelism,
   and any model-required engine arguments.
5. Run `scripts/deploy.py` without `--deploy` and review the complete plan.
6. Deploy, wait for `InService`, and invoke an OpenAI-compatible smoke request.
7. Report the model revision, S3 prefix, image, instance, endpoint, optional
   Inference Component, and smoke-test result.
8. Delete endpoint compute with `scripts/teardown.py --yes` when validation is
   complete.

Do not add per-model conditionals or profiles to the scripts.

## Direct-to-S3 staging

The staging controller resolves a Hugging Face repository to a commit SHA and
starts a CPU SageMaker Processing job:

```bash
python scripts/stage_model.py \
  --hf-model-id organization/model-name

python scripts/stage_model.py \
  --hf-model-id organization/model-name \
  --destination s3://bucket/models/model-name/ \
  --run
```

The worker streams each HTTP response into an S3 multipart upload, records the
repository, revision, blob ID, and size as object metadata, skips matching
objects on rerun, and writes `.hf-model-manifest.json` only after all files
verify. It never writes model weights to the caller's filesystem.

For a gated repository, store the Hugging Face token in AWS Secrets Manager and
pass `--hf-token-secret-id`. The caller and Processing execution role both need
`secretsmanager:GetSecretValue`. Do not commit a token, place it in command-line
arguments, or expose it in a Processing job environment.

## Deployment inputs

The agent must determine these values for the requested model:

| Input | Decision |
|---|---|
| `--model-s3` | Verified S3 prefix containing the uncompressed checkpoint |
| `--instance` | GPU architecture and memory sized to model plus runtime |
| `--num-gpu` | Inferred from known hardware or supplied explicitly |
| `--engine` | `vllm` or `sglang`, at a model-compatible version |
| `--deployment-mode` | `standard` for a whole-instance model, `inference-component` when intentionally packable |
| `--max-model-len` | Workload-driven context cap that fits the KV-cache budget |
| `--max-num-seqs` | vLLM concurrency cap, when an explicit cap is needed |
| `--env` | Model-required `SM_VLLM_*` or `SM_SGLANG_*` options |
| `--smoke-inputs` | Model-specific request fields, if required |

`scripts/deploy.py` accepts these as explicit arguments. It validates the S3
layout, endpoint quota, GPU count, image compatibility, and JSON inputs before
creating resources.

## Serving engine

Use the newest compatible canonical SageMaker DLC unless the user requests a
pinned image. Resolve images from AWS ECR registry `763104351884`, repository
`vllm` or `sglang`, and ignore EC2, SOCI, server aliases, and non-semantic tags.

| Engine | Base environment | Typical use |
|---|---|---|
| vLLM | `SM_VLLM_MODEL`, `SM_VLLM_TENSOR_PARALLEL_SIZE` | Broad model support and managed benchmark compatibility |
| SGLang | `SM_SGLANG_MODEL_PATH`, `SM_SGLANG_TP` | Alternate scheduler, kernels, and speculative decoding |

Current Blackwell containers require CUDA 13 and the AL2023 SageMaker GPU
inference AMI:

```text
al2023-ami-sagemaker-inference-gpu-4-1
```

Pass model-specific server options through `--env`; do not encode them in
Python conditionals. The DLC entrypoints convert each `SM_VLLM_*` or
`SM_SGLANG_*` variable directly into a CLI option. Use JSON booleans for
presence-only flags: `true` emits the bare flag and `false` omits it. Use
strings or numbers only for options that take a value.

## Endpoint pattern

Use a standard endpoint when one model consumes the full instance. The
production variant references the SageMaker Model directly.

Use an Inference Component when the model is intentionally packable. The
endpoint configuration provisions the host, then the Inference Component
declares accelerator count and minimum host memory.

Both paths use
`CreateModel.PrimaryContainer.ModelDataSource.S3DataSource` with
`S3DataType=S3Prefix` and `CompressionType=None`.

## Preflight guards

- The S3 prefix must contain `config.json` and at least one recognized weight
  file.
- Require `.hf-model-manifest.json` and verify every listed file size before
  deployment. Use `--allow-unverified-s3` only for an externally staged prefix
  that was independently verified.
- Prefer an official publisher or hardware-vendor checkpoint over an
  unverified community quantization. Confirm its license separately.
- Engine compatibility has three independent parts: model architecture,
  checkpoint quantization, and the minimum package version. Supporting the base
  architecture does not prove support for a particular quantized checkpoint.
- Check the checkpoint's required GPU microarchitecture. A model that fits in
  aggregate memory can still be incompatible with the selected GPU generation.
- Endpoint quota must be at least one. Quota does not guarantee capacity.
- Tensor parallelism must match the allocated GPU topology unless the engine's
  documented topology requires otherwise.
- The execution role must trust `sagemaker.amazonaws.com` and read the model
  prefix.
- Use `--capacity-reservation-arn` for time-sensitive scarce capacity.
- Configure a fallback only when it has enough GPU memory and a valid
  parallelism plan for the same checkpoint.
- Do not default to a model's maximum advertised context. Size context and
  concurrency from measured KV-cache usage.
- For a newly released architecture, verify that the selected DLC contains the
  required `transformers` version or native engine implementation. A newer
  engine version alone is not sufficient.
- Use a standard endpoint for a model that consumes the whole instance; do not
  add Inference Component packing semantics without a packing use case.
- Resolve the newest compatible DLC for exploration, then pin the image that
  passed validation for reproducible deployments.

## Reference commands

```bash
python scripts/config.py
python scripts/stage_model.py --hf-model-id organization/model-name
python scripts/stage_model.py --hf-model-id organization/model-name --run

python scripts/deploy.py \
  --model-id model-name \
  --model-s3 s3://bucket/models/model-name/ \
  --instance ml.gpu-instance \
  --engine vllm \
  --max-model-len 16384

python scripts/deploy.py <same-arguments> --deploy
python scripts/smoke_test.py --endpoint ENDPOINT
python scripts/smoke_test.py --endpoint ENDPOINT --ic INFERENCE_COMPONENT
python scripts/teardown.py --endpoint ENDPOINT --yes
```

## Primary references

Use the requested model publisher's model card as the source of model-specific
serving arguments. Browse the AWS model examples for a matching model family or
an architecturally similar SageMaker deployment:

- [AWS SageMaker GenAI hosting model examples](https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models)
- [AWS vLLM DLC documentation](https://aws.github.io/deep-learning-containers/vllm/)
- [AWS SGLang DLC documentation](https://aws.github.io/deep-learning-containers/sglang/)

## Handoff

Pass the endpoint and optional Inference Component name to
`sagemaker-benchmark`. Standard endpoints omit the component. Keep intentional
S3 model artifacts, but remove real-time endpoint compute after testing.
