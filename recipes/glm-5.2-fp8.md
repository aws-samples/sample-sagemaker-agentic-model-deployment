# GLM-5.2-FP8 deployment recipe

Runtime configuration for deploying **`zai-org/GLM-5.2-FP8`** to a SageMaker AI real-time
endpoint with `sagemaker-deploy`. Documentation only — copy these arguments; do not add them
to `deploy.py`.

## Model facts (research-derived)

| Property | Value | Consequence |
|---|---|---|
| Architecture | `GlmMoeDsaForCausalLM` (MoE + DeepSeek-style **sparse** MLA / DSA) | needs a **sparse-MLA** attention kernel |
| Checkpoint | FP8 (`quantization=fp8`), ~704 GiB weights | fits an 8-GPU node with room for KV cache |
| Type | reasoning model | output arrives in the `reasoning` field with `content: null` |
| Native context | very long | cap `--max-model-len` to the KV-cache budget |

## Key decisions and why

- **GPU: H200 (`ml.p5en.48xlarge`, 8 GPUs), NOT g7e.** GLM-5.2's DSA sparse attention
  (`use_sparse=True`) requires a sparse-MLA kernel. On RTX PRO 6000 Blackwell (g7e / SM120)
  no stock-DLC backend supports it — deploys fail either at attention-backend selection
  ("compute capability not supported") or with a flashinfer `kv_scale_format` kernel
  mismatch. On Hopper (H200) vLLM selects `FLASHMLA_SPARSE` and it works.
- **Engine: vLLM.** Broad support + managed-benchmark compatibility.
- **Image: pin vLLM `0.23.0` (cu130).** This tag passed validation on H200. Resolve the
  newest compatible DLC for exploration, but pin the validated tag for reproducibility; some
  newer vLLM builds hit the sparse-MLA `kv_scale_format` mismatch.
- **KV cache: `fp8`.** vLLM auto-selects DeepSeek's `fp8_ds_mla` format for this model.
- **Parsers:** `--tool-call-parser glm47`, `--reasoning-parser glm45`, auto tool choice on.
- **Endpoint pattern: `standard`.** One model consumes the whole instance (no packing).
- **Capacity:** scarce H200 — launch into a reserved SageMaker training plan with
  `--capacity-reservation-arn` (pass the **training-plan** ARN, not the reserved-capacity ARN).

## 1) Stage weights to S3

Gated repo — store the HF token in Secrets Manager and pass its id (never inline a token):

```bash
python scripts/stage_model.py \
  --hf-model-id zai-org/GLM-5.2-FP8 \
  --hf-token-secret-id <hf-token-secret> \
  --run
```

## 2) Deploy (dry-run, then --deploy)

```bash
python scripts/deploy.py \
  --model-id glm-5-2-fp8 \
  --model-s3 s3://<bucket>/models/GLM-5.2-FP8/ \
  --engine vllm \
  --instance ml.p5en.48xlarge \
  --image 763104351884.dkr.ecr.<region>.amazonaws.com/vllm:0.23.0-gpu-py312-cu130-ubuntu22.04-sagemaker \
  --capacity-reservation-arn <training-plan-arn> \
  --deployment-mode standard \
  --max-model-len 65535 \
  --timeout 3600 \
  --env '{"SM_VLLM_KV_CACHE_DTYPE":"fp8","SM_VLLM_TOOL_CALL_PARSER":"glm47","SM_VLLM_ENABLE_AUTO_TOOL_CHOICE":true,"SM_VLLM_REASONING_PARSER":"glm45"}'
```

Add `--deploy` to create the billable endpoint after reviewing the plan.

Notes:
- `SM_VLLM_ENABLE_AUTO_TOOL_CHOICE` is a JSON boolean `true` — `build_env` emits it as a
  presence-only flag.
- `--inference-ami-version` is only needed for Blackwell; p5en is Hopper, so leave it default.
- p5en is already in `deploy.py`'s `INSTANCE_GPUS` (8 GPUs → TP=8), so no `--num-gpu` needed.

## 3) Smoke test / teardown

```bash
python scripts/smoke_test.py --endpoint <endpoint-name>
python scripts/teardown.py --endpoint <endpoint-name> --yes
```

## Benchmarking note

GLM-5.2 is a reasoning model, so responses may carry text in `reasoning` with `content:null`.
When benchmarking, supply a request field such as `reasoning_effort` (via the benchmark
skill's request-shape input) so responses stay above AIPerf's validity gate.
