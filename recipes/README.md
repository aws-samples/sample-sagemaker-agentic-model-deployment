# Model recipes

Per-model **runtime recipes** for the `sagemaker-deploy` skill. Each file records the
research-derived arguments needed to deploy one checkpoint: the engine, image, GPU
architecture, topology, context cap, and any model-required serving options.

## Why this folder exists

`deploy.py` (and the other skill scripts) are intentionally **model-agnostic**. Per
`.kiro/steering/project.md` and `sagemaker-deploy/SKILL.md`, the scripts must not contain
model-name conditionals, model profiles, or model-specific defaults. Model-specific
knowledge is passed at runtime through generic CLI arguments (`--image`, `--instance`,
`--env`, `--smoke-inputs`, `--capacity-reservation-arn`).

These recipes are **documentation only**. They are not read by any script. Nothing here
selects behavior by model name — a recipe is just the reviewed set of arguments a human or
agent copies into `stage_model.py` / `deploy.py`. Keeping them out of the code preserves the
model-neutral contract while still capturing "how we deployed model X" for reuse.

## Conventions

- One file per checkpoint, named after the Hugging Face repo (e.g. `glm-5.2-fp8.md`).
- **Account-neutral**: use placeholders (`<bucket>`, `<region>`, `<training-plan-arn>`,
  `<hf-token-secret>`). Never commit real account IDs, ARNs, buckets, or tokens.
- Record *why* each non-obvious argument is needed (GPU generation, quantization, kernel
  support), not just the command.
- Pin the image tag that actually passed validation, for reproducibility.

## Available recipes

- [`glm-5.2-fp8.md`](glm-5.2-fp8.md) — GLM-5.2 (FP8, DSA sparse-MLA MoE) on H200.
