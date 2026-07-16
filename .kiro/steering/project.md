---
inclusion: always
---

# Project context for coding agents

This repository is an AWS sample for taking an open-weight language model from
its public checkpoint to a deployed, benchmarked, and optimized Amazon
SageMaker AI endpoint. Operational knowledge is captured in portable
`SKILL.md` contracts and implemented by bundled scripts.

This file is the single source of project steering. It is mirrored as root
`AGENTS.md` and `.claude/CLAUDE.md`.

## Scope

- In: direct-to-S3 model staging, real-time deployment, smoke testing, managed
  inference benchmarking, inference recommendations, observability, and cleanup.
- Out: model training, fine-tuning, and application-specific model evaluation.
- Deployment is model-agnostic. Model choices and serving parameters are
  researched at execution time and passed as explicit script arguments.

## Portable contracts

1. `sagemaker-deploy` stages an immutable Hugging Face revision into S3 without
   a workstation snapshot, selects compatible compute and a serving engine,
   deploys a standard or Inference Component endpoint, and smoke tests it.
2. `sagemaker-benchmark` runs SageMaker AI managed inference benchmarking and
   reports TTFT, ITL, request latency, and throughput.
3. `sagemaker-optimize` runs SageMaker AI inference recommendations, deploys a
   selected configuration, and enables a before/after benchmark.

Run the scripts bundled with each skill. Do not recreate their boto3 workflows
ad hoc.

## Engineering rules

- Never hardcode an AWS account ID, role ARN, bucket, or private URL in tracked
  files. Resolve runtime context through each skill's `scripts/config.py`.
- Keep billable operations opt-in through `--run`, `--deploy`, or `--yes`.
- Print and validate the complete plan before creating resources.
- Pin model sources to immutable revisions during staging.
- Do not download large model snapshots to the caller's local disk.
- Check service quota and plan for capacity separately. Quota does not guarantee
  an available instance.
- Match DLC CUDA support and the SageMaker inference AMI to the GPU generation.
- Preserve a standard endpoint path for models that consume a full instance and
  an Inference Component path for packable models.
- Tear down real-time endpoint compute after every validation run.
- Keep changes small, reviewable, account-neutral, and covered by focused tests.

## Model neutrality

- Do not add model-name conditionals, model profiles, or model-specific defaults
  to core scripts.
- Treat scripts as reusable AWS execution primitives, not as the place where
  model-selection intelligence lives.
- Research checkpoint format, engine support, memory, context, and request shape
  from primary sources when a model is requested.
- Pass model-specific serving options through generic CLI arguments such as
  `--env` and `--smoke-inputs`.
- Keep model-specific research outside the core implementation. The skill may
  link primary model cards and AWS examples without embedding a model profile.
- If a model needs a different deployment architecture, choose the appropriate
  SageMaker service path instead of adding a one-model branch to a generic
  script.

## Repository conventions

- `.kiro/skills` is canonical. `.agents/skills` and `.claude/skills` are
  symlinks to the same directory.
- Public documentation describes reusable customer workflows, not an event,
  presenter, rehearsal, or private environment.
- Run `python -m unittest discover -s tests -v` and compile all Python files
  before publishing.

## Safety and cost

Model staging creates a short-lived CPU Processing job and persistent S3
objects. Deployment creates a continuously billed GPU endpoint. Benchmark and
recommendation jobs add managed compute. Report created resource names, retain
only intentional S3 artifacts, and delete endpoint compute before finishing.
