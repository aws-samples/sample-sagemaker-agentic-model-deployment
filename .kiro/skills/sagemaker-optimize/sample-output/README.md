# Sample recommendation result

`recommendation.json` is a sanitized
`describe_ai_recommendation_job` response from a completed configuration-search
job for GPT-OSS-20B. Account-specific identifiers were replaced; performance
and deployment values were left unchanged.

The job evaluated a throughput target on `ml.g6.24xlarge` and returned:

- two model copies;
- tensor parallelism 2;
- maximum concurrency 88;
- expected output throughput of 1,893 tokens per second;
- TTFT p50 of approximately 270 ms;
- ITL p50 of approximately 43 ms.

The corresponding historical baseline bundle measured 218 output tokens per
second on a single-GPU configuration. The improvement came from the serving
configuration and additional compute, not a model-weight change.

Important fields:

| Field | Meaning |
|---|---|
| `DeploymentConfiguration` | Selected instance, copy count, environment, and image |
| `ExpectedPerformance` | Projected latency and throughput |
| `ModelDetails.ModelPackageArn` | Deployable model package |
| `OptimizationDetails` | Applied model-level optimizations, empty for this search |

A recommendation job registers a Model Package version inside a group named
after the job. Query it with
`list-model-packages --model-package-group-name <job-name>` or call
`describe-model-package` with the returned ARN.
