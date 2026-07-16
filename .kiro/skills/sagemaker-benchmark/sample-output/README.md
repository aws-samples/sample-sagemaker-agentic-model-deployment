# Sample benchmark output

This directory contains a sanitized output bundle from a completed SageMaker AI
managed inference benchmark. The target was GPT-OSS-20B on
`ml.g6.16xlarge`, using ShareGPT-style traffic with approximately 500 input
tokens, 256 output tokens, concurrency 10, and 300 requests.

The measured output-token throughput was 218 tokens per second. This result is
included to demonstrate the AIPerf output schema, not as a performance
commitment or a comparison with GLM-5.2.

Two large or redundant objects are omitted: `inputs.json` (approximately
125 MB) and the enclosing `output.tar.gz`.

| File | Contents |
|---|---|
| `profile_export_aiperf.json` | Aggregated metrics used by the result reader |
| `profile_export_aiperf.csv` | Aggregated metrics in CSV form |
| `profile_export.jsonl` | Per-request records |
| `outputs.json` | Model responses |
| `benchmark_summary.txt` | Run summary |
| `failure_reason.txt` | AIPerf validity-gate result |
| `plots/` | TTFT timeline plots |
| `logs/aiperf.log` | AIPerf execution log |

Inspect it without making an AWS call:

```bash
python scripts/benchmark_results.py --local sample-output
```

This historical run was marked `Failed` by AIPerf's validity gate because
10 of 300 reasoning-model responses returned HTTP 200 with reasoning text but
`content: null`. Metrics for the 290 valid requests remain in the bundle. The
current default curated dataset avoids 1-3-token output budgets that caused
those empty visible responses.
