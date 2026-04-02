# Compression Frontier Workflow

Use the frontier harness to find the highest compression rung that still feels calm.
Treat its OpenRouter probes as advisory validation, not as the thing that defines the Claude release default.

## Main command

```bash
tok dev compression-frontier \
  --model deepseek/deepseek-v3.2 \
  --benchmarks coding-loop-5,research-loop-5,research-loop-8 \
  --repeats 3 \
  --output tmp/compression_frontier
```

Artifacts:

- `tmp/compression_frontier/compression_frontier_report.json`
- `tmp/compression_frontier/compression_frontier_report.md`

## What it does

- compares three checkpoints by default: current head, `5aebb5d`, and one earlier commit from the prior 4 days
- walks a fixed ladder of compression profiles from conservative to extreme
- stops climbing after the first degraded rung for each checkpoint
- records benchmark verdicts plus a cheap OpenRouter probe for the current head when `OPENROUTER_API_KEY` is available
- keeps the benchmark-derived release lane separate from the OpenRouter confidence check

## Cheap probe example

```bash
OPENROUTER_API_KEY=... \
TOK_FRONTIER_PROFILE=balanced \
TOK_FRONTIER_ARTIFACT=tmp/openrouter_balanced.json \
uv run python3 examples/natural_first_openrouter.py
```

## Reading the report

- `stable`: calm enough for the default release lane
- `watch`: saves more, but recovery or pressure noise increased
- `degraded`: too brittle for default use

The release lane should track the highest rung that stays `stable`; anything above it stays experimental.

## Release gate integration

You can feed the report into `tok gate-check`:

```bash
tok gate-check tests/fixtures/replay \
  --stability-dir tests/fixtures/stability \
  --frontier-report tmp/compression_frontier/compression_frontier_report.json
```
