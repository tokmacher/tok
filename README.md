# Tok

[![CI](https://github.com/tokmacher/tok/actions/workflows/ci.yml/badge.svg)](https://github.com/tokmacher/tok/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/tok-protocol.svg)](https://pypi.org/project/tok-protocol/)
[![Python](https://img.shields.io/pypi/pyversions/tok-protocol.svg)](https://pypi.org/project/tok-protocol/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Tok is a local Claude Code bridge for deterministic context compression. It sits between
Claude Code and the upstream model API, reduces repeated file/search/tool context when
it can do so safely, and fails open to normal uncompressed behavior when fidelity is at
risk.

Tok `0.1.x` is deliberately narrow: Claude Code routed through a local bridge. It is not
a hosted service, agent framework, repo indexer, or general prompt-compression SDK.

## Quickstart

```bash
pip install tok-protocol
tok claude
```

Then, from another shell or after the session:

```bash
tok bridge status         # check bridge health
tok doctor                # explain current session state
tok stats                 # view savings
tok bridge stop           # stop cleanly
```

If you want an isolated CLI install and already use `pipx`, this works too:

```bash
pipx install tok-protocol
tok claude
```

`tok claude` starts the bridge if needed, routes Claude Code through it, and leaves your
shell rc files untouched. If you prefer legacy auto-routing, opt in explicitly:

```bash
tok install --wrap-claude
source ~/.zshrc           # or source ~/.bashrc
claude
```

## What Success Looks Like

A healthy bridge session usually has:

- `tok bridge status` showing the bridge running and Tok active
- `tok doctor` ending with `Recommendation: keep Tok on`
- `tok stats` showing `With Tok vs without Tok`, saved tokens, and estimated savings
- `Degraded to baseline` set to `no`

Representative output:

```text
Bridge running on :9090 (PID 12345)
Saved $0.0123 - 48.1% saved
Verdict                Tok active and helping
Tok active             yes
Degraded to baseline   no
Fallbacks              0
```

If `Degraded to baseline: yes` or fallback counts rise, Tok protected the session by
serving requests without compression.

## What Changed In 0.1.7

Tok `0.1.7` adds the first visible trace/audit layer around the supported bridge path:

- `tok audit` validates draft Tok Trace files and live bridge sidecars.
- `TOK_TRACE=1` writes opt-in metadata-only trace JSONL under `~/.tok/traces/`.
- `TOK_TRACE_CAPTURE_ARTIFACTS=1` writes sanitized metadata artifacts so `tok audit` can
  verify local hashes and byte sizes without storing raw prompts, responses, or tool
  outputs.
- New adversarial bridge-pressure tests cover large repeated reads, audit-heavy turns,
  overcompression risk, final-answer repair guards, and tool-pairing repair signals.

This is draft trace/audit groundwork, not universal protocol stability. Tok Capability,
Tok Session, resolver networking, binary encodings, and agent-to-agent protocol behavior
remain future work.

## Trace Audit

Enable trace sidecars only when you want to inspect what Tok did:

```bash
TOK_TRACE=1 TOK_TRACE_CAPTURE_ARTIFACTS=1 tok bridge start
tok claude
tok audit --latest
```

Trace mode is local. Tok does not send trace files to the model provider, and the
`0.1.7` live trace path does not store raw prompts, responses, or tool outputs.

`tok audit` is useful for checking bridge behavior and exactness metadata. It is not a
general protocol compliance certificate.

## Why Tok Exists

Long-running coding-agent sessions often resend verbose transcripts, file reads, search
results, and tool outputs on every turn. That is useful when a human reads the output,
but wasteful when the next reader is another model.

Tok tests a smaller runtime shape: compact, deterministic, model-facing state at the
machine-to-machine boundary, with human-facing output preserved at the edges.
Compression is rule-based rather than LLM-summarized, so behavior is repeatable and
auditable.

## What Tok Does

- **Semantic deduplication**: repeated file reads, search results, and tool outputs can
  be cached and replaced with compact references.
- **Delta compression**: changed content can be represented as a diff instead of a full
  repeated payload.
- **Bounded rolling state**: recent context stays available without unbounded history
  growth.
- **Fail-open safety**: when compression would risk fidelity, Tok serves the request in
  baseline mode and reports the fallback.
- **Diagnostics**: `status`, `doctor`, `stats`, logs, and optional trace audit explain
  what happened.

## Savings

Savings are workload-dependent. Tok tends to help most on sustained sessions with
repeated file reads, repeated searches, large tool outputs, or long-running debugging
loops. Very short sessions may intentionally run near baseline because compression
overhead is not worth paying.

Here is an upper-bound `tok stats` example from a long, highly repetitive 207-call
session. It is **not typical**:

![Tok Savings Output - upper-bound example from a high-repetition session](docs/images/tok_stats.png)

Use these as practical expectations:

- **Sustained sessions**: meaningful input-token savings when context repeats.
- **Short sessions**: little or no visible savings; Tok may stay baseline.
- **Risky compression cases**: fallback is preferred over corrupting context.

Pricing estimates depend on provider/model rates. See
[`docs/pricing_verification.md`](docs/pricing_verification.md) and
[`docs/claims_matrix.md`](docs/claims_matrix.md) for the current evidence trail.

## Prerequisites

- Python `3.10`-`3.12`
- macOS or Linux
- Claude Code installed and available as `claude`
- Claude Code already configured with provider credentials

Tok is a local proxy. It does not manage API keys. If Claude Code works without Tok,
`tok claude` should work too.

## Supported Surface

The public `0.1.x` workflow is:

```bash
tok init
tok install
tok claude
tok bridge status
tok doctor
tok stats
tok audit --latest
tok bridge stop
```

The supported mode is the default `tool-compatible` bridge mode. For comparison or
debugging, you can run without compression:

```bash
TOK_MODE=baseline tok bridge start
ANTHROPIC_BASE_URL=http://localhost:9090 claude
tok stats
```

For advanced routing or compatibility checks, you can still run the bridge and route a
client explicitly:

```bash
tok bridge start
ANTHROPIC_BASE_URL=http://localhost:9090 <your-client-command>
```

That path is useful for debugging and experiments, but the low-friction public install
story is `pip install tok-protocol` followed by `tok claude`.

Experimental Python submodule APIs and internal compression features exist, but they are
not part of the supported `0.1.x` contract and may change without compatibility
guarantees.

## How Tok Compares

- Claude Code `/compact` and auto-compaction are native conversation-management tools;
  Tok is a local bridge that compresses repeated machine-facing context before it
  reaches the model. See
  [`docs/claude-compaction-comparison.md`](docs/claude-compaction-comparison.md).
- Memory tools, code indexers, MCP servers, observability products, and prompt
  compressors solve adjacent problems. Tok's narrow job is deterministic bridge-layer
  context compression. See
  [`docs/positioning-context-tools.md`](docs/positioning-context-tools.md).

## Troubleshooting

| Symptom                                           | Check first                                                 | Likely fix                                                                                 |
| ------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `tok: command not found`                          | Was the package installed in the active Python environment? | Re-activate the environment and run `pip install tok-protocol`.                            |
| `claude: command not found` after wrapper install | Did your shell reload?                                      | Run `source ~/.zshrc` or `source ~/.bashrc`, or open a new shell.                          |
| `Bridge not running`                              | Did `tok bridge start` succeed?                             | Restart with `tok bridge start --foreground` and inspect `tok bridge logs`.                |
| No savings visible yet                            | Is the session short or non-repetitive?                     | Keep working for a few turns, then run `tok doctor` and `tok stats --last-session`.        |
| `Degraded to baseline: yes`                       | Did Tok fall back for safety?                               | Start with `tok doctor`, then follow [`docs/troubleshooting.md`](docs/troubleshooting.md). |

## Install Verification

Use this for a clean package sanity check:

```bash
python -m venv .venv
source .venv/bin/activate
pip install tok-protocol
tok --version
tok --help
tok claude --help
tok install
tok bridge start --help
tok bridge status --help
tok stats --help
tok audit --help
```

For local checkout work:

```bash
pip install .
tok --version
tok --help
```

## Docs Map

Start here:

- [`docs/bridge.md`](docs/bridge.md): full bridge tutorial
- [`docs/cli-reference.md`](docs/cli-reference.md): supported CLI surface
- [`docs/troubleshooting.md`](docs/troubleshooting.md): fallback, logs, degraded
  sessions, savings interpretation
- [`docs/diagnostics.md`](docs/diagnostics.md): detailed bridge health signals
- [`docs/claude-compaction-comparison.md`](docs/claude-compaction-comparison.md): Tok vs
  Claude Code compaction and baseline mode
- [`docs/positioning-context-tools.md`](docs/positioning-context-tools.md): Tok's place
  among memory, context, MCP, indexing, and observability tools

For release and architecture context:

- [`CHANGELOG.md`](CHANGELOG.md): release notes
- [`docs/public-release-decision.md`](docs/public-release-decision.md): supported
  workflows, limitations, and release bar
- [`docs/spec/README.md`](docs/spec/README.md): Tok Trace draft specification map
- [`docs/architecture.md`](docs/architecture.md): current architecture
- [`docs/architecture-0.2.md`](docs/architecture-0.2.md): roadmap, not the current
  runtime contract
- [`docs/production-readiness.md`](docs/production-readiness.md): advanced release
  posture

## Repo Map

- `src/tok/`: runtime, bridge, CLI, and library code
- `docs/`: public product docs plus release/reference docs
- `docs/spec/`: draft Tok Trace and protocol-layer specification work
- `docs/maintainers/`: maintainer roadmap and planning notes
- `examples/`: experimental wrapper/API examples outside the default bridge-first path
- `tests/`: unit, integration, replay, smoke, and stability coverage

## Development

For maintainer validation:

```bash
uv sync --frozen --extra dev
uv run pre-commit run --all-files
uv run ruff check src/tok tests
uv run mypy src/tok
uv run pytest tests/unit tests/integration -v --cov=src/tok --cov-fail-under=80
uv build
```

For release-specific checks, see
[`docs/release-checklist.md`](docs/release-checklist.md) and
[`docs/CICD_INTEGRATION.md`](docs/CICD_INTEGRATION.md).

## Privacy

Tok runs locally. No data leaves your machine except the model/API calls Claude Code
would already make. Optional `0.1.7` trace sidecars are local metadata files and do not
store raw prompts, responses, or tool outputs by default.

## Support Tok

Tok exists because repeated machine-facing context is a real cost in long coding-agent
sessions. The most useful support is practical feedback:

- Star the repo and share it with developers who use Claude Code heavily.
- File issues with bridge logs, `tok doctor`, and `tok stats` output.
- Share benchmark results from real sustained sessions.
- Contribute docs, tests, or focused fixes.
