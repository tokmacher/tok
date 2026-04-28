# Production Readiness

Runtime defaults and release posture for Tok.

## Supported Platforms

- Python 3.10, 3.11, 3.12
- macOS and Linux

## Runtime Defaults

| Setting               | Default                                            | Notes                                                                     |
| --------------------- | -------------------------------------------------- | ------------------------------------------------------------------------- |
| Compression path      | `tool-compatible` (`natural_first` request policy) | Claude-first default; validated on coding and research loops              |
| Legacy request policy | `legacy_tool_compatible`                           | Explicit rollback path; opt in only when you need the older behavior      |
| Frontier validation   | OpenRouter probes                                  | Advisory only; do not select the public default                           |
| Fallback mode         | `baseline`                                         | Requests pass through without compression                                 |
| Fail-open             | enabled                                            | Bridge errors are designed to pass through as upstream requests           |
| Port                  | 9090                                               | Configurable via `--port`                                                 |
| Bind address          | `127.0.0.1`                                        | Configurable via `TOK_BRIDGE_BIND_HOST`; client URL host remains separate |

## Security Note: Bind Address

The bridge binds to `127.0.0.1` by default, accepting connections only from the local
machine. Since the bridge proxies requests containing API keys, this is the safe default
for a single-user workstation.

To expose the bridge on a specific interface (e.g. for containerized or multi-machine
deployments), set the `TOK_BRIDGE_BIND_HOST` environment variable before starting the
bridge:

```bash
export TOK_BRIDGE_BIND_HOST=0.0.0.0   # listen on all interfaces (use with firewall rules)
export TOK_BRIDGE_BIND_HOST=192.168.1.100  # listen on specific interface
tok bridge start
```

For typical single-user workstation use (the intended 0.1.x scenario), the default
`127.0.0.1` bind is correct and no configuration is needed.

`TOK_BRIDGE_HOST` still controls the client-facing bridge URL used by helper scripts and
status probes when you intentionally connect through a non-default hostname.

## Release Posture

- Development status: Alpha (0.1.5)
- The bridge is the only supported product surface
- SDK/wrapper path exists but is experimental
- No breaking changes are guaranteed before 1.0
- The automated local gate for a release candidate is `uv sync --frozen --extra dev`,
  `uv run pre-commit`, `uv run ruff`, `uv run mypy`,
  `uv run pytest ... --cov-fail-under=80`, and `uv build`
- The current release candidate is gated on one final live Claude validation pass
  against the supported bridge workflow
- The release candidate should be tagged only from a quiet, clean tree after that
  validation passes

## Intentional Deferrals For 0.1.x

- `src/tok/cli/__init__.py` is still larger than ideal. The split helpers now own real
  implementation paths, but the final command-registration file will be reduced further
  after the first public release rather than during the release-candidate window.
- Published dependencies do not currently carry blanket upper bounds. For `0.1.x`, Tok
  treats the lockfile, CI matrix, and clean-room install/build verification as the
  source of truth for tested compatibility.

## Safety Guarantees

1. **Fail-open**: if the bridge encounters an error, requests are forwarded without
   compression
1. **Observable degradation**: `tok doctor` and `tok bridge status` report session
   health signals
1. **No data leaves your machine**: Tok runs locally; only your normal API calls leave

## Monitoring

Run `tok doctor` to get a one-shot health assessment. Key fields:

- `Verdict`: overall session health
- `Session quality`: `clean`, `watch`, or `degraded`
- `Degraded to baseline`: whether the session fell back to no compression
- `Fallbacks`: count of individual request-level fallbacks

For ongoing monitoring, use `tok stats` with time-window filters:

```bash
tok stats --recent 5
tok stats --last-session
tok stats --since 2026-03-01
```
