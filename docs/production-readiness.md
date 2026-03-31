# Production Readiness

Runtime defaults and release posture for Tok.

## Supported Platforms

- Python 3.10, 3.11, 3.12
- macOS and Linux

## Runtime Defaults

| Setting | Default | Notes |
|---------|---------|-------|
| Compression path | `tok-tool-compatible` | Validated on coding and research loops |
| Fallback mode | `baseline` | Requests pass through without compression |
| Fail-open | enabled | Bridge errors never block upstream requests |
| Port | 9090 | Configurable via `--port` |

## Release Posture

- Development status: Alpha (0.1.0)
- The bridge is the only supported product surface
- SDK/wrapper path exists but is experimental
- No breaking changes are guaranteed before 1.0
- The current release candidate is gated on one final live Claude validation pass against the supported bridge workflow
- The release candidate should be tagged only from a quiet, clean tree after that validation passes

## Intentional Deferrals For 0.1.0

- `src/tok/cli/__init__.py` is still larger than ideal. The split helpers now own real implementation paths, but the final command-registration file will be reduced further after the first public release rather than during the release-candidate window.
- Published dependencies do not currently carry blanket upper bounds. For `0.1.0`, Tok treats the lockfile, CI matrix, and clean-room install/build verification as the source of truth for tested compatibility.

## Safety Guarantees

1. **Fail-open**: if the bridge encounters an error, requests are forwarded without compression
2. **Observable degradation**: `tok doctor` and `tok bridge status` always report session health
3. **No data leaves your machine**: Tok runs locally; only your normal API calls leave

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
