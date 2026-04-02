# Public Release Decision

Supported workflows, limitations, and the release bar for Tok's first public release.

Tok is released as a Claude-first bridge. OpenRouter and frontier probes are useful
for validation, but they do not define the public default behavior.

## Supported Workflows

The first public release supports exactly this workflow:

1. `pip install tok-protocol`
2. `tok install` (adds `claude()` shell wrapper)
3. `tok bridge start`
4. Use Claude Code normally
5. `tok bridge status` / `tok doctor` / `tok stats` to monitor
6. `tok bridge stop` to end the session

The default CLI help surface should reinforce that path by centering:

- `tok install`
- `tok bridge start|status|logs|stop`
- `tok doctor`
- `tok stats` / `tok savings`

The only supported Python helper path in this release is:

1. create one `RuntimeSession`
2. call `tok.wrap(...)`
3. send the prepared request through your client
4. call `tok.process(...)`
5. reuse the same session for the next turn

## Unsupported Paths

The following are explicitly out of scope for the first release:

- broader SDK/decorator APIs beyond `RuntimeSession`, `tok.wrap(...)`, and `tok.process(...)`
- `tok-native` and `tok-minimal` compression paths
- Windows support
- Python 3.9 or earlier
- Broad agent-to-agent handoff
- Visual dashboards
- protocol playground/demo scripts that are not in `examples/`

## Limitations

- Token savings depend on conversation length; very short sessions may not show clear savings
- The bridge requires Claude Code to be installed and configured
- Model families beyond Anthropic, OpenAI, DeepSeek, and Qwen are not yet validated
- Frontier and OpenRouter probes are experimental validation, not release drivers
- `src/tok/cli/__init__.py` remains larger than we want for the long term; the extracted helper modules are now the canonical implementation, and further decomposition is deferred until after `0.1.0`
- Dependency policy for `0.1.0` is lockfile-backed validation plus CI coverage, not blanket upper bounds on every published requirement

## Release Bar

A public release requires:

1. All CI checks green on `main`
2. No regressions in `success_rate=1.0` on required benchmark families
3. Savings stay in the validated reference band (45-55%)
4. Onboarding docs are coherent and tested in a clean-room venv
5. No known security issues
6. Coverage for the supported release surface stays at or above 80%
7. Live Claude validation confirms the bridge-first workflow behaves correctly against a real session
8. The release shape is narrow, explicit, and defensible
9. The release candidate is cut from a clean, fully revalidated tree

The expected automated revalidation pass for that candidate is:

```bash
pre-commit run --all-files
ruff check src/tok tests
mypy src/tok
pytest tests/unit tests/integration -v --cov=src/tok --cov-fail-under=80
python -m build
```

## Paired IDL Audit Rule

The bounded IDL stress audit now has two required interpretations:

1. A baseline run is the source of truth for protocol-IDL coherence. It decides whether `src/tok/protocol/schema.py` and `src/tok/protocol/models.py` still describe one coherent canonical surface.
2. A Tok-captured run is a runtime-product audit. It decides whether caching, answer-repair heuristics, or fail-open behavior distort that same bounded audit.

Release review should classify findings this way:

- Protocol-schema drift: canonical IDL blocker
- Runtime tool-input drift: derived-contract blocker
- Bridge transport / canonicalization drift: transport blocker
- Tok-only capture anomalies: product blocker, but not evidence of a second canonical IDL by themselves

The release decision should remain red if a bounded read-only audit still degrades to `watch`
because of unsupported reread instructions, false answer-ready escalation, repair churn, or
ambiguous `fail-open compatibility` noise.

## Deferred Follow-Ups

These items are intentionally deferred after the first public release:

- Further decomposition of `src/tok/cli/__init__.py`. This is a maintainability improvement, not a release blocker, and further movement right before release would add churn around Typer registration and CLI test seams.
- Dependency upper bounds across the published requirements. For `0.1.0`, Tok relies on `uv.lock`, CI, and clean-room install checks to validate the tested dependency set. If live-release experience shows resolver churn or upstream breakage, we can add selective upper bounds in a follow-up release.
