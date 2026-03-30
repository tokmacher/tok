# Public Release Decision

Supported workflows, limitations, and the release bar for Tok's first public release.

## Supported Workflows

The first public release supports exactly this workflow:

1. `pip install tok-protocol`
2. `tok install` (adds `claude()` shell wrapper)
3. `tok bridge start`
4. Use Claude Code normally
5. `tok bridge status` / `tok doctor` / `tok stats` to monitor
6. `tok bridge stop` to end the session

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

## Deferred Follow-Ups

These items are intentionally deferred after the first public release:

- Further decomposition of `src/tok/cli/__init__.py`. This is a maintainability improvement, not a release blocker, and further movement right before release would add churn around Typer registration and CLI test seams.
- Dependency upper bounds across the published requirements. For `0.1.0`, Tok relies on `uv.lock`, CI, and clean-room install checks to validate the tested dependency set. If live-release experience shows resolver churn or upstream breakage, we can add selective upper bounds in a follow-up release.
