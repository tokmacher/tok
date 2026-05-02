# Public Release Decision

Supported workflows, limitations, and the release bar for Tok's first public release.

Tok is released as a Claude-first bridge. OpenRouter and frontier probes are useful for
validation, but they do not define the public default behavior.

## Supported Workflows

The first public release centers this default workflow:

1. `pip install tok-protocol`
1. Run Claude Code through Tok with `tok claude`
1. `tok bridge status` / `tok doctor` / `tok stats` to monitor
1. `tok bridge stop` to end the session

`tok install` remains a setup/migration helper and does not wrap `claude` by default.
Optional legacy behavior is still available via `tok install --wrap-claude`, but the
default release posture is `tok claude`.

The default CLI help surface should reinforce that path by centering:

- `tok claude`
- `tok install`
- `tok bridge start|status|logs|stop`
- `tok doctor`
- `tok stats`

The only supported product path in this release is the bridge-first workflow above.

The root `tok` namespace is intentionally narrow for `0.1.x`. Experimental Python APIs
may still be reachable through explicit submodules, but they are not part of the
supported public contract and should not be documented as canonical.

The release-surface manifest in `src/tok/release_surface.py` is the source of truth for
what counts as supported, experimental, and internal in this release.

Pricing and benchmark claims should be reconciled through:

- `docs/pricing_verification.md` for source-linked pricing review
- `docs/claims_matrix.md` for claim -> evidence -> status mapping

## Unsupported Paths

The following are explicitly out of scope for the first release:

- broader SDK/decorator APIs beyond the bridge-first public workflow
- `tok-native` and `tok-minimal` compression paths
- Windows support
- Python 3.9 or earlier
- Broad agent-to-agent handoff
- Visual dashboards
- protocol playground/demo scripts that are not in `examples/`

## Limitations

- Token savings depend on conversation length; very short sessions may not show clear
  savings
- The bridge requires Claude Code to be installed and configured
- Model families beyond Anthropic, OpenAI, DeepSeek, and Qwen are not yet validated
- Frontier and OpenRouter probes are experimental validation, not release drivers
- CLI decomposition: completed in 0.1.6. `_legacy_commands.py` removed; `__init__.py` is
  now 76 lines with no residual deferred work.
- Dependency policy for `0.1.x` is lockfile-backed validation plus CI coverage, not
  blanket upper bounds on every published requirement

## Release Bar

A public release requires:

1. All CI checks green on `main`
1. Tag-only release workflow enforces the same validation bar before publish
1. No regressions in `success_rate=1.0` on required benchmark families
1. Savings stay in the validated reference band (45-55%)
1. Pricing claims are sourced from `src/tok/utils/pricing.py` and reconciled in
   `docs/pricing_verification.md`
1. Onboarding docs are coherent and tested in a clean-room venv
1. Release artifacts build cleanly and the packaged wheel install path is revalidated
1. No known security issues
1. Coverage for the supported release surface stays at or above 80%
1. Live Claude validation confirms the bridge-first workflow behaves correctly against a
   real session
1. The release shape is narrow, explicit, and defensible
1. The release candidate is cut from a clean, fully revalidated tree

The expected automated revalidation pass for that candidate is:

```bash
uv sync --frozen --extra dev
uv run pre-commit run --all-files
uv run ruff check src/tok tests
uv run mypy src/tok
uv run pytest tests/unit tests/integration -v --cov=src/tok --cov-fail-under=80
uv build
```

## Paired IDL Audit Rule

The bounded IDL stress audit now has two required interpretations:

1. A baseline run is the source of truth for protocol-IDL coherence. It decides whether
   `src/tok/protocol/schema.py` and `src/tok/protocol/models.py` still describe one
   coherent canonical surface.
1. A Tok-captured run is a runtime-product audit. It decides whether caching,
   answer-repair heuristics, or fail-open behavior distort that same bounded audit.

Release review should classify findings this way:

- Protocol-schema drift: canonical IDL blocker
- Runtime tool-input drift: derived-contract blocker
- Bridge transport / canonicalization drift: transport blocker
- Tok-only capture anomalies: product blocker, but not evidence of a second canonical
  IDL by themselves

The release decision should remain red if a bounded read-only audit still degrades to
`watch` because of unsupported reread instructions, false answer-ready escalation,
repair churn, or ambiguous `fail-open compatibility` noise.

## Deferred Follow-Ups

These items are intentionally deferred after the first public release:

- CLI decomposition: completed in 0.1.6 (`_legacy_commands.py` removed, `__init__.py` at
  76 lines).
- Dependency upper bounds across the published requirements. For `0.1.x`, Tok relies on
  `uv.lock`, CI, and clean-room install checks to validate the tested dependency set. If
  live-release experience shows resolver churn or upstream breakage, we can add
  selective upper bounds in a follow-up release.
