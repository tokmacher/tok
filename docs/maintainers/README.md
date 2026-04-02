# Tok Maintainer Docs

Internal planning and release documents live here.

## Key Documents

- [Release Checklist](../release-checklist.md) - steps for cutting a release
- [Production Readiness](../production-readiness.md) - runtime defaults and release posture
- [Public Release Decision](../public-release-decision.md) - supported workflows and release bar
- [Repo-owned Tok skill draft](../../skills/public/tok/SKILL.md) - internal,
  launch-adjacent guidance for how an agent should operate around Tok

Local testing for the draft skill is manual: copy or symlink
`skills/public/tok` to `$CODEX_HOME/skills/tok` (or `~/.codex/skills/tok` when
`CODEX_HOME` is unset). This skill is internal collateral and is not part of
the `0.1.0` release bar.

## Contributing

See [`CONTRIBUTING.md`](../../CONTRIBUTING.md) for PR expectations and branch conventions.
