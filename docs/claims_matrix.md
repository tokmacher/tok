# Release Claims Matrix

Last updated: **2026-04-08**.

This matrix is the release input for pricing/benchmark/savings claims.

## Pricing Claims

| Claim                                               | Owner                          | Evidence Command                              | Artifact                | Status                            |
| --------------------------------------------------- | ------------------------------ | --------------------------------------------- | ----------------------- | --------------------------------- |
| Pricing lookup source-of-truth is centralized       | `src/tok/utils/pricing.py`     | `uv run pytest tests/unit/test_pricing.py -q` | Test output in CI/local | Verified                          |
| Anthropic and OpenAI rates are externally rechecked | `docs/pricing_verification.md` | Manual source review (dated)                  | Linked provider docs    | Verified                          |
| Non-Anthropic/OpenAI rates are release-defining     | `README.md` / docs             | N/A                                           | N/A                     | Demoted (aggregator-derived only) |

## Benchmark / Savings Claims

| Claim                                                            | Owner                                                                       | Evidence Command                                                                                                                                                                                                                                                                                                                                                                                                          | Artifact                                 | Tolerance / Rule                                  | Status                                            |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| Required benchmark families are stable in maintained fixtures    | `docs/public-release-decision.md`                                           | `uv run python -c "import json,sys; from pathlib import Path; from tok.testing.live_benchmark import check_stability_artifacts; out=check_stability_artifacts(Path('tests/fixtures/stability'), ['coding-loop-5','research-loop-5']); Path('tmp/stability_check.json').write_text(json.dumps(out, indent=2)); print('Wrote tmp/stability_check.json'); sys.exit(0 if all(v.get('passed') for v in out.values()) else 1)"` | `tmp/stability_check.json`               | `success_rate == 1.0` on required families        | Verified                                          |
| Global "60-70% savings" release claim                            | `README.md`                                                                 | gate-check command                                                                                                                                                                                                                                                                                                                                                                                                        | N/A                                      | Must be reproducible from current command outputs | Demoted                                           |
| Supported release reference band is 45-55% on validated sessions | `docs/public-release-decision.md` / `README.md` / `docs/troubleshooting.md` | Release validation pass + maintained benchmark workflows                                                                                                                                                                                                                                                                                                                                                                  | release artifacts and maintainer reports | Coherent band across docs                         | Verified as target band (not universal guarantee) |

Benchmark authority policy (release evidence):

- Release benchmark comparisons are **baseline vs tok-universal only**, executed via
  `UniversalTokRuntime` using OpenRouter (`https://openrouter.ai/api/v1`).
- Multi-mode comparisons (`tok-minimal`, `tok-native`, `tok-neuro`) are
  historical/experimental only and are not release claim inputs.

## Smoke / Boundary Claims

| Claim                                                       | Owner                                          | Evidence Command                                                | Artifact                   | Status   |
| ----------------------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------- | -------------------------- | -------- |
| Defended release surface remains narrow                     | `src/tok/release_surface.py`                   | `uv run pytest tests/unit/test_release_surface.py -q`           | test pass output           | Verified |
| Bridge non-streaming/streaming/malformed/install paths hold | `scripts/run_release_smoke.py` + smoke tests   | `uv run python scripts/run_release_smoke.py`                    | smoke harness output       | Verified |
| Added live smoke categories are automated                   | `tests/smoke/test_live_claude_smoke_matrix.py` | `uv run pytest tests/smoke/test_live_claude_smoke_matrix.py -q` | 4-pass smoke matrix output | Verified |
