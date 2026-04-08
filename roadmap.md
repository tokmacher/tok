# Tok Roadmap (Current)

<!-- markdownlint-disable MD036 -->

_Last updated: 2026-03-24_

This is a maintainer roadmap, not a first-run user guide.

If you are trying Tok for the first time, start with [`README.md`](README.md) and
[`docs/bridge.md`](docs/bridge.md). If you are maintaining the release, also read
[`docs/release-checklist.md`](docs/release-checklist.md) and
[`docs/public-release-decision.md`](docs/public-release-decision.md).

Tok is now bridge-first in practice, not just in intent:

- [`src/tok/universal_runtime.py`](src/tok/universal_runtime.py) owns the runtime
  semantics
- [`src/tok/gateway/__init__.py`](src/tok/gateway/__init__.py) is the primary production
  adapter
- `tok-tool-compatible` is the validated product path
- savings + invisibility remain the acceptance criteria

Tok's philosophy should be read in this order:

- near-term: invisible user-to-agent bridge
- medium-term: reliable agent-to-agent context handoff across surfaces
- long-term: universal runtime communication contract, if and only if the bridge path
  first becomes trusted infrastructure

This file is the canonical maintainer roadmap. Public onboarding docs should summarize
it, not replace it.

## Stage And Release Target

Current stage:

- validated internal-RC, bridge-first product

Next major external milestone:

- first public release of a bridge-first Tok that is trustworthy on real work

Long-term ambition:

- user-to-agent bridge first
- agent-to-agent handoff later
- universal runtime communication contract only after trust is earned

## Current Position

### What Is Proven

- `tok-tool-compatible` wins on both `coding-loop-5` and `research-loop-5`
- the win holds across DeepSeek, OpenAI, Qwen, and Anthropic families
- savings on the validated live benchmarks are roughly `45%` to `55%` vs baseline
- the bridge default/fallback path is hardened and observable
- the bridge UX is materially better:
  - `tok bridge status`
  - `tok doctor`
  - `tok bridge stop`
  - `tok stats` now make the runtime legible in normal usage

### What Is Ready

- bridge-first production path
- explicit mode policy (`tok-tool-compatible` default, `baseline` fallback)
- automated regression gate
- session-scoped fallback in the bridge path
- current and lifetime savings visibility
- shell integration that wraps `claude()` without overriding the real `tok` CLI
- a clearer path from operator trust signals to future cross-surface runtime semantics

### What Is Not Yet Proven

- replay-level parity beyond the current required release fixtures
- CI-enforced replay / stability gates on every required artifact path
- longer-horizon production confidence beyond the currently validated benchmark set
- a polished SDK/decorator story beyond the experimental wrapper API

### What Is No Longer The Main Work

- inventing more 5-turn benchmark mechanics
- making `tok-native` the default path
- treating `tok-minimal` as a production target
- presenting Tok primarily as a syntax or protocol before the bridge path is broadly
  trusted

## Completed Foundations

### Plan 1 – Mock Release Flow & Repo Cleanup

**Goal:** Make the internal-RC release path explicit, runnable, and easy to trust.

**Status:** ✅ COMPLETED (2026-03-24)

#### Tasks

- [x] add checked-in stability artifacts for:
  - `coding-loop-5`
  - `research-loop-5`
- [x] align CI and docs on one canonical `tok gate-check` release command
- [x] make `release_summary` part of the documented release contract
- [x] clean stale prompt-era and one-off planning files out of the repo root
- [x] run the full internal-RC rehearsal end to end and record the result

#### Verification

- [x] `tok gate-check ... --stability-dir tests/fixtures/stability` is sufficient for
  the internal-RC release check
- [x] CI blocks required replay or stability regressions on the same contract

______________________________________________________________________

### Plan 3 – Telemetry Gates, Replay, Billing Discipline

**Goal:** Extend confidence beyond the validated benchmark core while making the gate
noisy.

**Status:** ✅ COMPLETED (2026-03-24)

#### Tasks

- [x] Expand replay fixtures for coding loops, repeat reads/searches, malformed-response
  drift, recovery transitions
- [x] Integrate CICD gates with configurable thresholds
- [x] Trend dashboards & billing deltas
- [x] High-fidelity simulation infrastructure

______________________________________________________________________

### Plan 2 – Orchestrator Migration & Adapter Discipline

**Goal:** Ensure bridge parity and runtime-first behavior.

**Status:** ✅ COMPLETED (2026-03-24)

#### Tasks

- [x] Route orchestrator turn assembly through runtime
- [x] Delegate response finalization + telemetry to universal_runtime
- [x] Normalize adapters & add parity fixtures

______________________________________________________________________

### Plan 7 – Pattern Reactor & Protocol Consolidation

**Goal:** Deliver efficiency gains through macro patterns.

**Status:** ✅ COMPLETED (2026-03-24)

#### Tasks

- [x] Pattern Reactor Core (IR layer, Miner, Planner, Distiller, Integration, Monitor,
  Metrics, Memory, LLM Clients)
- [x] Macro Provenance Tracking
- [x] Global Macro Persistence
- [x] Protocol Consolidation (31% savings baseline, lazy Tok support)
- [x] Benchmarking & Verification (85% cumulative savings on 10-turn Tok-Tool-Compatible
  mode)

## Remaining Phases Before First Public Release

### Phase 1 – Production-Trusted Wedge

**Goal:** Make the current bridge path boring, trustworthy, and repeatedly usable on
real work.

**Status:** 🔄 IN PROGRESS

#### Tasks

- [ ] run recurring real-session capture on actual coding-agent workflows
- [ ] use `capture-review` and `evidence-gap` to rank repeated `watch` and `investigate`
  classes
- [ ] harden the top repeated degradation class first, especially fallback,
  reacquisition, or response-contract drift
- [ ] keep the internal-RC contract frozen unless required replay or stability evidence
  shows it is wrong
- [ ] keep tightening operator clarity so bridge behavior is self-explaining

#### Verification

- [ ] no regression in `success_rate=1.0` on required benchmark families
- [ ] validated savings stay in the current coding and research reference band
- [ ] repeated real-session degradations become explainable without raw log reading
- [ ] the next replay-promotion decision is evidence-backed and documented

______________________________________________________________________

### Phase 2 – Broader Production Proof

**Goal:** Extend confidence beyond the validated benchmark core without making the
release gate noisy.

**Status:** 🔄 IN PROGRESS

#### Tasks

- [~] keep expanding replay coverage only where it exposes real weakness classes:
  - [x] recovery / retry
  - [x] repeat-search / repeat-file-read
  - [~] cache-sensitivity / prompt-volatility
- [x] prefer replay-promotion candidates that come from repeated real-session capture or
  stress evidence, not speculative coverage
- [x] add replay-level bridge/orchestrator parity evidence
- [ ] keep aligning bridge/runtime/orchestrator semantics so future agent-to-agent
  handoff stays grounded in one canonical runtime contract
- [ ] extend real-session review into longer, noisier, more tool-heavy coding workflows
- [ ] keep `stress-language` advisory, but use it to prioritize proof gaps and hardening
  work
- [ ] decide which exploratory green fixtures are stable enough to promote after a clean
  internal-RC rehearsal

#### Verification

- [ ] broader release-proof fixtures stay green without making the gate noisy
- [ ] non-bridge surfaces no longer drift on savings/fallback semantics
- [ ] repeated uncovered evidence classes shrink over time
- [ ] the strongest open pre-release risks are known and ranked

______________________________________________________________________

### Phase 3 – Internal Productization

**Goal:** Make the bridge-first path coherent and usable for a strong new operator
without deep repo knowledge.

**Status:** 🔄 IN PROGRESS

#### Tasks

- [x] add recent-window and date-filtered stats views
- [ ] keep tightening the stats/dashboard surface around:
  - current session
  - last completed session
  - recent sessions
- [x] make captured real sessions part of the normal operator workflow and use them as
  the preferred source for future replay-fixture promotion candidates
- [x] continue repo/docs consolidation so canonical docs are obvious
- [ ] improve the SDK/decorator story after the bridge and release path settle
- [ ] define the smallest credible wrapper/SDK recipe for intentional adoption
- [ ] keep the product language bridge-first and invisible-first, so the future
  standard-contract story never outruns operator trust

#### Verification

- [ ] a new user can understand savings, fallback, and bridge state without reading deep
  docs
- [ ] operator actions for `keep on`, `watch`, and `investigate` are consistent across
  surfaces
- [ ] the wrapper path has a minimal, tested, readable recipe

### Phase 4 – Public Release Decision And Packaging

**Goal:** Freeze the public release shape and decide whether Tok is ready to release
publicly as a bridge-first product.

#### Tasks

- [ ] freeze the minimum supported workflows for the first public release
- [ ] define explicit non-goals and unsupported paths for the release
- [ ] confirm onboarding, release contract, and operator docs are coherent
- [ ] confirm the release bar is high enough for public use, not only internal RC
- [ ] decide whether the evidence and trust bar justify public release now

#### Verification

- [ ] `roadmap.md`, [`docs/production-readiness.md`](docs/production-readiness.md), and
  onboarding docs tell the same release story
- [ ] the public release shape is narrow, explicit, and defensible
- [ ] the release decision is recorded against evidence, not intuition

## Deferred Until After First Public Release

- `tok-native` as a default mode
- `tok-minimal` as a production mode
- broad new runtime modes
- long-horizon benchmark invention as the primary workstream
- visual dashboards before recent-window stats exist
- treating Tok as a universal protocol product before the default bridge path is trusted
  on real work
- broader agent-to-agent handoff as a primary product focus
- true cross-surface runtime expansion beyond what current parity proof can support
- universal communication contract or standardization branding
- broad document/artifact ingestion as a platform feature beyond what directly
  strengthens the current wedge

______________________________________________________________________

## Future Roadmap

### 2. Predictive Cache Warming

**Objective:** Use history as a heuristic for proactive state management.

**Mechanic:** Analyze the `rolling_cmds` and `hot_files` buckets to predict the next 2-3
likely file dependencies or search queries. Preemptively load these into the `@hot`
memory projection or prime the `ResultCache`.

**Benefit:** Ensures that when the LLM makes its next tool request, the context is
already "warm," preventing wait-states and cache-miss "reacquisition" costs.

**Status:** 📋 PENDING

### 3. Self-healing Serialization

**Objective:** Shield the bridge from model-driven schema drift.

**Mechanic:** Enhance `BridgeMemoryState.from_tok` with fuzzy-matching and semantic
redirection. If a model generates non-canonical headers (e.g., `@files:` instead of
`@f:` or `@history:` instead of `@c:`), the parser should autonomously map these back to
the canonical bridge state.

**Benefit:** Allows Tok to remain stable across a wider variety of models (including
smaller, faster models that may struggle with strict protocol adherence) without losing
state integrity.

**Status:** 📋 PENDING

## Immediate Recommendation

1. finish `Phase 1 – Production-Trusted Wedge`
1. continue into `Phase 2 – Broader Production Proof`
1. use `Phase 3 – Internal Productization` to prepare the actual public-release shape
1. make the public release decision only after
   `Phase 4 – Public Release Decision And Packaging`

That is now the shortest path from “bridge works well internally” to “we have a narrow,
trustworthy first public release.”

The philosophy behind that recommendation is deliberate: Tok should first win as an
invisible bridge between users and agents, then extend into reliable agent-to-agent
handoff, and only later claim universal-standard territory.
