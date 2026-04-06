# Architecture Diagrams

This page is the visual companion to [architecture.md](./architecture.md). It shows the post-refactor system as it exists after the universal runtime consolidation.

## 1. Runtime Topology

```mermaid
flowchart TB
    RT["UniversalTokRuntime<br/>request shaping<br/>response shaping<br/>memory projection<br/>tool normalization<br/>replay and gating<br/>telemetry"]

    CB["ClaudeBridgeAdapter<br/>gateway/__init__.py"]
    OA["OpenAIChatAdapter<br/>live_runner.py"]
    TL["TextLoopAdapter<br/>agent.py"]
    OR["OrchestratorAdapter<br/>adapters/orchestrator.py"]
    PS["Protocol Substrate<br/>format_bridge.py<br/>parser.py<br/>encoder.py<br/>schema.py"]

    CB --> RT
    OA --> RT
    TL --> RT
    OR --> RT
    PS -. explicit Tok language layer .-> RT
```

## 2. Primary Claude Bridge Request Flow

```mermaid
sequenceDiagram
    participant CC as Claude Code
    participant GW as gateway/__init__.py
    participant AD as ClaudeBridgeAdapter
    participant RT as UniversalTokRuntime
    participant MEM as BridgeMemoryState
    participant API as api.anthropic.com

    CC->>GW: POST /v1/messages
    GW->>AD: normalize HTTP request
    AD->>RT: prepare_request()
    RT->>RT: translate request results
    RT->>RT: normalize tool events
    RT->>RT: collect behavior signals
    RT->>RT: compress tool results
    RT->>RT: compress history
    RT->>MEM: refresh projected memory
    RT-->>AD: prepared request body
    AD-->>GW: transport-ready body
    GW->>API: forward request
    API-->>GW: model response
```

## 3. Primary Claude Bridge Response Flow

```mermaid
sequenceDiagram
    participant API as api.anthropic.com
    participant GW as gateway/__init__.py
    participant AD as ClaudeBridgeAdapter
    participant RT as UniversalTokRuntime
    participant MEM as BridgeMemoryState
    participant ST as SavingsTracker
    participant CC as Claude Code

    API-->>GW: SSE or JSON response
    GW->>AD: accumulated response text
    AD->>RT: process_response()
    RT->>RT: classify Tok-native vs fail-open
    RT->>RT: translate visible blocks
    RT->>MEM: update working memory
    RT->>RT: update family mode
    RT->>ST: emit behavior and savings signals
    RT-->>AD: processed content blocks
    AD-->>GW: transport-ready response
    GW-->>CC: readable output
```

## 4. Memory, Tools, and Telemetry Ontology

```mermaid
flowchart LR
    REQ["Normalized Request<br/>messages<br/>model<br/>adapter kind<br/>system"] --> TOOLS["Normalized Tool Events<br/>invocation<br/>result<br/>fidelity requirement<br/>compressibility class"]

    REQ --> MEMIN["Memory Projection<br/>hot state<br/>durable state<br/>&gt;&gt;&gt; wire state"]

    TOOLS --> SIG["Behavior Signals<br/>repeat reads<br/>repeat search<br/>fail-open drift<br/>blocker rediscovery"]

    MEMIN --> SIG
    SIG --> TEL["Telemetry and Replay<br/>invisible pressure<br/>family mode<br/>savings<br/>gate checks"]
```

## 5. Canonical IDL Ownership

```mermaid
flowchart LR
    subgraph Canonical["Canonical Protocol IDL"]
        SC["protocol/schema.py<br/>schema registry"]
        MD["protocol/models.py<br/>TokNode + AST/data model"]
    end

    subgraph Derived["Derived Runtime Contract"]
        TS["protocol/models.py::TOOL_SCHEMAS"]
        RTT["runtime/tools.py<br/>RuntimeToolExecutor._compiler_guard()"]
    end

    subgraph Wire["Bridge / Wire Adaptation"]
        RP["runtime/pipeline/request_preparation.py"]
        RV["runtime/pipeline/request_validation.py<br/>canonicalize_anthropic_bridge_body()"]
    end

    subgraph Public["Public Surface"]
        EX["src/tok/__init__.py<br/>convenience re-exports"]
        DOC["docs/architecture.md<br/>docs/architecture-diagrams.md"]
    end

    SC --> MD
    MD --> TS
    TS --> RTT
    MD --> RP
    RP --> RV
    SC -. documented by .-> DOC
    MD -. documented by .-> DOC
    SC -. re-exported for convenience only .-> EX
    MD -. re-exported for convenience only .-> EX
    RTT -. derived from protocol IDL .-> DOC
    RV -. transport translation only .-> DOC
```

## 6. Paired IDL Audit Gate

```mermaid
flowchart TB
    START["Bounded IDL stress prompt"] --> BASE["Baseline run<br/>source of truth for IDL coherence"]
    START --> TOK["Tok-captured run<br/>diagnose runtime interference"]

    BASE --> MAP["Canonical IDL map<br/>protocol/schema.py + protocol/models.py"]
    TOK --> SIG["Tok-only anomalies<br/>stable-result caching<br/>answer-repair churn<br/>fail-open compatibility"]

    MAP --> REL["Release decision"]
    SIG --> REL

    REL --> PASS["Release can proceed only if:<br/>baseline and Tok agree on major IDL findings<br/>and Tok anomalies are resolved or isolated"]
```

## 7. Deferred Orchestrator Migration Boundary

```mermaid
flowchart LR
    USER["User Input"] --> LOOP["Deferred orchestrator loop<br/>sliding window<br/>retry logic<br/>tool execution<br/>strict-mode validation"]

    LOOP --> BOUNDARY["OrchestratorAdapter<br/>runtime boundary now present"]
    BOUNDARY --> RT["UniversalTokRuntime"]

    LEGACY["Legacy-owned today<br/>multi-system prompt assembly<br/>large turn loop<br/>truncation policy"] -. still deferred .-> LOOP
```

The release-surface manifest in `src/tok/release_surface.py` defines which exports
and commands are supported versus experimental for the first public release.

## 8. Adoption Story

```mermaid
flowchart TB
    INV["Invisible-first adoption<br/>no Tok authoring required"] --> BR["Bridge wins as primary surface"]
    BR --> PUB["Narrow bridge-first public release"]
    PUB --> RT["Same runtime semantics expand across surfaces"]
    RT --> STD["Explicit Tok language surface<br/>standardizes later"]
    STD --> AA["Human-agent and agent-agent standardization"]
```

______________________________________________________________________

See the repository root `roadmap.md` for latest planning and phase status.
