# Architecture Diagrams

This page is the visual companion to [architecture.md](./architecture.md). It shows the post-refactor system as it exists after the universal runtime consolidation.

## 1. Runtime Topology

```mermaid
flowchart TB
    RT["UniversalTokRuntime<br/>request shaping<br/>response shaping<br/>memory projection<br/>tool normalization<br/>replay and gating<br/>telemetry"]

    CB["ClaudeBridgeAdapter<br/>gateway/__init__.py"]
    OA["OpenAIChatAdapter<br/>live_runner.py"]
    TL["TextLoopAdapter<br/>agent.py"]
    OR["OrchestratorAdapter<br/>tok_orchestrator.py boundary"]
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

## 5. Runtime Surfaces and Ownership

```mermaid
flowchart TB
    subgraph Canonical["Canonical Runtime"]
        RT["universal_runtime.py"]
    end

    subgraph Adapters["Adapters"]
        GW["gateway/__init__.py"]
        AD["adapters.py"]
        LR["live_runner.py"]
        AG["agent.py"]
        TO["tok_orchestrator.py"]
    end

    subgraph Protocol["Protocol Substrate"]
        FB["format_bridge.py"]
        PA["parser.py"]
        EN["encoder.py"]
        SC["schema.py"]
    end

    subgraph Shared["Shared Runtime Services"]
        CL["cli/__init__.py replay path"]
        ST["stats.py telemetry"]
        BM["bridge_memory.py"]
        CP["compression/__init__.py"]
    end

    GW --> RT
    AD --> RT
    LR --> AD
    AG --> AD
    TO --> AD
    CL --> RT
    ST --> RT
    BM --> RT
    CP --> RT
    FB -. protocol layer .-> RT
    PA -. protocol layer .-> RT
    EN -. protocol layer .-> RT
    SC -. protocol layer .-> RT
```

## 6. Deferred Orchestrator Migration Boundary

```mermaid
flowchart LR
    USER["User Input"] --> LOOP["TokOrchestrator main loop<br/>sliding window<br/>retry logic<br/>tool execution<br/>strict-mode validation"]

    LOOP --> BOUNDARY["OrchestratorAdapter<br/>runtime boundary now present"]
    BOUNDARY --> RT["UniversalTokRuntime"]

    LEGACY["Legacy-owned today<br/>multi-system prompt assembly<br/>large turn loop<br/>truncation policy"] -. still deferred .-> LOOP
```

## 7. Adoption Story

```mermaid
flowchart TB
    INV["Invisible-first adoption<br/>no Tok authoring required"] --> BR["Bridge wins as primary surface"]
    BR --> PUB["Narrow bridge-first public release"]
    PUB --> RT["Same runtime semantics expand across surfaces"]
    RT --> STD["Explicit Tok language surface<br/>standardizes later"]
    STD --> AA["Human-agent and agent-agent standardization"]
```

---

See [roadmap.md](../roadmap.md) for latest planning and phase status.
