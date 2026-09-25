# SIVAC: System Design & Planning Specification
**Project Title:** SIVAC — Self-Improving Vision Agent for Autonomous Computer Use  
**Document Identifier:** SIVAC-ENG-DES-002  
**Version:** 1.0.0  
**Author:** Abhishek Sahu (`abhisheksssss`)  
**Status:** Approved for Implementation  
**Date:** September 2026  

---

## 1. Executive Summary & Architectural Mission

Modern autonomous computer-use agents must reliably bridge human natural language objectives with graphical user interface (GUI) environments across both desktop operating systems and web browsers. Existing open-loop, vision-only agents suffer from four systemic vulnerabilities: **perception coordinate drift**, **amnesiac execution** (inability to learn from previous mistakes), **open-loop failure propagation** (blind execution without outcome verification), and **prohibitive VLM inference costs**.

**SIVAC** (*Self-Improving Vision Agent for Autonomous Computer Use*) resolves these challenges through a deterministic, closed-loop, and self-improving architecture. Rather than relying on simple end-to-end prompt-and-click loops, SIVAC implements an 8-stage cybernetic control loop:

$$\text{Observe} \longrightarrow \text{Perceive} \longrightarrow \text{Retrieve} \longrightarrow \text{Plan} \longrightarrow \text{Act} \longrightarrow \text{Verify} \longrightarrow \text{Diagnose/Recover} \longrightarrow \text{Reflect/Learn}$$

### Core Design Objectives
1. **Hybrid Deterministic Perception:** Fuse fast local perception (Tesseract OCR, Windows UI Automation, Playwright Web DOM) with multimodal Vision-Language Models (VLMs) using spatial Intersection-over-Union (IoU) deduplication to eliminate coordinate hallucination.
2. **Deterministic Closed-Loop Verification:** Every dispatched mouse or keyboard primitive is actively verified via visual diffing (SSIM / perceptual hash) and DOM state delta before proceeding to the next step.
3. **Dual-Layer Memory & Self-Improvement:** Maintain an episodic trajectory database (SQLite) for granular operational telemetry and a semantic vector memory (ChromaDB) that abstracts failure patterns into reusable heuristic rules.
4. **Guaranteed Zero-Cost Tier Viability:** Architected to run on high-performance free-tier API endpoints (Groq Llama 3 / Qwen-VL, Google Gemini 2.0 Flash, NVIDIA NIM, Cerebras) without compromising enterprise capability.
5. **Absolute Safety Guardrails:** Strict allowlisted JSON action schemas, screen boundary validation, and an immediate hardware/software emergency stop (Kill Switch) preventing unintended or destructive operations.

---

## 2. System Architecture

### 2.1 High-Level Architecture Diagram

```mermaid
flowchart TB
    subgraph UserInterface["1. Presentation & Telemetry Layer"]
        CLI["SIVAC CLI Client\n(Click / Rich)"]
        WebDashboard["React + Vite Telemetry Web UI\n(Live Screen / Trajectory / Controls)"]
        FastAPIServer["FastAPI Gateway & WebSocket Server\n(/api/v1/tasks, /ws/stream)"]
    end

    subgraph CoreEngine["2. Agent Orchestration Engine (LangGraph)"]
        Controller["LangGraph State Machine\n(AgentState)"]
        Planner["Hierarchical Planner\n(Reasoning Model)"]
        Verifier["Two-Tier Outcome Verifier\n(Visual Diff + DOM State)"]
        RecoveryEngine["Autonomous Recovery Engine\n(Backtracking & Replanning)"]
        ReflectionEngine["Post-Task Reflector\n(Lesson Extraction)"]
    end

    subgraph PerceptionLayer["3. Hybrid Multi-Modal Perception Engine"]
        ScreenCap["MSS Screen Capture\n(Multi-Monitor & DPI Aware)"]
        VLMGrounder["VLM Element Detector\n(Groq Qwen-VL / Gemini Flash)"]
        OCRScanner["OCR Text Engine\n(pytesseract)"]
        DOMInspect["Browser DOM Inspector\n(Playwright Accessibility Tree)"]
        UIAInspect["Windows UI Automation\n(pywinauto / uiautomation)"]
        SpatialMerger["Hybrid Spatial Merger\n(IoU Deduplication: DOM > UIA > VLM > OCR)"]
        UnifiedState["Unified UIState\n(Elements + BBoxes + Labels)"]
    end

    subgraph ActionSafetyLayer["4. Action Primitives & Safety Subsystem"]
        SafetyGuard["Safety Boundary Guard\n(Coordinate Validation & Allowlist)"]
        KillSwitch["Emergency Kill Switch\n(FailSafe Corner / Escape Key)"]
        ActionExecutor["Action Dispatcher\n(schema.py ActionCommand)"]
        MouseController["Mouse Driver\n(pyautogui smooth bezier)"]
        KeyController["Keyboard Driver\n(typewrite, hotkey, press)"]
        BrowserController["Playwright Browser Driver\n(click, fill, goto, evaluate)"]
        DesktopController["OS Window Manager\n(focus, minimize, launch)"]
    end

    subgraph StorageLayer["5. Dual-Layer Memory & Knowledge Store"]
        SQLiteDB[("SQLite Trajectory DB\n(SQLAlchemy 2.0)\nTasks, Steps, Verifications")]
        ChromaStore[("ChromaDB Vector Store\n(Embeddings: all-MiniLM-L6-v2)\nLearned Skills & Failure Heuristics")]
    end

    subgraph ModelGateway["6. Multi-Provider Model Abstraction Layer"]
        ModelFactory["ModelFactory\n(LangChain BaseChatModel)"]
        GroqProvider["Groq (Qwen 2.5-VL / Llama 3.3)"]
        GeminiProvider["Google Gemini 2.0 Flash"]
        NVIDIAProvider["NVIDIA NIM (Llama-3-Vision / DeepSeek)"]
        OpenRouterProvider["OpenRouter / Cerebras / Mistral"]
    end

    %% Communications
    CLI --> FastAPIServer
    WebDashboard --> FastAPIServer
    FastAPIServer --> Controller

    Controller --> PerceptionLayer
    ScreenCap --> VLMGrounder & OCRScanner
    DOMInspect & UIAInspect & VLMGrounder & OCRScanner --> SpatialMerger
    SpatialMerger --> UnifiedState
    UnifiedState --> Controller

    Controller --> Planner
    ModelGateway <--> Planner & VLMGrounder & Verifier & ReflectionEngine
    ChromaStore -.->|Few-Shot Relevant Skills| Planner

    Planner --> SafetyGuard
    KillSwitch -.->|Hardware Interrupt| ActionExecutor
    SafetyGuard --> ActionExecutor
    ActionExecutor --> MouseController & KeyController & BrowserController & DesktopController

    ActionExecutor --> Verifier
    Verifier -->|Step Succeeded| Controller
    Verifier -->|Step Failed| RecoveryEngine
    RecoveryEngine --> Planner

    Controller -->|Session Finished| ReflectionEngine
    ReflectionEngine --> SQLiteDB
    ReflectionEngine --> ChromaStore
```

### 2.2 Subsystem Decomposition

| Subsystem | Responsibilities | Key Technologies |
| :--- | :--- | :--- |
| **Presentation & Telemetry** | Provides real-time operator control, task input, visual overlay rendering, trajectory inspection, and instant emergency stop. | FastAPI, WebSockets, React, Vite, Lucide Icons, Vanilla CSS |
| **Orchestration (LangGraph)** | Governs the finite state machine of the agent, maintaining episodic state, managing transitions, and coordinating error recovery. | LangGraph, Pydantic v2, Python 3.12 |
| **Model Abstraction** | Unified multi-provider interface with automated fallback, rate-limit retries, and zero-cost endpoint routing. | LangChain Core, `ChatGoogleGenerativeAI`, `ChatGroq`, `ChatNVIDIA`, `ChatOpenAI` |
| **Hybrid Perception** | Captures multi-monitor high-DPI screenshots, extracts structured accessibility trees, runs OCR, detects visual affordances, and merges overlapping bboxes. | `mss`, Pillow, `pytesseract`, Playwright, `pywinauto`, `uiautomation` |
| **Action & Safety** | Validates structured action commands against screen geometry and security rules; dispatches mouse/keyboard/DOM events safely. | PyAutoGUI, Playwright, pywinauto, Pydantic |
| **Outcome Verifier** | Verifies execution outcomes through structural DOM comparison, OCR text assertion, and perceptual visual diffs. | Pillow, OpenCV/SSIM, DOM delta |
| **Dual-Layer Memory** | Stores exact step-by-step audit logs in SQLite and vectorizes failure reflections and operational heuristics in ChromaDB. | SQLAlchemy 2.0, SQLite, ChromaDB, Sentence-Transformers |

---

## 3. UML Diagrams

### 3.1 Use Case Diagram

```mermaid
flowchart LR
    Operator((Human Operator))
    BenchmarkRunner((QA / Benchmark Harness))
    Agent((SIVAC Agent System))
    TargetOS((Windows / Browser OS))

    subgraph SIVAC_Capabilities["SIVAC Autonomous Computer-Use Boundaries"]
        UC1["Submit Natural Language Goal"]
        UC2["Trigger Emergency Kill-Switch"]
        UC3["Live Screen & Trajectory Inspection"]
        UC4["Perceive Multi-Modal UI State"]
        UC5["Retrieve Relevant Past Experience"]
        UC6["Plan Step-by-Step Subgoals"]
        UC7["Validate Safety & Boundaries"]
        UC8["Dispatch Action Primitive"]
        UC9["Verify Post-Action Outcome"]
        UC10["Diagnose & Recover from Failure"]
        UC11["Reflect & Extract Generalizable Lessons"]
        UC12["Persist Trajectory & Update Knowledge Base"]
    end

    Operator --> UC1
    Operator --> UC2
    Operator --> UC3
    BenchmarkRunner --> UC1

    UC1 --> Agent
    Agent --> UC4
    Agent --> UC5
    Agent --> UC6
    Agent --> UC7
    Agent --> UC8
    Agent --> UC9
    Agent --> UC10
    Agent --> UC11
    Agent --> UC12

    UC2 -.->|Interrupts| UC8
    UC8 --> TargetOS
    TargetOS -.->|Screenshots & DOM| UC4
    UC9 -.->|If Step Fails| UC10
    UC10 -.->|Recovers| UC6
```

### 3.2 Class Diagram

```mermaid
classDiagram
    %% Core Perception Classes
    class BBox {
        +int x
        +int y
        +int width
        +int height
        +center() tuple[int, int]
        +area() int
        +iou(other: BBox) float
    }

    class UIElement {
        +str id
        +str source
        +str type
        +str text
        +BBox bbox
        +bool interactive
        +dict attributes
    }

    class UIState {
        +str screenshot_path
        +int screen_width
        +int screen_height
        +float dpi_scale
        +list~UIElement~ elements
        +str active_app
        +get_element_by_id(elem_id: str) UIElement
        +get_elements_by_text(query: str) list~UIElement~
    }

    class HybridPerceptionEngine {
        -ScreenshotEngine screenshot_engine
        -VLMElemDetector vlm_detector
        -OCRDetector ocr_detector
        -DOMInspector dom_inspector
        -UIAutomationInspector uia_inspector
        +capture_and_perceive(context: dict) UIState
        -merge_elements(elements: list~UIElement~) list~UIElement~
    }

    %% Action & Safety Classes
    class ActionType {
        <<enumeration>>
        CLICK
        DOUBLE_CLICK
        RIGHT_CLICK
        HOVER
        TYPE
        PRESS_KEY
        HOTKEY
        SCROLL
        DRAG
        WAIT
        DOM_CLICK
        DOM_TYPE
        NAVIGATE
        TERMINATE
    }

    class ActionCommand {
        +ActionType action_type
        +str target_id
        +tuple[int, int] coordinates
        +str text
        +str key
        +list~str~ keys
        +int scroll_amount
        +float duration
        +dict params
    }

    class ExecutionResult {
        +bool success
        +str action_name
        +dict details
        +str error
        +float execution_time
    }

    class SafetyGuard {
        +int screen_width
        +int screen_height
        +list~str~ blocked_keys
        +validate_action(cmd: ActionCommand) bool
        +check_boundaries(x: int, y: int) bool
    }

    class ActionExecutor {
        -SafetyGuard safety_guard
        -MouseController mouse
        -KeyboardController keyboard
        -BrowserController browser
        -DesktopController desktop
        +execute(cmd: ActionCommand, state: UIState) ExecutionResult
    }

    %% Orchestration & Verification Classes
    class AgentState {
        +str task_id
        +str goal
        +int step_count
        +int max_steps
        +UIState current_ui_state
        +list~ActionCommand~ action_history
        +list~ExecutionResult~ execution_history
        +list~dict~ verification_history
        +list~dict~ learned_rules_in_context
        +bool is_complete
        +bool is_failed
        +str error_summary
    }

    class OutcomeVerifier {
        +verify_step(before: UIState, after: UIState, cmd: ActionCommand) tuple[bool, str]
        -calculate_visual_diff(img1: str, img2: str) float
        -verify_dom_change(cmd: ActionCommand) bool
    }

    class RecoveryEngine {
        +diagnose_failure(state: AgentState, reason: str) dict
        +generate_recovery_plan(diagnosis: dict) list~ActionCommand~
    }

    class ReflectionEngine {
        +reflect_on_trajectory(task_id: str, history: list) dict
        +extract_lessons(reflections: dict) list~dict~
    }

    %% Relationships
    UIElement *-- BBox
    UIState o-- UIElement
    HybridPerceptionEngine ..> UIState : produces
    ActionCommand *-- ActionType
    ActionExecutor ..> ActionCommand : consumes
    ActionExecutor ..> ExecutionResult : produces
    ActionExecutor --> SafetyGuard
    AgentState o-- UIState
    AgentState o-- ActionCommand
    AgentState o-- ExecutionResult
    OutcomeVerifier ..> AgentState : validates
    RecoveryEngine ..> AgentState : recovers
```

### 3.3 Sequence Diagram: End-to-End Closed-Loop Execution

```mermaid
sequenceDiagram
    autonumber
    actor User as Human Operator
    participant API as FastAPI / UI Gateway
    participant LG as LangGraph Orchestrator
    participant Mem as Memory Store (Chroma + SQLite)
    participant Percept as Hybrid Perception Engine
    participant LLM as Model Gateway (Groq / Gemini)
    participant Safe as Safety Guard
    participant Act as Action Executor
    participant OS as OS / Browser Environment
    participant Verif as Outcome Verifier
    participant Refl as Reflection Engine

    User->>API: POST /api/v1/tasks {goal: "Download Q3 Report in Chrome and Open in Excel"}
    API->>LG: Initialize AgentState(task_id, goal)
    LG->>Mem: Query relevant past skills & failure heuristics
    Mem-->>LG: Return matched heuristic rules (e.g., "Wait 2s for download modal")
    
    rect rgb(240, 245, 255)
    note over LG, Verif: Main Execution Control Loop (Repeated per Step)
        LG->>Percept: capture_and_perceive(screen, dom, uia)
        Percept->>OS: Capture Screen & Accessibility Trees
        OS-->>Percept: Hardware Buffers + DOM Tree
        Percept-->>LG: Unified UIState (Labeled interactive elements + BBoxes)
        
        LG->>LLM: Generate Next Action(goal, UIState, retrieved_rules, history)
        LLM-->>LG: ActionCommand(type=CLICK, target_id="btn_download_q3")
        
        LG->>Safe: validate_action(cmd, screen_bounds)
        alt Action Violates Safety Boundary
            Safe-->>LG: SafetyException("Coordinates out of bounds")
            LG->>LG: Transition to Replan
        else Action is Safe
            Safe-->>LG: Approved
            LG->>Act: execute(ActionCommand, UIState)
            Act->>OS: Dispatch Hardware Click Event
            OS-->>Act: Input Acknowledged
            Act-->>LG: ExecutionResult(success=True)
            
            LG->>Percept: capture_after_action()
            Percept-->>LG: Post-Action UIState
            
            LG->>Verif: verify_step(pre_state, post_state, cmd)
            alt Outcome Verified (State Changed as Expected)
                Verif-->>LG: VerificationResult(passed=True)
                LG->>Mem: Log Step to SQLite Trajectory Store
            else Outcome Failed (UI Lag / Element not clicked)
                Verif-->>LG: VerificationResult(passed=False, "Download dialog not opened")
                LG->>LG: Trigger Autonomous Recovery (Retry with alternate coordinate / Wait)
            end
        end
    end

    rect rgb(245, 255, 240)
    note over LG, Mem: Post-Task Consolidation & Learning
        LG->>Refl: reflect_on_trajectory(task_history)
        Refl->>LLM: Analyze trajectory for reusable rules & friction points
        LLM-->>Refl: Extracted Rules (e.g., "Portal X download triggers Chrome shelf")
        Refl->>Mem: Store Structured Trajectory in SQLite
        Refl->>Mem: Vectorize & Insert Reusable Heuristic in ChromaDB
        LG-->>API: Task Completed Successfully
        API-->>User: Notification & Telemetry Report
    end
```

### 3.4 State Machine Diagram (LangGraph Workflow)

```mermaid
stateDiagram-v2
    [*] --> INITIALIZING : Task Goal Received
    
    INITIALIZING --> RETRIEVING_EXPERIENCE : Task ID & State Allocated
    RETRIEVING_EXPERIENCE --> PERCEIVING_UI : Past Heuristics Injected
    
    PERCEIVING_UI --> PLANNING_STEP : UIState Constructed
    
    PLANNING_STEP --> EVALUATING_GOAL : Plan Next Action
    EVALUATING_GOAL --> TASK_SUCCEEDED : Goal Criteria Met
    EVALUATING_GOAL --> VALIDATING_SAFETY : Next Subgoal Defined
    
    VALIDATING_SAFETY --> DISPATCHING_ACTION : Command Approved
    VALIDATING_SAFETY --> RECOVERY_MODE : Safety Violation Blocked
    
    DISPATCHING_ACTION --> VERIFYING_OUTCOME : Hardware Input Dispatched
    
    VERIFYING_OUTCOME --> RECORDING_STEP : State Delta Matches Expected
    VERIFYING_OUTCOME --> RECOVERY_MODE : Delta Null or Unexpected
    
    RECOVERY_MODE --> PLANNING_STEP : Retry with Backoff / Alternative
    RECOVERY_MODE --> TASK_FAILED : Max Retries (3) Exceeded
    
    RECORDING_STEP --> PERCEIVING_UI : Step Incremented (< Max Steps)
    RECORDING_STEP --> TASK_SUCCEEDED : Final Goal State Achieved
    
    TASK_SUCCEEDED --> REFLECTING_AND_LEARNING : Generate Trajectory Summary
    TASK_FAILED --> REFLECTING_AND_LEARNING : Diagnose Root Cause
    
    REFLECTING_AND_LEARNING --> CONSOLIDATING_MEMORY : Extract Heuristic Rules
    CONSOLIDATING_MEMORY --> [*] : Session Finalized & Persisted
```

---

## 4. Database & Storage Architecture

SIVAC employs a **dual-layer storage architecture**:
1. **Relational Database (SQLite / SQLAlchemy 2.0):** Guarantees strict transactional integrity, foreign-key relationships, and indexing for detailed audit logging of tasks, trajectory steps, raw actions, and visual verification results.
2. **Vector Database (ChromaDB):** Embeds high-level task goals, UI contexts, and post-task failure lessons into dense vector collections for fast semantic retrieval during the planning phase of future sessions.

### 4.1 Relational Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    TASKS ||--o{ TRAJECTORIES : "has"
    TRAJECTORIES ||--|{ STEPS : "contains"
    STEPS ||--|| VERIFICATIONS : "evaluated_by"
    TASKS ||--o| REFLECTIONS : "generates"
    REFLECTIONS ||--o{ LEARNED_SKILLS : "extracts"

    TASKS {
        string id PK "UUID4"
        string goal "Natural Language Goal"
        string status "PENDING | RUNNING | COMPLETED | FAILED"
        int total_steps "Total execution steps"
        float total_duration_seconds "Execution time"
        datetime created_at "Creation timestamp"
        datetime completed_at "Completion timestamp"
    }

    TRAJECTORIES {
        string id PK "UUID4"
        string task_id FK "References TASKS.id"
        string model_used "Model identifier (e.g. llama-3.3-70b)"
        int prompt_tokens "Total prompt tokens"
        int completion_tokens "Total completion tokens"
        float estimated_cost_usd "Estimated API cost"
        datetime created_at "Start timestamp"
    }

    STEPS {
        string id PK "UUID4"
        string trajectory_id FK "References TRAJECTORIES.id"
        int step_number "1-indexed step sequence"
        string screenshot_before "File path to pre-action image"
        string screenshot_after "File path to post-action image"
        json observation_elements "Captured UI elements snapshot"
        string planned_subgoal "Subgoal reasoning from planner"
        json action_command "Serialized ActionCommand JSON"
        json execution_result "Serialized ExecutionResult JSON"
        datetime executed_at "Timestamp of action"
    }

    VERIFICATIONS {
        string id PK "UUID4"
        string step_id FK "References STEPS.id"
        boolean passed "True if action verified"
        float visual_diff_score "SSIM or pixel delta ratio"
        json dom_delta "DOM tree mutation diff"
        string failure_reason "Explanation if verification failed"
        int retry_count "Number of recovery retries"
    }

    REFLECTIONS {
        string id PK "UUID4"
        string task_id FK "References TASKS.id"
        boolean overall_success "Task success indicator"
        string root_cause_analysis "Failure or friction explanation"
        json friction_points "Array of detected UI bottlenecks"
        datetime created_at "Timestamp of reflection"
    }

    LEARNED_SKILLS {
        string id PK "UUID4"
        string reflection_id FK "References REFLECTIONS.id"
        string application_name "Target app (e.g. Chrome, Excel)"
        string trigger_condition "Context where skill activates"
        string heuristic_rule "Extracted operational rule"
        int success_count "Times rule led to success"
        int failure_count "Times rule led to failure"
        float confidence_score "Heuristic reliability [0.0 - 1.0]"
        datetime updated_at "Last update timestamp"
    }
```

### 4.2 SQL DDL Schema Specification

```sql
-- SIVAC Relational Schema (SQLite 3 compatible)

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY NOT NULL,
    goal TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    total_steps INTEGER NOT NULL DEFAULT 0,
    total_duration_seconds REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trajectories (
    id TEXT PRIMARY KEY NOT NULL,
    task_id TEXT NOT NULL,
    model_used TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    id TEXT PRIMARY KEY NOT NULL,
    trajectory_id TEXT NOT NULL,
    step_number INTEGER NOT NULL,
    screenshot_before TEXT NOT NULL,
    screenshot_after TEXT,
    observation_elements TEXT, -- JSON blob of UI elements
    planned_subgoal TEXT NOT NULL,
    action_command TEXT NOT NULL, -- JSON blob of ActionCommand
    execution_result TEXT NOT NULL, -- JSON blob of ExecutionResult
    executed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(id) ON DELETE CASCADE,
    UNIQUE(trajectory_id, step_number)
);

CREATE TABLE IF NOT EXISTS verifications (
    id TEXT PRIMARY KEY NOT NULL,
    step_id TEXT NOT NULL UNIQUE,
    passed INTEGER NOT NULL CHECK(passed IN (0, 1)),
    visual_diff_score REAL NOT NULL DEFAULT 0.0,
    dom_delta TEXT, -- JSON blob
    failure_reason TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (step_id) REFERENCES steps(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reflections (
    id TEXT PRIMARY KEY NOT NULL,
    task_id TEXT NOT NULL UNIQUE,
    overall_success INTEGER NOT NULL CHECK(overall_success IN (0, 1)),
    root_cause_analysis TEXT NOT NULL,
    friction_points TEXT, -- JSON array
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS learned_skills (
    id TEXT PRIMARY KEY NOT NULL,
    reflection_id TEXT,
    application_name TEXT NOT NULL,
    trigger_condition TEXT NOT NULL,
    heuristic_rule TEXT NOT NULL,
    success_count INTEGER NOT NULL DEFAULT 1,
    failure_count INTEGER NOT NULL DEFAULT 0,
    confidence_score REAL NOT NULL DEFAULT 0.8,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (reflection_id) REFERENCES reflections(id) ON DELETE SET NULL
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_steps_trajectory ON steps(trajectory_id, step_number);
CREATE INDEX IF NOT EXISTS idx_skills_app ON learned_skills(application_name);
```

### 4.3 Vector Database Schema (ChromaDB Collections)

ChromaDB is utilized as a local, zero-infrastructure vector store embedded directly in the agent runtime. It maintains two specialized collections:

#### Collection 1: `episodic_trajectories`
* **Purpose:** Enables few-shot retrieval of past task demonstrations given a new natural language user objective.
* **Document Text:** Concatenation of `goal` and `initial_active_app` (e.g., `"Goal: Download invoice PDF in Google Chrome. App: Chrome"`).
* **Metadata Schema:**
  ```json
  {
    "task_id": "str (UUID4)",
    "status": "COMPLETED | FAILED",
    "total_steps": "int",
    "duration_seconds": "float",
    "model_used": "str",
    "timestamp": "float"
  }
  ```
* **Distance Metric:** Cosine Similarity (`hnsw:space: cosine`).

#### Collection 2: `learned_heuristics`
* **Purpose:** Dynamic retrieval of application-specific interaction rules injected directly into the Hierarchical Planner's context.
* **Document Text:** `trigger_condition` + `" | Application: "` + `application_name` (e.g., `"Trigger: File download modal in Chrome | Rule: Wait 2.5s for file shelf animation to finish before clicking"`).
* **Metadata Schema:**
  ```json
  {
    "skill_id": "str (UUID4)",
    "application_name": "str",
    "confidence_score": "float",
    "success_count": "int",
    "failure_count": "int"
  }
  ```
* **Embedding Model:** Local default `all-MiniLM-L6-v2` (running locally via ONNX without API token usage) or optional `text-embedding-004`.

---

## 5. Technology Stack Selection & Tradeoff Analysis

| Layer | Selected Technology | Alternative Considered | Technical Tradeoff & Rationale |
| :--- | :--- | :--- | :--- |
| **Language & Runtime** | **Python 3.12** | Python 3.10 / Node.js / Rust | Python is the uncontested standard for AI/VLM orchestration, LangChain/LangGraph, and OS automation bindings (`pywinauto`, `pyautogui`). Python 3.12 provides substantial runtime speedups and native typing features. |
| **Package Management** | **`uv` + `hatchling`** | Poetry / Pipenv / conda | `uv` is 10-100x faster than pip/poetry, handles cross-platform lockfiles cleanly (`uv.lock`), and integrates directly with modern PEP 621 `pyproject.toml`. |
| **Orchestration Engine** | **LangGraph** | AutoGen / CrewAI / Raw While-Loops | LangGraph models the agent as a cyclic, deterministic Directed Acyclic Graph (DAG) with explicit checkpointing, state rollbacks, and recovery branches. AutoGen/CrewAI lack fine-grained deterministic step verification. |
| **Model Abstraction** | **LangChain BaseChatModel** | Raw REST / LiteLLM | Provides standardized multi-modal message formats across OpenAI, Google GenAI, Groq, and NVIDIA NIM with unified streaming and tool-calling interfaces. |
| **Vision-Language Model** | **Groq Qwen 2.5-VL / Gemini 2.0 Flash** | GPT-4o / Claude 3.5 Sonnet | Groq and Google AI Studio provide high-throughput free-tier inference (sub-500ms TTFT), allowing cost-effective high-frequency visual ground loops that would cost $50+/hr on commercial proprietary models. |
| **Screen Perception** | **`mss` + PIL + `pytesseract`** | OpenCV raw / PyGetWindow | `mss` captures native multi-monitor screen buffers in <15ms without OS overhead. Local `pytesseract` provides zero-cost, instant text detection fallback when cloud VLMs downsample tiny fonts. |
| **Accessibility & DOM** | **Playwright + `pywinauto`** | Selenium / WinAppDriver | Playwright provides deep, headless/headful CDP DOM access with native element coordinate mapping. `pywinauto` gives native Windows accessibility trees without requiring heavy server daemons. |
| **Action Automation** | **`pyautogui` + OS Virtual Inputs** | Native C Win32 `SendInput` | PyAutoGUI provides cross-platform mouse/keyboard automation with built-in fail-safes (moving cursor to corner triggers `FailSafeException`). |
| **Relational Storage** | **SQLite + SQLAlchemy 2.0** | PostgreSQL / MySQL | Zero configuration, file-based portability, fully ACID compliant, and zero operational overhead for desktop users. Can be swapped for PostgreSQL in enterprise multi-agent deployments. |
| **Vector Storage** | **ChromaDB (Embedded)** | Pinecone / Weaviate / Milvus | Pure Python embedded vector store running in-process; requires no external Docker container or hosted cloud service, maintaining zero setup friction. |
| **API & Web Server** | **FastAPI + Uvicorn** | Flask / Django | Asynchronous native support for high-throughput WebSockets (streaming live desktop frames and real-time execution logs) with automatic OpenAPI documentation. |
| **Web Dashboard UI** | **React + Vite + Modern CSS** | Streamlit / Gradio | Streamlit and Gradio re-render entire components upon state change, causing unmanageable visual flickering during live 30fps screen streaming. React provides fluid, low-latency UI updates. |

---

## 6. Project Plan & Implementation Timeline

The project follows a rigorous 9-phase progressive engineering roadmap. Each phase delivers fully tested, decoupled modules before proceeding to high-level orchestration.

### 6.1 Phase Breakdown & Status Matrix

```mermaid
gantt
    title SIVAC Engineering Implementation Roadmap
    dateFormat  YYYY-MM-DD
    section Completed
    Phase 1 Model Abstraction Layer       :done, p1, 2026-08-01, 2026-08-07
    Phase 2 Hybrid Perception Engine      :done, p2, 2026-08-08, 2026-08-20
    section Active
    Phase 3 Action Primitives & Safety    :active, p3, 2026-08-21, 2026-08-28
    section Planned
    Phase 4 Outcome Verifier & Recovery  :crit, p4, 2026-08-29, 2026-09-07
    Phase 5 Dual-Layer Memory Store      :p5, 2026-09-08, 2026-09-15
    Phase 6 LangGraph Orchestration      :p6, 2026-09-16, 2026-09-24
    Phase 7 Self-Improvement & Reflection:p7, 2026-09-25, 2026-10-03
    Phase 8 Telemetry API & Dashboard    :p8, 2026-10-04, 2026-10-12
    Phase 9 E2E Benchmarking & Hardening :p9, 2026-10-13, 2026-10-22
```

### 6.2 Detailed Phase Deliverables

#### Phase 1: Model Abstraction Layer [COMPLETED ✅]
* **Deliverables:**
  * Pydantic configuration settings (`src/vision_agent/config.py`) loading multi-provider keys (`GEMINI`, `GROQ`, `NVIDIA`, `OPENROUTER`).
  * Unified `ModelFactory` (`src/vision_agent/model/`) wrapping LangChain chat models with automated fallback.
  * Integration test suite (`test_models.py`) validating model reasoning and VLM capabilities.

#### Phase 2: Hybrid Multi-Modal Perception Engine [COMPLETED ✅]
* **Deliverables:**
  * `state.py`: Domain models for `BBox`, `UIElement`, and `UIState`.
  * `screenshot.py`: Multi-monitor, DPI-aware capture engine via `mss` and PIL.
  * `vlm_detector.py`: Visual grounding parser prompting VLMs to return bounding boxes.
  * `ocr.py`: Local `pytesseract` OCR text locator.
  * `dom.py` & `ui_automation.py`: Playwright DOM inspector and Windows UI Automation tree traversal.
  * `hybrid_merger.py`: Spatial deduplication engine with IoU hierarchy (`DOM > UIA > VLM > OCR`).
  * Test suite (`test_perception.py`) verifying element detection and spatial merging.

#### Phase 3: Action Primitive Layer & Safety Subsystem [UNDERWAY 🔄]
* **Deliverables:**
  * `schema.py`: Strict Pydantic action definitions (`ActionType`, `ActionCommand`, `ExecutionResult`).
  * `safety.py`: Screen boundary checks, forbidden hotkey blocking, and emergency corner failsafe.
  * `mouse.py` & `keyboard.py`: Human-like smooth Bézier mouse movement and typed keystrokes.
  * `browser.py` & `desktop.py`: Playwright DOM element actions and OS window management.
  * `executor.py`: Unified dispatcher resolving element IDs to physical screen coordinates.
  * Verification suite (`test_actions.py`) validating mouse clicks, typing, and safety aborts.

#### Phase 4: Outcome Verifier & Closed-Loop Recovery Engine [UPCOMING]
* **Deliverables:**
  * Two-tier verifier combining visual pixel/SSIM delta and DOM mutation tracking.
  * Heuristic failure classifier distinguishing UI lag from missing elements or unclickable layers.
  * Backtracking recovery engine implementing retry with exponential backoff and alternate strategies.

#### Phase 5: Dual-Layer Memory & Trajectory Storage [UPCOMING]
* **Deliverables:**
  * SQLAlchemy 2.0 SQLite schema (`tasks`, `trajectories`, `steps`, `verifications`, `reflections`).
  * ChromaDB vector collections for task trajectories and learned application heuristics.
  * High-speed semantic similarity retrieval pipeline for planning prompts.

#### Phase 6: Agent Orchestration Engine (LangGraph) [UPCOMING]
* **Deliverables:**
  * LangGraph cyclic state machine managing `AgentState`.
  * Hierarchical planner node decomposing high-level tasks into discrete subgoals.
  * Integrated safety check, execution, verification, and recovery nodes.

#### Phase 7: Self-Improvement & Reflection Engine [UPCOMING]
* **Deliverables:**
  * Post-task trajectory analyzer identifying friction points and recurring errors.
  * Heuristic lesson extraction module synthesizing natural language rules into structured skill objects.
  * Dynamic rule confidence updater adjusting weights based on subsequent task success.

#### Phase 8: Telemetry, API Gateway & Web Dashboard [UPCOMING]
* **Deliverables:**
  * FastAPI REST API (`/api/v1/tasks`, `/api/v1/trajectories`, `/api/v1/skills`).
  * Low-latency WebSocket streaming live screen captures with annotated element bounding boxes.
  * Modern React + Vite web dashboard featuring live stream, manual override, and audit playback.

#### Phase 9: End-to-End Evaluation & Benchmark Hardening [UPCOMING]
* **Deliverables:**
  * Benchmark test harness evaluating success rate against standard computer-use suites (OSWorld, WebArena).
  * Robustness stress testing under UI lag, modal interruptions, and multi-monitor setups.
  * Final production packaging via Hatchling wheels and `sivac` CLI entry point.

---

## 7. Risk Analysis & Mitigation Strategies

| Risk Factor | Severity | Probability | Impact Description | Architectural Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **VLM Coordinate Drift** | High | High | VLM hallucinates click coordinates by 20-50px, clicking empty space. | **Hybrid Spatial Merger:** Prioritize exact DOM and Windows UI Automation coordinates over VLM estimates. Use VLM only for semantic grounding. |
| **Asynchronous UI Latency** | High | High | Agent clicks an element before page rendering finishes, causing action loss. | **Post-Action Verifier:** Enforce mandatory wait-and-verify loops with configurable timeout (up to 3s) before declaring step completion. |
| **Infinite Failure Death Loops** | Critical | Med | Agent gets stuck repeating the same failing action indefinitely. | **Recovery Circuit Breaker:** Track step action signatures. If identical action fails twice, trigger autonomous backtracking or abort task. |
| **API Rate Limiting / Downtime** | Med | Med | Free-tier endpoints (Groq/Gemini) throttle requests during intense execution. | **Multi-Provider Fallback:** LangChain `ModelFactory` dynamically cascades requests (`Groq -> Gemini Flash -> NVIDIA NIM -> OpenRouter`). |
| **Accidental Destructive Actions** | Critical | Low | Agent clicks system delete, terminal commands, or sensitive settings. | **Safety Filter & Failsafe:** Hard blocklist of dangerous commands (`rmdir`, `del`, format); physical mouse corner fail-safe immediately aborts execution. |

---

## 8. Summary of Engineering Artifacts Produced

The following blueprint specifications have been formalized and committed to the repository:
1. **Architecture Blueprint:** Complete system and subsystem interaction diagram.
2. **UML Artifacts:** Use Case, Class Diagram, Sequence Diagram, and State Machine.
3. **Database Specifications:** Relational DDL (SQLite) and Vector Schema (ChromaDB).
4. **Technology Stack Selection:** Justified matrix with tradeoff evaluations.
5. **Project Implementation Timeline:** 9-phase Gantt schedule and risk mitigation matrix.
