# Implementation Plan - SIVAC (Self-Improving Vision Agent for Autonomous Computer Use)

SIVAC is an autonomous vision-based computer control agent featuring a closed-loop perception-action cycle and a core **self-improvement mechanism**. SIVAC observes the desktop/browser screen, constructs a unified multi-modal UI state (VLM, OCR, DOM, UIA), plans structured safe actions, executes them via system primitives, verifies outcomes, diagnoses failures, recovers autonomously, and extracts reusable lessons and skills into episodic and semantic vector memory to improve on future tasks.

---

## User Review Required

> [!IMPORTANT]
> **Key Architecture & Technology Choices**:
> 1. **Framework & Orchestration**: LangGraph + FastAPI + Pydantic v2.
> 2. **Model Layer**: Flexible provider interface supporting Gemini (default via `google-genai`), Qwen-VL, OpenAI, NVIDIA NIM, and local Ollama options configurable via `.env`.
> 3. **Perception Engine**:
>    - **Visual**: VLM bounding box detection & UI classification.
>    - **OCR**: `pytesseract` / `easyocr` fallback for text location.
>    - **DOM**: Playwright browser state inspector.
>    - **UIA**: `pywinauto` / `uiautomation` for native Windows accessibility tree extraction.
> 4. **Computer Control & Safety**:
>    - `pyautogui` + `playwright` + `pywinauto`.
>    - **Safety Mechanisms**: Global emergency hotkey (`Ctrl+Alt+Esc`), coordinate bounds checking, confirmation filters for destructive actions (e.g. delete file, send email, checkout), max steps limit.
> 5. **Memory System**:
>    - SQLite / PostgreSQL for trajectory metadata and structured logs.
>    - ChromaDB / FAISS for fast local vector storage of episodic trajectories and extracted lessons.
> 6. **Dashboard UI**: React + Vite + TailwindCSS + Lucide icons + Socket.IO / WebSockets for live screenshot monitoring, step visualization, and memory state inspection.

> [!NOTE]
> We will construct SIVAC modularly so that Phase 1 (Perception & Model Abstractions) through Phase 8 (Self-Improvement & Dashboard UI) can be progressively built and verified.

---

## Proposed System Architecture & Repository Structure

```text
Vision-Agent/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic Settings & Env setup
│   ├── main.py                   # FastAPI server entry point & WebSockets
│   │
│   ├── models/                   # Modular LLM / VLM abstractions
│   │   ├── base.py               # VisionModel and ReasoningModel abstract base classes
│   │   ├── gemini.py             # Google Gemini VLM & LLM implementation
│   │   ├── openai_provider.py    # OpenAI / Qwen / NIM compatibility provider
│   │   └── factory.py            # Model factory based on env configuration
│   │
│   ├── perception/               # Multi-modal UI State Perception
│   │   ├── state.py              # UIState & Element BBox data classes
│   │   ├── screenshot.py         # Screen capture engine (mss / PIL)
│   │   ├── vlm_detector.py       # VLM vision analysis & element extraction
│   │   ├── ocr.py                # OCR fallback text detector
│   │   ├── dom.py                # Playwright DOM element inspector
│   │   ├── ui_automation.py      # Windows UI Automation (pywinauto) inspector
│   │   └── hybrid_merger.py      # Merges VLM, OCR, DOM, and UIA into unified UIState
│   │
│   ├── actions/                  # Structured Action Primitive Layer
│   │   ├── schema.py             # Action Pydantic schemas (Click, Type, Hotkey, etc.)
│   │   ├── mouse.py              # PyAutoGUI mouse controller
│   │   ├── keyboard.py           # PyAutoGUI keyboard controller
│   │   ├── browser.py            # Playwright browser controller
│   │   ├── desktop.py            # Pywinauto application launcher/controller
│   │   ├── executor.py           # Unified ActionExecutor with safety guards & target resolution
│   │   └── safety.py             # Action safety filter & emergency stop monitor
│   │
│   ├── agent/                    # LangGraph Controller & Core Closed-Loop
│   │   ├── state.py              # AgentState definitions for LangGraph
│   │   ├── planner.py            # Next Action Planner
│   │   ├── verifier.py           # Cheap & Visual Outcome Verifier
│   │   ├── recovery.py           # Failure Diagnosis & Strategy Selector
│   │   └── graph.py              # LangGraph state machine definition
│   │
│   ├── memory/                   # Short-term, Episodic & Semantic Memory
│   │   ├── storage.py            # SQLite / Relational store for trajectories & lessons
│   │   ├── vector_store.py       # ChromaDB / FAISS vector storage for episodic recall
│   │   ├── episodic.py           # Trajectory logger & episodic memory manager
│   │   ├── semantic.py           # Knowledge & Lesson store
│   │   └── retrieval.py          # Context-aware experience retriever
│   │
│   ├── learning/                 # Self-Improvement & Reflection Pipeline
│   │   ├── evaluator.py          # Trajectory & step success evaluator
│   │   ├── reflection.py         # Failure & trajectory reflection generator
│   │   ├── lesson_extractor.py   # Extracts actionable rules & avoids past mistakes
│   │   └── strategy.py           # Application-specific perception/action strategy manager
│   │
│   └── dashboard/                # API endpoints for frontend UI
│       ├── router.py
│       └── websocket.py
│
├── frontend/                     # React + Vite + Tailwind Dashboard
│   ├── src/
│   │   ├── components/           # ScreenViewer, ActionHistory, MemoryBrowser, AgentControl
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
│
├── evaluation/                   # Benchmark Suite
│   ├── benchmark_tasks.json      # 50+ benchmark computer tasks
│   ├── runner.py                 # Benchmark evaluation script
│   └── metrics.py                # Task success rate, recovery rate, efficiency calculations
│
├── tests/                        # Unit and integration tests
├── .env.example
├── requirements.txt
├── README.md
└── run.py                        # CLI runner script
```

---

## Detailed Phased Implementation Roadmap

### Phase 1: Core Architecture, Config & Model Abstraction Layer
- Create `requirements.txt` with essential libraries (`fastapi`, `uvicorn`, `pydantic`, `langgraph`, `pyautogui`, `playwright`, `pywinauto`, `mss`, `pillow`, `pytesseract`, `chromadb`, `sqlalchemy`, `google-genai`).
- Implement `app/config.py` using Pydantic Settings for environment variables (`VISION_PROVIDER`, `REASONING_PROVIDER`, `API_KEY`, etc.).
- Implement `app/models/base.py`, `gemini.py`, `openai_provider.py`, and `factory.py` for decoupled vision/reasoning models.

### Phase 2: Hybrid Perception Engine
- Implement screen capture engine in `app/perception/screenshot.py` using `mss` and `Pillow`.
- Define data schemas in `app/perception/state.py` for `Element`, `BBox`, and `UIState`.
- Build VLM detector (`vlm_detector.py`), OCR text finder (`ocr.py`), Playwright DOM inspector (`dom.py`), and Windows UI Automation parser (`ui_automation.py`).
- Implement `app/perception/hybrid_merger.py` to compile elements from all sources into a normalized, de-duplicated `UIState`.

### Phase 3: Controlled Action Primitive Layer & Safety Subsystem
- Define explicit action schemas in `app/actions/schema.py` (`click`, `double_click`, `type`, `press`, `hotkey`, `scroll`, `wait`, `browser_open`, `application_open`).
- Build safety filter & emergency stop handler in `app/actions/safety.py`.
- Implement action execution drivers in `app/actions/mouse.py`, `keyboard.py`, `browser.py`, `desktop.py`.
- Implement semantic target resolution in `app/actions/executor.py` (translating target element IDs / bounding boxes into precise, safe click points).

### Phase 4: Closed-Loop Controller (LangGraph Agent), Verification & Recovery Engine
- Build `AgentState` schema in `app/agent/state.py`.
- Implement `app/agent/planner.py` to prompt LLM with Goal + UIState + Previous Actions + Retrieved Memories -> returns structured action JSON.
- Implement `app/agent/verifier.py` to check expected vs actual outcome using screenshot/DOM/OCR state.
- Implement `app/agent/recovery.py` to analyze failure diagnostics and output corrective actions (preventing repeated identical failed actions).
- Assemble state machine graph in `app/agent/graph.py`:
  `Observe -> Perceive -> Retrieve Memory -> Plan -> Act -> Verify -> (Success -> Plan / Finish | Failure -> Reflect & Recover -> Observe)`

### Phase 5: Self-Improvement & Multi-Level Memory System
- Build SQLite database models in `app/memory/storage.py` for storing Trajectories, Actions, Errors, and Lessons.
- Build ChromaDB vector indexing in `app/memory/vector_store.py` for embedding tasks and UI states.
- Build `app/memory/episodic.py`, `semantic.py`, and `retrieval.py` for retrieving past relevant trajectories and lessons during planning.

### Phase 6: Skill Extraction & Adaptive Strategy Learning
- Implement `app/learning/reflection.py` and `lesson_extractor.py` to auto-generate reusable rules ("If [Condition] then [Prefer/Avoid Action]") after each run.
- Build `app/learning/strategy.py` to dynamically adjust perception/action strategy priorities based on application context (e.g. Chrome -> DOM preference, Native App -> UIA preference).

### Phase 7: Real-Time Web Dashboard UI
- Build FastAPI WebSocket and HTTP endpoints in `app/dashboard/` and `app/main.py`.
- Implement React + Vite dashboard in `frontend/` featuring:
  - **Live Screen Feed**: Real-time screenshot display with overlaid bounding boxes.
  - **Action & Verification Stream**: Step-by-step tree visualization of planning, execution, and verification states.
  - **Self-Improvement Inspector**: Trajectory history viewer, extracted lesson database, and skill library browser.

### Phase 8: Evaluation Benchmark Suite
- Create `evaluation/benchmark_tasks.json` with 50+ benchmark scenarios across web, file management, and desktop apps.
- Implement `evaluation/runner.py` and `metrics.py` to measure Task Success Rate, Recovery Success Rate, Action Efficiency, and Learning Improvement across repeated benchmark passes.

---

## Verification Plan

### Automated Testing
- **Unit Tests**: Test model factory, schema validation, perception merger, action safety guards, memory storage & vector retrieval using `pytest tests/`.
- **Mock State Graph Verification**: Test LangGraph state transitions with synthetic UI states and mocked model responses.

### Manual & Integrated Testing
- **Action Primitive Verification**: Test desktop app launch, semantic element clicking, key typing, and window switching.
- **Closed-Loop Task Run**: Execute end-to-end tasks (e.g. "Open Chrome and search for NVIDIA NIM"), confirming that verification and failure recovery operate correctly.
- **Self-Improvement & Learning Validation**: Verify that after a failure and recovery, a lesson is stored in ChromaDB/SQLite, and subsequent executions of the task successfully retrieve and apply the lesson.
- **Dashboard UI Testing**: Verify real-time WebSocket screenshot streaming, bounding box overlays, and control buttons (Start, Pause, Emergency Stop) in the browser dashboard.
