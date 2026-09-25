# Agentic ADC – Two-Stage Multi-Agent PCB Inspection System

An end-to-end agentic visual inspection and explainability pipeline for printed circuit board (PCB) assembly manufacturing.

The system decouples **Fast-Path Automated Defect Classification (Agent 1 Orchestrator)** from deep **Explainability & Root-Cause Auditing (Agent 2)** via a persistent REST and Vector Database layer (Qdrant).

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        AGENT 1: ORCHESTRATOR                           │
│  • Dataset Preparation & Verification                                  │
│  • Two-Stage Inference (Feature Classifier → Defect Classifier)        │
│  • Deterministic Policy Engine & LLM Planner                           │
│  • Interactive GUI (Tkinter) or Headless CLI (main.py)                 │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Saves Run State (JSON)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   SHARED PERSISTENCE & DATA LAYER                      │
│  • REST Data API (:8000) (FastAPI + Uvicorn)                           │
│  • Qdrant Vector Database (:6333) (Historical Precedents & Telemetry)  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Automatic Review Trigger (POST)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 AGENT 2: EXPLAINABILITY REVIEW AGENT                   │
│  • Service Endpoint (:8001)                                            │
│  • 4 Model Context Protocol (MCP) Tools:                               │
│     1. case_context_retrieval_tool  → Precedent Qdrant Vector Search   │
│     2. visual_evidence_tool         → Local LLaVA ROI Inspection       │
│     3. measurement_evidence_tool    → ICT / Laser Height Telemetry     │
│     4. grounding_and_self_check_tool→ OpenAI GPT-4o Strict Verification│
└────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```text
pcb_agentic_inspector/
├── .env                              # API keys and environment configuration
├── compose.qdrant.yaml               # Docker Compose file for Qdrant vector database
├── requirements.txt                  # Core project dependencies
├── requirements-rest.txt             # REST API and data layer dependencies
├── auto_review.py                    # Standalone batch review runner
├── adc_rest.py                       # CLI utility for database queries & runs
├── main.py                           # Headless batch inspection pipeline
├── adc_shared/                       # Shared microservices layer
│   ├── client.py                     # Shared DataClient interface
│   ├── data_api.py                   # FastAPI service for run & review state (Port 8000)
│   ├── repository.py                 # Qdrant persistence repository
│   └── agent2_api.py                 # Agent 2 REST service wrapper (Port 8001)
├── src/
│   ├── agent1_orchestrator/          # Agent 1 (Orchestrator & GUI)
│   │   ├── ui.py                     # Main Tkinter graphical user interface
│   │   ├── agents/                   # Orchestrator agent logic & planner
│   │   └── services/                 # Dataset preparation & verification
│   └── agent2_explainability/        # Agent 2 (Explainability & MCP tools)
│       ├── mcp/
│       │   └── agent2_mcp_server.py  # FastMCP server & 4 review tools
│       └── pipeline/
│           └── review_graph.py       # LangGraph review graph executor
└── sample_data/                      # Golden & Defect image datasets
```

---

## Prerequisites

1. **Windows 10/11** with **Anaconda / Miniconda**.
2. **Docker Desktop** (required to run Qdrant).
3. **OpenAI API Key** (for LLM planning and Agent 2 GPT-4o grounding).
4. *(Optional)* **Ollama with LLaVA** (for local vision analysis; fallback enabled if offline):
   ```cmd
   ollama run llava
   ```

---

## Environment Setup

Open the **Anaconda Prompt** and navigate to your project root:

```cmd
cd /d "C:\Users\<YourUsername>\Desktop\Semicon Agents\pcb_agentic_inspector"
```

### 1. Create and Activate Conda Environment
```cmd
conda create -n pcb_inspector python=3.11 -y
conda activate pcb_inspector
```

### 2. Install Dependencies
```cmd
pip install -r requirements.txt -r requirements-rest.txt
```

### 3. Configure `.env`
Create or update the `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
QDRANT_URL=http://127.0.0.1:6333
ADC_DATA_URL=http://127.0.0.1:8000
ADC_AGENT2_URL=http://127.0.0.1:8001
ADC_ENABLE_AGENT2=1
```

---

## Running the System (3-Terminal Workflow)

### Terminal 1: Start Qdrant and the Shared Data API
1. Open **Docker Desktop** and wait until it is running.
2. In your terminal, start the Qdrant container:
   ```cmd
   conda activate pcb_inspector
   docker compose -f compose.qdrant.yaml up -d
   ```
3. Start the Shared Data API on port **8000**:
   ```cmd
   set QDRANT_URL=http://127.0.0.1:6333
   python -m uvicorn adc_shared.data_api:app --host 127.0.0.1 --port 8000 --workers 1
   ```
   *Verify in browser: `http://127.0.0.1:8000/health`*

---

### Terminal 2: Start Agent 2 Explainability API
Open a second Anaconda Prompt:
```cmd
cd /d "C:\Users\<YourUsername>\Desktop\Semicon Agents\pcb_agentic_inspector"
conda activate pcb_inspector
set ADC_ENABLE_AGENT2=1
set ADC_DATA_URL=http://127.0.0.1:8000
python -m uvicorn adc_shared.agent2_api:app --host 127.0.0.1 --port 8001 --workers 1
```
*Verify in browser: `http://127.0.0.1:8001/health` (should return `{"status": "ok", "execution_enabled": true}`)*

---

### Terminal 3: Launch Agent 1 Orchestrator GUI
Open a third Anaconda Prompt:
```cmd
cd /d "C:\Users\<YourUsername>\Desktop\Semicon Agents\pcb_agentic_inspector"
conda activate pcb_inspector
set ADC_DATA_URL=http://127.0.0.1:8000
set ADC_AGENT2_URL=http://127.0.0.1:8001
python src/agent1_orchestrator/ui.py
```

---

## Inspecting and Operating the UI

1. **Select Inputs**:
   - **Dataset CSV**: `sample_data/dataset.csv`
   - **Inspection XML**: `sample_data/inspection.xml`
   - **Image Folder**: `sample_data`
   - **Output JSON**: `outputs/result.json`
2. Click **1. Prepare** → Validates and matches images with inspection records.
3. Click **2. Prepare + Verify** → Checks golden pairing and defect ROI crops.
4. Click **3. Run Agentic Workflow**:
   - Agent 1 executes feature and defect classification.
   - Saves results and indexes to Qdrant (Port 8000).
   - If any sample is uncertain or marked `REVIEW_REQUIRED`, it is **automatically escalated to Agent 2 (Port 8001)**.
   - The review diagnosis, evidence verification, and final verdict stream directly into the UI log.

---

## Headless CLI Execution

### Running Batch CLI Inspections
To run the inspection pipeline without the GUI:
```cmd
python main.py --dataset sample_data/dataset.csv --xml sample_data/inspection.xml --image-root sample_data --agent2-url http://127.0.0.1:8001
```

### Inspecting Runs with `adc_rest.py`
Query stored runs and review cases from the database:
```cmd
# List all review cases for a specific run ID
python adc_rest.py reviews <RUN_ID>

# Import an existing result JSON into the database
python adc_rest.py import example_result.json --run-id run-test-001
```

### Triggering Reviews via `auto_review.py`
If running outside the UI, trigger all pending reviews from the latest run in `outputs/`:
```cmd
python auto_review.py
```

---

## Agent 2 MCP Tools Specification

Agent 2 (`src/agent2_explainability/mcp/agent2_mcp_server.py`) defines 4 tools:

| Tool Name | Technology | Description |
|---|---|---|
| `case_context_retrieval_tool` | Qdrant Vector Store | Retrieves historical precedents for the component reference and cross-references IPC-A-610 Class 3 standards. |
| `visual_evidence_tool` | Local LLaVA / Ollama | Analyzes the ROI crop using a multimodal vision model across the 7 IPC defect classes. |
| `measurement_evidence_tool` | Telemetry Engine | Fetches ICT electrical telemetry (resistance, capacitance, laser profile height). |
| `grounding_and_self_check_tool` | OpenAI GPT-4o | Synthesizes visual and physical evidence to ensure strict JSON output and avoid hallucinations. |

---

## Troubleshooting

### 1. `[WinError 10061] No connection could be made because target machine actively refused it`
* **Cause**: Qdrant is not running.
* **Fix**: Ensure Docker Desktop is active, then run `docker compose -f compose.qdrant.yaml up -d`. Check `http://127.0.0.1:6333/dashboard`.

### 2. `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'`
* **Cause**: Running from Anaconda's global `(base)` environment instead of the project's environment.
* **Fix**: Run `conda activate pcb_inspector`.

### 3. `HTTP 422: No defect classification available`
* **Cause**: Agent 1 flagged `FEATURE_CLASSIFICATION_UNCERTAIN` and skipped defect classification.
* **Fix**: Ensure `adc_shared/agent2_api.py` includes the fallback to `sample.get('machine_defect')` so Agent 2 has a defect category to audit.
