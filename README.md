# Agentic ADC – Two-Stage Multi-Agent PCB Inspection System

An end-to-end multi-agent visual inspection and explainability pipeline for printed circuit board (PCB) surface mount assembly manufacturing.

The system decouples **Fast-Path Automated Defect Classification (Agent 1 Orchestrator)** from deep **Explainability & Root-Cause Auditing (Agent 2)** via a persistent REST and Vector Database layer (Qdrant), complete with an interactive **Human-in-the-Loop (HitL)** conflict resolution engine.

---

## 1. Directory Structure

```text
pcb_agentic_inspector/
├── .env                              # API keys and environment variables
├── compose.qdrant.yaml               # Docker Compose file for Qdrant vector database
├── requirements.txt                  # Core dependencies (Tkinter, PyTorch, etc.)
├── requirements-rest.txt             # REST API and data layer dependencies
├── auto_review.py                    # Standalone batch review script
├── adc_rest.py                       # CLI utility for database queries & runs
├── main.py                           # Headless batch inspection pipeline
├── inputs/                           # PCB image directory (Golden & Defect crops)
├── outputs/                          # Generated inspection results & run ID records
├── adc_shared/                       # Shared microservices layer
│   ├── client.py                     # DataClient interface for API communication
│   ├── data_api.py                   # FastAPI service for run & review state (Port 8000)
│   ├── repository.py                 # Qdrant persistence repository
│   └── agent2_api.py                 # Agent 2 REST service wrapper (Port 8001)
├── src/
│   ├── agent1_orchestrator/          # Agent 1 (Orchestrator & GUI)
│   │   ├── ui.py                     # Tkinter UI with HitL conflict resolution
│   │   ├── agents/                   # Orchestrator agent logic & planner
│   │   └── services/                 # Dataset preparation & verification
│   └── agent2_explainability/        # Agent 2 (Explainability & MCP tools)
│       ├── mcp/
│       │   └── agent2_mcp_server.py  # FastMCP server & 4 review tools
│       └── pipeline/
│           └── review_graph.py       # LangGraph review graph executor
└── sample_data/                      # Example CSV and AOI XML datasets
```

---

## 2. System Architecture & Workflow

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                          AGENT 1: ORCHESTRATOR                            │
│  • src/agent1_orchestrator/ui.py (Interactive GUI) or main.py (CLI)       │
│  • Dataset Preparation & Verification (Relative 'inputs/...' paths)       │
│  • Two-Stage Inference (Feature Classifier → Defect Classifier)           │
│  • Policy Engine & LLM Planner (OpenAI / Deterministic)                   │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ Saves Run State (JSON)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     SHARED PERSISTENCE & DATA LAYER                       │
│  • adc_shared/data_api.py (FastAPI on Port 8000)                          │
│  • compose.qdrant.yaml (Qdrant Vector DB on Port 6333)                    │
│  • adc_shared/repository.py & client.py (Vector storage & retrieval)      │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ Auto-Dispatches "REVIEW_REQUIRED" Cases
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                   AGENT 2: EXPLAINABILITY REVIEW AGENT                    │
│  • adc_shared/agent2_api.py (REST Service on Port 8001)                   │
│  • src/agent2_explainability/pipeline/review_graph.py (LangGraph Flow)    │
│  • src/agent2_explainability/mcp/agent2_mcp_server.py (4 MCP Tools):      │
│     1. case_context_retrieval_tool  → Precedent Qdrant Vector Search      │
│     2. visual_evidence_tool         → Local LLaVA ROI Multimodal Audit    │
│     3. measurement_evidence_tool    → ICT / Laser Profile Telemetry       │
│     4. grounding_and_self_check_tool→ OpenAI GPT-4o Strict Verification   │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ Returns Verdict + Detailed Diagnosis
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      HUMAN-IN-THE-LOOP (HitL) ENGINE                      │
│  • If Agent 1 == Agent 2  ──► CONSENSUS (Auto-Approved into Database)     │
│  • If Agent 1 != Agent 2  ──► MODAL CONFLICT RESOLUTION DIALOG            │
│     - Side-by-side evidence inspection                                    │
│     - Operator selects verdict (Agent 1, Agent 2, or custom IPC class)   │
│     - Operator notes recorded and saved to Qdrant (HUMAN_RESOLVED)        │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Prerequisites

1. **Windows 10/11** or **Linux** with **Anaconda / Miniconda**.
2. **Git**.
3. **Docker Desktop** (required to run the Qdrant vector database container).
4. **OpenAI API Key** (for LLM planning and Agent 2 GPT-4o grounding).
5. *(Optional)* **Ollama with LLaVA** (for local vision analysis; heuristic fallback enabled if offline):
   ```bash
   ollama run llava
   ```

---

## 4. Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/kjxlau/pcb_agentic_inspector.git
cd pcb_agentic_inspector
```

### 2. Create and Activate Conda Environment
```bash
conda create -n pcb_inspector python=3.11 -y
conda activate pcb_inspector
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt -r requirements-rest.txt
```

### 4. Configure `.env`
Create a `.env` file in the root of the cloned repository (or copy `.env.example` if available):
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
QDRANT_URL=http://127.0.0.1:6333
ADC_DATA_URL=http://127.0.0.1:8000
ADC_AGENT2_URL=http://127.0.0.1:8001
ADC_ENABLE_AGENT2=1
```

---

## 5. Local Execution (3-Terminal Workflow)

Open three separate terminals in the `pcb_agentic_inspector` directory:

### Terminal 1: Start Qdrant & Shared Data API
1. Ensure **Docker Desktop** is open and active.
2. In your project root:
   ```cmd
   conda activate pcb_inspector
   docker compose -f compose.qdrant.yaml up -d
   ```
3. Launch the Shared Data API on port **8000**:
   ```cmd
   set QDRANT_URL=http://127.0.0.1:6333
   python -m uvicorn adc_shared.data_api:app --host 127.0.0.1 --port 8000 --workers 1
   ```
   *Verify in browser:* `http://127.0.0.1:8000/health` (returns `{"status":"ok","database":"qdrant"}`).

---

### Terminal 2: Start Agent 2 Explainability API
In your project root:
```cmd
conda activate pcb_inspector
set ADC_ENABLE_AGENT2=1
set ADC_DATA_URL=http://127.0.0.1:8000
python -m uvicorn adc_shared.agent2_api:app --host 127.0.0.1 --port 8001 --workers 1
```
*Verify in browser:* `http://127.0.0.1:8001/health` (returns `{"status":"ok","execution_enabled":true}`).

---

### Terminal 3: Launch Agent 1 Orchestrator GUI
In your project root:
```cmd
conda activate pcb_inspector
set ADC_DATA_URL=http://127.0.0.1:8000
set ADC_AGENT2_URL=http://127.0.0.1:8001
python src/agent1_orchestrator/ui.py
```

---

## 6. Using the UI & Human-in-the-Loop Conflict Resolution

1. **Select Inputs in UI**:
   - **Dataset CSV**: `sample_data/dataset.csv` (uses relative `inputs/...` paths)
   - **Inspection XML**: `sample_data/inspection.xml`
   - **Image Folder**: `.` (or leave empty if CSV paths are relative to root)
   - **Output JSON**: `outputs/result.json`
2. Click **1. Prepare** → Matches images with inspection records.
3. Click **2. Prepare + Verify** → Validates Golden-Defect ROI pairs.
4. Click **3. Run Agentic Workflow**:
   - Agent 1 executes feature and defect classification.
   - Saves run state to Qdrant via `adc_shared/data_api.py`.
   - Any sample marked `REVIEW_REQUIRED` is **automatically dispatched to Agent 2 (`:8001`)**.

### Conflict Resolution Flow
* **Consensus**: When Agent 1 and Agent 2 agree on the classification, the result is auto-approved and saved to Qdrant.
* **Conflict**: If Agent 1 and Agent 2 disagree, a **Human Review Dialog** modal opens:
  - Displays side-by-side model outputs (`Agent 1` vs `Agent 2`).
  - Displays the full multimodal diagnosis from LLaVA and GPT-4o.
  - Allows the operator to:
    - Accept Agent 2
    - Accept Agent 1
    - Override with an IPC-A-610 class (`Missing Part`, `Wrong Part`, `Shifted`, `Tombstone`, `Solder Insufficient`, `Foreign Material`, `No Defect / Pass`)
    - Enter operator notes.
  - Submitting writes `HUMAN_RESOLVED` directly back to the database (`PUT /runs/{run_id}/reviews/{sample_id}`) and updates the UI log.

---

## 7. Headless & Cloud Execution (CLI / Google Colab)

> **Note:** Tkinter requires an active X11 display. For headless environments or Google Colab, use the CLI scripts.

### Running Batch Inspections via `main.py`
```bash
python main.py \
  --dataset sample_data/dataset.csv \
  --xml sample_data/inspection.xml \
  --image-root inputs \
  --agent2-url http://127.0.0.1:8001 \
  --output outputs/result.json
```

### Inspecting Database Runs via `adc_rest.py`
```bash
# View review cases for a specific run ID
python adc_rest.py reviews <RUN_ID>

# Import an existing result JSON into Qdrant
python adc_rest.py import example_result.json --run-id imported-001
```

### Standalone Batch Review Trigger via `auto_review.py`
Trigger Agent 2 reviews for the latest run generated in `outputs/`:
```bash
python auto_review.py
```

### Running in Google Colab (All 3 Services in Background)
```python
import subprocess, time, httpx, os
from pathlib import Path

os.environ["PYTHONPATH"] = "."
os.environ["QDRANT_URL"] = "http://127.0.0.1:6333"
os.environ["ADC_DATA_URL"] = "http://127.0.0.1:8000"
os.environ["ADC_AGENT2_URL"] = "http://127.0.0.1:8001"
os.environ["ADC_ENABLE_AGENT2"] = "1"

# Kill lingering processes on ports
!fuser -k 6333/tcp 8000/tcp 8001/tcp 2>/dev/null || true

# 1. Start Qdrant Standalone Binary
if not Path("qdrant").is_file():
    !curl -s -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar -xz
    !chmod +x qdrant
subprocess.Popen(["./qdrant"], stdout=open("qdrant.log", "w"), stderr=subprocess.STDOUT)

# 2. Start Data API
subprocess.Popen(["python", "-m", "uvicorn", "adc_shared.data_api:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"], stdout=open("data_api.log", "w"), stderr=subprocess.STDOUT)

# 3. Start Agent 2 API
subprocess.Popen(["python", "-m", "uvicorn", "adc_shared.agent2_api:app", "--host", "127.0.0.1", "--port", "8001", "--workers", "1"], stdout=open("agent2.log", "w"), stderr=subprocess.STDOUT)

time.sleep(5)
print("Background services online! Run 'python main.py' to process inspections.")
```

---

## 8. Agent 2 MCP Tools Specification

Agent 2 (`src/agent2_explainability/mcp/agent2_mcp_server.py`) exposes 4 Model Context Protocol tools:

| Tool Name | Engine / Backend | Purpose |
|---|---|---|
| `case_context_retrieval_tool` | Qdrant Vector Store | Searches historical precedent embeddings and cross-references IPC-A-610 Class 3 specifications. |
| `visual_evidence_tool` | Local LLaVA / Ollama | Analyzes the ROI crop using a multimodal vision model across 7 IPC defect classes. |
| `measurement_evidence_tool` | Telemetry Engine | Fetches ICT electrical telemetry (resistance, capacitance, laser profile height). |
| `grounding_and_self_check_tool` | OpenAI GPT-4o | Synthesizes visual and physical evidence to ensure strict JSON output and prevent hallucination. |

---

## 9. Troubleshooting

### 1. `[WinError 10061] No connection could be made because target machine actively refused it`
* **Cause**: Docker or Qdrant container is not running on port 6333.
* **Fix**: Ensure Docker Desktop is active and run `docker compose -f compose.qdrant.yaml up -d`. Verify at `http://127.0.0.1:6333/dashboard`.

### 2. `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'`
* **Cause**: Running from Anaconda's global `(base)` environment with mismatched FastAPI/Starlette packages.
* **Fix**: Activate the dedicated environment: `conda activate pcb_inspector`.

### 3. `HTTP 422: No defect classification available`
* **Cause**: Agent 1 encountered feature uncertainty (`FEATURE_CLASSIFICATION_UNCERTAIN`) and skipped defect classification.
* **Fix**: Ensure `adc_shared/agent2_api.py` includes the fallback to `sample.get('machine_defect')` so Agent 2 has a defect candidate to evaluate.
