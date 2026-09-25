# Agentic ADC – Two-Stage Multi-Agent PCB Inspection System

An end-to-end multi-agent visual inspection and explainability pipeline for printed circuit board (PCB) surface mount assembly manufacturing.

The system decouples **Fast-Path Automated Defect Classification (Agent 1 Orchestrator)** from deep **Explainability & Root-Cause Auditing (Agent 2)** via a persistent REST and Vector Database layer (Qdrant), complete with an interactive **Human-in-the-Loop (HitL)** conflict resolution engine.

---

## 1. System Architecture

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                          AGENT 1: ORCHESTRATOR                            │
│  • Dataset Preparation & Verification (Relative "inputs/..." paths)       │
│  • Two-Stage Inference (Feature Classifier → Defect Classifier)           │
│  • Deterministic Policy Engine & LLM Planner                              │
│  • Interactive GUI (Tkinter) or Headless CLI (main.py)                    │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ Saves Run State (JSON)
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     SHARED PERSISTENCE & DATA LAYER                       │
│  • REST Data API (:8000) (FastAPI + Uvicorn)                              │
│  • Qdrant Vector Database (:6333) (Historical Precedents & Telemetry)     │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │ Auto-Dispatches "REVIEW_REQUIRED" Cases
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                   AGENT 2: EXPLAINABILITY REVIEW AGENT                    │
│  • Service Endpoint (:8001)                                               │
│  • 4 Model Context Protocol (MCP) Tools:                                  │
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
│  • If Agent 1 != Agent 2  ──► MODAL CONFLICT RESOLUTION WINDOW            │
│     - Inspect side-by-side evidence & full diagnosis                      │
│     - Operator selects verdict (Agent 1, Agent 2, or custom IPC class)   │
│     - Records justification notes & updates Qdrant (HUMAN_RESOLVED)       │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

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

## 3. Prerequisites

1. **Windows 10/11** or **Linux** with **Anaconda / Miniconda**.
2. **Docker Desktop** (required to run the Qdrant vector store).
3. **OpenAI API Key** (for LLM planning and Agent 2 GPT-4o grounding).
4. *(Optional)* **Ollama with LLaVA** (for local vision analysis; heuristic fallback enabled if offline):
   ```cmd
   ollama run llava
   ```

---

## 4. Installation & Environment Setup

Open your **Anaconda Prompt** and navigate to the project root:

```cmd
cd /d "C:\Users\<YourUsername>\Desktop\Semicon Agents\pcb_agentic_inspector"
```

### 1. Create and Activate the Dedicated Conda Environment
```cmd
conda create -n pcb_inspector python=3.11 -y
conda activate pcb_inspector
```

### 2. Install Dependencies
```cmd
pip install -r requirements.txt -r requirements-rest.txt
```

### 3. Configure `.env`
Ensure a `.env` file exists in the root folder with the following variables:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
QDRANT_URL=http://127.0.0.1:6333
ADC_DATA_URL=http://127.0.0.1:8000
ADC_AGENT2_URL=http://127.0.0.1:8001
ADC_ENABLE_AGENT2=1
```

---

## 5. Running the System (3-Terminal Workflow)

### Terminal 1: Start Qdrant & Shared Data API
1. Open **Docker Desktop** and wait until the engine is active.
2. In your terminal, start the Qdrant container:
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
Open a second Anaconda Prompt:
```cmd
cd /d "C:\Users\<YourUsername>\Desktop\Semicon Agents\pcb_agentic_inspector"
conda activate pcb_inspector
set ADC_ENABLE_AGENT2=1
set ADC_DATA_URL=http://127.0.0.1:8000
python -m uvicorn adc_shared.agent2_api:app --host 127.0.0.1 --port 8001 --workers 1
```
*Verify in browser:* `http://127.0.0.1:8001/health` (returns `{"status":"ok","execution_enabled":true}`).

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

## 6. Using the UI & Human-in-the-Loop Conflict Resolution

1. **Select Inputs in UI**:
   - **Dataset CSV**: `sample_data/dataset.csv` (using relative `inputs/...` paths)
   - **Inspection XML**: `sample_data/inspection.xml`
   - **Image Folder**: `.` or leave blank (if CSV contains `inputs/...`)
   - **Output JSON**: `outputs/result.json`
2. Click **1. Prepare** → Validates and matches images with inspection records.
3. Click **2. Prepare + Verify** → Validates Golden-Defect ROI pairs.
4. Click **3. Run Agentic Workflow**:
   - Agent 1 executes feature and defect classification.
   - Run results are indexed into Qdrant (`:8000`).
   - Any sample marked `REVIEW_REQUIRED` is **automatically sent to Agent 2 (`:8001`)** for multimodal explainability review.
   
### Conflict Resolution (HitL)
* **Consensus**: If Agent 1 and Agent 2 agree, the verdict is auto-approved and logged:
  ```text
  ✅ [CONSENSUS] Both agents agree on 'WrongPart'. Auto-approved.
  ```
* **Discrepancy**: If Agent 1 and Agent 2 disagree (e.g. `WrongPart` vs `MissingPart`), execution pauses and a **Conflict Resolution Dialog** appears:
  - Displays side-by-side model outputs.
  - Displays Agent 2's detailed diagnosis explanation (LLaVA + ICT measurements + GPT-4o reasoning).
  - Allows the operator to:
    - `[•] Accept Agent 2`
    - `[ ] Accept Agent 1`
    - `[ ] Manual Override (IPC Class: Shifted, Tombstone, Solder Insufficient, etc.)`
    - Input optional operator notes.
  - Submitting resolution updates the UI log and writes `HUMAN_RESOLVED` directly back to the database (`PUT /runs/{run_id}/reviews/{sample_id}`).

---

## 7. Headless & Cloud Execution (CLI / Google Colab)

> **Note on Google Colab:** Tkinter requires an active X11 desktop display and will throw `_tkinter.TclError: no display name` in Colab. For cloud/headless execution, use the CLI runner.

### Running Batch Inspections via CLI
```bash
python main.py \
  --dataset sample_data/dataset.csv \
  --xml sample_data/inspection.xml \
  --image-root inputs \
  --agent2-url http://127.0.0.1:8001 \
  --output outputs/result.json
```

### Querying Database Runs via `adc_rest.py`
```bash
# View review cases for a specific run ID
python adc_rest.py reviews <RUN_ID>

# Import pre-existing result JSON into Qdrant
python adc_rest.py import example_result.json --run-id imported-001
```

### Standalone Batch Review Trigger (`auto_review.py`)
To trigger Agent 2 reviews for the most recent run generated in `outputs/`:
```bash
python auto_review.py
```

---

## 8. Agent 2 MCP Tools Suite

Agent 2 (`src/agent2_explainability/mcp/agent2_mcp_server.py`) defines 4 Model Context Protocol tools:

| Tool Name | Engine / Backend | Description |
|---|---|---|
| `case_context_retrieval_tool` | Qdrant Vector Store | Retrieves historical precedents for the component reference and matches IPC-A-610 Class 3 specifications. |
| `visual_evidence_tool` | Local LLaVA / Ollama | Analyzes the ROI crop using a multimodal vision model across 7 IPC defect classes. |
| `measurement_evidence_tool` | Telemetry Engine | Fetches In-Circuit Test (ICT) resistance, capacitance, and laser profile height data. |
| `grounding_and_self_check_tool` | OpenAI GPT-4o | Synthesizes visual and physical evidence to ensure strict JSON output and avoid hallucinations. |

---

## 9. Troubleshooting

### 1. `[WinError 10061] No connection could be made because target machine actively refused it`
* **Cause**: Docker or Qdrant container is not running on port 6333.
* **Fix**: Ensure Docker Desktop is active and run `docker compose -f compose.qdrant.yaml up -d`. Verify at `http://127.0.0.1:6333/dashboard`.

### 2. `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'`
* **Cause**: Running from Anaconda's global `(base)` environment with mismatched FastAPI/Starlette packages.
* **Fix**: Run `conda activate pcb_inspector`.

### 3. `HTTP 422: No defect classification available`
* **Cause**: Agent 1 flagged `FEATURE_CLASSIFICATION_UNCERTAIN` and skipped defect classification.
* **Fix**: Ensure `adc_shared/agent2_api.py` includes the fallback to `sample.get('machine_defect')` so Agent 2 has a defect candidate to evaluate.
