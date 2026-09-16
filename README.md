# 🔬 Unified Multi-Agent PCB Defect Inspection & Explainability System

An industrial-grade, multi-agent inspection architecture for Printed Circuit Boards (PCBs). This system couples **Edge Baseline Classification** (multi-stage ONNX inference) with **Multimodal Physical Explainability** (Local LLaVA VLM, 3D Laser height profiles, In-Circuit Testing electrical measurements, and IPC-A-610 Class 2/3 standards) governed by the **Agent2Agent (A2A)** and **Model Context (MCP)** protocols.

---

## 🏛️ System Architecture

```text
               AOI Inspection XML + CSV Metadata + Image Pairs
                                      │
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      AGENT 1: CORE ADC & ORCHESTRATOR                       │
 │                                                                             │
 │  1. Dataset Preparation: Extracts failed feature & inspection criteria      │
 │  2. Verification: Schema validation, geometric alignment, image readability │
 │  3. Stage 1: Feature Classifier (224x224 ONNX) ──► Body, Lead, or Text      │
 │  4. Dynamic Routing: Dispatches to Body (640x640), Lead (640x640), or Text   │
 │  5. Deterministic Policy Gate: Evaluates confidence & metadata consistency  │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        │                                                               │
   [Confidence >= Threshold]                                     [REVIEW_REQUIRED]
        │                                                               │
        ▼                                                               │ A2A Protocol (HTTP POST :8001)
   AUTO-ACCEPTED                                                        │ Task: pcb.explainability.audit
   (Status: COMPLETED)                                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                 AGENT 2: MULTIMODAL EXPLAINABILITY & REVIEW                 │
 │                           (A2A Server on Port 8001)                         │
 │                                                                             │
 │   Receives A2A Task ──► Executes LangGraph Pipeline with MCP Tools:         │
 │                                                                             │
 │   ┌───────────────────────┐   ┌───────────────────────┐   ┌─────────────┐   │
 │   │  1. Vector DB RAG     │   │  2. Local LLaVA VLM   │   │ 3. Telemetry│   │
 │   │  Qdrant IPC-A-610-C2  │   │  Ollama Visual Inspect│   │ 3D Laser/ICT│   │
 │   └───────────┬───────────┘   └───────────┬───────────┘   └──────┬──────┘   │
 │               └─────────────────────┬─────┴──────────────────────┘          │
 │                                     ▼                                       │
 │                       ┌───────────────────────────┐                         │
 │                       │ 4. Grounding Self-Check   │                         │
 │                       │ GPT-4o Physics Arbitrator │                         │
 │                       └─────────────┬─────────────┘                         │
 └─────────────────────────────────────┼───────────────────────────────────────┘
                                       │
                                       ▼
                              Unified Audit JSON
       (Resolved Verdict, IPC Citations, Contradiction Flags, Diagnosis)
```

---

## 🏷️ Normalized IPC-A-610 Defect Taxonomy

All predictions from both agents are normalized into 7 standard industrial categories:

| Defect Class | Visual Criteria | Physical Telemetry Ground Truth |
|---|---|---|
| **`missing part`** | Empty land pattern / bare solder pads | ICT Open Circuit ($R > 10\text{ M}\Omega$); Laser Height $\approx 0\,\mu\text{m}$ |
| **`shifted`** | Component rotated or displaced | Side overhang $> 50\%$ (IPC Class 2 violation) |
| **`foreign material`**| Extraneous solder balls or debris | Surface coplanarity disruption; non-standard conductive path |
| **`tombstone`** | Part detached, standing vertically | Open circuit with elevated laser profile height ($>100\,\mu\text{m}$) |
| **`solder insufficient`**| Poor wetting fillet; incomplete pad coverage | Laser fillet thickness below minimum IPC Class 2 threshold |
| **`wrong part`** | Mismatched component package or markings | Measured capacitance / resistance out of tolerance band |
| **`no defect`** | Optimal solder fillet and alignment | All electrical and dimensional tolerances satisfied |

---

## 📁 Project Structure

```text
pcb_agentic_inspection_system/
├── config/
│   ├── models.yaml                      # Fast ONNX model specs (224x224, 640x640, 480x480)
│   ├── policy.yaml                      # Confidence thresholds & loop safety limits
│   └── agent2_config.yaml               # Qdrant, Ollama, & IPC tolerance thresholds
├── data/
│   ├── inputs/                          # Hierarchical PCB defect/golden image folders
│   └── sample_data/                     # dataset.csv and inspection.xml
├── models/                              # Baseline Single-Image RGB ONNX Classifiers
│   ├── feature/feature_classifier.onnx  (224x224)
│   ├── body/body_classifier.onnx        (640x640)
│   ├── lead/lead_classifier.onnx        (640x640)
│   └── text/text_classifier.onnx        (480x480)
├── outputs/
│   ├── telemetry_by_image.json          # Fast O(1) physical telemetry lookup index
│   └── result.json                      # Unified final diagnostic audit output
├── qdrant_db/                           # Persistent local vector store for IPC standards
├── src/
│   ├── agent1_orchestrator/             # AGENT 1: BASELINE & CONTROL
│   │   ├── services/
│   │   │   ├── dataset_preparation.py   # XML/CSV measurement extraction
│   │   │   ├── dataset_verification.py  # Image quality & alignment checker
│   │   │   └── model_lifecycle.py       # ONNX Runtime session cache
│   │   ├── policy/policy_engine.py      # Deterministic gating engine
│   │   └── ui.py                        # Tkinter Operator Desktop UI
│   └── agent2_explainability/           # AGENT 2: MULTIMODAL REVIEW & EXPLAINABILITY
│       ├── a2a/
│       │   ├── protocol.py              # A2A Task & AgentCard Pydantic schemas
│       │   └── agent2_a2a_server.py     # FastAPI A2A Server (Port 8001)
│       ├── pipeline/
│       │   └── review_graph.py          # LangGraph Multimodal State Machine
│       └── generate_telemetry.py        # 3D AOI & ICT profile builder
├── main.py                              # Unified CLI runner (Batch & Slicing enabled)
├── requirements.txt
└── .env                                 # Local API keys (OPENAI_API_KEY)
```

---

## ⚡ Setup & Installation

### 1. Environment Setup
Python 3.10+ is required:

```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Pull the Local Vision Model (Ollama)
Ensure [Ollama](https://ollama.com/) is installed and running locally:
```bash
ollama pull llava
```

### 3. Configure `.env`
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-proj-yourActualKeyHere
OPENAI_MODEL=gpt-4o
```

---

## 🚀 Two-Terminal Execution Workflow

Agent 1 and Agent 2 run as decoupled services communicating over HTTP via the A2A protocol.

### Terminal 1: Start Agent 2 (The Explainability Server)
```bash
python -m src.agent2_explainability.a2a.agent2_a2a_server
```
Wait until you see:
```text
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

---

### Terminal 2: Run Agent 1 (The Orchestrator)

#### Option A: Headless Command-Line Runner
```bash
python main.py \
  --dataset sample_data/dataset.csv \
  --xml sample_data/inspection.xml \
  --image-root data/inputs \
  --confidence-threshold 0.85
```

#### Option B: Operator Desktop UI (Tkinter)
```bash
python src/agent1_orchestrator/ui.py
# On Windows, you can also double-click: run_ui.bat
```

---

## 📦 Batch Execution & Sample Size Control

`main.py` provides flags to control processing volume, memory usage, and execution chunking:

| Flag | Default | Description |
|---|---|---|
| `--limit N` | `None` (All) | Restricts execution to the first $N$ samples (ideal for smoke testing). |
| `--batch-size B` | `10` | Splits samples into mini-batches of size $B$ to prevent timeouts. |
| `--confidence-threshold` | `0.85` | Baseline confidence gate threshold; below this triggers Agent 2 review. |
| `--image-root` | `data/inputs` | Base folder used for recursive image discovery and path remapping. |
| `--agent2-url` | `http://127.0.0.1:8001` | A2A server URL for explainability delegation. |
| `--output` | `outputs/result.json` | Destination path for the unified diagnostic JSON report. |

### Batch Usage Examples

```bash
# 1. Quick test run with only the first 5 samples
python main.py --limit 5

# 2. Process all samples in mini-batches of 20
python main.py --batch-size 20

# 3. Process 100 samples in batches of 25 with a strict 0.90 confidence gate
python main.py --limit 100 --batch-size 25 --confidence-threshold 0.90
```

---

## 🔍 Robust 3-Stage Image Path Resolution

The runner automatically recovers image paths when files are moved or when `dataset.csv` contains legacy absolute paths from another machine (e.g., `C:\Users\...\Desktop\usi\...`):

1. **Direct Match:** Checks if the path exists directly on the local filesystem.
2. **Subpath Preservation:** Preserves directory hierarchies below the `usi` folder or Board ID (e.g., `data/inputs/35-900032-AAA-RV1/Text/Golden/image.jpg`).
3. **Recursive Basename Match:** Scans recursively across `--image-root`, `inputs/`, `data/inputs/`, and `sample_data/` by filename.
4. **Canonical Absolute Path Transmission:** All relative paths are resolved to absolute paths before being dispatched over A2A HTTP, preventing missing-image errors between independent terminal sessions.

---

## 📊 Sample Unified Output (`outputs/result.json`)

```json
{
  "summary": {
    "total_samples": 3,
    "agent1_auto_accepted": 1,
    "escalated_to_agent2": 2,
    "agent2_resolved": 2,
    "human_review_required": 0
  },
  "results": [
    {
      "sample_id": "Sample_C636_Missing",
      "component_id": "C636",
      "resolved_image_path": "D:/Project/data/inputs/Board1_C636_Body_06-200036-02_MissingPart_3.jpg",
      "baseline_inference": {
        "feature_class": "Body",
        "defect_class": "MissingPart",
        "confidence": 0.62
      },
      "gate_decision": "REVIEW_REQUIRED",
      "agent2_review": {
        "predicted_defect": "missing part",
        "confidence": 0.98,
        "self_check_passed": true,
        "contradiction_detected": false,
        "diagnosis": "Physical open circuit (ICT FAIL) and laser profile height (0.80 µm) confirm missing part.",
        "ipc_citations": [
          "IPC-A-610 Class 2 Section 8.3"
        ],
        "visual_evidence": "Solder land pads are bare silver finish. Ceramic body is absent."
      },
      "final_verdict": "missing part",
      "workflow_status": "COMPLETED"
    },
    {
      "sample_id": "Sample_R102_Shift",
      "component_id": "R102",
      "resolved_image_path": "D:/Project/data/inputs/Board1_R102_Body_06-200036-02_Shift_1.jpg",
      "baseline_inference": {
        "feature_class": "Body",
        "defect_class": "Shift",
        "confidence": 0.71
      },
      "gate_decision": "REVIEW_REQUIRED",
      "agent2_review": {
        "predicted_defect": "shifted",
        "confidence": 0.95,
        "self_check_passed": true,
        "contradiction_detected": false,
        "diagnosis": "Side overhang (62.0%) exceeds IPC-A-610 Class 2 maximum 50% threshold.",
        "ipc_citations": [
          "IPC-A-610 Class 2 Section 8.3.2"
        ],
        "visual_evidence": "Component is displaced laterally beyond pad boundaries."
      },
      "final_verdict": "shifted",
      "workflow_status": "COMPLETED"
    },
    {
      "sample_id": "Sample_C314_Normal",
      "component_id": "C314",
      "resolved_image_path": "D:/Project/data/inputs/Board1_C314_Body_06-200036-02_Normal_0.jpg",
      "baseline_inference": {
        "feature_class": "Body",
        "defect_class": "NoDefect",
        "confidence": 0.98
      },
      "gate_decision": "PASS",
      "agent2_review": null,
      "final_verdict": "no defect",
      "workflow_status": "COMPLETED"
    }
  ]
}
```

---

## 🛠️ Production Extensibility

* **Human-in-the-Loop (HITL):** If Agent 2 detects an irreconcilable visual/physical contradiction or if `self_check_passed == False`, the workflow status becomes `HUMAN_QA_REQUIRED` and routes the case to the QA review queue.
* **Continuous Learning:** Verified escalated cases can be indexed back into the local `qdrant_db` collection to expand historical retrieval accuracy for future runs.
* **Industrial SMT Line Integration:** In `generate_telemetry.py`, replace simulated/lookup files with direct SECS/GEM or OPC-UA protocols streaming real-time measurements from inline 3D AOI and ICT equipment.
