import json
import os
import threading
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Make direct UI launch resolve both project and Agent 1 imports.
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (_PROJECT_ROOT, _PROJECT_ROOT / "src" / "agent1_orchestrator"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from adc_shared.client import DataClient
from agents.orchestrator import OrchestratorAgent
from services.dataset_preparation import DatasetPreparationService
from services.dataset_verification import DatasetVerificationService


def normalize_defect(label: str) -> str:
    """Normalize defect labels for accurate comparison (e.g. 'WrongPart_13' -> 'wrong part')."""
    if not label:
        return "no defect"
    import re
    s = label.strip()
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', s).lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    for canonical in ("missing part", "shifted", "foreign material", "tombstone", "solder insufficient", "wrong part", "no defect"):
        if canonical.replace(" ", "") in s.replace(" ", "") or canonical in s:
            return canonical
    return s


class HumanReviewDialog(tk.Toplevel):
    """Modal dialog prompting the operator when Agent 1 and Agent 2 disagree."""
    IPC_CLASSES = [
        "Missing Part",
        "Wrong Part",
        "Shifted",
        "Tombstone",
        "Solder Insufficient",
        "Foreign Material",
        "No Defect / Pass",
    ]

    def __init__(self, parent, sample_id, a1_verdict, a2_verdict, diagnosis, event, result_holder):
        super().__init__(parent)
        self.title(f"⚠️ Conflict Resolution – Sample {sample_id}")
        self.geometry("680x580")
        self.minsize(600, 500)
        self.grab_set()  # Make window modal

        self.event = event
        self.result_holder = result_holder
        self.a1_verdict = a1_verdict
        self.a2_verdict = a2_verdict

        # Layout
        frame = ttk.Frame(self, padding=16)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text=f"Defect Discrepancy for Sample: {sample_id}", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Agent 1 and Agent 2 arrived at conflicting classifications. Please review the evidence and select the correct verdict.",
            wraplength=640,
            foreground="#444444"
        ).pack(anchor="w", pady=(2, 10))

        # Comparison Grid
        comp_box = ttk.LabelFrame(frame, text="Model Comparison", padding=10)
        comp_box.pack(fill="x", pady=6)
        ttk.Label(comp_box, text="Agent 1 (Fast-Path/AOI):", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(comp_box, text=f"{self.a1_verdict}", foreground="#0055aa").grid(row=0, column=1, sticky="w", padx=10)

        ttk.Label(comp_box, text="Agent 2 (Explainability Audit):", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(comp_box, text=f"{self.a2_verdict}", foreground="#aa5500").grid(row=1, column=1, sticky="w", padx=10, pady=(4, 0))

        # Agent 2 Diagnosis Text Box
        diag_box = ttk.LabelFrame(frame, text="Agent 2 Detailed Diagnosis & Evidence", padding=8)
        diag_box.pack(fill="both", expand=True, pady=6)
        txt = tk.Text(diag_box, wrap="word", height=6, font=("Segoe UI", 9))
        txt.insert("1.0", diagnosis if diagnosis else "No detailed diagnosis provided by Agent 2.")
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)

        # Human Choice Selection
        choice_box = ttk.LabelFrame(frame, text="Operator Decision", padding=10)
        choice_box.pack(fill="x", pady=6)

        self.choice_var = tk.StringVar(value="a2")
        ttk.Radiobutton(choice_box, text=f"Accept Agent 2: '{self.a2_verdict}'", variable=self.choice_var, value="a2").pack(anchor="w")
        ttk.Radiobutton(choice_box, text=f"Accept Agent 1: '{self.a1_verdict}'", variable=self.choice_var, value="a1").pack(anchor="w", pady=2)
        
        custom_frame = ttk.Frame(choice_box)
        custom_frame.pack(anchor="w", fill="x", pady=2)
        ttk.Radiobutton(custom_frame, text="Manual Override with IPC standard:", variable=self.choice_var, value="custom").pack(side="left")
        self.custom_combo = ttk.Combobox(custom_frame, values=self.IPC_CLASSES, state="readonly", width=22)
        self.custom_combo.set(self.IPC_CLASSES[0])
        self.custom_combo.pack(side="left", padx=8)

        # Operator Notes
        notes_frame = ttk.Frame(choice_box)
        notes_frame.pack(fill="x", pady=(6, 0))
        ttk.Label(notes_frame, text="Operator Notes:").pack(side="left")
        self.notes_entry = ttk.Entry(notes_frame)
        self.notes_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))

        # Action Buttons
        btn_bar = ttk.Frame(frame)
        btn_bar.pack(fill="x", pady=(10, 0))
        submit_btn = ttk.Button(btn_bar, text="Submit Resolution", command=self._submit, width=20)
        submit_btn.pack(side="right")

        # Handle window close (X button) safely so worker thread doesn't hang
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _submit(self):
        c = self.choice_var.get()
        if c == "a1":
            verdict = self.a1_verdict
        elif c == "a2":
            verdict = self.a2_verdict
        else:
            verdict = self.custom_combo.get()

        self.result_holder["verdict"] = verdict
        self.result_holder["notes"] = self.notes_entry.get().strip() or "Resolved via Human Operator Review"
        self.event.set()
        self.destroy()

    def _on_close(self):
        if not self.event.is_set():
            self.result_holder["verdict"] = self.a2_verdict
            self.result_holder["notes"] = "Window closed without change; accepted Agent 2 by default"
            self.event.set()
        self.destroy()


class ADCApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Agentic ADC – LLM Orchestrator")
        self.geometry("1120x820")
        self.minsize(1000, 720)

        self.project_root = Path(__file__).resolve().parents[2]
        self.dataset_var = tk.StringVar()
        self.xml_var = tk.StringVar()
        self.image_root_var = tk.StringVar()
        self.output_var = tk.StringVar(value=str(self.project_root / "outputs" / "result.json"))

        self.feature_threshold_var = tk.DoubleVar(value=0.70)
        self.defect_threshold_var = tk.DoubleVar(value=0.70)
        self.use_llm_var = tk.BooleanVar(value=False)
        self.llm_model_var = tk.StringVar(value=os.getenv("OPENAI_MODEL", "gpt-4o"))
        self.llm_fallback_var = tk.BooleanVar(value=True)

        self.status_var = tk.StringVar(value="Ready")
        self.input_count_var = tk.StringVar(value="0")
        self.prepared_var = tk.StringVar(value="0")
        self.verified_var = tk.StringVar(value="0")
        self.inference_var = tk.StringVar(value="0")
        self.accepted_var = tk.StringVar(value="0")
        self.review_var = tk.StringVar(value="0")

        self._build_ui()

    def _build_ui(self):
        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Agentic ADC – LLM Planning + Deterministic Policy + Two-Stage Inference",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(
            outer,
            text=(
                "ADC Inputs → Workflow State → LLM Planner → Observation / Constraint / Decision / Reason "
                "→ Policy Engine → Tool Execution → State Update → Re-plan"
            ),
        ).pack(anchor="w", pady=(0, 14))

        input_box = ttk.LabelFrame(outer, text="Input Selection", padding=12)
        input_box.pack(fill="x")
        self._file_row(input_box, 0, "Dataset CSV", self.dataset_var, self._browse_dataset)
        self._file_row(input_box, 1, "Inspection XML", self.xml_var, self._browse_xml)
        self._file_row(input_box, 2, "Image Folder (Optional)", self.image_root_var, self._browse_image_root)
        self._file_row(input_box, 3, "Output JSON", self.output_var, self._browse_output)
        ttk.Label(
            input_box,
            text=(
                "Image Folder remaps paths below the original 'usi' root. Example: "
                "...\\usi\\35-... → D:\\Image\\35-..."
            ),
            foreground="#555555",
        ).grid(row=4, column=1, sticky="w", padx=(4, 8), pady=(0, 4))

        options = ttk.LabelFrame(outer, text="Policy + LLM Planner", padding=12)
        options.pack(fill="x", pady=(12, 0))

        ttk.Label(options, text="Feature confidence threshold").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            options, from_=0.0, to=1.0, increment=0.05,
            textvariable=self.feature_threshold_var, width=9
        ).grid(row=0, column=1, sticky="w", padx=(8, 26))

        ttk.Label(options, text="Defect confidence threshold").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(
            options, from_=0.0, to=1.0, increment=0.05,
            textvariable=self.defect_threshold_var, width=9
        ).grid(row=0, column=3, sticky="w", padx=(8, 26))

        ttk.Checkbutton(
            options,
            text="Use real LLM Planner",
            variable=self.use_llm_var,
        ).grid(row=1, column=0, sticky="w", pady=(10, 0))

        ttk.Label(options, text="OpenAI model").grid(row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Entry(options, textvariable=self.llm_model_var, width=24).grid(
            row=1, column=3, sticky="w", padx=(8, 26), pady=(10, 0)
        )

        ttk.Checkbutton(
            options,
            text="Fallback to deterministic planner if LLM call fails",
            variable=self.llm_fallback_var,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        api_state = "OPENAI_API_KEY detected" if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY not detected"
        ttk.Label(options, text=api_state, foreground="#555555").grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(6, 0)
        )

        button_bar = ttk.Frame(outer)
        button_bar.pack(fill="x", pady=14)
        self.prepare_btn = ttk.Button(button_bar, text="1. Prepare", command=self._prepare_async)
        self.prepare_btn.pack(side="left")
        self.verify_btn = ttk.Button(button_bar, text="2. Prepare + Verify", command=self._verify_async)
        self.verify_btn.pack(side="left", padx=(8, 0))
        self.run_btn = ttk.Button(button_bar, text="3. Run Agentic Workflow", command=self._run_async)
        self.run_btn.pack(side="left", padx=(8, 0))
        ttk.Button(button_bar, text="Clear Log", command=self._clear_log).pack(side="right")

        summary_box = ttk.LabelFrame(outer, text="Workflow Summary", padding=10)
        summary_box.pack(fill="x")
        labels = [
            ("Status", self.status_var),
            ("Input", self.input_count_var),
            ("Prep Ready", self.prepared_var),
            ("Verified", self.verified_var),
            ("Inference", self.inference_var),
            ("Accepted", self.accepted_var),
            ("Review", self.review_var),
        ]
        for i, (name, var) in enumerate(labels):
            ttk.Label(summary_box, text=f"{name}:").grid(row=0, column=i * 2, sticky="w")
            ttk.Label(
                summary_box,
                textvariable=var,
                font=("Segoe UI", 10, "bold") if name == "Status" else None,
            ).grid(row=0, column=i * 2 + 1, sticky="w", padx=(5, 18))

        log_box = ttk.LabelFrame(outer, text="Execution Log / Planner History / Results", padding=8)
        log_box.pack(fill="both", expand=True, pady=(12, 0))
        self.log = tk.Text(log_box, wrap="word", font=("Consolas", 9))
        scroll = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self._log(
            "Agent loop:\n"
            "State → Planner → Policy → Tool → State Update → Re-plan\n"
            "Enable 'Use real LLM Planner' to make the next-action decision through OpenAI.\n\n"
        )

    def _file_row(self, parent, row, label, variable, command):
        ttk.Label(parent, text=label, width=22).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(4, 8), pady=5)
        ttk.Button(parent, text="Browse...", command=command, width=12).grid(row=row, column=2, pady=5)
        parent.columnconfigure(1, weight=1)

    def _browse_dataset(self):
        path = filedialog.askopenfilename(title="Select dataset.csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.dataset_var.set(path)

    def _browse_xml(self):
        path = filedialog.askopenfilename(title="Select AOI Inspection XML", filetypes=[("XML files", "*.xml"), ("All files", "*.*")])
        if path:
            self.xml_var.set(path)

    def _browse_image_root(self):
        path = filedialog.askdirectory(title="Select new image root folder")
        if path:
            self.image_root_var.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(title="Select output JSON", defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if path:
            self.output_var.set(path)

    def _validate_inputs(self):
        dataset = Path(self.dataset_var.get().strip())
        xml = Path(self.xml_var.get().strip())
        if not dataset.is_file():
            messagebox.showerror("Input Error", "Please select a valid dataset CSV file.")
            return None
        if not xml.is_file():
            messagebox.showerror("Input Error", "Please select a valid inspection XML file.")
            return None
        image_root_text = self.image_root_var.get().strip()
        image_root = image_root_text if image_root_text else None
        return str(dataset), str(xml), image_root

    def _set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.prepare_btn.configure(state=state)
        self.verify_btn.configure(state=state)
        self.run_btn.configure(state=state)

    def _run_background(self, target, extra=()):
        inputs = self._validate_inputs()
        if not inputs:
            return
        self._set_busy(True)
        self.status_var.set("Running...")
        threading.Thread(target=self._safe_worker, args=(target, inputs + tuple(extra)), daemon=True).start()

    def _safe_worker(self, target, args):
        try:
            target(*args)
        except Exception:
            error = traceback.format_exc()
            self.after(0, lambda: self._log(error))
            self.after(0, lambda: self.status_var.set("ERROR"))
        finally:
            self.after(0, lambda: self._set_busy(False))

    def _prepare_async(self):
        self._run_background(self._prepare)

    def _verify_async(self):
        self._run_background(self._verify)

    def _run_async(self):
        settings = (
            float(self.feature_threshold_var.get()),
            float(self.defect_threshold_var.get()),
            bool(self.use_llm_var.get()),
            self.llm_model_var.get().strip() or "gpt-4o",
            bool(self.llm_fallback_var.get()),
            self.output_var.get().strip(),
        )
        self._run_background(self._run_full, settings)

    def _prepare(self, dataset, xml, image_root):
        self.after(0, lambda: self._log("\n=== DATASET PREPARATION ===\n"))
        result = DatasetPreparationService().prepare(dataset, xml, image_root)
        self.after(0, lambda: self.input_count_var.set(str(result.metrics.get("total_samples", 0))))
        self.after(0, lambda: self.prepared_var.set(str(result.metrics.get("ready_samples", 0))))
        self.after(0, lambda: self.status_var.set(result.status))
        payload = {
            "status": result.status,
            "message": result.message,
            "metrics": result.metrics,
            "errors": result.errors[:30],
        }
        self.after(0, lambda: self._log(json.dumps(payload, indent=2) + "\n"))

    def _verify(self, dataset, xml, image_root):
        self.after(0, lambda: self._log("\n=== PREPARE + VERIFY ===\n"))
        prep = DatasetPreparationService().prepare(dataset, xml, image_root)
        samples = prep.data.get("samples", [])
        result = DatasetVerificationService().verify(samples)
        self.after(0, lambda: self.input_count_var.set(str(prep.metrics.get("total_samples", 0))))
        self.after(0, lambda: self.prepared_var.set(str(prep.metrics.get("ready_samples", 0))))
        self.after(0, lambda: self.verified_var.set(str(result.metrics.get("passed_samples", 0))))
        self.after(0, lambda: self.status_var.set(result.status))
        payload = {
            "preparation": {"status": prep.status, "metrics": prep.metrics},
            "verification": {
                "status": result.status,
                "metrics": result.metrics,
                "sample_results": result.data.get("sample_results", []),
            },
        }
        self.after(0, lambda: self._log(json.dumps(payload, indent=2) + "\n"))

    def _run_full(self, dataset, xml, image_root, feature_threshold, defect_threshold, use_llm, llm_model, llm_fallback, output_text):
        self.after(0, lambda: self._log("\n=== AGENTIC ADC WORKFLOW ===\n"))
        self.after(0, lambda: self._log(f"Planner: {'OpenAI LLM' if use_llm else 'Deterministic'} | Model: {llm_model if use_llm else 'N/A'}\n"))

        agent = OrchestratorAgent(
            self.project_root,
            feature_threshold=feature_threshold,
            defect_threshold=defect_threshold,
            use_llm=use_llm,
            planner_model=llm_model,
            allow_llm_fallback=llm_fallback,
            enable_a2a=False,
            populate_vector_db=False,
        )
        state = agent.run(dataset, xml, image_root)

        output = Path(output_text) if output_text else self.project_root / "outputs" / "result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")

        # ================= ADC REST PERSISTENCE + AGENT 2 LINK =================
        import uuid
        run_id = str(uuid.uuid4())
        output.with_suffix(".run_id.txt").write_text(run_id, encoding="utf-8")
        
        saved_ok = False
        try:
            saved = DataClient().save_run(asdict(state), run_id=run_id)
            saved_ok = True
        except Exception as exc:
            message = f"Shared DB save failed; JSON retained. Retry run ID: {run_id}\n{exc}\n"
            self.after(0, lambda message=message: self._log(message))
        else:
            message = f"Shared Qdrant save complete. Run ID: {saved['run_id']}\n"
            self.after(0, lambda message=message: self._log(message))

        # --- Automatic Agent 2 Explainability Review Trigger with HitL ---
        if saved_ok:
            def _dispatch_agent2_reviews():
                import httpx
                agent2_url = os.getenv("ADC_AGENT2_URL", "http://127.0.0.1:8001")
                client = DataClient()
                try:
                    cases = list(client.review_cases(run_id))
                    if not cases:
                        self.after(0, lambda: self._log("[Agent 2] No samples require explainability review.\n"))
                        return

                    self.after(0, lambda c=len(cases): self._log(
                        f"\n[Agent 2] Escalating {c} sample(s) for Explainability Review to {agent2_url}...\n"
                    ))

                    for case in cases:
                        sid = case.get("sample_id")
                        sample_obj = case.get("sample", {})
                        inference_obj = case.get("inference", {})
                        details = inference_obj.get("details", {})
                        defect_pred = details.get("defect_classification", {}).get("prediction")
                        
                        # Agent 1's verdict (model prediction or machine defect fallback)
                        a1_verdict = defect_pred or sample_obj.get("machine_defect", "Unknown")

                        self.after(0, lambda s=sid: self._log(f"[Agent 2] Reviewing evidence for sample {s}...\n"))

                        try:
                            resp = httpx.post(
                                f"{agent2_url}/reviews",
                                json={"run_id": run_id, "sample_id": sid},
                                timeout=180.0
                            )
                            if resp.status_code == 200:
                                res_json = resp.json()
                                result_wrap = res_json.get("result", res_json)
                                out = result_wrap.get("output", {})
                                a2_verdict = out.get("predicted_defect") or result_wrap.get("review_status") or "Unclassified"
                                diagnosis = out.get("diagnosis") or out.get("reasoning") or "Evidence verified."

                                # Log Agent 2 Diagnosis back to user
                                self.after(0, lambda s=sid, a1=a1_verdict, v=a2_verdict, d=diagnosis: self._log(
                                    f"\n──────────────────────────────────────────────────────────\n"
                                    f"[Agent 2 Diagnosis for {s}]\n"
                                    f"• Agent 1 Baseline : {a1}\n"
                                    f"• Agent 2 Audit    : {v}\n"
                                    f"• Full Explanation :\n  {d}\n"
                                    f"──────────────────────────────────────────────────────────\n"
                                ))

                                # ── CHECK AGREEMENT ──
                                if normalize_defect(a1_verdict) == normalize_defect(a2_verdict):
                                    self.after(0, lambda s=sid, v=a2_verdict: self._log(
                                        f"✅ [CONSENSUS] Both agents agree on '{v}'. Auto-approved.\n\n"
                                    ))
                                else:
                                    # ── DISCREPANCY DETECTED -> ASK HUMAN OPERATOR ──
                                    self.after(0, lambda s=sid, a1=a1_verdict, a2=a2_verdict: self._log(
                                        f"⚠️ [CONFLICT DETECTED] Agent 1 ('{a1}') != Agent 2 ('{a2}').\n"
                                        f"Waiting for Operator input in dialog...\n"
                                    ))

                                    event = threading.Event()
                                    result_holder = {}

                                    # Trigger popup dialog on main Tkinter UI thread
                                    self.after(0, lambda s=sid, a1=a1_verdict, a2=a2_verdict, diag=diagnosis: HumanReviewDialog(
                                        self, s, a1, a2, diag, event, result_holder
                                    ))

                                    # Wait for human to make choice and submit
                                    event.wait()

                                    chosen = result_holder.get("verdict", a2_verdict)
                                    notes = result_holder.get("notes", "")

                                    self.after(0, lambda s=sid, c=chosen, n=notes: self._log(
                                        f"👤 [HUMAN RESOLUTION] Sample {s} confirmed as: '{c}'\n"
                                        f"   Notes: {n}\n\n"
                                    ))

                                    # ── SEND HUMAN DECISION BACK TO SHARED DATABASE ──
                                    human_result = {
                                        "review_status": "HUMAN_RESOLVED",
                                        "final_defect": chosen,
                                        "operator_notes": notes,
                                        "agent1_verdict": a1_verdict,
                                        "agent2_verdict": a2_verdict,
                                        "agent2_diagnosis": diagnosis,
                                        "resolved_timestamp": datetime.now().isoformat()
                                    }
                                    client.request(
                                        "PUT",
                                        f"/runs/{run_id}/reviews/{sid}",
                                        json={"result": human_result}
                                    )

                            else:
                                err = resp.json().get("detail", resp.text)
                                self.after(0, lambda s=sid, e=err: self._log(f"[Agent 2 Error] Sample {s}: {e}\n"))
                        except httpx.ConnectError:
                            self.after(0, lambda: self._log(
                                f"[Agent 2] Could not connect to {agent2_url}. Make sure Agent 2 API is running.\n"
                            ))
                            break
                        except Exception as exc:
                            self.after(0, lambda s=sid, e=str(exc): self._log(f"[Agent 2 Exception] Sample {s}: {e}\n"))

                except Exception as ex:
                    self.after(0, lambda e=str(ex): self._log(f"[Agent 2 Query Error]: {e}\n"))

            threading.Thread(target=_dispatch_agent2_reviews, daemon=True).start()

        # ================= UPDATE UI COUNTERS =================
        self.after(0, lambda: self.status_var.set(state.status))
        self.after(0, lambda: self.input_count_var.set(str(state.input_samples)))
        self.after(0, lambda: self.prepared_var.set(str(state.preparation_ready)))
        self.after(0, lambda: self.verified_var.set(str(state.verification_passed)))
        self.after(0, lambda: self.inference_var.set(str(state.inference_attempted)))
        self.after(0, lambda: self.accepted_var.set(str(state.accepted)))
        self.after(0, lambda: self.review_var.set(str(state.review_required)))

        summary = {
            "workflow_status": state.status,
            "termination_reason": state.termination_reason,
            "planner_backend": state.planner_backend,
            "planner_model": state.planner_model,
            "input_samples": state.input_samples,
            "preparation_ready": state.preparation_ready,
            "preparation_failed": state.preparation_failed,
            "verification_passed": state.verification_passed,
            "verification_failed": state.verification_failed,
            "inference_attempted": state.inference_attempted,
            "inference_completed": state.inference_completed,
            "accepted": state.accepted,
            "review_required": state.review_required,
            "inference_aborted": state.inference_aborted,
            "plan_history": state.plan_history,
            "observations": state.observations,
            "results": state.inference_results,
            "output_json": str(output),
        }
        self.after(0, lambda: self._log(json.dumps(summary, indent=2) + "\n"))
        self.after(0, lambda: messagebox.showinfo(
            "ADC Workflow Finished",
            f"Status: {state.status}\nReason: {state.termination_reason}\n\nResult saved to:\n{output}",
        ))

    def _log(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def _clear_log(self):
        self.log.delete("1.0", "end")


if __name__ == "__main__":
    ADCApplication().mainloop()