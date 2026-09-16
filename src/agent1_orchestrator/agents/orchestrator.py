from planner.llm_planner import Planner
from policy.policy_engine import PolicyEngine
from state.workflow_state import WorkflowState
from services.dataset_preparation import DatasetPreparationService
from services.dataset_verification import DatasetVerificationService
from services.model_lifecycle import ModelLifecycleService
from services.multimodal_inference import TwoStageInferenceService
from services.result_comparison import final_decision


class OrchestratorAgent:
    """
    Agentic ADC workflow controller.

    Loop:
      Workflow State -> LLM Planner -> Observation/Constraint/Decision/Reason
      -> Policy Engine -> Tool Execution -> State Update -> Re-plan

    The LLM decides WHAT should happen next. Domain services and policy remain
    deterministic so model inference and safety constraints are reproducible.
    """

    def __init__(
        self,
        project_root,
        feature_threshold=0.70,
        defect_threshold=0.70,
        use_llm=False,
        planner_model=None,
        allow_llm_fallback=True,
        max_replans=8,
    ):
        self.planner = Planner(
            use_llm=use_llm,
            model=planner_model,
            allow_fallback=allow_llm_fallback,
        )
        self.policy = PolicyEngine()
        self.preparation = DatasetPreparationService()
        self.verification = DatasetVerificationService()
        self.lifecycle = ModelLifecycleService(project_root)
        self.inference = TwoStageInferenceService(self.lifecycle, feature_threshold)
        self.defect_threshold = defect_threshold
        self.max_replans = max_replans
        self.use_llm = use_llm

    def run(self, dataset_csv, inspection_xml, image_root=None):
        state = WorkflowState(
            status="RUNNING",
            inputs={
                "dataset_csv": dataset_csv,
                "inspection_xml": inspection_xml,
                "image_root": image_root,
            },
            planner_backend="openai" if self.use_llm else "deterministic",
            llm_enabled=self.use_llm,
            planner_model=self.planner.model if self.use_llm else None,
        )

        last_action = None
        repeated_action_count = 0

        while state.status == "RUNNING":
            if state.replan_count >= self.max_replans:
                state.status = "ABORTED"
                state.termination_reason = "MAX_REPLAN_LIMIT_REACHED"
                state.observations.append(
                    f"Stopped after {self.max_replans} planner iterations to prevent loops."
                )
                break

            plan = self.planner.plan(state)
            state.replan_count += 1

            plan_record = {
                "plan_version": state.replan_count,
                "observation": plan.observation,
                "constraint": plan.constraint,
                "decision": plan.decision,
                "reason": plan.reason,
                "planner_source": plan.source,
                "policy": None,
                "tool_status": None,
            }

            state.observations.append(plan.observation)

            if plan.decision == last_action:
                repeated_action_count += 1
            else:
                repeated_action_count = 0
            last_action = plan.decision

            if repeated_action_count >= 2:
                plan_record["policy"] = {
                    "allowed": False,
                    "reason": "Repeated planner action detected; loop prevention activated.",
                }
                state.plan_history.append(plan_record)
                state.status = "ABORTED"
                state.termination_reason = "PLANNER_LOOP_DETECTED"
                break

            allowed, policy_reason = self.policy.validate_action(plan.decision, state)
            plan_record["policy"] = {"allowed": allowed, "reason": policy_reason}

            if not allowed:
                state.plan_history.append(plan_record)
                state.observations.append(f"Policy rejected {plan.decision}: {policy_reason}")
                # Bounded re-plan: planner sees the new observation on next iteration.
                continue

            state.current_step = plan.decision

            try:
                if plan.decision == "dataset_preparation":
                    result = self._execute_preparation(state)
                elif plan.decision == "dataset_verification":
                    result = self._execute_verification(state)
                elif plan.decision == "multimodal_inference":
                    result = self._execute_inference(state)
                elif plan.decision == "finalize":
                    self._finalize(state)
                    result = {"status": state.status, "success": state.status != "ABORTED"}
                elif plan.decision == "abort":
                    state.status = "ABORTED"
                    state.termination_reason = "PLANNER_ABORTED_WORKFLOW"
                    result = {"status": "ABORTED", "success": False}
                else:
                    result = {"status": "UNKNOWN_ACTION", "success": False}
            except Exception as exc:
                error = f"{plan.decision} failed: {type(exc).__name__}: {exc}"
                state.errors.append(error)
                state.observations.append(error)
                result = {"status": "TOOL_EXECUTION_FAILED", "success": False, "error": error}

            plan_record["tool_status"] = result.get("status")
            state.plan_history.append(plan_record)
            state.tool_history.append({
                "step": plan.decision,
                "status": result.get("status"),
                "success": result.get("success", False),
            })

            # If an unrecoverable tool execution produced no path forward, abort.
            if result.get("fatal"):
                state.status = "ABORTED"
                state.termination_reason = result.get("termination_reason", "UNRECOVERABLE_TOOL_FAILURE")

        state.current_step = None
        return state

    def _execute_preparation(self, state):
        prep = self.preparation.prepare(
            state.inputs["dataset_csv"],
            state.inputs["inspection_xml"],
            state.inputs.get("image_root"),
        )
        state.prepared_samples = prep.data.get("samples", [])
        state.input_samples = prep.metrics.get("total_samples", len(state.prepared_samples))
        state.preparation_ready = prep.metrics.get("ready_samples", 0)
        state.preparation_failed = prep.metrics.get("failed_samples", 0)

        state.observations.append(
            f"Dataset preparation: {state.preparation_ready}/{state.input_samples} READY; "
            f"{state.preparation_failed} failed. Status={prep.status}."
        )
        state.errors.extend(prep.errors[:20])

        if not state.prepared_samples:
            return {
                "status": prep.status,
                "success": False,
                "fatal": True,
                "termination_reason": "NO_INPUT_SAMPLES",
            }

        # Partial preparation is recoverable; verification decides what can proceed.
        return {"status": prep.status, "success": prep.success}

    def _execute_verification(self, state):
        verify = self.verification.verify(state.prepared_samples)
        state.verified_samples = verify.data.get("verified_samples", [])
        state.verification_passed = verify.metrics.get("passed_samples", 0)
        state.verification_failed = verify.metrics.get("failed_samples", 0)

        state.observations.append(
            f"Dataset verification: {state.verification_passed}/{verify.metrics.get('total_samples', 0)} "
            f"PASSED; {state.verification_failed} failed. Status={verify.status}."
        )

        allowed, reason = self.policy.allow_inference(verify)
        state.observations.append(reason)

        if not allowed:
            return {
                "status": verify.status,
                "success": False,
                "fatal": True,
                "termination_reason": "NO_VERIFIED_SAMPLES",
            }

        return {"status": verify.status, "success": verify.success}

    def _execute_inference(self, state):
        state.inference_attempted = len(state.verified_samples)

        for sample in state.verified_samples:
            result = self.inference.infer_sample(sample)

            if result.success:
                payload = result.data
                payload["status"] = result.status
                payload["final_decision"] = final_decision(result, self.defect_threshold)
                state.inference_results.append(payload)
                state.inference_completed += 1
            else:
                decision = "REVIEW_REQUIRED" if result.recoverable else "ABORTED"
                payload = {
                    "sample_id": sample.get("sample_id"),
                    "status": result.status,
                    "errors": result.errors,
                    "details": result.data,
                    "final_decision": decision,
                }
                state.inference_results.append(payload)
                if decision == "ABORTED":
                    state.inference_aborted += 1

        state.accepted = sum(
            1 for r in state.inference_results if r.get("final_decision") == "ACCEPTED"
        )
        state.review_required = sum(
            1 for r in state.inference_results if r.get("final_decision") == "REVIEW_REQUIRED"
        )
        state.inference_aborted = sum(
            1 for r in state.inference_results if r.get("final_decision") == "ABORTED"
        )

        state.observations.append(
            f"Inference attempted {state.inference_attempted} sample(s): "
            f"{state.accepted} accepted, {state.review_required} review-required, "
            f"{state.inference_aborted} aborted."
        )

        return {
            "status": "INFERENCE_EXECUTED",
            "success": bool(state.inference_results),
            "fatal": not bool(state.inference_results),
            "termination_reason": "NO_INFERENCE_RESULTS",
        }

    def _finalize(self, state):
        if not state.inference_results:
            state.status = "ABORTED"
            state.termination_reason = "NO_INFERENCE_RESULTS"
            return

        if state.inference_aborted > 0:
            state.status = "ABORTED"
            state.termination_reason = "INFERENCE_ABORTED"
        elif state.review_required > 0:
            state.status = "REVIEW_REQUIRED"
            uncertain = [
                r.get("status") for r in state.inference_results
                if r.get("final_decision") == "REVIEW_REQUIRED"
            ]
            state.termination_reason = uncertain[0] if uncertain else "HUMAN_REVIEW_REQUIRED"
        else:
            state.status = "COMPLETED"
            state.termination_reason = "ALL_INFERENCE_RESULTS_ACCEPTED"

        state.observations.append(
            f"Workflow finalized as {state.status}. Reason={state.termination_reason}."
        )

# Inside agents/orchestrator.py
from agents.a2a_dispatcher import Agent2ReviewClient

a2a_client = Agent2ReviewClient(agent2_url="http://127.0.0.1:8001")

def execute_review_escalation(state, sample, inference_result):
    if not a2a_client.discover():
        state.set_termination("REVIEW_REQUIRED", "AGENT2_UNAVAILABLE")
        return

    # Call Agent 2 via A2A
    review_response = a2a_client.delegate_review(sample, inference_result)
    artifact = review_response.get("artifact", {})

    if artifact.get("self_check_passed"):
        # The multimodal agent resolved the ambiguity with high confidence
        inference_result["final_defect"] = artifact.get("predicted_defect")
        inference_result["diagnosis"] = artifact.get("diagnosis")
        inference_result["resolved_by"] = "Agent2_Multimodal"
        state.mark_accepted(sample["sample_id"])
    else:
        # Physical or visual contradiction persists; route to human
        inference_result["diagnosis"] = artifact.get("diagnosis")
        inference_result["resolved_by"] = "HUMAN_QA_REQUIRED"
        state.mark_for_human_review(sample["sample_id"])