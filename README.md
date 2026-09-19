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
      "sample_id": "06-200036-02_C636_Board1_C636_Body_06-200036-02_MissingPart_3",
      "board_id": "06-200036-02",
      "component_id": "C636",
      "resolved_image_path": "C:/Users/.../data/inputs/06-200036-02/Body/Passed/Board1_C636_Body_...MissingPart_3.jpg",
      "resolved_golden_path": "C:/Users/.../data/inputs/06-200036-02/Body/Golden/Board1_C636_Body_...Golden.jpg",
      "baseline_inference": {
        "feature_class": "Body",
        "defect_class": "MissingPart",
        "confidence": 0.65
      },
      "gate_decision": "REVIEW_REQUIRED",
      "agent2_review": {
        "predicted_defect": "missing part",
        "confidence": 0.98,
        "self_check_passed": true,
        "contradiction_detected": false,
        "diagnosis": "Physical open circuit (ICT FAIL) and laser height (0.80 µm) confirm missing part.",
        "ipc_citations": [
          "IPC-A-610 Class 2 Section 8.3"
        ],
        "visual_evidence": "Rectangular solder land pattern is bare silver. Ceramic component body is absent."
      },
      "final_verdict": "missing part",
      "workflow_status": "COMPLETED"
    },
    {
      "sample_id": "06-200036-02_C978_Board1_C978_Body_06-200036-02_Shift_4",
      "board_id": "06-200036-02",
      "component_id": "C978",
      "resolved_image_path": "C:/Users/.../data/inputs/06-200036-02/Body/Passed/Board1_C978_Body_...Shift_4.jpg",
      "resolved_golden_path": "C:/Users/.../data/inputs/06-200036-02/Body/Golden/Board1_C978_Body_...Golden.jpg",
      "baseline_inference": {
        "feature_class": "Body",
        "defect_class": "Shift",
        "confidence": 0.70
      },
      "gate_decision": "REVIEW_REQUIRED",
      "agent2_review": {
        "predicted_defect": "shifted",
        "confidence": 0.95,
        "self_check_passed": true,
        "contradiction_detected": false,
        "diagnosis": "Side overhang (62.0%) exceeds IPC-A-610 Class 2 maximum 50% limit.",
        "ipc_citations": [
          "IPC-A-610 Class 2 Section 8.3.2"
        ],
        "visual_evidence": "Component body is misaligned laterally past the termination pad edge."
      },
      "final_verdict": "shifted",
      "workflow_status": "COMPLETED"
    }
  ]
}
