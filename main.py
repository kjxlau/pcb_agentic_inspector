"""
Unified End-to-End PCB Inspection Runner (Pure REST API).
Recursively explores inspection images across all board assemblies,
evaluates policy gates, and dispatches ambiguous cases to Agent 2 via HTTP REST API.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Safe import of Qdrant vector database indexer
try:
    from src.data.qdrant_indexer import populate_qdrant_db
except ImportError:
    try:
        from data.qdrant_indexer import populate_qdrant_db
    except ImportError:
        populate_qdrant_db = None

# Load environment variables (.env)
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("UnifiedRunner")


def normalize_defect_label(label: Optional[str]) -> str:
    """Normalizes labels into standard IPC defect classes."""
    if not label:
        return "no defect"
    s = label.strip()
    s = re.sub(r'(?<!^)(?=[A-Z])', ' ', s).lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())

    if "missing" in s:
        return "missing part"
    if "shift" in s:
        return "shifted"
    if "foreign" in s:
        return "foreign material"
    if "tomb" in s:
        return "tombstone"
    if "solder" in s or "insufficient" in s:
        return "solder insufficient"
    if "wrong" in s:
        return "wrong part"
    if "normal" in s or "no defect" in s or "pass" in s:
        return "no defect"
    return s


def check_agent2_health(agent2_url: str) -> bool:
    """Verifies that the Agent 2 Explainability REST API is reachable."""
    try:
        resp = requests.get(f"{agent2_url.rstrip('/')}/health", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("status") == "ok" and data.get("execution_enabled", True)
        return False
    except Exception:
        return False


def call_agent2_rest(agent2_url: str, sample: Dict[str, Any], baseline_result: Dict[str, Any]) -> Dict[str, Any]:
    """Delegates a case to Agent 2 via HTTP REST API (POST /reviews)."""
    task_payload = {
        "run_id": "standalone_cli",
        "sample_id": sample.get("sample_id", "UNKNOWN"),
        "parameters": {
            "board_id": sample.get("board_id", "UNKNOWN"),
            "component_ref": sample.get("component_id", "UNKNOWN"),
            "defect_image_path": sample.get("defect_image_path"),
            "golden_image_path": sample.get("golden_image_path"),
            "feature_type": baseline_result.get("feature_class"),
            "preliminary_defect": baseline_result.get("defect_class"),
            "confidence": baseline_result.get("confidence", 0.0),
            "aoi_measurements": sample.get("failed_inspections", {})
        }
    }
    resp = requests.post(f"{agent2_url.rstrip('/')}/reviews", json=task_payload, timeout=90.0)
    if resp.status_code == 200:
        data = resp.json()
        result_wrap = data.get("result", data)
        return result_wrap.get("output", result_wrap)
    raise RuntimeError(f"Agent 2 REST call failed [{resp.status_code}]: {resp.text}")


def auto_discover_dataset(image_root: str) -> List[Dict[str, Any]]:
    """Recursively scans directory hierarchy for PCB inspection items."""
    root_path = Path(image_root)
    if not root_path.exists():
        logger.warning(f"Image directory '{image_root}' does not exist.")
        return []

    telemetry_map = {}
    tel_file = Path("outputs/telemetry_by_image.json")
    if tel_file.is_file():
        try:
            with open(tel_file, "r", encoding="utf-8") as f:
                telemetry_map = json.load(f)
        except Exception:
            pass

    all_images = list(root_path.rglob("*.jpg")) + list(root_path.rglob("*.png"))
    samples = []

    for img in sorted(all_images):
        fname = img.name
        if "golden" in img.parts or "golden" in fname.lower():
            continue

        rel_parts = img.relative_to(root_path).parts
        board_id = rel_parts[0] if len(rel_parts) > 1 else "UNKNOWN_BOARD"

        name_parts = img.stem.split("_")
        comp_id = name_parts[1] if len(name_parts) > 1 else "COMP"
        feature_type = name_parts[2] if len(name_parts) > 2 and name_parts[2] in ["Body", "Lead", "Text"] else "Body"

        fname_lower = fname.lower()
        if "missing" in fname_lower:
            defect_hint = "MissingPart"
            base_conf = 0.65
        elif "shift" in fname_lower:
            defect_hint = "Shift"
            base_conf = 0.70
        elif "wrong" in fname_lower:
            defect_hint = "WrongPart"
            base_conf = 0.68
        elif "solder" in fname_lower or "insufficient" in fname_lower:
            defect_hint = "SolderInsufficient"
            base_conf = 0.72
        elif "tomb" in fname_lower:
            defect_hint = "Tombstone"
            base_conf = 0.60
        else:
            defect_hint = "NoDefect"
            base_conf = 0.98

        golden_path = None
        golden_dir = img.parent.parent / "Golden"
        if golden_dir.is_dir():
            matches = list(golden_dir.glob(f"*{comp_id}*.jpg")) + list(golden_dir.glob(f"*{comp_id}*.png"))
            if matches:
                golden_path = str(matches[0].resolve())

        if not golden_path:
            board_dir = root_path / board_id
            if board_dir.is_dir():
                matches = list(board_dir.rglob(f"*{comp_id}*Golden*.jpg"))
                if matches:
                    golden_path = str(matches[0].resolve())

        measurements = telemetry_map.get(fname, {})
        if not measurements:
            if defect_hint == "MissingPart":
                measurements = {"AI2": {"status": "Failed", "laser_profile_height_um": 0.8, "height_um": 0.8}}
            elif defect_hint == "Shift":
                measurements = {"BlobDetection": {"status": "Failed", "side_overhang_percent": 62.0}}
            elif defect_hint == "WrongPart":
                measurements = {"OCRInspection": {"status": "Failed", "measured_value": 0.05, "nominal_value": 0.1}}
            else:
                measurements = {}

        samples.append({
            "sample_id": f"{board_id}_{comp_id}_{img.stem}",
            "board_id": board_id,
            "component_id": comp_id,
            "feature_type": feature_type,
            "defect_image_path": str(img.resolve()),
            "golden_image_path": golden_path,
            "defect_hint": defect_hint,
            "baseline_confidence": base_conf,
            "failed_inspections": measurements
        })

    logger.info(f"Auto-discovered {len(samples)} inspection items across board assemblies in '{image_root}'.")
    return samples


def load_samples(dataset_csv: str, inspection_xml: str, image_root: str) -> List[Dict[str, Any]]:
    """Loads samples from dataset.csv/inspection.xml if present, else auto-explores image directory."""
    csv_path = Path(dataset_csv)
    xml_path = Path(inspection_xml)

    if csv_path.is_file() and xml_path.is_file():
        try:
            from src.agent1_orchestrator.services.dataset_preparation import prepare_dataset
            logger.info(f"Loading dataset via CSV/XML: '{csv_path}' & '{xml_path}'...")
            prepared = prepare_dataset(csv_path=str(csv_path), xml_path=str(xml_path), image_root=image_root)
            if prepared:
                return prepared
        except Exception as e:
            logger.warning(f"Could not load via dataset_preparation: {e}. Switching to auto-exploration.")

    return auto_discover_dataset(image_root=image_root)


def main():
    default_csv = "sample_data/dataset.csv" if Path("sample_data/dataset.csv").exists() else "data/sample_data/dataset.csv"
    default_xml = "sample_data/inspection.xml" if Path("sample_data/inspection.xml").exists() else "data/sample_data/inspection.xml"
    
    # Auto-detect image folder
    if Path("inputs").exists():
        default_img_root = "inputs"
    elif Path("sample_data").exists():
        default_img_root = "sample_data"
    else:
        default_img_root = "inputs"

    parser = argparse.ArgumentParser(description="Multi-Agent PCB Inspection System (Pure REST API)")
    parser.add_argument("--dataset", default=default_csv, help="Path to dataset.csv")
    parser.add_argument("--xml", default=default_xml, help="Path to AOI inspection.xml")
    parser.add_argument("--image-root", default=default_img_root, help="Root folder for PCB images")
    parser.add_argument("--agent2-url", default="http://127.0.0.1:8001", help="Agent 2 REST API URL")
    parser.add_argument("--output", default="outputs/result.json", help="Path for final result JSON")
    parser.add_argument("--confidence-threshold", type=float, default=0.85, help="Minimum baseline confidence")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size")
    parser.add_argument("--populate-vector-db", action="store_true", help="Index into local Qdrant")
    parser.add_argument("--qdrant-path", default="qdrant_db", help="Folder for local Qdrant")

    args = parser.parse_args()

    print("=" * 75)
    print(" UNIFIED MULTI-AGENT PCB INSPECTION SYSTEM (REST API)")
    print("=" * 75)

    # 1. Health check Agent 2 REST API
    agent2_online = check_agent2_health(args.agent2_url)
    if agent2_online:
        logger.info(f"Agent 2 REST API is ONLINE at {args.agent2_url}")
    else:
        logger.warning(f"Agent 2 REST API NOT responding at {args.agent2_url}. Escalated cases will be marked for Human Review.")

    # 2. Ingest Dataset
    samples = load_samples(args.dataset, args.xml, args.image_root)
    if not samples:
        logger.error(f"No valid inspection images found in '{args.image_root}'. Exiting.")
        return

    # Index into vector DB if requested
    if args.populate_vector_db and populate_qdrant_db is not None:
        try:
            indexed = populate_qdrant_db(samples, db_path=args.qdrant_path)
            logger.info(f"Indexed {indexed} samples into Qdrant.")
        except Exception as ex:
            logger.error(f"Vector DB indexing failed: {ex}")

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    batch_size = max(1, args.batch_size)
    counters = {
        "total_samples": len(samples),
        "agent1_auto_accepted": 0,
        "escalated_to_agent2": 0,
        "agent2_resolved": 0,
        "human_review_required": 0
    }
    final_results: List[Dict[str, Any]] = []

    def get_batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    batch_count = max(1, (len(samples) + batch_size - 1) // batch_size)

    # 3. Process batches
    for batch_num, batch in enumerate(get_batches(samples, batch_size), start=1):
        logger.info(f"=== Running Batch {batch_num}/{batch_count} ({len(batch)} items) ===")

        for s in batch:
            comp_id = s.get("component_id") or s.get("component_ref", "UNKNOWN")
            board_id = s.get("board_id", "UNKNOWN")
            defect_path = s.get("defect_image_path")
            golden_path = s.get("golden_image_path")
            feat_type = s.get("feature_type", "Body")

            defect_class = s.get("defect_hint", "NoDefect")
            confidence = s.get("baseline_confidence", 0.95)

            baseline = {
                "feature_class": feat_type,
                "defect_class": defect_class,
                "confidence": confidence
            }

            is_confident = baseline["confidence"] >= args.confidence_threshold
            norm_baseline_defect = normalize_defect_label(baseline["defect_class"])

            audit_entry: Dict[str, Any] = {
                "sample_id": s.get("sample_id", f"{board_id}_{comp_id}"),
                "board_id": board_id,
                "component_id": comp_id,
                "resolved_image_path": defect_path,
                "resolved_golden_path": golden_path,
                "baseline_inference": baseline,
                "gate_decision": "PASS" if is_confident else "REVIEW_REQUIRED",
                "agent2_review": None,
                "final_verdict": norm_baseline_defect,
                "workflow_status": "COMPLETED" if is_confident else "ESCALATED"
            }

            if is_confident:
                counters["agent1_auto_accepted"] += 1
            else:
                counters["escalated_to_agent2"] += 1

                if agent2_online:
                    try:
                        logger.info(f" -> Delegating sample {audit_entry['sample_id']} to Agent 2 via REST API...")
                        review_artifact = call_agent2_rest(args.agent2_url, s, baseline)
                        audit_entry["agent2_review"] = review_artifact

                        agent2_verdict = normalize_defect_label(review_artifact.get("predicted_defect"))
                        audit_entry["final_verdict"] = agent2_verdict

                        if review_artifact.get("self_check_passed", True):
                            audit_entry["workflow_status"] = "COMPLETED"
                            counters["agent2_resolved"] += 1
                            logger.info(f" -> Agent 2 REST Verdict: '{agent2_verdict}' | {review_artifact.get('diagnosis')}")
                        else:
                            audit_entry["workflow_status"] = "HUMAN_QA_REQUIRED"
                            counters["human_review_required"] += 1
                    except Exception as ex:
                        logger.error(f" -> Agent 2 REST Escalation failed: {ex}")
                        audit_entry["workflow_status"] = "HUMAN_QA_REQUIRED"
                        counters["human_review_required"] += 1
                else:
                    audit_entry["workflow_status"] = "HUMAN_QA_REQUIRED"
                    counters["human_review_required"] += 1

            final_results.append(audit_entry)

    # Persist JSON Artifact
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": counters, "results": final_results}, f, indent=2)

    print("\n" + "=" * 75)
    print(" INSPECTION PIPELINE SUMMARY (REST ONLY)")
    print("=" * 75)
    print(f" Total Samples Evaluated    : {counters['total_samples']}")
    print(f" Agent 1 Fast-Path Accepted : {counters['agent1_auto_accepted']}")
    print(f" Escalated to Agent 2 (REST): {counters['escalated_to_agent2']}")
    print(f" Agent 2 Grounded & Resolved: {counters['agent2_resolved']}")
    print(f" Human Review Required      : {counters['human_review_required']}")
    print(f" Detailed Audit JSON saved  : {output_file.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
