import os
from pathlib import Path
import httpx
from adc_shared.client import DataClient

AGENT2_URL = os.getenv("ADC_AGENT2_URL", "http://127.0.0.1:8001")

def main():
    # 1. Automatically find the most recent .run_id.txt file
    run_id_files = sorted(Path("outputs").rglob("*.run_id.txt"), key=os.path.getmtime, reverse=True)
    if not run_id_files:
        print("No .run_id.txt files found in outputs/.")
        return

    latest_file = run_id_files[0]
    run_id = latest_file.read_text(encoding="utf-8").strip()
    print(f"Latest Run ID detected: {run_id}")

    # 2. Get review cases using client's built-in review_cases method
    client = DataClient()
    try:
        cases = list(client.review_cases(run_id))
    except Exception as e:
        print(f"Failed to fetch review cases: {e}")
        return

    if not cases:
        print("No samples in this run require Agent 2 review.")
        return

    print(f"Found {len(cases)} sample(s) requiring review.\n")

    # 3. Post reviews sequentially to Agent 2
    for case in cases:
        sample_id = case.get("sample_id")
        print(f"-> Sending review for sample: {sample_id} to Agent 2...")
        
        try:
            resp = httpx.post(
                f"{AGENT2_URL}/reviews",
                json={"run_id": run_id, "sample_id": sample_id},
                timeout=180.0
            )
            if resp.status_code == 200:
                print(f"   [SUCCESS] Review generated.")
            else:
                detail = resp.json().get('detail', resp.text)
                print(f"   [RESPONSE {resp.status_code}]: {detail}")
        except httpx.ConnectError:
            print(f"   [ERROR] Could not connect to Agent 2 at {AGENT2_URL}.")
            print("   Make sure Agent 2 API is running on port 8001 with ADC_ENABLE_AGENT2=1.")
            break
        except Exception as exc:
            print(f"   [ERROR] {exc}")

if __name__ == "__main__":
    main()