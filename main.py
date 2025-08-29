import subprocess
import time
import requests
from api.orchestrator_tool import Orchestrator
from api.synthesizer import Synthesizer

def start_ml_server():
    """Start FastAPI ML server in a subprocess."""
    process = subprocess.Popen(
        ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("🚀 Starting ML server at http://localhost:8000 ...")
    time.sleep(3)  # give server a few seconds to spin up
    return process

def wait_for_server(url="http://localhost:8000/predict", timeout=30):
    """Wait until ML server is responding."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.post(url, json={}, timeout=2)
            if r.status_code in (200, 422):  # 422 = validation error (expected if empty payload)
                print("✅ ML server is live.")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("❌ ML server failed to start.")

def main():
    # Start ML server
    server_proc = start_ml_server()
    try:
        wait_for_server()

        orch = Orchestrator(api_base_url="http://localhost:8000")
        synth = Synthesizer(api_base_url="http://localhost:8000")

        queries = [
            "which estate had the least BTO in the past 5 years, for this estate, recommend a BTO price for low floor, 3-room flat in Bedok with an area of 100 sq m and lease commencement in 2019. the flat model is premium maisonette",
            "How does the predicted price for a 3-room flat in Bedok with an area of 100 sq m and lease commencement in 2019. the flat model is premium maisonette?"
        ]

        for q in queries:
            print("=" * 80)
            print("USER:", q)

            # 1. orchestrator
            orch_out = orch.run_two_pass(q)
            final_text = synth.synthesize(str(orch_out["response"]))
            print("FINAL ANSWER:\n", final_text)
            print("=" * 80)

    finally:
        print("🛑 Stopping ML server...")
        server_proc.terminate()
        server_proc.wait()

if __name__ == "__main__":
    main()
