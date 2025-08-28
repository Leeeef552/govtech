import subprocess
import time
import requests

from api.analyst import HDBDataAnalyst
from api.orchestrator import Orchestrator
from api.predictor import PredictorClient
from api.synthesizer import Synthesizer

def start_ml_server():
    """Start FastAPI ML server in a subprocess."""
    process = subprocess.Popen(
        ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print("🚀 Starting ML server at http://localhost:8000 ...")
    time.sleep(3)
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

        # Init components
        orchestrator = Orchestrator()
        analyst = HDBDataAnalyst("data/hdb_prices.db")
        predictor = PredictorClient("http://localhost:8000/predict")
        synthesizer = Synthesizer()

        # Example queries
        queries = [
            "What’s the predicted price of a 4-room flat in Ang Mo Kio?",
            "Which towns had the least BTO launches in the past decade?",
            "How do current resale predictions compare with past BTO trends in Ang Mo Kio?",
        ]

        for q in queries:
            print(f"\n🟢 User query: {q}")
            result = orchestrator.run(q, analyst, predictor, synthesizer)
            print(f"🔵 Final result:\n{result}")

    finally:
        print("🛑 Stopping ML server...")
        server_proc.terminate()
        server_proc.wait()

if __name__ == "__main__":
    main()
