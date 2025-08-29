import requests, os
from dotenv import load_dotenv

class Predictor:
    def __init__(self, url: str = None):
        load_dotenv
        self.url = url or os.getenv("PREDICTOR_URL")

    def predict(self, payload: dict) -> dict:
        try:
            resp = requests.post(self.url, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
