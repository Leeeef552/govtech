from google import genai
import os
from dotenv import load_dotenv
import json
from api.analyst import Analyst
from api.predictor import Predictor
from api.synthesizer import Synthesizer
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Orchestrator that decides whether to call Prediction, Analysis, or Both,
    and synthesizes the outputs into a final answer.
    """

    PROMPT_TEMPLATE = """
    You are a housing market intelligence assistant designed to support analysis of BTO (Build-To-Order) pricing in Singapore, leveraging trends from HDB resale transactions. Provide explanations and predictions for BTO prices based on past resale data and contextual factors.
    1. **Pricing Framework**
    - BTO flat prices are informed by nearby resale flat transactions: HDB benchmarks BTO “market value” by comparing transacted resale flat prices in the vicinity

    2. **Trends in BTO–Resale Price Gaps**
    - In **mature estates**, resale prices have risen faster than BTO prices
    - In **non-mature estates**, BTO price growth sometimes outpaces resale growth

    3. **Other Market Influences**
    - BTO supply levels significantly affect resale demand and prices. Over-supply of BTO units historically led to resale price stagnation, while undersupply—or demand exceeding supply—increases pressure on resale prices
   - Micro-level price drivers include: location/town attributes, estate maturity, flat type and size, remaining lease, floor level, and time of year. Empirical analyses confirm that higher floors,central/mature locations, larger flat sizes, newer builds (longer leases) command higher resale prices

    You are an orchestration assistant. Your task is to use the above knowledge about BTO and Resale relationship, understand the user intent and requirements, and provide a decision as to what analytical method is required and then decide which endpoints to call for a given user query:
    1. "prediction" → call the resale price prediction model
    2. "analysis"   → call the SQL+analyst model
    3. "both"       → call both models, then combine the outputs
    
    Rules:
    - Always return JSON only, no explanations.
    - Keys: {{"action": "prediction" | "analysis" | "both"}}
    - If "both", also explain how to combine them.
    - Base your choice ONLY on the query intent.
    
    Example:
    Q: "What’s the predicted price of a 4-room flat in Ang Mo Kio?"
    A: {{"action": "prediction"}}
    
    Q: "Which towns had the least BTO launches in the past decade?"
    A: {{"action": "analysis"}}
    
    Q: "How do current resale predictions compare with past BTO trends in Ang Mo Kio?"
    A: {{"action": "both"}}
    
    ---
    Query: {user_query}
    """


    TOWNS = [
        "woodlands", "jurong west", "tampines", "yishun", "bedok",
        "sengkang", "hougang", "ang mo kio", "bukit batok", "bukit merah",
        "choa chu kang", "pasir ris", "bukit panjang", "toa payoh",
        "kallang/whampoa", "geylang", "queenstown", "punggol", "clementi",
        "jurong east", "sembawang", "serangoon", "bishan", "marine parade",
        "central area", "bukit timah"
    ]

    FLAT_TYPES = [
        "4-room", "3-room", "5-room", "executive",
        "2-room", "1-room", "multi-generation"
    ]

    STOREY_RANGES = [
        "04 to 06", "07 to 09", "01 to 03", "10 to 12", "13 to 15",
        "01 to 05", "06 to 10", "16 to 18", "11 to 15", "19 to 21",
        "22 to 24", "16 to 20", "25 to 27", "28 to 30", "21 to 25",
        "26 to 30", "34 to 36", "37 to 39", "31 to 33", "40 to 42",
        "36 to 40", "31 to 35"
    ]

    FLAT_MODELS = [
        "model a", "improved", "new generation", "premium apartment",
        "simplified", "apartment", "maisonette", "standard", "dbss",
        "model a2", "model a-maisonette", "adjoined flat", "type s1",
        "2-room", "type s2", "premium apartment loft", "terrace",
        "multi generation", "3gen", "improved-maisonette", "premium maisonette"
    ]

    MIN_AREA, MAX_AREA = 31, 266


    def __init__(self, model="gemini-2.5-flash"):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        self.client = genai.Client(api_key=api_key)
        self.model = model


    def decide(self, user_query: str) -> dict:
        """Return structured decision JSON: {"action": "..."}"""
        prompt = self.PROMPT_TEMPLATE.format(user_query=user_query)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        raw = response.text or ""
        json_str = re.search(r'\{.*\}', raw, flags=re.S)
        if not json_str:
            raise ValueError("No JSON object found in model answer")

        result = json.loads(json_str.group(0))
        return result
    

    def build_prediction_payload(self, user_query: str) -> dict:
        """
        Given a free-form user query, return a dictionary with the 6 required keys
        for the prediction endpoint.  Any missing values are filled with the
        statistically most common / reasonable choice.
        """
        prompt = f"""
            You are an assistant that extracts structured attributes for a Singapore HDB resale-price prediction model.

            Allowed values:
            - month: in YYYY-MM format, if not specified you may use 2025-01
            - town: {self.TOWNS}
            - flat_type: {self.FLAT_TYPES}
            - flat_model: {self.FLAT_MODELS}
            - storey_range: {self.STOREY_RANGES}
            - floor_area_sqm: integer between {self.MIN_AREA} and {self.MAX_AREA}
            - lease_commence_date: integer year (YYYY).  If not mentioned, use 2025.

            For any attribute *not explicitly stated* in the user query, choose the SINGLE most likely value based on overall Singapore resale-market frequency.  Never return null.

            Return valid JSON only, no extra words or keys.
            Example JSON: {{"month": "2025-01", "town": "ang mo kio", "flat_type": "4-room", "flat_model": "improved", "storey_range": "10 to 12", "floor_area_sqm": 90, "lease_commence_date": 2025}}

            User query: {user_query}
        """

        response = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")).models.generate_content(
                model=self.model,
                contents=prompt
            )
        raw_text = response.text.strip()

        # Attempt to extract JSON from response
        json_match = re.search(r'\{.*\}', raw_text, re.S)
        if not json_match:
            raise ValueError(f"No valid JSON found in model response: {raw_text[:200]}...")

        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from model response: {e}\nResponse: {raw_text}")

        # Validate expected keys
        required_keys = ["month", "town", "flat_type", "flat_model", "storey_range", "floor_area_sqm", "lease_commence_date"]
        for key in required_keys:
            if key not in payload:
                payload[key] = self._default_value_for(key)

        return payload
    
    
    def run(self, user_query: str, analyst: Analyst,
            predictor: Predictor, synthesizer: Synthesizer) -> dict:

        decision = self.decide(user_query)
        action = decision["action"]

        # Log the decision
        logger.info(f"Orchestrator decision: {action} | Query: {user_query}")

        if action == "prediction":
            payload = self.build_prediction_payload(user_query)
            logger.debug(f"Prediction payload: {payload}")
            prediction_result = predictor.predict(payload)
            synthesized = synthesizer.synthesize(
                user_query=user_query,
                prediction=prediction_result,
                analysis=None,
                payload=payload
            )
            return synthesized

        elif action == "analysis":
            analysis = analyst.query(user_query, display=False)
            logger.debug(f"Analysis result: {analysis}")
            synthesized = synthesizer.synthesize(
                user_query=user_query,
                prediction=None,
                analysis=str(analysis),
                payload=None
            )
            return synthesized

        elif action == "both":
            payload = self.build_prediction_payload(user_query)
            logger.debug(f"Prediction payload: {payload}")
            prediction = predictor.predict(payload)
            analysis = analyst.query(user_query, display=False)
            logger.debug(f"Analysis result: {analysis}")
            synthesized = synthesizer.synthesize(
                user_query=user_query,
                prediction=prediction,
                analysis=analysis,
                payload=payload
            )
            return synthesized