from google import genai
import os
from dotenv import load_dotenv
import json


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
    - Keys: {"action": "prediction" | "analysis" | "both"}
    - If "both", also explain how to combine them.
    - Base your choice ONLY on the query intent.
    
    Example:
    Q: "What’s the predicted price of a 4-room flat in Ang Mo Kio?"
    A: {"action": "prediction"}
    
    Q: "Which towns had the least BTO launches in the past decade?"
    A: {"action": "analysis"}
    
    Q: "How do current resale predictions compare with past BTO trends in Ang Mo Kio?"
    A: {"action": "both"}
    
    ---
    Query: {user_query}
    """

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
        # Expecting valid JSON back
        return json.loads(response.text.strip())
