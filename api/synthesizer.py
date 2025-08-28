from google import genai
import os
from dotenv import load_dotenv


class Synthesizer:
    """
    Synthesizer that combines outputs (prediction, analysis, etc.)
    into a natural language final answer.
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

    4. **BTO prices discounted against Resale prices**
    - The system will generate predictions by calculating resale prices with a standard discount (20-30%)


    You will be given one or more outputs from different models:
    - A resale price prediction model
    - An SQL/analyst model explaining BTO/resale trends

    Your task:
    1. Read the available outputs
    2. Combine them into a coherent, user-facing explanation
    3. Keep the answer concise, factual, and helpful

    Inputs:
    Payload: {payload}
    Prediction: {prediction}
    Analysis: {analysis}
    Query: {user_query}

    Respond with natural language only. Remember to quote the predicted resale price and the discount applied to provide the final recommended BTO price.
    """

    def __init__(self, model="gemini-2.5-flash"):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def synthesize(self, user_query, payload, prediction, analysis) -> str:
        prompt = self.PROMPT_TEMPLATE.format(
            prediction=prediction or "N/A",
            analysis=analysis or "N/A",
            user_query = user_query,
            payload=payload or "N/A"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip()
