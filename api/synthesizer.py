from google import genai
import os
from dotenv import load_dotenv
from api.orchestrator_tool import Orchestrator
from utils.utils import get_defaults, get_valid_values, create_function_declarations


class Synthesizer(Orchestrator):
    """
    Synthesizer that combines outputs (prediction, analysis, etc.)
    into a natural language final answer.
    """

    def __init__(self, api_base_url="http://localhost:8000", model="gemini-2.5-flash"):
        super().__init__(api_base_url=api_base_url, model=model)

        self.final_template  = """
            You are a housing market intelligence assistant designed to support analysis of BTO (Build-To-Order) pricing in Singapore,
            leveraging trends from HDB resale transactions. Provide explanations and predictions for BTO prices based on past resale data
            and contextual factors.

            1. **Pricing Framework**
            - BTO flat prices are informed by nearby resale flat transactions: HDB benchmarks BTO “market value” by comparing transacted resale flat prices in the vicinity

            2. **Trends in BTO–Resale Price Gaps**
            - In **mature estates**, resale prices have risen faster than BTO prices
            - In **non-mature estates**, BTO price growth sometimes outpaces resale growth

            3. **Other Market Influences**
            - BTO supply levels significantly affect resale demand and prices.
            - Micro-level price drivers: location/town attributes, estate maturity, flat type and size, remaining lease, floor level, time of year.

            4. **BTO prices discounted against Resale prices**
            - Predictions use resale benchmarks with a 20–30% discount for BTO recommendations.

            You will be given one or more outputs from different models:
            - A resale price prediction model
            - An SQL/analyst model explaining BTO/resale trends

            Your task:
            1. Read the available outputs
            2. Combine them into a coherent, user-facing explanation
            3. Keep the answer concise, factual, and helpful

            Provided outputs:
            {outputs}

            Respond with natural language only. Remember to quote the predicted resale price and
            the discount applied to provide the final recommended BTO price.
            """
        
    # ----------  final plain-language synthesis ----------
    def synthesize(self, outputs: str) -> str:
        prompt = self.final_template.format(outputs=outputs)
        return self.model.generate_content(contents=prompt).text.strip()