import os
import logging
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union
import google.generativeai as genai
from utils.utils import get_defaults, get_valid_values, create_function_declarations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, api_base_url="http://localhost:8000", model="gemini-2.5-flash"):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.api_base_url = api_base_url  # where FastAPI is running 

        # Load valid values and defaults from helper functions
        self.valid_values = get_valid_values()
        self.DEFAULTS = get_defaults()

        # Create function schemas using helper
        self.function_declarations = create_function_declarations(self.valid_values)
    
        self.system_prompt = """
        You are a housing market intelligence assistant for Singapore's BTO (Build-To-Order) and HDB resale market.
        
        Your task is to understand the user's intent and determine which API(s) to call:
        
        1. Use call_prediction_api when the user asks for:
           - Price predictions for specific properties
           - "How much is a flat worth in X town?"
           - "Predict the price for a Y-room flat in Z"
           - Any query requesting a specific price estimation
        
        2. Use call_analysis_api when the user asks for:
           - Historical data analysis
           - Market trends
           - Comparative studies between towns or flat types
           - General information about the housing market
           - "Which towns have the highest prices?"
           - "Show me price trends over time"
        
        3. Use both APIs when the user wants:
           - A prediction along with historical context
           - Comparison between predicted and historical prices
           - "How does the predicted price compare to historical data?"
        
        For the prediction API, extract all relevant parameters from the user's query. 
        The function schema includes all valid options as enums, so you should only use values from those lists.
        If a parameter is not mentioned in the query, don't include it in the function call - the system will use appropriate defaults.
        
        For the analysis API, pass the original user query as the parameter.
        """

    def _ensure_prediction_params(self, params: Dict) -> Dict:
        """Ensure required parameters are present and fill defaults for missing optional ones"""
        # Start with a copy of the params
        final_params = params.copy()
        
        # Ensure required parameter 'town' is present
        if "town" not in final_params or not final_params["town"]:
            logger.warning(f"No town specified, using default: {self.DEFAULTS['town']}")
            final_params["town"] = self.DEFAULTS["town"]
        
        # Fill defaults for any missing optional parameters
        for key, default_value in self.DEFAULTS.items():
            if key not in final_params or final_params[key] is None or final_params[key] == "":
                final_params[key] = default_value
        
        return final_params
      
    def process_query(self, user_query: str) -> dict:
        """
        Process the user query using function calling to determine intent and extract parameters
        """
        try:
            response = self.model.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": self.system_prompt + "\n\nUser query: " + user_query}
                        ]
                    }
                ],
                tools=[{"function_declarations": self.function_declarations}],
                tool_config={"function_calling_config": {"mode": "any"}}
            )
            
            # Extract function calls from response
            function_calls = []
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append({
                            "name": part.function_call.name,
                            "args": dict(part.function_call.args) if part.function_call.args else {}
                        })
            
            # If no function calls were made, return a helpful response
            if not function_calls:
                return {
                    "response": "I'm not sure how to answer your query. Could you please clarify what information you're looking for about Singapore's HDB market?",
                    "sources": []
                }
            
            # Execute the function calls and collect results
            results = []
            for call in function_calls:
                if call["name"] == "call_prediction_api":
                    # Ensure parameters are complete with defaults
                    final_params = self._ensure_prediction_params(call["args"])
                    result = self._call_predict_endpoint(final_params)
                    results.append({
                        "type": "prediction",
                        "data": result,
                        "parameters": final_params,
                        "original_parameters": call["args"]
                    })
                elif call["name"] == "call_analysis_api":
                    result = self._call_analyze_endpoint(call["args"]["query"])
                    results.append({
                        "type": "analysis",
                        "data": result,
                        "parameters": call["args"]
                    })
            
            # Generate a natural language response based on the results
            return self._generate_response(user_query, results)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "sources": []
            }
    
    def _generate_response(self, user_query: str, results: list) -> dict:
        """
        Generate a natural language response based on the API results
        """
        # Prepare context for the model
        context_parts = []
        
        for result in results:
            if result["type"] == "prediction":
                predicted_price = result["data"].get("predicted_price")
                if predicted_price is not None:
                    context_parts.append(f"Prediction result: The estimated price is ${predicted_price:,.2f}")
                else:
                    context_parts.append("Prediction result: Unable to generate a price prediction")
                
                # Add parameters used
                params = result["parameters"]
                param_details = []
                for key, value in params.items():
                    param_details.append(f"{key}: {value}")
                
                context_parts.append(f"Prediction parameters: {', '.join(param_details)}")
                
            elif result["type"] == "analysis":
                analysis_data = result["data"]
                if analysis_data.get("results"):
                    context_parts.append(f"Analysis result: {analysis_data.get('explanation', 'Data analysis completed')}")
                    
                    # Include sample data if available
                    results_data = analysis_data.get("results", [])
                    if results_data and len(results_data) > 0:
                        # Show first few rows as examples
                        sample_data = results_data[:3]
                        context_parts.append(f"Sample data: {sample_data}")
                else:
                    context_parts.append("Analysis result: No data found for the query")
        
        # Create prompt for response generation
        prompt = f"""
        User query: {user_query}
        
        Context from API calls:
        {' '.join(context_parts)}
        
        Generate a helpful, natural language response to the user's query based on the context above.
        If there are multiple results, synthesize them into a coherent answer.
        Be specific and include numbers and data points when available.
        """
        
        try:
            response = self.model.generate_content(
                contents=[{"role": "user", "parts": [{"text": prompt}]}]
            )
            
            if response.candidates and response.candidates[0].content.parts:
                response_text = response.candidates[0].content.parts[0].text
                return {
                    "response": response_text,
                    "sources": results
                }
            
            return {
                "response": "I'm sorry, I couldn't generate a response to your query.",
                "sources": results
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": f"I encountered an error while generating a response: {str(e)}",
                "sources": results
            }
    
    def _call_predict_endpoint(self, payload: dict) -> dict:
        """Call /predict endpoint with structured payload"""
        url = f"{self.api_base_url}/predict"
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()  # {"predicted_price": ...}
        except Exception as e:
            logger.error(f"Error calling /predict: {e}")
            return {"predicted_price": None}
    
    def _call_analyze_endpoint(self, query: str) -> dict:
        """Call /analyze endpoint with query"""
        url = f"{self.api_base_url}/analyze"
        try:
            resp = requests.post(url, json={"query": query}, timeout=10)
            resp.raise_for_status()
            return resp.json()  # {sql, results, columns, explanation}
        except Exception as e:
            logger.error(f"Error calling /analyze: {e}")
            return {"sql": "", "results": [], "columns": [], "explanation": f"Error calling analyze: {str(e)}"}

    def run_two_pass(self, user_query: str) -> dict:
        # ---------- Pass 1 ----------
        result1 = self.process_query(user_query)
        sources1 = result1.get("sources", [])

        # ---------- Pass 2 ----------
        # Get the raw response text from the first pass
        first_response_text = result1["response"]

        # Prompt the model again with the first result and ask if more info is needed
        prompt = f"""
        User query: {user_query}

        First-pass response: {first_response_text}

        Based on the initial response, do you need additional information?
        If yes, suggest relevant API calls (e.g., call_analysis_api or call_prediction_api).
        If no, return the final answer directly without making further API calls.
        """

        try:
            second_response = self.model.generate_content(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                tools=[{"function_declarations": self.function_declarations}],
                tool_config={"function_calling_config": {"mode": "any"}}
            )

            # Extract function calls from the second response
            function_calls = []
            if second_response.candidates and second_response.candidates[0].content.parts:
                for part in second_response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append({
                            "name": part.function_call.name,
                            "args": dict(part.function_call.args) if part.function_call.args else {}
                        })

            # Execute any function calls made in the second pass
            results2 = []
            for call in function_calls:
                if call["name"] == "call_prediction_api":
                    final_params = self._ensure_prediction_params(call["args"])
                    result = self._call_predict_endpoint(final_params)
                    results2.append({
                        "type": "prediction",
                        "data": result,
                        "parameters": final_params,
                        "original_parameters": call["args"]
                    })
                elif call["name"] == "call_analysis_api":
                    result = self._call_analyze_endpoint(call["args"]["query"])
                    results2.append({
                        "type": "analysis",
                        "data": result,
                        "parameters": call["args"]
                    })

            # Combine all results
            all_results = sources1 + results2

            # Generate final natural language response
            final_response = self._generate_response(user_query, all_results)

            return final_response

        except Exception as e:
            logger.error(f"Error during second pass: {e}")
            return {
                "response": f"An error occurred during the second pass: {str(e)}",
                "sources": sources1
            }

def main():
    # Optional: set log level to DEBUG while testing
    logging.getLogger().setLevel(logging.DEBUG)
    
    orch = Orchestrator(api_base_url="http://localhost:8000")

    test_queries = [
        "which estate had the least BTO in the past 5 years, for this estate, recommend a BTO price for low floor, 3 room flat",
        "How does the predicted price for a 3-room flat in Bedok with an area ofd 100 sq m and lease commencement in 2019. the flat model is premium maisonette?"
    ]

    for q in test_queries:
        print("="*80)
        print(f"Query: {q}")
        result = orch.run_two_pass(q)
        print(result["response"])
        print("="*80)

if __name__ == "__main__":
    main()