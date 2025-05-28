# import google.generativeai as genai
# from dotenv import load_dotenv
# import os
# import json
# import re

# # Load .env variables
# load_dotenv()
# api_key = os.getenv("GEMINI_API_KEY")
# print(f"Loaded Gemini API key: {api_key[:5]}...")  # Debug: confirm API key loaded

# # Configure Gemini
# genai.configure(api_key=api_key)

# def get_precautions_and_water_level(disease_name: str) -> dict:
#     prompt = f"""
#     You are a plant disease expert. A plant leaf image was classified as suffering from "{disease_name}".

#     Based on this disease:
#     1. What precautions should a farmer take to manage or prevent it?
#     2. What is the ideal water level (e.g., low, moderate, high) to maintain for a plant suffering from this?

#     Respond strictly in this JSON format:
#     {{
#         "precautions": "...",
#         "water_level": "..."
#     }}
#     """

#     model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
#     response = model.generate_content(prompt)

#     print("Gemini raw response:")
#     print(response.text)  # Debug: see full response

#     # Try direct JSON parse first
#     try:
#         return json.loads(response.text)
#     except json.JSONDecodeError:
#         print("Failed to parse Gemini response as JSON, trying to extract JSON block...")

#         # Try to extract JSON substring from response
#         try:
#             json_str = re.search(r"\{.*\}", response.text, re.DOTALL).group()
#             parsed = json.loads(json_str)
#             print("Successfully extracted JSON from response.")
#             return parsed
#         except Exception as e:
#             print(f"Failed to extract/parse JSON: {e}")
#             print("Returning default fallback response.")
#             return {"precautions": "N/A", "water_level": "N/A"}
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(f"Loaded Gemini API key: {api_key[:5]}...")  # Debug

# Configure Gemini
genai.configure(api_key=api_key)

def get_precautions_and_water_level(disease_name: str) -> dict:
    prompt = f"""
    You are a plant disease expert.

    A plant leaf has been diagnosed with the disease: "{disease_name}".

    Respond ONLY in strict JSON format with the following structure, without markdown, comments, or explanations:

    {{
        "precautions": "<precise, clear steps to prevent/manage the disease in points and if it is not diseased, say 'No precautions needed'>",
        "water_level": "<low | moderate | high>"
    }}
    """

    model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

    # Reduce creativity for more structured outputs
    response = model.generate_content(prompt, generation_config={"temperature": 0.4})

    print("Gemini raw response:")
    print(response.text)

    # Attempt direct JSON parse
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        print("❌ Failed to parse directly. Trying to extract JSON block...")

        try:
            # Extract JSON using regex
            json_str = re.search(r"\{.*\}", response.text, re.DOTALL).group()
            parsed = json.loads(json_str)
            print("✅ Successfully extracted JSON.")
            return parsed
        except Exception as e:
            print(f"❌ Final fallback error: {e}")
            return {
                "precautions": "N/A",
                "water_level": "N/A"
            }
