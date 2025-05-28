import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load your environment variables to get API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def list_models():
    models = genai.list_models()
    print("Available models:")
    for model in models:
        print(f"- {model.name} (supports: {model.supported_generation_methods})")

list_models()
