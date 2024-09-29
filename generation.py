import os
import requests
from dotenv import load_dotenv


# Load the environment variables from .env file
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")

def get_gemini_response(response,query):
    
    prompt = (
            "You are a question answering bot."
            f"This is the query from the user: {query} "
            f"These are some contexts from our db (A Physics textbook for 9th Grade) for the given query: {response}"
            "Generate an appropriate, easy to understand response for the user's query according to the book."
            
        )
    
    headers = {
        "Content-Type": "application/json",
    }

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={gemini_api_key}",
        headers=headers,
        json=data
    )

    if response.status_code == 200:
        response_json = response.json()
        content = response_json.get('candidates', [{}])[0].get('content', '')
        return content
    else:
        raise Exception(f"Failed to get response from Gemini API: {response.status_code}, {response.text}")
    
    