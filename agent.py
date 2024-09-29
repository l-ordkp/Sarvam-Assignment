import google.generativeai as genai
from dotenv import load_dotenv
import os
from spotify_retrieval import get_songs_by_artist
from chain_of_thought import load_generator  
from text_retrieval import retrieve_text_from_vector_db
from google.ai import generativelanguage as glm
import requests

load_dotenv()

def general_call(query):
    prompt = (
            "You are a generic question answering bot."
            f"This is the query from the user: {query} "
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

def db_call(query):
    ret_text = retrieve_text_from_vector_db(query)
    context = "\n".join(ret_text)
    generator = load_generator()
    response = generator(query, context)
    return response

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Function declarations
get_songs_by_artist_declaration = glm.FunctionDeclaration(
    name="get_songs_by_artist",
    description="Fetches songs by a given artist from Spotify",
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        properties={
            "artist_name": glm.Schema(type=glm.Type.STRING)
        },
        required=["artist_name"]
    )
)

db_call_declaration = glm.FunctionDeclaration(
    name="db_call",
    description="Retrieves text from a vector database",
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        properties={
            
            "query": glm.Schema(type=glm.Type.STRING)
            
        },
        required=[ "query"]
    )
)
general_call_declaration = glm.FunctionDeclaration(
    name="general_call",
    description="Returns answers for general queries",
    parameters=glm.Schema(
        type=glm.Type.OBJECT,
        properties={
            
            "query": glm.Schema(type=glm.Type.STRING)
            
        },
        required=[ "query"]
    )
)
# Configure the Generative AI API with the loaded key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model with function declarations
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",  # Changed to pro for better function handling
    tools=[get_songs_by_artist_declaration, db_call_declaration, general_call_declaration]
)

# Custom prompt
custom_prompt = """
You are a helpful AI agent with access to two functions:
1. get_songs_by_artist: Use this to fetch songs by a specific artist from Spotify.
2. db_call: Use this to retrieve relevant text from a vector database that contains info about physics (typically sound).
3. general_call: Use this to answer questions when the query is neither physics related nor for fetching songs of a specific artist.

When asked about songs or artists, use the get_songs_by_artist function and return the list of songs.
When asked about any physics question, use the db_call function.
When asked about any general question, use the general_call function.


Always try to provide helpful and accurate information based on the user's request.
"""

# Start a chat with the custom prompt
chat = model.start_chat(enable_automatic_function_calling=True, history=[])
chat.send_message(custom_prompt)

# Function to handle the model's function calls
def handle_function_call(function_call):
    if function_call.name == "get_songs_by_artist":
        return get_songs_by_artist(**function_call.args)
    elif function_call.name == "retrieve_text_from_vector_db":
        return retrieve_text_from_vector_db(**function_call.args)
    elif function_call.name == "general_call":
        return general_call(**function_call.args)
    else:
        return "Function not implemented"

# Send a message and handle any function calls
def send_message_and_handle_functions(message):
    response = chat.send_message(message)
    
    if response.parts:
        for part in response.parts:
            if part.function_call:
                function_response = handle_function_call(part.function_call)
                response = chat.send_message(str(function_response))
    
    return response.text

