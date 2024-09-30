import requests
import os
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the function for text-to-speech conversion
def text_to_speech(text, output_file):
    # Get API key from environment variables
    sarvam_key = os.getenv('SARVAMAI_API_KEY')
    
    # Set up the API URL and payload
    url = "https://api.sarvam.ai/text-to-speech"
    payload = {
        "inputs": [text],
        "target_language_code": "hi-IN"
    }
    headers = {
        "api-subscription-key": sarvam_key,
        "Content-Type": "application/json"
    }
    
    # Make the API request
    response = requests.post(url, json=payload, headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Extract the base64 audio data from the response
        audio_base64 = response.json()["audios"][0]
        
        # Decode the base64-encoded audio
        audio_data = base64.b64decode(audio_base64)
        
        # Save the decoded audio to a file (e.g., "output_audio.mp3")
        with open(output_file, "wb") as audio_file:
            audio_file.write(audio_data)
        
        return (f"Audio saved successfully as '{output_file}'!")
    else:
        return (f"Error: {response.status_code} - {response.text}")


