from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from chain_of_thought import load_generator  
from text_retrieval import retrieve_text_from_vector_db
from image_retrieval import retrieve_images
from generation import get_gemini_response  
from agent import send_message_and_handle_functions
from tts import text_to_speech

# Define the input model for FastAPI
class QueryRequest(BaseModel):
    query: str
class TextToSpeechRequest(BaseModel):
    query: str
    output_file_path: str

app = FastAPI()

# Initialize the generator (Chain of Thought)
generator = load_generator()
@app.post("/process_query/")
async def process_query(request: QueryRequest):
    query = request.query
    print(query)
    ret_text = retrieve_text_from_vector_db(query)
    context = "\n".join(ret_text)
    
    # Generate context using Chain of Thought reasoning
    response = generator(query, context)
    print(response)
    # Get the final response from Gemini API
    response_text = get_gemini_response(response, query)
    image_path = retrieve_images(query)
    return JSONResponse(content={"response": response_text, "image_path": image_path})

# Define the FastAPI endpoint to handle user queries
@app.post("/AI-agent/")
async def ask_agent(query_request: QueryRequest):
    try:
        # Get the user query from the request
        user_query = query_request.query
        
        # Send the message to the agent and get the response
        agent_response = send_message_and_handle_functions(user_query)
        
        # Return the agent's response
        return {"response": agent_response}
    
    except Exception as e:
        # Handle any errors and return a 500 status code
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/AI-Voice-agent/")
async def ask_agent(query_request: TextToSpeechRequest):
    try:
        # Get the user query from the request
        user_query = query_request.query
        path = query_request.output_file_path
        
        # Send the message to the agent and get the response
        agent_response = send_message_and_handle_functions(user_query)
        
        voice = text_to_speech(text=agent_response, output_file=path)
        
        # Return the agent's response
        return {"response": voice}
    
    except Exception as e:
        # Handle any errors and return a 500 status code
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
