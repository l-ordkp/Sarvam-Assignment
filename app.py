from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from chain_of_thought import load_generator  
from text_retrieval import retrieve_text_from_vector_db
from image_retrieval import retrieve_images
from generation import get_gemini_response  
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Define the input model for FastAPI
class QueryRequest(BaseModel):
    query: str

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
