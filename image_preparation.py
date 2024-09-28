import google.generativeai as genai
import numpy as np
import os
from PIL import Image
import io
import base64
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the Google API key from the environment
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Configure the Generative AI API with the loaded key
genai.configure(api_key=GOOGLE_API_KEY)

def summarize_image(image_path):
    try:
        myfile = genai.upload_file(image_path)
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content([myfile, "\n\n", "This image is being taken from a physics textbook, elaborate the image in terms of the physics concept (sound waves) that is being used there"])
        return result.text
    except Exception as e:
        print(f"Error summarizing {image_path}: {e}")
        return ""

def encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format)
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding {image_path}: {e}")
        return ""

def process_images_to_vector_db(image_folder, index_path):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize an empty list to store summaries and corresponding metadata
    summaries = []
    metadata_list = []
    
    # Check if the image folder exists and contains images
    if not os.path.exists(image_folder):
        print(f"Image folder '{image_folder}' does not exist.")
        return None
    
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    if not image_files:
        print(f"No valid images found in folder '{image_folder}'.")
        return None
    
    # Process each image to generate a summary and metadata
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        
        # Generate summary for the image
        summary = summarize_image(image_path)
        print(f"Summary for {filename}: {summary}")
        
        if summary:
            # Encode the image
            encoded_image = encode_image(image_path)
            
            if encoded_image:
                # Add the summary and metadata (image path and encoded image) to the lists
                summaries.append(summary)
                metadata_list.append({
                    "image_path": image_path,
                    "encoded_image": encoded_image  # Storing the encoded image in base64 format
                })
    
    if not summaries:
        print("No summaries generated for any image.")
        return None
    
    # Generate embeddings for the summaries
    text_embeddings = embeddings.embed_documents(summaries)  # Use LangChain's embedding functionality
    
    # Check the shape of the embeddings to avoid the IndexError
    if not text_embeddings:
        print("No embeddings generated for the summaries.")
        return None
    
    # Create the FAISS index with the embeddings and associated metadata
    image_vectorstore = FAISS.from_texts(summaries, embeddings, metadatas=metadata_list)
    
    print("Image vector store created with image data stored.")
    
    # Save the FAISS index for image data
    image_vectorstore.save_local(index_path)
    
    return image_vectorstore

# Usage
image_folder = "images"
image_index_path = "vector_db//image_faiss_index"
image_vectorstore = process_images_to_vector_db(image_folder, image_index_path)
