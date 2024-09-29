from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import base64
from PIL import Image
import io

def load_image_vectorstore(index_path, embeddings):
    # Load the FAISS index from the saved path, passing the embeddings model
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def display_image_from_base64(encoded_image):
    # Decode the base64 image
    image_data = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(image_data))
    image.show()

def retrieve_images(query, index_path = "vector_db\\image_faiss_index", top_k=1):
    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the vector store from the saved FAISS index
    image_vectorstore = load_image_vectorstore(index_path, embeddings)
    
    # Generate an embedding for the query text
    query_embedding = embeddings.embed_query(query)
    
    # Perform similarity search on the FAISS index
    results = image_vectorstore.similarity_search_by_vector(query_embedding, k=top_k)
    
    if not results:
        print("No results found.")
        return []

    # Collect the image paths from the results
    image_paths = []
    for result in results:
        metadata = result.metadata
        image_path = metadata.get("image_path")  # Get the image path from metadata
        
        if image_path:
            image_paths.append(image_path)  # Append the image path to the list
        else:
            print("No image path found in metadata.")
    
    return image_paths  # Return the list of image paths
