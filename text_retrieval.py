from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def retrieve_text_from_vector_db(query, index_path = "vector_db\\text_faiss_index", top_k=3):
    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the FAISS vector store from the saved index
    text_vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Generate an embedding for the query text
    query_embedding = embeddings.embed_query(query)
    
    # Perform similarity search on the FAISS index
    text_results = text_vectorstore.similarity_search_by_vector(query_embedding, k=top_k)
    retrieved_texts = [result.page_content for result in text_results]   
    return retrieved_texts



