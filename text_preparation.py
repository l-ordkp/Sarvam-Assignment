import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def process_text_to_vector_db(pdf_path, index_path):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings using Sentence Transformers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the FAISS index for text data
    text_vectorstore = FAISS.from_texts(chunks, embeddings)
    
    # Save the FAISS index for text data
    text_vectorstore.save_local(index_path)
    print("Text_db created")
    
    return text_vectorstore

# Usage
pdf_path = "C:\\Users\\Kshit\\Desktop\\Sarvam.ai\\iesc111.pdf"
text_index_path = "vector_db\\text_faiss_index"
text_vectorstore = process_text_to_vector_db(pdf_path, text_index_path)