import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Step 2: Create Chunks
def create_chunks(extracted_data, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Step 3 & 4: Create and Store Embeddings
def create_vector_store():
    # Load PDFs
    documents = load_pdf_files(DATA_PATH)
    print(f"Loaded {len(documents)} PDF pages")
    
    # Create chunks
    text_chunks = create_chunks(documents)
    print(f"Created {len(text_chunks)} text chunks")
    
    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create and save FAISS vector store
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store saved to {DB_FAISS_PATH}")
    
    return db

if __name__ == "__main__":
    create_vector_store()