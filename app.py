import os
import streamlit as st
from src.chatbot import MedicalChatbot
from src.utils import load_environment_variables, check_data_folder
from src.data_ingestion import create_vector_store

# Page configuration
st.set_page_config(
    page_title="Medical AI Chatbot",
    page_icon="🏥",
    layout="wide"
)

# Load environment variables
load_environment_variables()

@st.cache_resource
def load_chatbot():
    try:
        # Check if vector store exists
        if not os.path.exists("vectorstore/db_faiss/index.faiss"):
            st.error("Vector store not found. Please process PDFs first.")
            return None
            
        chatbot = MedicalChatbot(use_groq=True)
        return chatbot
    except Exception as e:
        st.error(f"Error loading chatbot: {str(e)}")
        return None

def main():
    st.title("🏥 AI Medical Chatbot with RAG")
    st.markdown("Ask questions about medical information from your PDF documents")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Vector store management
        st.subheader("Vector Store")
        if st.button("Process PDFs and Build Vector Store"):
            if check_data_folder():
                with st.spinner("Processing PDFs and building vector store..."):
                    try:
                        create_vector_store()
                        st.success("Vector store built successfully!")
                        st.cache_resource.clear()  # Clear cache to reload chatbot
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Please add PDF files to the data folder first.")
        
        st.divider()
        
        # Information
        st.info(
            "This chatbot uses RAG to answer questions based on your medical PDFs."
        )
    
    # Check if vector store exists
    if not os.path.exists("vectorstore/db_faiss/index.faiss"):
        st.warning("⚠️ Vector database not found.")
        st.write("Please:")
        st.write("1. Add PDF files to the 'data' folder")
        st.write("2. Click 'Process PDFs and Build Vector Store' in the sidebar")
        return
    
    # Initialize chatbot
    chatbot = load_chatbot()
    
    if chatbot is None:
        st.error("Failed to initialize chatbot. Please rebuild the vector store.")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching medical documents..."):
                try:
                    response = chatbot.query(prompt)
                    
                    # Display answer
                    st.markdown(response["answer"])
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response["answer"]
                    })
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()