import os
from dotenv import load_dotenv

def load_environment_variables():
    """Load environment variables from .env file or Streamlit secrets"""
    # Try to load from .env file (local development)
    load_dotenv()
    
    # For Streamlit Cloud, secrets are automatically available
    # No need to manually load them

def check_data_folder():
    """Check if data folder exists and contains PDFs - deployment friendly"""
    data_path = "data/"
    
    # Create data folder if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        return False
    
    # Check for PDF files
    try:
        pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
        if not pdf_files:
            return False
        return True
    except FileNotFoundError:
        return False

def setup_directories():
    """Create necessary directories for deployment"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("vectorstore", exist_ok=True)
    os.makedirs("src", exist_ok=True)