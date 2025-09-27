import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
DB_FAISS_PATH = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
You are a medical AI assistant. Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context. Be precise and professional in your responses.

Context: {context}
Question: {question}

Start the answer directly with the medical information requested.
"""

class MedicalChatbot:
    def __init__(self, use_groq=True):
        self.use_groq = use_groq
        self.vectorstore = None
        self.qa_chain = None
        self.setup_chain()
    
    def set_custom_prompt(self):
        prompt = PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE, 
            input_variables=["context", "question"]
        )
        return prompt
    
    def load_llm(self):
        if self.use_groq:
            # Using Groq for faster inference
            return ChatGroq(
                model_name="llama-3.1-8b-instant",
                temperature=0.1,
                groq_api_key=os.getenv("GROQ_API_KEY"),
            )
        else:
            # Using HuggingFace
            return HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                temperature=0.1,
                model_kwargs={
                    "token": os.getenv("HF_TOKEN"),
                    "max_length": 512
                }
            )
    
    def setup_chain(self):
        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector store
        self.vectorstore = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.load_llm(),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': self.set_custom_prompt()}
        )
    
    def query(self, question):
        if not self.qa_chain:
            raise Exception("Chatbot chain not initialized. Call setup_chain() first.")
        
        response = self.qa_chain.invoke({'query': question})
        return {
            "answer": response["result"],
            "source_documents": response["source_documents"]
        }

# Utility function for command line testing
def test_chatbot():
    chatbot = MedicalChatbot()
    
    print("Medical Chatbot Ready! Type 'quit' to exit.")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() == 'quit':
            break
        
        response = chatbot.query(user_query)
        print("\nAnswer:", response["answer"])
        print("\nSources:", [doc.metadata for doc in response["source_documents"]])

if __name__ == "__main__":
    test_chatbot()