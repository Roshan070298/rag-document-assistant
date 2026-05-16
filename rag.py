import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import shutil

load_dotenv()

# Step 1: Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

# Step 2: Split into chunks
def split_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 500 characters per chunk
        chunk_overlap=50     # 50 char overlap — context lost nahi ho
    )
    chunks = splitter.split_documents(pages)
    return chunks

# Step 3: Create embeddings + store in ChromaDB
def create_vectorstore(chunks):
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Local storage
    )
    return vectorstore

# Step 4: Build RAG chain
def build_rag_chain(vectorstore):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",   # Free Groq model
        temperature=0
    )
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Top 3 relevant chunks
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # Kahan se aaya answer
    )
    return chain

# Main function — sab ek saath
def process_pdf(file_path):
    print("Loading PDF...")
    pages = load_pdf(file_path)
    
    print(f"Splitting into chunks...")
    chunks = split_documents(pages)
    print(f"Total chunks: {len(chunks)}")
    
    print("Creating embeddings...")
    vectorstore = create_vectorstore(chunks)
    
    print("Building RAG chain...")
    chain = build_rag_chain(vectorstore)
    
    return chain

def ask_question(chain, question):
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    return answer, sources