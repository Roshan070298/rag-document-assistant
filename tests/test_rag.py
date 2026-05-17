# tests/test_rag.py

def test_chunk_splitting():
    """Test that text splitting works correctly"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10
    )
    
    text = "Hello world. " * 50  # Long text
    from langchain_core.documents import Document
    docs = [Document(page_content=text)]
    chunks = splitter.split_documents(docs)
    
    # Chunks bane hain?
    assert len(chunks) > 1
    
    # Har chunk 100 chars se zyada nahi
    for chunk in chunks:
        assert len(chunk.page_content) <= 120  # slight buffer

def test_environment():
    """Test that required packages are importable"""
    import langchain
    import chromadb
    import streamlit
    assert True  # Agar import hua toh pass