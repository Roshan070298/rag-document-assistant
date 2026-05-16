# rag-document-assistant

A simple Streamlit-based Retrieval-Augmented Generation (RAG) assistant for PDF documents.

## What it does

- Uploads a PDF via a web UI.
- Splits the PDF into chunks using `langchain_text_splitters.RecursiveCharacterTextSplitter`.
- Creates sentence embeddings for each chunk with `HuggingFaceEmbeddings`.
- Stores embeddings locally in `chroma_db` using `Chroma`.
- Uses `ChatGroq` from `langchain_groq` to answer user queries over the relevant PDF chunks.
- Displays answers with source chunks so users can inspect where responses came from.

## Core files

- `app.py` - Streamlit app for uploading PDFs, asking questions, and showing chat history.
- `rag.py` - PDF loading, chunking, embedding creation, vector store setup, and RAG chain construction.
- `requirements.txt` - Python dependency list.
- `chroma_db/` - Local Chroma database directory containing persisted embeddings.

## Technical details

### Document processing

- PDF loading: `langchain_community.document_loaders.PyPDFLoader`
- Splitting: `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
- Embeddings: `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`
- Vector store: `Chroma.from_documents(...)` persisted in `./chroma_db`

### RAG chain

- Retriever: `vectorstore.as_retriever(search_kwargs={"k": 3})`
- QA chain: `RetrievalQA.from_chain_type(...)`
- LLM: `ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant", temperature=0)`

### Runtime behavior

- User uploads a PDF; `app.py` saves it to a temporary file.
- `process_pdf` builds or refreshes the local vector store with document embeddings.
- User queries are answered by retrieval-enhanced generation with source documents returned.
- Chat history is maintained in Streamlit session state.

## Setup

1. Create a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set `GROQ_API_KEY` in your environment.
4. Run the app:

```bash
streamlit run app.py
```

## Notes

- The app deletes and recreates `./chroma_db` on each new PDF upload.
- The source content shown in the UI is limited to the first 200 characters of each chunk.
- This project targets local RAG workflows with a cloud LLM backend via Groq.
