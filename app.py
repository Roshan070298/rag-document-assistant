import streamlit as st
import tempfile
import os
from rag import process_pdf, ask_question

# Page config
st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Ask My Docs")
st.caption("Upload a PDF and ask anything about it")

# Session state — chain aur history store karne ke liye
if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# PDF Upload
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Pehle se process nahi hua toh process karo
    if st.session_state.chain is None:
        with st.spinner("Processing PDF... please wait ⏳"):
            # Temp file banao — Streamlit direct path nahi deta
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # RAG chain banao
            st.session_state.chain = process_pdf(tmp_path)
            os.unlink(tmp_path)  # Temp file delete karo

        st.success("PDF processed! Ask your questions below 🎉")

# Chat interface
if st.session_state.chain is not None:

    # Chat history dikhao
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            with st.expander("📚 Sources"):
                for i, source in enumerate(chat["sources"]):
                    st.caption(f"Chunk {i+1}: {source.page_content[:200]}...")

    # Question input
    question = st.chat_input("Ask something about your PDF...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                answer, sources = ask_question(
                    st.session_state.chain, question
                )
            st.write(answer)
            with st.expander("📚 Sources"):
                for i, source in enumerate(sources):
                    st.caption(
                        f"Chunk {i+1}: {source.page_content[:200]}..."
                    )

        # History mein save karo
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": sources
        })

# New PDF button
if st.session_state.chain is not None:
    if st.button("🔄 Upload New PDF"):
        st.session_state.chain = None
        st.session_state.chat_history = []
        st.rerun()