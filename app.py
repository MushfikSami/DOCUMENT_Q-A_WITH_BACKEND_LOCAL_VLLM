import streamlit as st 
import os 
import threading
import subprocess
# Correct imports
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import ChatOpenAI  # Use LangChain's wrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

LOCAL_API_BASE = "http://localhost:5000/v1"
LOCAL_MODEL = "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"

# 1. Improved Server Start (prevents multiple triggers)
@st.cache_resource
def start_ollama_server():
    try:
        # Check if ollama is already running or just try to start it once
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "Server initialized"
    except Exception as e:
        return f"Error: {e}"

start_ollama_server()

st.title("📄 Document Q&A (Local Qwen3 + Ollama)")

# 2. Initialize the LLM via LangChain's OpenAI wrapper
llm = ChatOpenAI(
    base_url=LOCAL_API_BASE, 
    api_key="no_key", 
    model=LOCAL_MODEL
)

# 3. Define the Prompt
prompt = ChatPromptTemplate.from_template('''
Answer the question based only on the provided context.
If the answer is not in the context, say "I don't know."

<CONTEXT>: {context}
<QUESTION>: {input}
''')

def vector_embedding():
    if 'vector' not in st.session_state:
        with st.spinner("Processing documents..."):
            st.session_state.embeddings = OllamaEmbeddings(model='nomic-embed-text')
            st.session_state.loader = PyPDFDirectoryLoader('./us_census')
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vector = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
            st.success('Vector store created!')

if st.button('Generate Embeddings'):
    vector_embedding()

user_question = st.text_input('Ask a question about the document:') 

if user_question:
    if 'vector' not in st.session_state:
        st.warning('Please generate embeddings first!')
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        retriever = st.session_state.vector.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        with st.spinner("Thinking..."):
            response = retrieval_chain.invoke({"input": user_question})
            st.write('### Answer:')
            st.write(response['answer'])