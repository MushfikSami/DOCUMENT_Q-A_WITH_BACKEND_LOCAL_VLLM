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
from pydantic import BaseModel 
from fastapi import FastAPI


LOCAL_API_BASE = "http://localhost:5000/v1"
LOCAL_MODEL = "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
vector_store = None 

llm=ChatOpenAI(
    base_url=LOCAL_API_BASE, 
    api_key="no_key", 
    model=LOCAL_MODEL) 

app=FastAPI() 

class QueryRequest(BaseModel):
    query:str 

@app.on_event('startup')
def startup_event():
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)    


@app.post('/ingest')
def ingest_documents():
    global vector_store 
    embedding=OllamaEmbeddings(model='nomic-embed-text')
    loader=PyPDFDirectoryLoader('./us_census')
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs=text_splitter.split_documents(docs)
    vector_store=FAISS.from_documents(final_docs, embedding) 
    return {"message": "Documents ingested and vector store created."}


@app.post('/query')
def query_documents(request:QueryRequest):
    if vector_store is None:
        return {"error": "Vector store not initialized. Please ingest documents first."}
    
    prompt=ChatPromptTemplate.from_template('''
    Answer the question based only on the provided context.
    If the answer is not in the context, say "I don't know."
    <CONTEXT>: {context}
    <QUESTION>: {input}                                           
    '''
    ) 

    document_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever=vector_store.as_retriever()
    retriever_chain=create_retrieval_chain(retriever, document_chain)
    response=retriever_chain.invoke({'input': request.query})
    return {'answer': response['answer']}