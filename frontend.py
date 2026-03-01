import streamlit as st 
import requests 


BACKEND_URL = "http://localhost:8000"


st.title("End to End Document Q&A with Backend")
if st.button("Embed and Store Document"):
    response=requests.post(f"{BACKEND_URL}/ingest")
    if response.status_code==200:
        st.success(response.json()['message'])



query=st.text_input("Enter your question: ")
if query and st.button('Get Answer'):
    response=requests.post(f'{BACKEND_URL}/query',json={'query':query})
    if response.status_code==200:
        data=response.json()
        st.write("Answer:")
        st.write(data['answer'])