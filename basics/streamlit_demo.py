from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.globals import set_debug

import streamlit as st

set_debug(True)
load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
# llm = ChatOpenAI(model="gpt-5-nano", api_key=OPENAI_API_KEY)   
llm = ChatOllama(model="gemma:2b")

st.title("Ask Anything")

question = st.text_input("Enter the question:  ")
if question:
    response = llm.invoke(question)
    st.write(response.content)