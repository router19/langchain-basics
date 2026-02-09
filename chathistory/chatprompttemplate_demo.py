from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st




load_dotenv()
  
llm = ChatOllama(model="llama3.2:3b")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a Agine Coach.Answer any questions "
         "related to agile process"),
        ("human","{input}")
    ]
)


st.title("Agile Guide")

input = st.text_input("Enter the question:  ")


chain = prompt_template | llm
if input:
    response = chain.invoke({"input": input})
    st.write(response.content)