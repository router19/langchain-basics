from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
# from langchain_core.globals import set_debug
from langchain_core.prompts import PromptTemplate
import streamlit as st



# set_debug(True)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
llm = ChatOpenAI(model="gpt-5-nano", api_key=OPENAI_API_KEY)   
# llm = ChatOllama(model="gemma:2b")

prompt_template = PromptTemplate(
    input_variables=["country","no_of_paragraphs","language"],
    template="""
        You are an expert in traditional cuisines.
        You provide information about a specific dish from a specific country.
        Avoid giving information about fictional places. If the country is fictional
        or non-existent answer: I don't know.
        Answer the question: What is the traditional cuisine of {country}?
        Answer in {no_of_paragraphs} paragraphs and in {language} language.
        """)
st.title("Cuisine Explorer")

country = st.text_input("Enter the country:  ")
no_of_paragraphs = st.number_input("Enter the number of paragraphs:  ", min_value=1, max_value=5, value=3)
language = st.selectbox("Select the language:  ", options=["Hindi","English", "Spanish", "French", "German", "Chinese"])
if country:
    response = llm.invoke(prompt_template.format(
        country=country, 
        no_of_paragraphs=no_of_paragraphs, 
        language=language  ))
    st.write(response.content)