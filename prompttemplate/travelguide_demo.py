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
    input_variables=["city","month","language","budget"],
    template="""
       Welcome to the {city} travel guide!
        If you're visiting in {month}, here's what you can do:
        1. Must-visit attractions.
        2. Local cuisine you must try.
        3. Useful phrases in {language}.
        4. Tips for traveling on a {budget} budget.
        Enjoy your trip!
        """)
st.title("Travel Guide")

city = st.text_input("Enter the city:  ")
month = st.text_input("Enter the month:  ")
language = st.text_input("Enter the language:")
budget = st.selectbox("Select the budget:  ", options=["Low", "Medium", "High"])
if city and month and language and budget:
    response = llm.invoke(prompt_template.format(
        city=city, 
        month=month, 
        language=language,
        budget=budget
        ))
    st.write(response.content)