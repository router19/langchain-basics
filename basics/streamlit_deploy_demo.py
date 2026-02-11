from langchain_openai import ChatOpenAI
import streamlit as st

st.title("Ask Anything")

with st.sidebar:
    st.title("Provide API Key")
    OPENAI_API_KEY = st.text_input("OpenAI API Key: ", type="password")
    
if not OPENAI_API_KEY:
    st.warning("Please provide your OpenAI API Key in the sidebar.")
    st.stop()


llm = ChatOpenAI(model="gpt-5-nano",api_key=OPENAI_API_KEY)



question = st.text_input("Enter the question:  ")
if question:
    response = llm.invoke(question)
    st.write(response.content)