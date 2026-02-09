from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
import streamlit as st




load_dotenv()
  
llm = ChatOllama(model="llama3.2:3b")

title_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
       You are an experienced speech writer.
        You need to craft an impactful title for a speech
        on the following topic: {topic}
        Answer exactly with one title.
        """)

speech_prompt = PromptTemplate(
    input_variables=["title","emotion"],
    template="""
       You need to write a powerful {emotion} speech of atleast 350 words for the following title: {title}
       Return ONLY valid JSON format with exactly these 2 keys. Do not include any other text.
       Example format:
       {{"title": "Your Title Here", "speech": "Your speech text here..."}}
        """)

first_chain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title),title)[1]) 
second_chain = speech_prompt | llm | JsonOutputParser() 
final_chain = first_chain | (lambda title : {"title": title, "emotion": emotion}) | second_chain

st.title("Speech Generator")

topic = st.text_input("Enter the speech topic:  ")
emotion = st.selectbox("Select the emotion for the speech: ", ["Inspiring", "Motivational", "Emotional", "Humorous"])


if topic and emotion:
    response = final_chain.invoke({
        "topic": topic
        })
    st.write(response)