from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st




load_dotenv()
  
llm = ChatOllama(model="llama3.2:3b")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a Agine Coach.Answer any questions "
         "related to agile process"),
         MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ]
)


st.title("Agile Guide")

input = st.text_input("Enter the question:  ")


chain = prompt_template | llm

history_for_chain = StreamlitChatMessageHistory()

chain_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history")

if input:
    response = chain_history.invoke(
        {"input": input},
        config={"configurable": {"session_id": "default"}}
    )
    st.write(response.content)

st.write("Chat History:")
st.write(history_for_chain.messages)