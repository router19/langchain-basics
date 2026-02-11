from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate,MessagesPlaceholder
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    
llm = ChatOpenAI(model="gpt-5-nano", api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
# llm = ChatOllama(model="llama3.2")
# embedding = OllamaEmbeddings(model="llama3.2")

document = TextLoader("Legal_Document_Analysis_Data.txt").load()
text_splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitters.split_documents(document)

vector_store = Chroma.from_documents(chunks, embedding)
retreiver = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","""
            You are an assistant for answering questions.
            Use the provided context to respond.If the answer
            isn't clear, acknowledge that you don't know.
            Limit your response to three concise sentences.
            {context} """),
            MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ]
    )
history_aware_retriever = create_history_aware_retriever(llm, retreiver, prompt_template)
qa_chain = create_stuff_documents_chain(llm,prompt_template)
rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)    

history_for_chain = StreamlitChatMessageHistory()

chain_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
    )
st.write("Chat with Legal Document: ")
question = st.text_input("Enter your question: ")
if question:
    response = chain_history.invoke({"input": question}, {"configurable": {"session_id": "default"}})
    st.write(response["answer"])

