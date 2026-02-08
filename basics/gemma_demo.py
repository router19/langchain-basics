import os

from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# model = gpt-4o-mini
llm = ChatOllama(model="gemma:2b")
question = input("Enter the question:  ")
response = llm.invoke(question)
print(response.content)