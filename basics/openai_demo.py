import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# model = gpt-4o-mini
llm = ChatOpenAI(model="gpt-5-nano", api_key=OPENAI_API_KEY)
question = input("Enter the question:  ")
response = llm.invoke(question)
print(response.content)