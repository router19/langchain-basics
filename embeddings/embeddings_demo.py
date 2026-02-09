import os

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# model = gpt-4o-mini
llm = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
question = input("Enter the text:  ")
response = llm.embed_query(question)
print(response)