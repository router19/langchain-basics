from langchain_ollama import OllamaEmbeddings
import numpy as np

llm = OllamaEmbeddings(model="llama3.2")

text = input("Enter the text1:  ")

response = llm.embed_documents(
    [
        "Hello World",
        "Hi there",
        "How are you doing?"
    ]
)
print(len(response))
print(response[0])                                                     